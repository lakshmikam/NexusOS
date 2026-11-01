import sqlite3
import pytest
from fastapi.testclient import TestClient
from nexus import auth
from nexus.api import create_app
from nexus.db import connect, initialize, seed_demo


@pytest.fixture
def database(tmp_path):
    path=tmp_path/'auth.db'
    initialize(path)
    seed_demo(path)
    return path


@pytest.fixture
def client(database):
    with TestClient(create_app(database)) as client:
        yield client


def sign_in(client,email='asha@example.test',password=auth.DEMO_CUSTOMER_PASSWORD):
    response=client.post('/api/auth/login',json={'email':email,'password':password})
    assert response.status_code==200,response.text
    return response.json()


def headers(login):
    return {'Authorization':'Bearer '+login['access_token']}


def test_password_hashes_are_salted_and_verified():
    first=auth.hash_password(auth.DEMO_CUSTOMER_PASSWORD)
    second=auth.hash_password(auth.DEMO_CUSTOMER_PASSWORD)
    assert first!=second
    assert auth.DEMO_CUSTOMER_PASSWORD not in first
    assert auth.verify_password(auth.DEMO_CUSTOMER_PASSWORD,first)
    assert not auth.verify_password('wrong-password',first)
    assert not auth.verify_password('wrong-password','invalid-hash')


def test_login_stores_only_token_digest_and_no_password_leaks(client,database):
    result=sign_in(client)
    assert result['token_type']=='bearer' and result['expires_in']==3600
    assert result['user']['role']=='customer' and result['user']['account_id']=='acct_duplicate'
    assert 'password' not in result and 'password_hash' not in result['user']
    assert len(result['access_token'])==43
    with connect(database) as db:
        row=db.execute('SELECT * FROM sessions').fetchone()
        assert row['token_hash']==auth.token_digest(result['access_token'])
        assert row['token_hash']!=result['access_token']
        assert db.execute('SELECT password_hash FROM users WHERE id=?',('usr_asha',)).fetchone()[0].startswith('scrypt$')
    me=client.get('/api/auth/me',headers=headers(result))
    assert me.status_code==200 and me.json()==result['user']


@pytest.mark.parametrize('path',['/api/accounts','/api/accounts/acct_duplicate',
    '/api/accounts/acct_duplicate/invoices','/api/accounts/acct_duplicate/tickets',
    '/api/tickets/invented','/api/demo/scenarios','/api/auth/me'])
def test_business_records_require_a_session(client,path):
    response=client.get(path)
    assert response.status_code==401
    assert response.headers['www-authenticate']=='Bearer'


def test_customer_only_sees_owned_account_and_cannot_spoof_ticket_owner(client,database):
    session=headers(sign_in(client))
    assert client.get('/api/accounts',headers=session).json()['accounts'][0]['id']=='acct_duplicate'
    assert len(client.get('/api/accounts',headers=session).json()['accounts'])==1
    assert client.get('/api/accounts/acct_duplicate',headers=session).status_code==200
    assert client.get('/api/accounts/acct_duplicate/invoices',headers=session).status_code==200
    for path in ['/api/accounts/acct_healthy','/api/accounts/acct_healthy/invoices','/api/accounts/acct_healthy/tickets']:
        assert client.get(path,headers=session).status_code==404
    assert client.get('/api/demo/scenarios',headers=session).status_code==403
    ticket={'account_id':'acct_healthy','subject':'Spoofed owner','description':'A customer attempts to create another account ticket.'}
    assert client.post('/api/tickets',json=ticket,headers=session).status_code==404
    with connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM tickets').fetchone()[0]==0


def test_ticket_ownership_cannot_be_bypassed_by_guessing_its_id(client):
    asha=headers(sign_in(client))
    ticket=client.post('/api/tickets',headers=asha,json={'account_id':'acct_duplicate','subject':'Missing Pro access',
              'description':'My successful payment did not enable the features.'}).json()
    assert client.get('/api/tickets/'+ticket['id'],headers=asha).status_code==200
    meera=headers(sign_in(client,'meera@example.test'))
    hidden=client.get('/api/tickets/'+ticket['id'],headers=meera)
    missing=client.get('/api/tickets/invented',headers=meera)
    assert hidden.status_code==missing.status_code==404
    assert hidden.json()==missing.json()
    operator=headers(sign_in(client,'support@example.test',auth.DEMO_OPERATOR_PASSWORD))
    assert client.get('/api/tickets/'+ticket['id'],headers=operator).status_code==200
    assert len(client.get('/api/accounts',headers=operator).json()['accounts'])==3
    assert client.get('/api/demo/scenarios',headers=operator).status_code==200


def test_wrong_unknown_and_disabled_logins_have_generic_errors(client,database):
    wrong=client.post('/api/auth/login',json={'email':'asha@example.test','password':'wrong'})
    unknown=client.post('/api/auth/login',json={'email':'missing@example.test','password':'wrong'})
    with connect(database) as db:
        db.execute('UPDATE users SET is_active=0 WHERE id=?',('usr_asha',))
    disabled=client.post('/api/auth/login',json={'email':'asha@example.test','password':auth.DEMO_CUSTOMER_PASSWORD})
    assert wrong.status_code==unknown.status_code==disabled.status_code==401
    assert wrong.json()==unknown.json()==disabled.json()


def test_expiration_tampering_and_logout(client,monkeypatch):
    result=sign_in(client)
    original=auth.now()
    bad=('A' if result['access_token'][0]!='A' else 'B')+result['access_token'][1:]
    assert client.get('/api/auth/me',headers={'Authorization':'Bearer '+bad}).status_code==401
    monkeypatch.setattr(auth,'now',lambda:original+3601)
    assert client.get('/api/auth/me',headers=headers(result)).status_code==401
    monkeypatch.setattr(auth,'now',lambda:original)
    assert client.post('/api/auth/logout',headers=headers(result)).status_code==200
    assert client.get('/api/auth/me',headers=headers(result)).status_code==401


def test_disabling_user_invalidates_existing_session(client,database):
    session=headers(sign_in(client))
    with connect(database) as db:
        db.execute('UPDATE users SET is_active=0 WHERE id=?',('usr_asha',))
    assert client.get('/api/accounts',headers=session).status_code==401


def test_login_throttling_and_window_reset(client,monkeypatch):
    instant=auth.now()
    # Exercise the intended window independently of machine load / password hashing time.
    monkeypatch.setattr(auth,'now',lambda:instant)
    for _ in range(auth.LOGIN_LIMIT):
        assert client.post('/api/auth/login',json={'email':'asha@example.test','password':'wrong'}).status_code==401
    blocked=client.post('/api/auth/login',json={'email':'asha@example.test','password':auth.DEMO_CUSTOMER_PASSWORD})
    assert blocked.status_code==429 and int(blocked.headers['retry-after'])>0
    monkeypatch.setattr(auth,'now',lambda:instant+auth.LOGIN_WINDOW+1)
    sign_in(client)


def test_sessions_and_existing_passwords_survive_restart(database):
    with TestClient(create_app(database)) as first:
        result=sign_in(first)
    with connect(database) as db:
        before=db.execute('SELECT password_hash FROM users WHERE id=?',('usr_asha',)).fetchone()[0]
    with TestClient(create_app(database)) as restarted:
        assert restarted.get('/api/auth/me',headers=headers(result)).status_code==200
    with connect(database) as db:
        assert db.execute('SELECT password_hash FROM users WHERE id=?',('usr_asha',)).fetchone()[0]==before
        assert db.execute('SELECT COUNT(*) FROM users').fetchone()[0]==4


def test_role_injection_rejected_and_login_response_not_cached(client):
    injected=client.post('/api/auth/login',json={'email':'asha@example.test','password':auth.DEMO_CUSTOMER_PASSWORD,'role':'operator'})
    assert injected.status_code==422
    result=client.post('/api/auth/login',json={'email':' ASHA@EXAMPLE.TEST ','password':auth.DEMO_CUSTOMER_PASSWORD})
    assert result.status_code==200 and result.json()['user']['role']=='customer'
    assert result.headers['cache-control']=='no-store'


def test_user_role_and_account_relationship_enforced(database):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(database) as db:
            db.execute('INSERT INTO users(id,email,name,password_hash,role,account_id) VALUES (?,?,?,?,?,?)',
                       ('invalid','bad@example.test','Bad','irrelevant','customer',None))
