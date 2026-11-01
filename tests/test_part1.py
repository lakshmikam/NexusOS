import sqlite3
from contextlib import contextmanager
import pytest
from fastapi.testclient import TestClient
from nexus.api import create_app
from nexus.db import connect, initialize, seed_demo
from nexus.auth import DEMO_OPERATOR_PASSWORD


@contextmanager
def operator_client(database):
    """Part 1 billing tests now exercise the same records through authorized access."""
    with TestClient(create_app(database)) as client:
        response=client.post('/api/auth/login',json={'email':'support@example.test','password':DEMO_OPERATOR_PASSWORD})
        assert response.status_code==200
        client.headers['Authorization']='Bearer '+response.json()['access_token']
        yield client


@pytest.fixture
def database(tmp_path):
    path=tmp_path/'nexus.db'
    initialize(path)
    seed_demo(path)
    return path


def test_startup_and_seed_are_non_destructive(database):
    initialize(database)
    seed_demo(database)
    with connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM accounts').fetchone()[0]==3
        assert db.execute('SELECT COUNT(*) FROM invoices').fetchone()[0]==3
        assert db.execute('SELECT COUNT(*) FROM payments').fetchone()[0]==5
        assert db.execute('SELECT COUNT(*) FROM schema_migrations').fetchone()[0]==5


def test_accounts_and_health_are_explicitly_demo(database):
    with operator_client(database) as client:
        assert client.get('/health').json()['authentication']=='expiring bearer sessions'
        assert len(client.get('/api/accounts').json()['accounts'])==3
        assert client.get('/api/accounts/missing').status_code==404
        healthy=client.get('/api/accounts/acct_healthy').json()
        assert healthy['subscription']['status']=='active'
        assert all(e['enabled'] for e in healthy['entitlements'])


def test_duplicate_charge_and_missing_access_are_visible(database):
    with operator_client(database) as client:
        invoice=client.get('/api/accounts/acct_duplicate/invoices').json()['invoices'][0]
        assert invoice['amount_minor']==99900
        assert invoice['successful_paid_minor']==199800
        assert invoice['excess_paid_minor']==99900
        assert invoice['outstanding_minor']==0
        assert len({p['provider_reference'] for p in invoice['payments']})==2
        account=client.get('/api/accounts/acct_duplicate').json()
        assert account['subscription']['status']=='active'
        assert all(not e['enabled'] for e in account['entitlements'])


def test_failed_and_pending_attempts_do_not_count_as_paid(database):
    with operator_client(database) as client:
        invoice=client.get('/api/accounts/acct_pending/invoices').json()['invoices'][0]
        assert len(invoice['payments'])==2
        assert invoice['successful_paid_minor']==0
        assert invoice['excess_paid_minor']==0
        assert invoice['outstanding_minor']==99900


def test_ticket_survives_app_restart(database):
    payload={'account_id':'acct_duplicate','subject':'Duplicate charge and missing access',
             'description':'I upgraded yesterday, was charged twice, and cannot access Pro features.'}
    with operator_client(database) as client:
        response=client.post('/api/tickets',json=payload)
        assert response.status_code==201
        ticket=response.json()
        assert ticket['status']=='open'
    with operator_client(database) as restarted:
        assert restarted.get('/api/tickets/'+ticket['id']).json()==ticket
        assert restarted.get('/api/accounts/acct_duplicate/tickets').json()['tickets']==[ticket]


def test_ticket_input_is_validated_and_sql_text_is_data(database):
    with operator_client(database) as client:
        assert client.post('/api/tickets',json={'account_id':'acct_healthy','subject':'x','description':'short'}).status_code==422
        payload={'account_id':'missing','subject':'Example ticket','description':'A sufficiently long description.'}
        assert client.post('/api/tickets',json=payload).status_code==404
        payload.update(account_id='acct_healthy',status='resolved')
        assert client.post('/api/tickets',json=payload).status_code==422
        assert client.get("/api/accounts/anything' OR '1'='1").status_code==404
        assert client.get('/api/tickets/missing').status_code==404


def test_payment_cannot_reference_another_accounts_invoice(database):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(database) as db:
            db.execute("INSERT INTO payments(id,invoice_id,account_id,provider_reference,amount_minor,status,received_at) VALUES (?,?,?,?,?,?,?)",
                       ('bad','inv_healthy','acct_duplicate','bad-ref',99900,'succeeded','2026-11-01T00:00:00Z'))


def test_invalid_money_rolls_back_the_transaction(database):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(database) as db:
            db.execute('INSERT INTO accounts(id,name,email) VALUES (?,?,?)',('rollback','Demo','rollback@example.test'))
            db.execute('INSERT INTO invoices(id,account_id,reference,amount_minor,status,issued_at) VALUES (?,?,?,?,?,?)',
                       ('bad','rollback','BAD',-1,'open','2026-11-01T00:00:00Z'))
    with connect(database) as db:
        assert db.execute("SELECT * FROM accounts WHERE id='rollback'").fetchone() is None


def test_empty_app_can_start_without_seed_data(tmp_path):
    path=tmp_path/'empty.db'
    with TestClient(create_app(path,demo=False)) as client:
        assert client.get('/health').status_code==200
        assert client.get('/api/accounts').status_code==401
    with connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM users').fetchone()[0]==0
