"""Password hashing, opaque database sessions, and reusable authorization checks."""
import hashlib
import hmac
import re
import secrets
import time
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from nexus.db import connect

SCRYPT_N, SCRYPT_R, SCRYPT_P = 32768, 8, 1
SESSION_TTL = 3600
LOGIN_WINDOW = 60
LOGIN_LIMIT = 5
DEMO_CUSTOMER_PASSWORD = 'OrbitDesk-Demo-2026!'
DEMO_OPERATOR_PASSWORD = 'OrbitDesk-Operator-2026!'
bearer = HTTPBearer(auto_error=False, scheme_name='SessionToken',
                    description='Paste access_token from POST /api/auth/login. Expires after one hour by default.')


def now():
    return int(time.time())


def hash_password(password):
    if not 12 <= len(password) <= 256:
        raise ValueError('Provisioned passwords must contain 12..256 characters')
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                            p=SCRYPT_P, dklen=32, maxmem=64*1024*1024)
    return f'scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}'


def verify_password(password, encoded):
    if not 1 <= len(password) <= 256:
        return False
    try:
        scheme, n, r, p, salt, expected = encoded.split('$')
        if (scheme, int(n), int(r), int(p)) != ('scrypt', SCRYPT_N, SCRYPT_R, SCRYPT_P):
            return False
        salt, expected = bytes.fromhex(salt), bytes.fromhex(expected)
        if len(salt) != 16 or len(expected) != 32:
            return False
        actual = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                                p=SCRYPT_P, dklen=32, maxmem=64*1024*1024)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


# Unknown addresses still perform one password verification, with the same generic error.
DUMMY_HASH = hash_password('This-is-a-dummy-password!')


class Principal(BaseModel):
    id: str
    email: str
    name: str
    role: Literal['customer', 'operator']
    account_id: str | None


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value):
        value = value.strip().lower()
        if '@' not in value:
            raise ValueError('Enter an email address')
        return value


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    expires_in: int
    expires_at: int
    user: Principal


def unauthorized():
    return HTTPException(401, 'Missing, invalid, expired, or revoked session',
                         headers={'WWW-Authenticate': 'Bearer'})


def token_digest(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def throttle(path, scope):
    instant = now()
    key = hashlib.sha256(scope.encode('utf-8')).hexdigest()
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM login_throttles WHERE scope = ?', (key,)).fetchone()
        if row is None or instant - row['window_started'] >= LOGIN_WINDOW:
            db.execute('INSERT INTO login_throttles(scope,window_started,attempts) VALUES (?,?,1) '
                       'ON CONFLICT(scope) DO UPDATE SET window_started=excluded.window_started, attempts=1', (key, instant))
        elif row['attempts'] >= LOGIN_LIMIT:
            retry = max(1, LOGIN_WINDOW - (instant-row['window_started']))
            raise HTTPException(429, 'Too many login attempts; try again shortly', headers={'Retry-After': str(retry)})
        else:
            db.execute('UPDATE login_throttles SET attempts=attempts+1 WHERE scope = ?', (key,))
    return key


def login(path, payload, client_host, ttl=SESSION_TTL):
    scope = throttle(path, payload.email + '\0' + client_host)
    with connect(path) as db:
        row = db.execute('SELECT * FROM users WHERE email = ?', (payload.email,)).fetchone()
    valid = verify_password(payload.password, row['password_hash'] if row else DUMMY_HASH)
    if not row or not valid or not row['is_active']:
        raise HTTPException(401, 'Invalid email or password', headers={'WWW-Authenticate': 'Bearer'})
    token = secrets.token_urlsafe(32)
    created = now()
    with connect(path) as db:
        db.execute('INSERT INTO sessions(token_hash,user_id,created_at,expires_at) VALUES (?,?,?,?)',
                   (token_digest(token),row['id'],created,created+ttl))
        db.execute('DELETE FROM login_throttles WHERE scope = ?', (scope,))
    return LoginResponse(access_token=token, expires_in=ttl, expires_at=created+ttl,
                         user=Principal(**dict(row)))


def authenticated_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if credentials is None or credentials.scheme.lower() != 'bearer' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', credentials.credentials):
        raise unauthorized()
    with connect(request.app.state.db_path) as db:
        row = db.execute('SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id '
                         'WHERE s.token_hash=? AND s.expires_at>? AND s.revoked_at IS NULL AND u.is_active=1',
                         (token_digest(credentials.credentials),now())).fetchone()
    if row is None:
        raise unauthorized()
    return Principal(**dict(row))


def require_account_access(user, account_id):
    # Return the same error as an unknown account to avoid disclosing its existence.
    if user.role != 'operator' and user.account_id != account_id:
        raise HTTPException(404, 'Account not found')


def require_operator(user: Principal = Depends(authenticated_user)):
    if user.role != 'operator':
        raise HTTPException(403, 'Operator access required')
    return user


def revoke(path, token):
    with connect(path) as db:
        db.execute('UPDATE sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL',
                   (now(),token_digest(token)))


def seed_demo_users(path):
    users = [
        ('usr_meera','meera@example.test','Meera Shah','customer','acct_healthy',DEMO_CUSTOMER_PASSWORD),
        ('usr_kabir','kabir@example.test','Kabir Sen','customer','acct_pending',DEMO_CUSTOMER_PASSWORD),
        ('usr_asha','asha@example.test','Asha Rao','customer','acct_duplicate',DEMO_CUSTOMER_PASSWORD),
        ('usr_operator','support@example.test','Demo Support Operator','operator',None,DEMO_OPERATOR_PASSWORD),
    ]
    with connect(path) as db:
        for user_id,email,name,role,account_id,password in users:
            if db.execute('SELECT 1 FROM users WHERE id=? OR email=?', (user_id,email)).fetchone():
                continue
            db.execute('INSERT INTO users(id,email,name,password_hash,role,account_id) VALUES (?,?,?,?,?,?)',
                       (user_id,email,name,hash_password(password),role,account_id))
