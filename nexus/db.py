"""SQLite connections, transactional migrations and non-destructive demo seeding."""
import argparse
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_path():
    return Path(os.environ.get('NEXUS_DB_PATH', str(ROOT / 'data' / 'nexus.db')))


@contextmanager
def connect(path):
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
        versions = {row['version'] for row in db.execute('SELECT version FROM schema_migrations')}
        for migration in sorted((ROOT / 'nexus' / 'migrations').glob('*.sql')):
            if migration.stem in versions:
                continue
            statement = ''
            for line in migration.read_text(encoding='utf-8').splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    db.execute(statement)
                    statement = ''
            if statement.strip():
                raise ValueError(f'Incomplete SQL in {migration.name}')
            db.execute('INSERT INTO schema_migrations(version) VALUES (?)', (migration.stem,))


def seed_demo(path):
    """Fixed synthetic identities. Existing records and tickets are never reset."""
    with connect(path) as db:
        for account_id, name, email in [
            ('acct_healthy', 'Meera Shah', 'meera@example.test'),
            ('acct_pending', 'Kabir Sen', 'kabir@example.test'),
            ('acct_duplicate', 'Asha Rao', 'asha@example.test'),
        ]:
            db.execute('INSERT OR IGNORE INTO accounts(id,name,email) VALUES (?,?,?)', (account_id,name,email))
            suffix = account_id.removeprefix('acct_')
            db.execute('INSERT OR IGNORE INTO subscriptions(id,account_id,plan,status,monthly_amount_minor) VALUES (?,?,?,?,?)',
                       (f'sub_{suffix}',account_id,'pro','pending' if suffix == 'pending' else 'active',99900))
            db.execute('INSERT OR IGNORE INTO invoices(id,account_id,reference,amount_minor,status,issued_at) VALUES (?,?,?,?,?,?)',
                       (f'inv_{suffix}',account_id,f'ORB-2026-{suffix.upper()}',99900,'open' if suffix == 'pending' else 'paid','2026-10-31T10:00:00Z'))
            for feature in ('team_workspace','advanced_reports'):
                db.execute('INSERT OR IGNORE INTO entitlements(account_id,feature,enabled) VALUES (?,?,?)',
                           (account_id,feature,int(suffix == 'healthy')))
            payment_states = ['succeeded','succeeded'] if suffix == 'duplicate' else ['pending','failed'] if suffix == 'pending' else ['succeeded']
            for index, status in enumerate(payment_states, 1):
                db.execute('INSERT OR IGNORE INTO payments(id,invoice_id,account_id,provider_reference,amount_minor,status,received_at) VALUES (?,?,?,?,?,?,?)',
                           (f'pay_{suffix}_{index}',f'inv_{suffix}',account_id,f'sim_{suffix}_{index}',99900,status,f'2026-10-31T10:0{index}:00Z'))
    # Seed users only after the account transaction commits. Existing hashes are preserved.
    from nexus.auth import seed_demo_users
    seed_demo_users(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=default_path())
    parser.add_argument('--seed', action='store_true')
    args = parser.parse_args()
    initialize(args.path)
    if args.seed:
        seed_demo(args.path)
    print(f'Database ready: {args.path.resolve()}')


if __name__ == '__main__':
    main()
