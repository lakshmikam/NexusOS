"""Owner-scoped durable conversation history; it does not change model weights."""
import json
import sqlite3
import time
import uuid
from fastapi import HTTPException
from nexus.auth import require_account_access
from nexus.db import connect


def owned_conversation(db,user,conversation_id):
    row=db.execute('SELECT * FROM conversations WHERE id=? AND owner_user_id=?',(conversation_id,user.id)).fetchone()
    if row is None or (user.role!='operator' and row['account_id']!=user.account_id):
        raise HTTPException(404,'Conversation not found')
    return dict(row)


class ConversationMemory:
    def __init__(self,path): self.path=path

    def create(self,user,title,account_id=None):
        account_id=account_id or user.account_id
        if account_id:
            require_account_access(user,account_id)
        instant=int(time.time())
        conversation_id='conv_'+uuid.uuid4().hex
        with connect(self.path) as db:
            if account_id and not db.execute('SELECT 1 FROM accounts WHERE id=?',(account_id,)).fetchone():
                raise HTTPException(404,'Account not found')
            db.execute('INSERT INTO conversations(id,owner_user_id,account_id,title,created_at,updated_at) VALUES (?,?,?,?,?,?)',
                       (conversation_id,user.id,account_id,title,instant,instant))
            return owned_conversation(db,user,conversation_id)

    def list(self,user):
        with connect(self.path) as db:
            rows=db.execute('SELECT * FROM conversations WHERE owner_user_id=? ORDER BY updated_at DESC,id',(user.id,))
            return [dict(r) for r in rows if user.role=='operator' or r['account_id']==user.account_id]

    def inspect(self,user,conversation_id):
        with connect(self.path) as db:
            conversation=owned_conversation(db,user,conversation_id)
            turns=[dict(r) for r in db.execute('SELECT * FROM conversation_turns WHERE conversation_id=? ORDER BY created_at,rowid',(conversation_id,))]
        for turn in turns:
            raw=turn.pop('answer_json')
            turn['answer']=json.loads(raw) if raw else None
        return {'conversation':conversation,'turns':turns}

    def snapshot(self,user,conversation_id):
        if not conversation_id:
            return {'conversation_id':None,'revision':None,'history':[],'stored_completed_turns':0}
        with connect(self.path) as db:
            conversation=owned_conversation(db,user,conversation_id)
            if db.execute("SELECT 1 FROM conversation_turns WHERE conversation_id=? AND status='running'",(conversation_id,)).fetchone():
                raise HTTPException(409,'An answer is already running in this conversation')
            count=db.execute("SELECT COUNT(*) FROM conversation_turns WHERE conversation_id=? AND status='completed'",(conversation_id,)).fetchone()[0]
            rows=db.execute("SELECT question,answer_json FROM conversation_turns WHERE conversation_id=? AND status='completed' ORDER BY created_at DESC,rowid DESC LIMIT 3",(conversation_id,)).fetchall()
        history=[]
        used=0
        for row in rows:
            answer=json.loads(row['answer_json'])
            item={'question':row['question'],'previous_answer':answer['answer']}
            size=len(json.dumps(item,ensure_ascii=False))
            if used+size>4000: break
            history.append(item)
            used+=size
        return {'conversation_id':conversation_id,'revision':conversation['revision'],
                'history':list(reversed(history)),'stored_completed_turns':count}

    def begin(self,user,snapshot,question):
        turn_id='turn_'+uuid.uuid4().hex
        if not snapshot['conversation_id']: return turn_id
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            conversation=owned_conversation(db,user,snapshot['conversation_id'])
            if conversation['revision']!=snapshot['revision']:
                raise HTTPException(409,'Conversation changed while preparing the answer; retry the request')
            try:
                db.execute("INSERT INTO conversation_turns(id,conversation_id,question,status,created_at) VALUES (?,?,?,'running',?)",
                    (turn_id,snapshot['conversation_id'],question,int(time.time())))
            except sqlite3.IntegrityError as exc:
                raise HTTPException(409,'An answer is already running in this conversation') from exc
            db.execute('UPDATE conversations SET revision=revision+1,updated_at=? WHERE id=?',(int(time.time()),snapshot['conversation_id']))
        return turn_id

    def finish(self,turn_id,result=None,error=None):
        with connect(self.path) as db:
            row=db.execute("SELECT conversation_id FROM conversation_turns WHERE id=? AND status='running'",(turn_id,)).fetchone()
            if row is None: return
            db.execute('UPDATE conversation_turns SET status=?,answer_json=?,error_code=?,completed_at=? WHERE id=?',
                       ('completed' if result is not None else 'failed',json.dumps(result,allow_nan=False) if result is not None else None,
                        error,int(time.time()),turn_id))
            db.execute('UPDATE conversations SET revision=revision+1,updated_at=? WHERE id=?',(int(time.time()),row['conversation_id']))

    def recover_interrupted(self):
        # This local application supports one server worker. No job is silently retried.
        with connect(self.path) as db:
            affected=[r['conversation_id'] for r in db.execute("SELECT conversation_id FROM conversation_turns WHERE status='running'")]
            db.execute("UPDATE conversation_turns SET status='failed',error_code='server_restarted',completed_at=? WHERE status='running'",(int(time.time()),))
            for conversation_id in affected:
                db.execute('UPDATE conversations SET revision=revision+1 WHERE id=?',(conversation_id,))
