"""Inspectable policy ingestion, BM25, local dense retrieval and reciprocal rank fusion."""
import argparse
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from nexus.db import ROOT, connect, default_path, initialize
from nexus.embeddings import LocalEncoder, RetrievalUnavailable

STOP_WORDS = set('a an the and or of to for is are was were be been in on at with can could would should do does did it that this me we our you your i my am as from by have has had what how when which'.split())


def today_india():
    return datetime.now(timezone(timedelta(hours=5,minutes=30))).date()


def tokenize(text):
    return [word for word in re.findall(r'[a-z0-9_]+',text.lower()) if word not in STOP_WORDS]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_policy(path):
    path=Path(path)
    if path.stat().st_size>32000:
        raise ValueError(f'{path.name}: policy exceeds the 32 KB demo limit')
    source=path.read_text(encoding='utf-8')
    metadata, sections = {}, []
    title, heading, lines, body_started = '', 'Overview', [], False
    for line in source.splitlines():
        if line.startswith('# ') and not title:
            title=line[2:].strip()
            continue
        match=re.fullmatch(r'(Policy ID|Version|Effective from|Effective until|Product):\s*(.+)',line)
        if not body_started and match:
            metadata[match[1]]=match[2].strip()
            continue
        if not line.strip() and not body_started:
            continue
        body_started=True
        if line.startswith('## '):
            if '\n'.join(lines).strip():
                sections.append((heading,'\n'.join(lines).strip()))
            heading,lines=line[3:].strip(),[]
        else:
            lines.append(line)
    if '\n'.join(lines).strip():
        sections.append((heading,'\n'.join(lines).strip()))
    required=('Policy ID','Version','Effective from')
    if not title or not sections or any(key not in metadata for key in required):
        raise ValueError(f'{path.name}: missing title, body, or policy metadata')
    policy_id=metadata['Policy ID']
    match=re.fullmatch(r'([a-z][a-z0-9-]{0,60})-v([1-9][0-9]*)',policy_id)
    version=int(metadata['Version'])
    if not match or int(match[2])!=version:
        raise ValueError(f'{path.name}: policy ID version and Version field must agree')
    effective=date.fromisoformat(metadata['Effective from']).isoformat()
    until=date.fromisoformat(metadata['Effective until']).isoformat() if metadata.get('Effective until') else None
    if until and until<=effective:
        raise ValueError(f'{path.name}: end date must follow effective date')
    return {'policy_id':policy_id,'family':match[1],'version':version,'title':title,'source':path.name,
            'source_hash':digest(source),'effective_from':effective,'effective_until':until,'sections':sections}


def split_text(text,prefix,encoder,max_tokens,overlap_words=0):
    """Greedy word boundaries with actual embedding-token counts; never silently truncate."""
    words=list(re.finditer(r'\S+',text))
    start=0
    while start<len(words):
        end=start
        while end<len(words):
            candidate=text[words[start].start():words[end].end()]
            if encoder.count(prefix+'\n'+candidate)>max_tokens:
                break
            end+=1
        if end==start:
            raise ValueError('A heading or individual word exceeds the chunk budget')
        body=text[words[start].start():words[end-1].end()]
        yield body,encoder.count(prefix+'\n'+body)
        if end==len(words):
            break
        start=max(start+1,end-overlap_words)


def make_chunks(document,encoder,strategy='heading',max_tokens=180,overlap_words=12):
    if strategy not in ('heading','fixed') or not 40<=max_tokens<=220:
        raise ValueError('Use heading/fixed strategy and a 40..220 token budget')
    if not 0<=overlap_words<=30:
        raise ValueError('Overlap must be 0..30 words')
    sections=document['sections'] if strategy=='heading' else [('Whole document','\n\n'.join(text for _,text in document['sections']))]
    result=[]
    for heading,text in sections:
        prefix=document['title']+'\n'+heading
        for part,count in split_text(text,prefix,encoder,max_tokens,overlap_words if strategy=='fixed' else 0):
            chunk_id='chk_'+digest(document['policy_id']+document['source_hash']+strategy+heading+str(len(result))+part)[:20]
            result.append({'chunk_id':chunk_id,'policy_id':document['policy_id'],'heading':heading,
                           'text':part,'token_count':count,'embedding_text':prefix+'\n'+part})
    return result


def bm25_scores(query,passages,k1=1.5,b=.75):
    documents=[Counter(tokenize(text)) for text in passages]
    if not documents:
        return []
    lengths=[sum(doc.values()) for doc in documents]
    avg=sum(lengths)/len(lengths) or 1
    scores=[0.0]*len(documents)
    for term in set(tokenize(query)):
        df=sum(term in doc for doc in documents)
        if not df:
            continue
        idf=math.log(1+(len(documents)-df+.5)/(df+.5))
        for i,doc in enumerate(documents):
            frequency=doc.get(term,0)
            scores[i]+=idf*frequency*(k1+1)/(frequency+k1*(1-b+b*lengths[i]/avg))
    return scores


def eligible_documents(documents,as_of):
    """Choose latest effective version, then apply its expiry; never revive superseded versions."""
    grouped={}
    for doc in documents:
        if doc['effective_from']<=as_of:
            grouped.setdefault(doc['family'],[]).append(doc)
    chosen=[]
    for versions in grouped.values():
        latest=max(versions,key=lambda d:(d['effective_from'],d['version']))
        if not latest['effective_until'] or latest['effective_until']>as_of:
            chosen.append(latest['policy_id'])
    return set(chosen)


class RetrievalService:
    def __init__(self,path,policies_dir=None,encoder=None):
        self.path=Path(path)
        self.policies_dir=Path(policies_dir) if policies_dir else ROOT/'policies'
        self.encoder=encoder if encoder is not None else LocalEncoder()
        self._index_lock=threading.Lock()

    def _index_id(self,index_id=None):
        if index_id:
            return index_id
        with connect(self.path) as db:
            row=db.execute('SELECT active_index_id FROM retrieval_state WHERE id=1').fetchone()
        if row is None:
            raise RetrievalUnavailable('Policies are not indexed. Run python -m nexus.retrieval index or use the operator reindex endpoint.')
        return row['active_index_id']

    def status(self):
        try:
            index_id=self._index_id()
        except RetrievalUnavailable:
            return {'ready':False,'message':'Build the policy index first: python -m nexus.retrieval index'}
        with connect(self.path) as db:
            index=dict(db.execute('SELECT * FROM retrieval_indexes WHERE id=?',(index_id,)).fetchone())
            index['documents']=db.execute('SELECT COUNT(*) FROM policy_documents WHERE index_id=?',(index_id,)).fetchone()[0]
            index['chunks']=db.execute('SELECT COUNT(*) FROM policy_chunks WHERE index_id=?',(index_id,)).fetchone()[0]
        return {'ready':True,**index,'note':'Shared policy text only; customer records are not embedded.'}

    def build(self,strategy='heading',max_tokens=180,overlap_words=12,publish=True):
        started=time.perf_counter()
        with self._index_lock:
            documents=[parse_policy(path) for path in sorted(self.policies_dir.glob('*.md'))]
            if not documents:
                raise ValueError('No policy Markdown files found')
            identities=[(doc['family'],doc['version']) for doc in documents]
            if len(set(identities))!=len(identities):
                raise ValueError('Duplicate policy family/version')
            chunks=[chunk for doc in documents for chunk in make_chunks(doc,self.encoder,strategy,max_tokens,overlap_words)]
            vectors=np.asarray(self.encoder.encode([chunk['embedding_text'] for chunk in chunks]),dtype=np.float32)
            if vectors.shape!=(len(chunks),self.encoder.dimensions) or not np.isfinite(vectors).all():
                raise ValueError('Embedding dimensions or values are invalid')
            norms=np.linalg.norm(vectors,axis=1,keepdims=True)
            if np.any(norms<=0):
                raise ValueError('Embedding vectors cannot be zero')
            vectors=vectors/norms
            index_id='idx_'+uuid.uuid4().hex
            corpus_hash=digest(''.join(doc['policy_id']+doc['source_hash'] for doc in documents))
            with connect(self.path) as db:
                db.execute('INSERT INTO retrieval_indexes VALUES (?,?,?,?,?,?,?,?,?)',
                    (index_id,int(time.time()),strategy,max_tokens,overlap_words if strategy=='fixed' else 0,
                     self.encoder.model_id,self.encoder.revision,self.encoder.dimensions,corpus_hash))
                for doc in documents:
                    db.execute('INSERT INTO policy_documents VALUES (?,?,?,?,?,?,?,?,?)',
                        (index_id,doc['policy_id'],doc['family'],doc['version'],doc['title'],doc['source'],
                         doc['source_hash'],doc['effective_from'],doc['effective_until']))
                for chunk,vector in zip(chunks,vectors):
                    db.execute('INSERT INTO policy_chunks VALUES (?,?,?,?,?,?,?)',
                        (index_id,chunk['chunk_id'],chunk['policy_id'],chunk['heading'],chunk['text'],
                         chunk['token_count'],json.dumps(vector.tolist(),allow_nan=False)))
                if publish:
                    db.execute('INSERT INTO retrieval_state(id,active_index_id) VALUES (1,?) '
                               'ON CONFLICT(id) DO UPDATE SET active_index_id=excluded.active_index_id',(index_id,))
            return {'index_id':index_id,'published':publish,'strategy':strategy,'documents':len(documents),
                    'chunks':len(chunks),'model_id':self.encoder.model_id,'revision':self.encoder.revision,
                    'dimensions':self.encoder.dimensions,'build_seconds':time.perf_counter()-started,'corpus_hash':corpus_hash}

    def _load(self,index_id=None,as_of=None,family=None):
        index_id=self._index_id(index_id)
        as_of=(as_of or today_india()).isoformat()
        with connect(self.path) as db:
            row=db.execute('SELECT * FROM retrieval_indexes WHERE id=?',(index_id,)).fetchone()
            if row is None:
                raise RetrievalUnavailable('Index not found')
            index=dict(row)
            docs=[dict(r) for r in db.execute('SELECT * FROM policy_documents WHERE index_id=?',(index_id,))]
            allowed=eligible_documents(docs,as_of)
            rows=[dict(r) for r in db.execute('SELECT c.*, d.family,d.version,d.title,d.source,d.source_hash,d.effective_from,d.effective_until '
                'FROM policy_chunks c JOIN policy_documents d ON c.index_id=d.index_id AND c.policy_id=d.policy_id '
                'WHERE c.index_id=? ORDER BY c.chunk_id',(index_id,))
                if r['policy_id'] in allowed and (family is None or r['family']==family)]
        return index,rows,as_of

    def chunks(self,as_of=None,family=None):
        index,rows,as_of=self._load(as_of=as_of,family=family)
        return {'index_id':index['id'],'as_of':as_of,'chunks':[
            {k:v for k,v in row.items() if k!='embedding_json'} for row in rows]}

    def search(self,query,method='hybrid',top_k=3,as_of=None,family=None,index_id=None):
        started=time.perf_counter()
        query=query.strip()
        if not 3<=len(query)<=1200 or method not in ('bm25','dense','hybrid') or not 1<=top_k<=10:
            raise ValueError('Invalid query, method, or top_k')
        index,rows,as_of=self._load(index_id,as_of,family)
        response={'query':query,'method':method,'index_id':index['id'],'as_of':as_of,'eligible_chunks':len(rows),
                  'score_is_probability':False,'results':[],
                  'note':'Retrieved candidates, not an answer or proof of sufficient evidence. Scores are not confidence probabilities.'}
        if not rows:
            response.update(status='no_applicable_policy',elapsed_ms=(time.perf_counter()-started)*1000)
            return response
        passages=[r['title']+'\n'+r['heading']+'\n'+r['text'] for r in rows]
        lexical=bm25_scores(query,passages) if method!='dense' else [None]*len(rows)
        semantic=[None]*len(rows)
        if method!='bm25':
            if (index['model_id'],index['model_revision'],index['dimensions'])!=(self.encoder.model_id,self.encoder.revision,self.encoder.dimensions):
                raise RetrievalUnavailable('The query embedding model differs from the index; rebuild with the configured model')
            matrix=np.asarray([json.loads(r['embedding_json']) for r in rows],dtype=np.float32)
            vector=np.asarray(self.encoder.encode([query]),dtype=np.float32)
            if matrix.shape!=(len(rows),self.encoder.dimensions) or vector.shape!=(1,self.encoder.dimensions) or not np.isfinite(matrix).all() or not np.isfinite(vector).all():
                raise RetrievalUnavailable('Invalid vector dimensions or values; rebuild the index')
            norm=np.linalg.norm(vector)
            if norm<=0:
                raise ValueError('Query embedding cannot be zero')
            semantic=np.clip(matrix @ (vector[0]/norm),-1,1).tolist()
        lexical_order=sorted([i for i,s in enumerate(lexical) if s is not None and s>0],key=lambda i:(-lexical[i],rows[i]['chunk_id']))[:20]
        dense_order=sorted([i for i,s in enumerate(semantic) if s is not None],key=lambda i:(-semantic[i],rows[i]['chunk_id']))[:20]
        lr={i:rank for rank,i in enumerate(lexical_order,1)}
        dr={i:rank for rank,i in enumerate(dense_order,1)}
        fused={i:(1/(60+lr[i]) if i in lr else 0)+(1/(60+dr[i]) if i in dr else 0) for i in set(lr)|set(dr)}
        order=lexical_order if method=='bm25' else dense_order if method=='dense' else sorted(fused,key=lambda i:(-fused[i],rows[i]['chunk_id']))
        for rank,i in enumerate(order[:top_k],1):
            row={k:v for k,v in rows[i].items() if k not in ('embedding_json','index_id')}
            row.update(rank=rank,score=lexical[i] if method=='bm25' else semantic[i] if method=='dense' else fused[i],
                       bm25_score=lexical[i],cosine_similarity=semantic[i],bm25_rank=lr.get(i),dense_rank=dr.get(i),
                       rrf_score=fused.get(i) if method=='hybrid' else None)
            response['results'].append(row)
        response.update(status='candidates_only' if order else 'no_lexical_match',elapsed_ms=(time.perf_counter()-started)*1000)
        return response


def normalized(text):
    return ' '.join(text.replace('`','').casefold().split())


def evaluate(service,index_id=None,cases_path=None,top_k=3):
    cases=json.loads(Path(cases_path or ROOT/'evaluation'/'retrieval_cases.json').read_text(encoding='utf-8'))
    result={'label':'small development set; not a held-out quality guarantee','top_k':top_k,'methods':{}}
    for method in ('bm25','dense','hybrid'):
        rows=[]
        for case in cases:
            response=service.search(case['query'],method,top_k,date.fromisoformat(case['as_of']),index_id=index_id)
            relevant=[r['rank'] for r in response['results'] if any(r['policy_id']==gold['policy_id'] and normalized(gold['quote']) in normalized(r['text']) for gold in case['relevant'])]
            first=min(relevant) if relevant else None
            rows.append({'id':case['id'],'query':case['query'],'first_relevant_rank':first,
                         'retrieved_embedding_tokens':sum(r['token_count'] for r in response['results']),
                         'retrieved':[{'chunk_id':r['chunk_id'],'policy_id':r['policy_id'],'heading':r['heading']} for r in response['results']],
                         'elapsed_ms':response['elapsed_ms']})
        result['methods'][method]={'hit_at_k':sum(r['first_relevant_rank'] is not None for r in rows)/len(rows),
                                 'mrr_at_k':sum(1/r['first_relevant_rank'] if r['first_relevant_rank'] else 0 for r in rows)/len(rows),
                                 'mean_retrieved_embedding_tokens':sum(r['retrieved_embedding_tokens'] for r in rows)/len(rows),
                                 'cases':rows}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['download-model','index','status','search','evaluate','compare'])
    parser.add_argument('--strategy',choices=['heading','fixed'],default='heading')
    parser.add_argument('--max-tokens',type=int,default=180)
    parser.add_argument('--query',default='I was charged twice. What evidence is needed for a refund?')
    parser.add_argument('--method',choices=['bm25','dense','hybrid'],default='hybrid')
    parser.add_argument('--as-of',type=date.fromisoformat,default=today_india())
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='download-model':
        encoder=LocalEncoder(download=True)
        encoder.prepare()
        print(json.dumps({'model_id':encoder.model_id,'revision':encoder.revision,'cached':True}))
        return
    path=default_path()
    initialize(path)
    service=RetrievalService(path)
    if args.command=='index': result=service.build(args.strategy,args.max_tokens)
    elif args.command=='status': result=service.status()
    elif args.command=='search': result=service.search(args.query,args.method,as_of=args.as_of)
    elif args.command=='evaluate': result=evaluate(service)
    else:
        service.encoder.prepare()
        result={'note':'Same corpus, model, queries and token budget. Heading chunks do not overlap; fixed chunks overlap by 12 words. This is a small development comparison.','strategies':{}}
        for strategy in ('heading','fixed'):
            built=service.build(strategy,args.max_tokens,publish=False)
            result['strategies'][strategy]={'build':built,'evaluation':evaluate(service,built['index_id'])}
    output=json.dumps(result,indent=2,allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(output,encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
