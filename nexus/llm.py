"""Gemini REST transport. Credentials remain local and errors never echo provider bodies."""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import httpx
from dotenv import dotenv_values
from nexus.db import ROOT


class GenerationError(RuntimeError):
    def __init__(self,code,message,status=502):
        super().__init__(message)
        self.code,self.status=code,status


@dataclass
class LLMConfig:
    provider: str = 'gemini'
    model: str = 'gemini-3.8-flash'
    api_key: str = field(default='',repr=False)

    @classmethod
    def load(cls,path=None):
        values=dotenv_values(path or ROOT/'.env',encoding='utf-8-sig')
        def get(key,default=''):
            return os.environ.get(key,values.get(key) or default)
        return cls(get('NEXUS_LLM_PROVIDER','gemini'),get('NEXUS_LLM_MODEL','gemini-3.8-flash'),get('GEMINI_API_KEY'))


def provider_error(status):
    if status in (401,403):
        return GenerationError('provider_auth','Gemini rejected the credential or model access. Check the local API key and model configuration.',503)
    if status==404:
        return GenerationError('model_unavailable','The configured Gemini model is unavailable. Select an accessible model in .env.',503)
    if status==429:
        return GenerationError('provider_quota','Gemini quota or rate limit reached. Check your provider quota before retrying.',429)
    if status==400:
        return GenerationError('provider_request','Gemini rejected the request configuration. Check model/schema compatibility.',502)
    return GenerationError('provider_unavailable','Gemini is temporarily unavailable. No answer was accepted.',502)


class GeminiProvider:
    def __init__(self,config=None,transport=None):
        self.config=config or LLMConfig.load()
        self.transport=transport

    def status(self):
        return {'provider':self.config.provider,'model':self.config.model,
                'configured':bool(self.config.api_key),'network_destination':'generativelanguage.googleapis.com'}

    def ensure_ready(self):
        if self.config.provider!='gemini':
            raise GenerationError('configuration','Set NEXUS_LLM_PROVIDER=gemini in the local .env file.',503)
        if not re.fullmatch(r'[A-Za-z0-9._-]{1,100}',self.config.model):
            raise GenerationError('configuration','Invalid model name in local configuration.',503)
        if not self.config.api_key:
            raise GenerationError('configuration','Add GEMINI_API_KEY to the ignored local .env file and save it. Never paste the key into chat.',503)

    def stream(self,system,context,schema):
        self.ensure_ready()
        url=f'https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:streamGenerateContent?alt=sse'
        body={'systemInstruction':{'parts':[{'text':system}]},
              'contents':[{'role':'user','parts':[{'text':json.dumps(context,ensure_ascii=False)}]}],
              'generationConfig':{'temperature':0.1,'maxOutputTokens':2048,
                                  'responseMimeType':'application/json','responseJsonSchema':schema}}
        size,finish,usage=0,None,{}
        try:
            timeout=httpx.Timeout(90,connect=10,write=15,pool=5)
            with httpx.Client(timeout=timeout,transport=self.transport,follow_redirects=False) as client:
                with client.stream('POST',url,headers={'x-goog-api-key':self.config.api_key},json=body) as response:
                    if response.status_code!=200:
                        raise provider_error(response.status_code)
                    frame=[]
                    for line in response.iter_lines():
                        if line.startswith('data:'):
                            frame.append(line[5:].lstrip())
                        elif not line and frame:
                            events,ended,new_usage=self._frame('\n'.join(frame))
                            frame=[]
                            finish=ended or finish
                            usage=new_usage or usage
                            for text in events:
                                size+=len(text)
                                if size>24000:
                                    raise GenerationError('output_limit','Model output exceeded the application limit.')
                                yield {'type':'delta','text':text}
                    if frame:
                        events,ended,new_usage=self._frame('\n'.join(frame))
                        finish=ended or finish
                        usage=new_usage or usage
                        for text in events:
                            size+=len(text)
                            if size>24000:
                                raise GenerationError('output_limit','Model output exceeded the application limit.')
                            yield {'type':'delta','text':text}
        except httpx.TimeoutException as exc:
            raise GenerationError('provider_timeout','Gemini timed out. No answer was accepted.',504) from exc
        except httpx.HTTPError as exc:
            raise GenerationError('provider_connection','Could not reach Gemini. Check the network connection.',502) from exc
        if finish!='STOP':
            raise GenerationError('provider_incomplete','Gemini did not finish a complete response. No partial answer was accepted.')
        if not size:
            raise GenerationError('provider_empty','Gemini returned no answer text.')
        yield {'type':'usage','usage':{k:v for k,v in usage.items() if k in (
            'promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')}}

    @staticmethod
    def _frame(raw):
        try:
            payload=json.loads(raw)
        except ValueError as exc:
            raise GenerationError('provider_protocol','Gemini returned an invalid stream frame.') from exc
        if not isinstance(payload,dict) or payload.get('error'):
            raise GenerationError('provider_protocol','Gemini returned an unsuccessful stream frame.')
        if payload.get('promptFeedback',{}).get('blockReason'):
            raise GenerationError('provider_blocked','Gemini declined this request. No answer was accepted.')
        candidates=payload.get('candidates',[])
        candidate=candidates[0] if candidates else {}
        parts=candidate.get('content',{}).get('parts',[])
        text=[p['text'] for p in parts if isinstance(p,dict) and isinstance(p.get('text'),str) and not p.get('thought')]
        return text,candidate.get('finishReason'),payload.get('usageMetadata',{})
