import hashlib, json, math, re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

def tokens(s): return [x.lower() for x in TOKEN.findall(s)]

def embed(text, dims=384):
    v=[0.0]*dims
    ts=tokens(text)
    for t in ts:
        h=int(hashlib.sha256(t.encode()).hexdigest()[:12],16)
        v[h%dims]+=1.0
    n=math.sqrt(sum(x*x for x in v)) or 1
    return [x/n for x in v]

def similarity(a,b): return sum(x*y for x,y in zip(a,b))

def chunk_markdown(text, size=420, overlap=80):
    sections=[]
    title="document"
    for part in re.split(r"(?m)^##?\s+", text):
        if not part.strip(): continue
        lines=part.strip().splitlines(); title=lines[0].strip(); body=" ".join(lines[1:]) or title
        step=max(1,size-overlap)
        for i in range(0,len(body),step):
            piece=body[i:i+size]
            if piece.strip(): sections.append({"id":f"D{len(sections)+1}","title":title,"text":piece})
            if i+size>=len(body): break
    return sections

class RAG:
    STRONG_SCOPE_TERMS=(
        "attention","注意力","編碼器","解碼器","位置編碼","positional","前饋","殘差",
        "layer normalization","query","key","value","qkv","softmax","sqrt","d_k","d_v",
        "d_model","d_ff","multi-head","self-attention","masked","head","embedding","relu",
        "optimizer","adam","warmup","dropout","label smoothing","learning-rate","beta1","beta2",
        "wmt","bleu","英德","英法","p100","constituency parsing","複雜度","o(n","長序列",
        "recurrence","recurrent","convolution","vaswani",
    )
    TRANSFORMER_INTENTS=(
        "什麼","提出","何時","誰","核心","創新","架構","組成","包含","幾層","運作",
        "原理","使用","用途","優點","限制","效率","平行","訓練多久","泛化",
    )
    OUT_OF_SCOPE_TERMS=(
        "天氣","氣溫","下雨","颱風","股票","股價","匯率","投資","新聞","總統",
        "住在哪","城市","地址","電話","生日","年齡","電影","音樂","食譜","交通",
        "幾點","醫療","法律","運動比賽","星座","笑話",
    )
    def __init__(self, docs, top_k=3):
        self.docs=docs; self.top_k=top_k; self.vecs=[embed(d["title"]+" "+d["text"]) for d in docs]; self.history=[]
    def rewrite(self,q):
        if self.history and any(x in q for x in ["它","這個","那它","其"]):
            previous=self.history[-1][0]
            subjects=["Transformer","編碼器","解碼器","Multi-Head Attention","Self-Attention","注意力","前饋網路","位置編碼"]
            subject=next((x for x in subjects if x.lower() in previous.lower()),previous)
            rewritten=q
            for pronoun in ["那它","這個","它","其"]: rewritten=rewritten.replace(pronoun,subject)
            return rewritten
        return q
    def retrieve(self,q,k=None):
        rq=self.rewrite(q); qv=embed(rq)
        ranked=sorted(zip(self.docs,self.vecs),key=lambda x:similarity(qv,x[1]),reverse=True)[:k or self.top_k]
        return [{**d,"score":round(similarity(qv,v),4)} for d,v in ranked]
    def supported(self,q):
        normalized=q.lower().replace(" ","")
        if any(term.replace(" ","") in normalized for term in self.OUT_OF_SCOPE_TERMS): return False
        if any(term.replace(" ","") in normalized for term in self.STRONG_SCOPE_TERMS): return True
        return "transformer" in normalized and any(term in normalized for term in self.TRANSFORMER_INTENTS)
    def answer(self,q,prompt="grounded"):
        rewritten=self.rewrite(q); hits=self.retrieve(q); qt=set(tokens(rewritten))
        if not self.supported(rewritten):
            ans="現有 Transformer 論文語料不足以回答此問題。"
            self.history.append((q,ans))
            return {"question":q,"answer":ans,"references":[]}
        candidates=[]
        for h in hits:
            for sent in re.split(r"(?<=[。！？])",h["text"]):
                score=len(qt & set(tokens(sent)))+h["score"]
                candidates.append((score,sent.strip(),h["id"]))
        candidates.sort(reverse=True)
        chosen=[]
        for _,s,i in candidates:
            if s and s not in [x[0] for x in chosen]: chosen.append((s,i))
            if len(chosen)>=2: break
        if prompt=="direct": ans=chosen[0][0] if chosen else "資料不足。"
        elif prompt=="cot-plan": ans="根據檢索內容比對關鍵概念後，"+" ".join(f"{s} [{i}]" for s,i in chosen)
        else: ans=" ".join(f"{s} [{i}]" for s,i in chosen) or "現有資料不足以回答。"
        self.history.append((q,ans)); return {"question":q,"answer":ans,"references":hits}

def load_rag(root, size=420, overlap=80, top_k=3):
    text=(Path(root)/"data/corpus.md").read_text(encoding="utf-8")
    return RAG(chunk_markdown(text,size,overlap),top_k)

