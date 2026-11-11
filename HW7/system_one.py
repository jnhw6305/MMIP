"""A real trained System-One character n-gram Naive Bayes router."""
from __future__ import annotations
import json, math
from collections import Counter, defaultdict
from pathlib import Path
from ml_config import SYSTEM_ONE_PATH, ROUTES

def grams(text,n=2):
    clean="".join(text.lower().split())
    return [clean[i:i+n] for i in range(max(1,len(clean)-n+1))] or [clean]

class SystemOne:
    def __init__(self,model=None): self.model=model
    def fit(self,rows):
        counts={r:Counter() for r in ROUTES};docs=Counter()
        for row in rows: counts[row["route"]].update(grams(row["text"]));docs[row["route"]]+=1
        vocab=sorted(set().union(*(set(c) for c in counts.values())))
        self.model={"routes":ROUTES,"vocab_size":len(vocab),"docs":dict(docs),"counts":{r:dict(counts[r]) for r in ROUTES},"totals":{r:sum(counts[r].values()) for r in ROUTES}}
        return self
    def predict(self,text):
        m=self.model;total_docs=sum(m["docs"].values());scores={}
        for r in m["routes"]:
            s=math.log((m["docs"].get(r,0)+1)/(total_docs+len(m["routes"])))
            denom=m["totals"][r]+m["vocab_size"]
            for g in grams(text):s+=math.log((m["counts"][r].get(g,0)+1)/denom)
            scores[r]=s
        best=max(scores,key=scores.get); exps={r:math.exp(v-max(scores.values())) for r,v in scores.items()};conf=exps[best]/sum(exps.values())
        return best,round(conf,4)
    def save(self,path=SYSTEM_ONE_PATH):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(self.model,ensure_ascii=False),encoding="utf-8")
    @classmethod
    def load(cls,path=SYSTEM_ONE_PATH):return cls(json.loads(Path(path).read_text(encoding="utf-8")))

if __name__=="__main__":
    rows=[json.loads(x) for x in Path("data/train.jsonl").read_text(encoding="utf-8").splitlines()]
    model=SystemOne().fit(rows);model.save();print(f"saved {SYSTEM_ONE_PATH}")
