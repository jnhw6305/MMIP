"""Actual LoRA SFT for route classification; designed for a 16 GB GPU."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np, torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from ml_config import TEXT_MODEL, ADAPTER_DIR, ROUTES, SEED
from real_models import INTENTS

SYSTEM=("你是3C電商客服分流器。只輸出一個 JSON 物件，不要 Markdown。必要欄位為 route、intent、confidence、"
        "needs_clarification、missing_fields、entities、safety、method。合法 route: "+", ".join(ROUTES))
def answer_for(row):
    r=row["route"];answer=json.dumps({"route":r,"intent":INTENTS[r],"confidence":0.95,
      "needs_clarification":r=="other","missing_fields":[],"entities":{},
      "safety":{"blocked":r=="security","signals":[]},"method":"sft"},ensure_ascii=False,separators=(",",":"))
    return answer

class ChatDataset(Dataset):
    def __init__(self, rows, tok): self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row=self.rows[i]; answer=answer_for(row)
        prompt=self.tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":row["text"]}],tokenize=False,add_generation_prompt=True)
        full=prompt+answer+self.tok.eos_token
        enc=self.tok(full,truncation=True,max_length=512)
        prompt_len=len(self.tok(prompt,truncation=True,max_length=512)["input_ids"])
        labels=enc["input_ids"].copy(); labels[:prompt_len]=[-100]*min(prompt_len,len(labels))
        return {"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],"labels":labels}

def collate(batch,tok):
    n=max(len(x["input_ids"]) for x in batch); out={k:[] for k in ("input_ids","attention_mask","labels")}
    for x in batch:
        pad=n-len(x["input_ids"])
        out["input_ids"].append(x["input_ids"]+[tok.pad_token_id]*pad)
        out["attention_mask"].append(x["attention_mask"]+[0]*pad)
        out["labels"].append(x["labels"]+[-100]*pad)
    return {k:torch.tensor(v) for k,v in out.items()}

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); losses=[]
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}; losses.append(model(**batch).loss.item())
    model.train(); return float(np.mean(losses))
def main():
    set_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    rows=[json.loads(x) for x in Path("data/train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows=[json.loads(x) for x in Path("data/validation.jsonl").read_text(encoding="utf-8").splitlines()]
    tok=AutoTokenizer.from_pretrained(TEXT_MODEL);tok.pad_token=tok.eos_token
    train_ds=ChatDataset(rows,tok); val_ds=ChatDataset(val_rows,tok)
    collator=lambda b: collate(b,tok)
    train_loader=DataLoader(train_ds,batch_size=4,shuffle=True,collate_fn=collator)
    val_loader=DataLoader(val_ds,batch_size=4,shuffle=False,collate_fn=collator)
    model=AutoModelForCausalLM.from_pretrained(TEXT_MODEL,dtype=torch.bfloat16,device_map="auto")
    peft=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model=get_peft_model(model,peft); model.train(); device=next(model.parameters()).device
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
    start=time.perf_counter(); history=[]; accum=4; opt.zero_grad()
    for epoch in range(3):
        losses=[]
        for step,batch in enumerate(train_loader,1):
            batch={k:v.to(device) for k,v in batch.items()}; loss=model(**batch).loss/accum; loss.backward(); losses.append(loss.item()*accum)
            if step%accum==0 or step==len(train_loader): opt.step(); opt.zero_grad()
        val_loss=evaluate(model,val_loader,device); history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val_loss})
        print(history[-1])
    ADAPTER_DIR.mkdir(parents=True,exist_ok=True); model.save_pretrained(str(ADAPTER_DIR));tok.save_pretrained(str(ADAPTER_DIR))
    metrics={"train":{"runtime_seconds":time.perf_counter()-start,"epochs":3,"examples":len(train_ds)},"validation":{"examples":len(val_ds),"final_loss":history[-1]["validation_loss"]},"history":history}
    (ADAPTER_DIR/"training_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(metrics)
if __name__=="__main__":main()
