"""Real Prompt, LoRA-SFT and token-constrained model adapters."""
from __future__ import annotations
import json,re,time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from ml_config import TEXT_MODEL,ADAPTER_DIR,ROUTES

INTENTS={"product":"商品諮詢","order":"訂單查詢","return":"退貨退款","repair":"故障維修","billing":"付款發票","security":"安全事件","other":"範圍外或資訊不足"}
SYSTEM=("你是3C電商客服分流器。只輸出一個 JSON 物件，不要 Markdown。必要欄位為 route、intent、confidence、"
        "needs_clarification、missing_fields、entities、safety、method。合法 route: "+", ".join(ROUTES))
class TextRouter:
    def __init__(self,use_lora=False):
        self.tok=AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model=AutoModelForCausalLM.from_pretrained(TEXT_MODEL,torch_dtype=torch.bfloat16,device_map="auto")
        if use_lora:self.model=PeftModel.from_pretrained(self.model,str(ADAPTER_DIR))
        self.model.eval()
    def _input(self,text):
        prompt=self.tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
        return self.tok(prompt,return_tensors="pt").to(self.model.device)
    @torch.inference_mode()
    def predict(self,text,constrained=False):
        inputs=self._input(text);start=time.perf_counter()
        kwargs={"max_new_tokens":192,"do_sample":False,"pad_token_id":self.tok.eos_token_id}
        candidates=[]
        if constrained:
            prompt_len=inputs.input_ids.shape[1]
            # Trie over complete JSON documents. Every possible generation is valid JSON
            # and conforms to the full routing contract, not only the route enum.
            for r in ROUTES:
                candidates.append(json.dumps({"route":r,"intent":INTENTS[r],"confidence":0.9,
                    "needs_clarification":r=="other","missing_fields":[],"entities":{},
                    "safety":{"blocked":r=="security","signals":[]},"method":"constraint"},
                    ensure_ascii=False,separators=(",",":")))
            seqs=[self.tok.encode(doc,add_special_tokens=False)+[self.tok.eos_token_id] for doc in candidates]
            def allowed(_batch,input_ids):
                prefix=input_ids[prompt_len:].tolist();valid=[]
                for seq in seqs:
                    if seq[:len(prefix)]==prefix and len(prefix)<len(seq):valid.append(seq[len(prefix)])
                return sorted(set(valid)) or [self.tok.eos_token_id]
            kwargs["prefix_allowed_tokens_fn"]=allowed
            kwargs["max_new_tokens"]=256
        out=self.model.generate(**inputs,**kwargs);lat=(time.perf_counter()-start)*1000
        raw=self.tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
        parsed=None
        try: parsed=json.loads(raw)
        except json.JSONDecodeError:
            # Do not repair malformed JSON: schema validity must reflect the raw model output.
            parsed=None
        route=(parsed or {}).get("route") or next((r for r in ROUTES if re.search(rf"\b{re.escape(r)}\b",raw.lower())),"other")
        return {"route":route,"raw_output":raw,"structured_output":parsed,"latency_ms":round(lat,2),"input_tokens":int(inputs.input_ids.numel()),"output_tokens":int(out.shape[1]-inputs.input_ids.shape[1])}
