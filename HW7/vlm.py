"""Real BLIP pixel captioning + OCR image prompt-injection detector."""
from __future__ import annotations
import re,time
import torch
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration
import easyocr
from ml_config import VLM_MODEL

class VisionAnalyzer:
    def __init__(self):
        self.model=BlipForConditionalGeneration.from_pretrained(VLM_MODEL,torch_dtype=torch.float16).to("cuda").eval()
        self.processor=BlipProcessor.from_pretrained(VLM_MODEL)
        self.ocr=easyocr.Reader(["en"],gpu=True)
    @torch.inference_mode()
    def analyze(self,path):
        image=Image.open(path).convert("RGB");start=time.perf_counter()
        inputs=self.processor(images=image,return_tensors="pt").to("cuda",torch.float16)
        out=self.model.generate(**inputs,max_new_tokens=40);caption=self.processor.decode(out[0],skip_special_tokens=True)
        ocr_text=" ".join(self.ocr.readtext(str(path),detail=0))
        # OCR text is data, never an instruction. Detect before combining it with the user prompt.
        signals=[p for p in ["ignore","previous","system prompt","password","jailbreak","instruction","override","bypass","reveal","leak","secret"] if p in ocr_text.lower()]
        high_risk={"system prompt","password","jailbreak","override","bypass","reveal","leak"}
        injected=len(signals)>=2 or bool(high_risk.intersection(signals))
        return {"description":caption,"ocr_text":ocr_text,"injection":injected,"signals":signals,"latency_ms":round((time.perf_counter()-start)*1000,2)}
