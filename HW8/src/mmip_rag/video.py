import json
from pathlib import Path
from .rag import RAG

def stamp(sec): return f"{int(sec)//60:02d}:{int(sec)%60:02d}"

class VideoRAG(RAG):
    VIDEO_SCOPE_TERMS=("streamlit","mmipraglab","文字rag","videorag","graphrag","頁面","heads","檢索","來源","引用","文件","關係","路徑","實驗結果","下載","transformer","attention","注意力","query","key","value","d_k","位置編碼","循環","卷積")
    def supported_video(self,q):
        normalized=self.rewrite(q).lower().replace(" ","")
        if any(term.replace(" ","") in normalized for term in self.OUT_OF_SCOPE_TERMS): return False
        return any(term.replace(" ","") in normalized for term in self.VIDEO_SCOPE_TERMS)
    def answer(self,q,prompt="grounded"):
        if not self.supported_video(q):
            ans="現有影片語料不足以回答此問題。"
            self.history.append((q,ans))
            return {"question":q,"answer":ans,"references":[]}
        hits=self.retrieve(q); parts=[]
        for h in hits[:1]: parts.append(f'{h["text"]} [{h["id"]} {stamp(h["start"])}-{stamp(h["end"])}]')
        ans=" ".join(parts) or "影片資料不足。"; self.history.append((q,ans))
        return {"question":q,"answer":ans,"references":hits}

def ingest(root):
    root=Path(root); raw=root/"video_corpus/raw"; side=root/"video_corpus/sidecars"; docs=[]
    raw.mkdir(parents=True,exist_ok=True)
    files=sorted(raw.glob("*.mp4"),key=lambda path:path.name.casefold())
    for video in files:
        sf=side/(video.stem+".json")
        if not sf.exists():
            try:
                import whisper
                result=whisper.load_model("base").transcribe(str(video))
                generated=[{"start":s["start"],"end":s["end"],"text":s["text"].strip(),"modality":"speech"} for s in result["segments"]]
                sf.write_text(json.dumps(generated,ensure_ascii=False,indent=2),encoding="utf-8")
            except (ImportError,RuntimeError):
                continue
        for seg in json.loads(sf.read_text(encoding="utf-8")):
            docs.append({"id":f"V{len(docs)+1}","title":video.name,"text":seg["text"],"start":seg["start"],"end":seg["end"],"modality":seg.get("modality","speech")})
    if not docs:
        sidecar_files=sorted(side.glob("*.json"),key=lambda path:path.name.casefold())
        for sf in sidecar_files:
            title=sf.name.replace(".json",".mp4 (recorded sidecar)")
            for seg in json.loads(sf.read_text(encoding="utf-8")):
                docs.append({"id":f"V{len(docs)+1}","title":title,"text":seg["text"],"start":seg["start"],"end":seg["end"],"modality":seg.get("modality","unknown")})
    (root/"outputs").mkdir(exist_ok=True); (root/"outputs/video_index.json").write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding="utf-8")
    return docs

def load_video(root): return VideoRAG(ingest(root),top_k=3)

