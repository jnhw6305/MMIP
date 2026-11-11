"""Dependency-free local demo server."""
from __future__ import annotations
import base64, json, os, tempfile
import mimetypes
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from chat import Session

ROOT = Path(__file__).parent
SESSIONS: dict[str, Session] = {}
VISION = None
TEXT_MODELS = {}

def real_text_route(text: str, method: str):
    if os.environ.get("ENABLE_ML") != "1" or method not in {"prompt", "sft", "constraint"}: return None
    from real_models import TextRouter
    key = "sft" if method == "sft" else "base"
    if key not in TEXT_MODELS: TEXT_MODELS[key] = TextRouter(use_lora=(key == "sft"))
    return TEXT_MODELS[key].predict(text, constrained=(method == "constraint"))

def analyze_image(data_url: str):
    """Decode actual pixels and call BLIP+OCR when launched with ENABLE_ML=1."""
    global VISION
    if not data_url or os.environ.get("ENABLE_ML") != "1": return None
    from vlm import VisionAnalyzer
    if VISION is None: VISION = VisionAnalyzer()
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"): raise ValueError("Only image data URLs are accepted")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 5_000_000: raise ValueError("Image exceeds 5 MB")
    suffix = ".png" if "png" in header else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f: f.write(raw); path=f.name
    try: return VISION.analyze(path)
    finally: Path(path).unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200, sid=None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if sid:
            self.send_header("X-Session-ID", sid)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        target = ROOT / ("web/index.html" if path == "/" else path.lstrip("/"))
        if not target.is_file() or ROOT not in target.resolve().parents:
            self.send_error(404); return
        data = target.read_bytes(); self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 7_000_000:
                self.send_json({"error":"request too large"}, 413); return
            payload = json.loads(self.rfile.read(size) or b"{}")
            sid = self.headers.get("X-Session-ID") or str(uuid.uuid4())
            session = SESSIONS.setdefault(sid, Session())
            text=str(payload.get("text", ""))[:2000]; image_name=str(payload.get("image_name", ""))[:200]
            vision=analyze_image(str(payload.get("image_data", "")))
            if vision:
                if vision["injection"]: text += " prompt injection system prompt"
                else: text += " 圖片分析：" + vision["description"] + "；OCR：" + vision["ocr_text"]
            out = session.reply(text, image_name, str(payload.get("method", "constraint")))
            # In full mode, replace the quick baseline with the actual GPU model result.
            if not out["routing"]["safety"]["blocked"]:
                actual=real_text_route(text, str(payload.get("method", "constraint")))
                if actual:
                    labels={"product":"商品諮詢","order":"訂單查詢","return":"退貨退款","repair":"故障維修","billing":"付款發票","security":"安全事件","other":"範圍外或資訊不足"}
                    out["routing"]["route"]=actual["route"];out["routing"]["intent"]=labels[actual["route"]];out["routing"]["model_trace"]=actual
                    out["message"]=(f"模型判定為「{labels[actual['route']]}」，已分流至 {actual['route']}。" if actual["route"]!="other" else "模型無法可靠分流，請補充商品、訂單或問題細節。")
            out["vision"] = vision
            self.send_json(out, sid=sid)
        except Exception as exc:
            self.send_json({"error":str(exc)}, 400)


if __name__ == "__main__":
    print("MMIP HW7 running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
