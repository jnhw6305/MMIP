"""Extract actual BLIP captions and OCR text once for the A-D benchmark."""
import json
from pathlib import Path
from vlm import VisionAnalyzer

FILES={
 "broken_monitor.jpg":"broken_monitor.png",
 "laptop_led.jpg":"laptop_error.png",
 "earphone.jpg":"earphone_noise.png",
 "statement.png":"duplicate_charge.png",
}
def main():
    analyzer=VisionAnalyzer();features={}
    for logical,actual in FILES.items():
        r=analyzer.analyze(Path("data/vlm_cases")/actual)
        features[logical]={"source_file":actual,**r}
        print(logical,r["description"],r["ocr_text"])
    Path("data/multimodal_features.json").write_text(json.dumps(features,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
