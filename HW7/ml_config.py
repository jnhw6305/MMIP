from pathlib import Path

TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLM_MODEL = "Salesforce/blip-image-captioning-base"
ROOT = Path(__file__).parent
ADAPTER_DIR = ROOT / "artifacts" / "lora-router"
SYSTEM_ONE_PATH = ROOT / "artifacts" / "system_one.json"
SEED = 42
ROUTES = ["product", "order", "return", "repair", "billing", "security", "other"]
