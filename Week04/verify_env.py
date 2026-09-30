"""Fail fast with an actionable environment diagnostic."""
import importlib
import platform
import sys

if sys.version_info >= (3, 13):
    raise SystemExit("請改用 Python 3.10–3.12；目前為 " + platform.python_version())

modules = ["torch", "torchvision", "transformers", "datasets", "pyarrow", "pandas", "sklearn"]
errors = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
if errors:
    raise SystemExit("環境檢查失敗：\n" + "\n".join(errors))
print(f"環境檢查成功：Python {platform.python_version()}；PyTorch 可用。")
