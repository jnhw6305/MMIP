import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(*args): subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); p.add_argument("--gemini", action="store_true")
    args = p.parse_args()
    if args.smoke:
        call("src/sentiment.py", "--epochs", "1", "--train-size", "256", "--test-size", "64")
        call("src/vision.py", "--epochs", "1", "--train-size", "128", "--test-size", "64")
        cap = ["src/captioning.py", "--epochs", "1", "--train-size", "16", "--test-size", "4"]
    else:
        call("src/sentiment.py"); call("src/vision.py"); cap = ["src/captioning.py"]
    if args.gemini: cap.append("--gemini")
    call(*cap); call("src/make_report.py")

