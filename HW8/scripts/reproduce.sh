#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m src.mmip_rag.cli all
python -m src.mmip_rag.ragas_eval
python -m src.mmip_rag.reproduce
python -m unittest discover -s tests -v
