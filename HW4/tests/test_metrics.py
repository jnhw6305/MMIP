import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from common import classification_metrics


def test_perfect_binary_metrics():
    result = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0

