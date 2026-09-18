# Week 01 程式碼

四支主程式均放在此資料夾，預設輸入路徑依儲存庫根目錄計算。因此可在任何工作目錄執行，但下列指令以儲存庫根目錄為例：

```bat
python Week01\Code\quiz1_grayscale.py
python Week01\Code\quiz2_histogram.py
python Week01\Code\quiz3_perspective.py --output-dir Week01\Data\Quiz3\advanced
python Week01\Code\quiz3_stress_test.py
python Week01\Code\quiz4_stitching.py
```

| 程式 | 預設輸入 | 預設輸出 |
| --- | --- | --- |
| `quiz1_grayscale.py` | `Week01/Data/Quiz1/color.jpg` | `Week01/Data/Quiz1/advanced/` |
| `quiz2_histogram.py` | `Week01/Data/Quiz2/dark.jpg` | `Week01/Data/Quiz2/advanced/` |
| `quiz3_perspective.py` | `Week01/Data/Quiz3/document*` | `results/quiz3_auto/`；上例改為 `Week01/Data/Quiz3/advanced/` |
| `quiz3_stress_test.py` | `Week01/Data/Quiz3/document_0?.jpg` | `Week01/Data/Quiz3/advanced/stress.csv` |
| `quiz4_stitching.py` | `Week01/Data/Quiz4/` 中的兩組照片 | `results/quiz4_two_pairs/experiments_時間戳/` |

Quiz 1／2 可用 `--input`、`--output-dir`、`--repeat` 指定影像、輸出位置與計時次數；Quiz 3 可用 `--input-dir`、`--pattern`、`--output-dir`；Quiz 4 可用 `--images-dir`、`--results-dir`。可在程式後加 `--help` 查看參數。Quiz 4 每次執行會產生大量圖片，正式提交資料以 [進階報告](../Data/Quiz4/advanced/summary.md)中的代表性證據為準。
