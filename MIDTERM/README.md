# CreditGuard-TS 期中專案

結合 MMIP Week02 信用卡違約分類成果、六個月時間特徵與統計合成資料的互動式違約預警網站。

## 線上使用

- GitHub Pages：<https://jnhw6305.github.io/MMIP/>

## 內容

- `index.html`、CSS、JavaScript：互動式成果網站
- `model.json`：瀏覽器端 Logistic Regression 基準模型
- `run_final_experiments.py`：Random Forest、時間特徵與合成資料消融實驗
- `experiment_results.json`：固定測試集的完整實驗結果
- `train_credit_model.py`：瀏覽器端模型訓練與匯出程式

## 主要結果

合成資料實驗將 Recall 由 0.5441 提升至 0.5735，漏判 FN 由 605 降至 566；同時 Accuracy 略降，呈現提高違約偵測率與增加誤報之間的取捨。

> 本網站僅供課程研究與教學，不構成授信決策或金融建議。
