# 信用卡違約預測：Quiz 1–3 完整結果

資料：[Kaggle 指定資料集](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)；原始來源：[UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)。
共 30,000 筆，23 個特徵；違約比例 22.12%。移除 ID，以 1 表示次月違約。
隨機種子 42；分層切分：模型訓練 19,200 筆、門檻校準 4,800 筆、最終驗證 6,000 筆。StandardScaler 僅以模型訓練集擬合。
門檻以校準集 F1 最大化選擇（0.05 至 0.80，間距 0.01）；最終驗證集不參與門檻選擇。

## Quiz 1：兩種 Scikit-Learn 分類模型

| 模型／門檻 | Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic 0.50 | 0.50 | 4525 | 148 | 989 | 338 | 0.8105 | 0.6955 | 0.2547 | 0.3729 | 0.7080 |
| Logistic 0.30 | 0.30 | 4174 | 499 | 721 | 606 | 0.7967 | 0.5484 | 0.4567 | 0.4984 | 0.7080 |
| Logistic 校準門檻 | 0.29 | 4141 | 532 | 701 | 626 | 0.7945 | 0.5406 | 0.4717 | 0.5038 | 0.7080 |
| Random forest 0.50 | 0.50 | 4423 | 250 | 847 | 480 | 0.8172 | 0.6575 | 0.3617 | 0.4667 | 0.7711 |
| Random forest 校準門檻 | 0.28 | 3942 | 731 | 562 | 765 | 0.7845 | 0.5114 | 0.5765 | 0.5420 | 0.7711 |

混淆矩陣定義：TN＝正確判定未違約、FP＝誤報違約、FN＝漏判違約、TP＝正確判定違約。
校準門檻下，Logistic 的 Precision / Recall 為 0.541 / 0.472；Random forest 為 0.511 / 0.576。
Logistic 的誤報 FP=532、漏判 FN=701；Random forest 的誤報 FP=731、漏判 FN=562。
提高門檻通常減少誤報、增加漏判；降低門檻則相反。

## Quiz 2：MLP 基礎與改善

使用 PyTorch，23→64→32→1、ReLU、BCEWithLogitsLoss、Adam、學習率 0.001、batch size 256、60 epochs。
改善版加入兩層 Dropout(0.30) 與 Adam weight decay=0.0001；其他設定及資料切分相同。
訓練與驗證損失曲線：[quiz2_loss.svg](quiz2_loss.svg)。每個 epoch 的原始數值見兩份 history CSV。

| 模型 | Training loss (epoch 60) | Validation loss (epoch 60) | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| MLP baseline | 0.3990 | 0.4441 | 0.7643 | 0.4724 | 0.5614 | 0.5131 |
| MLP improved | 0.4157 | 0.4331 | 0.7890 | 0.5225 | 0.5343 | 0.5283 |

單筆驗證集預測：ID=6908，實際違約=0，原版 MLP 違約機率=0.0744，改善版=0.0931；改善版依其校準門檻預測=0。
原版驗證損失在第 19 epoch 達最低值 0.4364，之後訓練損失仍下降、驗證損失卻回升，顯示過擬合。改善版第 60 epoch 驗證損失由 0.4441 降至 0.4331；訓練與驗證損失差距由 +0.0451 縮至 +0.0174，後段曲線也較平穩。改善版 Precision 和 F1 上升，但 Recall 下降，並非所有指標都改善。

## Quiz 3：ROC 與 AUC

同一張 ROC 圖：[quiz3_roc.svg](quiz3_roc.svg)。橫軸 FPR=FP/(FP+TN)，縱軸 TPR=TP/(TP+FN)。ROC 顯示各門檻下偵測違約與誤報之間的關係；AUC 是曲線下面積，越高表示對違約與未違約樣本的排序區分能力越好。

| 模型 | Validation AUC |
|---|---:|
| Logistic regression | 0.7080 |
| Random forest | 0.7711 |
| MLP baseline | 0.7692 |
| MLP improved | 0.7746 |

本次驗證集中，MLP improved 的 AUC 最高（0.7746），因此在這次固定切分上有最佳的正負樣本排序區分能力。AUC 差異並不代表已證明在其他資料或實際部署中必然較佳。

## 重現方式

```powershell
python -m pip install numpy pandas scikit-learn torch matplotlib
python credit_default_quiz1_3.py --data UCI_Credit_Card.csv
```

所有表格均由 `results.json` 的實際訓練結果生成。
