# MMIP Quiz 1–4 實驗報告

本報告由實際執行結果自動產生。

## Quiz 1–2：Sentiment Analysis

IMDb：train=10000、test=2500

| 模型 | Accuracy | Macro-F1 | 訓練秒數 |
|---|---:|---:|---:|
| RNN | 0.6156 | 0.6148 | 5.3 |
| LSTM | 0.7740 | 0.7740 | 6.0 |

## Quiz 3：Vision Transformer

CIFAR-10：train=10000、test=2000

| 模型 | Accuracy | Macro-AUC | 訓練秒數 |
|---|---:|---:|---:|
| VIT | 0.9655 | 0.9988 | 256.4 |
| RESNET | 0.9150 | 0.9962 | 52.7 |

## Quiz 4：Image Captioning

Flickr8k：train=2000、test=200

| 方法 | BLEU-1 | BLEU-4 | BERTScore F1 | 平均延遲(秒) |
|---|---:|---:|---:|---:|
| greedy | 0.2686 | 0.0737 | 0.9119 | 0.1579 |
| beam5 | 0.2890 | 0.0831 | 0.9143 | 0.2054 |