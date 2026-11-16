# Transformer 論文教學語料

## 背景與貢獻
Transformer 是 Vaswani 等人在 2017 年提出的序列轉換架構。它不使用循環神經網路或卷積，而以注意力機制建立編碼器與解碼器。模型能平行處理序列位置，因此訓練效率較傳統循環模型好。

## 編碼器與解碼器
編碼器由六個相同層堆疊。每層包含多頭自注意力子層與位置式前饋網路。每個子層外都有殘差連接，之後進行 layer normalization。解碼器也有六層，另外加入 masked self-attention，避免位置看到後續 token，並用 encoder-decoder attention 讀取編碼器輸出。

## Scaled dot-product attention
注意力輸入為 query、key、value。分數為 Q 乘 K 的轉置，再除以 sqrt(d_k)，經 softmax 後加權 V。縮放可避免 key 維度很大時內積過大，使 softmax 進入梯度很小的區域。

## Multi-head attention
多頭注意力先以不同線性投影取得多組 Q、K、V，各頭平行執行注意力，再串接並投影。不同頭可關注不同表示子空間與位置關係。論文基礎模型使用 8 個 head，d_model 為 512，每頭 d_k 與 d_v 為 64。

## 位置資訊
因模型沒有 recurrence 或 convolution，必須注入位置。論文在輸入 embedding 加上正弦與餘弦 positional encoding。不同頻率讓模型學習相對位置；作者也測試 learned positional embedding，結果相近。

## 前饋、正則化與訓練
位置式前饋網路對每個位置分別套用兩層線性轉換，中間為 ReLU；d_model 512，內層 d_ff 2048。訓練使用 Adam，beta1=0.9、beta2=0.98、epsilon=1e-9，搭配 warmup 4000 steps 的特殊 learning-rate schedule。使用 dropout 0.1 與 label smoothing 0.1。

## 實驗結果
在 WMT 2014 英德翻譯，big Transformer 達到 28.4 BLEU。在 WMT 2014 英法翻譯，模型達到 41.8 BLEU。英法模型在 8 張 P100 GPU 訓練 3.5 天。論文也在 English constituency parsing 展示泛化能力。

## 複雜度
Self-attention 每層複雜度約為 O(n^2 d)，recurrent layer 約為 O(n d^2)。當序列長度 n 小於表示維度 d 時，自注意力通常較有利，而且 sequential operations 為 O(1)，有利平行化；但非常長序列的 n^2 成本仍是限制。

