# Week 05：多模態深度學習神經網路

本週完成影像檢索、多模態分類與 CLIP Image Captioning 三項實驗。三份 notebook 均保留正式訓練輸出、量化指標、比較圖、推論案例與實驗結論，可直接在 GitHub 檢視。

## 作業內容

| 題目 | 資料集與模型 | 評估與進階內容 | Notebook |
| --- | --- | --- | --- |
| Quiz 1：影像檢索 | CIFAR-10；ResNet-18／ViT-Tiny | Recall@1/5/10、Precision@5、mAP@10；CNN 與 ViT 比較 | [開啟](MMIP_Week05_Quiz1_Image_Retrieval.ipynb) |
| Quiz 2：多模態分類 | MM-IMDb；影像 backbone＋DistilBERT＋fusion MLP | Micro/Macro-F1、Subset Accuracy、Hamming Loss；更換影像 backbone | [開啟](MMIP_Week05_Quiz2_Multimodal_Classification.ipynb) |
| Quiz 3：CLIP Image Captioning | Flickr8k；CLIP ViT-B/32＋GPT-2 Decoder | BLEU-1/4、BERTScore、推論案例；與 Week 4 BLIP 比較 | [開啟](MMIP_Week05_Quiz3_CLIP_Image_Captioning.ipynb) |

## Quiz 1：CNN 與 ViT 影像檢索

- 使用 CIFAR-10 的 10,000 張訓練影像與 2,000 張 query/gallery 影像。
- 以類別監督訓練 256 維、L2-normalized embedding，使用 cosine similarity 排序。
- ViT-Tiny 的 Recall@1／mAP@10 為 **0.8975／0.8698**，略高於 ResNet-18 的 **0.8875／0.8650**。
- ResNet-18 訓練較快；ViT-Tiny 則在多數前段檢索指標略占優勢。

## Quiz 2：影像與文字融合分類

- 使用 MM-IMDb 的電影海報、劇情文字與多重 genre 標籤，取 4,000 筆訓練、1,000 筆測試資料及最常見的 10 個 genres。
- 比較 ResNet-18＋DistilBERT 與 ViT-Base＋DistilBERT；兩個模態經 projection 後 concatenate，再交由 fusion MLP 分類。
- ViT-Base 組合的 Micro-F1／Macro-F1 為 **0.5857／0.5480**，優於 ResNet-18 組合的 **0.5421／0.5075**。
- ViT-Base 將 Hamming Loss 從 **0.2532** 降至 **0.2109**，但訓練時間較長。

## Quiz 3：CLIP Image Captioning

- 使用 Flickr8k 的 2,000 張訓練影像與 200 張測試影像。
- 凍結 CLIP ViT-B/32 Image Encoder，訓練 GPT-2 Decoder 與新增的 cross-attention，並用 beam search（beam size=5）生成描述。
- CLIP-GPT2 的 BLEU-1／BLEU-4／BERTScore F1 為 **0.1598／0.0106／0.8140**。
- Week 4 BLIP beam-5 的對應結果為 **0.2890／0.0831／0.9143**；差距顯示 image-text alignment 與生成式跨模態預訓練的重要性。

## 重現性與檔案範圍

- 隨機種子固定為 42，三題皆使用正式資料規模與 3 epochs 訓練。
- Notebook 內含資料下載、模型建立、訓練、評估、視覺化及結果保存程式。
- 資料集與預訓練權重由 torchvision／Hugging Face 於執行時下載，不納入 repository。
- 約 961 MB 的 CLIP-GPT2 checkpoint 不屬於繳交必要內容，因此不提交；正式評估結果與推論範例已保留在 notebook 輸出中。

