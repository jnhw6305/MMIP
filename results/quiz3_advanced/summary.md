# Quiz 3 進階：多張實拍影像自動透視校正

執行方式（於儲存庫根目錄）：

```bat
python quiz3_perspective.py --pattern "document*.jpg" --output-dir "results\quiz3_advanced"
python results\quiz3_advanced\stress_test.py
```

程式不使用手動點選或針對個別照片寫死座標。先嘗試一般四邊形輪廓與原有雙色書封方法；失敗後，依序使用較低 Canny 門檻、藍色／青色封面分割。候選四邊形須通過面積、凸性、邊長和邊界檢查。輸出每張照片的角點標記圖、校正圖及 [summary.csv](summary.csv)。

## 實拍結果與目視檢查

| 照片／條件 | 偵測方式 | 程式結果 | 目視檢查 |
| --- | --- | --- | --- |
| [document.jpg](../../Week01/Data/Quiz3/document.jpg)：青色書封、布面背景 | 彩色封面與底邊 | 成功 | [角點](document_corners.jpg)大致沿書封；[校正圖](document_corrected.jpg)仍略有邊界誤差。 |
| [document_01.jpg](../../Week01/Data/Quiz3/document_01.jpg)：低視角、明亮桌面、彩色書封 | 藍色封面區域 | 成功 | [角點](document_01_corners.jpg)沿封面四邊；[校正圖](document_01_corrected.jpg)內容完整。 |
| [document_02.jpg](../../Week01/Data/Quiz3/document_02.jpg)：旋轉斜拍、木紋桌面、深藍書封 | 藍色封面區域 | 成功 | [角點](document_02_corners.jpg)大致正確；[校正圖](document_02_corrected.jpg)右下邊緣有少量裁切，不視為完美校正。 |
| [document_03.jpg](../../Week01/Data/Quiz3/document_03.jpg)：斜拍白紙、淺色桌面 | 低門檻四邊形輪廓 | 成功 | [角點](document_03_corners.jpg)沿紙張邊緣；[校正圖](document_03_corrected.jpg)形狀變直。白紙無文字，無法評估文字可讀性。 |

本次四張實拍照片均產出校正圖（4/4），但這不代表對任意拍攝條件都有效。原演算法在新增三張照片上為 0/3：預設邊緣門檻漏掉白紙低對比邊界，書封輪廓受背景或封面內部線條干擾；原本以座標和／差排列四角，在強旋轉影像上也不夠穩定。修訂版改用上下兩點分組，再按水平位置排列。此排列仍假設照片未旋轉到上下邊顛倒。

## 受控劣化測試（非額外實拍）

使用同三張新增照片，先縮至最長邊 1000 px，再分別調整亮度係數或套用 Gaussian Blur。詳細參數、程式與 27 筆輸出見 [stress_test.py](stress_test.py)、[stress.csv](stress.csv)。下列「成功」僅表示找到四角，未逐張人工評估劣化後的校正品質。

| 照片 | 首次找不到角點的已測亮度係數 | 首次找不到角點的已測模糊核 |
| --- | --- | --- |
| document_01.jpg | 0.15（0.30 仍找到） | 3、7、15、31 均找到 |
| document_02.jpg | 1.0、0.7、0.5、0.3、0.15 均找到 | 3、7、15、31 均找到 |
| document_03.jpg | 0.70（1.0 找到） | 7（3 找到） |

白紙與淺色桌面的對比本來就較低；變暗或模糊後，紙張邊緣更不連續，較低 Canny 門檻也無法形成可信的封閉四邊形。深藍封面的顏色差異較大，因此在本次測試範圍內較穩定。亮度係數及模糊核是軟體模擬條件，**不能當成實際拍攝角度**。三張新增照片沒有記錄相機與紙面夾角，因此無法宣稱精確的角度失敗閾值。

目前藍色分割只涵蓋特定色相；遇到其他顏色的封面、同色背景、物件被裁切或極端旋轉時，仍須新增樣本再驗證。`summary.csv` 的「成功」代表程式輸出成功，品質判斷以上述目視檢查為準。
