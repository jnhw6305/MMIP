# MMIP HW7 繳交檢核表

最後檢核：2026-11-11

| 作業要求 | 狀態 | 可檢查證據 |
|---|---|---|
| 定義業務情境、下游任務與結構化 Spec | 完成 | `REPORT.md` 作業一、`router.py` |
| 完整技術架構與角落情境 | 完成 | `REPORT.md` Mermaid 架構圖、`services.py` |
| A：Prompt 結構化輸出 | 完成 | `real_models.py`、正式結果 JSON |
| B：SFT/RLHF | 完成（採 SFT） | `train_lora.py`、LoRA adapter、training metrics |
| C：Decoding 引導 | 完成 | 完整 JSON prefix-trie constrained decoding |
| D：System One | 完成 | `system_one.py`、`artifacts/system_one.json` |
| 至少 10 筆 Benchmark | 完成（12 筆） | `benchmark.py`、`benchmark_real.py` |
| 比較正確性、時間、成本 | 完成 | route accuracy、schema validity、P50/P95、能源成本估計 |
| 多模態＋多輪＋分流聊天機器人 | 完成 | `chat.py`、`vlm.py`、`server.py`、公開 Demo |
| 合理聊天室 UI | 完成 | `web/` 與公開 Demo |
| UX 優化並證明有效 | 完成 | `ux_experiment.py`、`results/ux_results.json` |
| 無脈絡問題處理並證明有效 | 完成 | `other` 路徑、`test_out_of_scope` |
| 文字／圖片攻擊防禦並證明有效 | 完成 | Safety Gate、13 張 VLM 測試、單元測試 |
| 公開網址可直接操作 | 完成 | https://mmip-nova-care-hw7.legendaryhunter87.chatgpt.site |

## 誠信與範圍

- 公開網址是免伺服器、瀏覽器端互動展示，頁面標示 `DEMO ENGINE`；不把它宣稱為雲端 GPU 推論。
- 正式 A–D 數據由 RTX 5080 上的本機模型實測，逐筆原始輸出與延遲位於 `results/real_benchmark_results.json`。
- 圖片攻擊結果來自小型合成測試集，不宣稱可代表真實世界零漏判。
- 成本是依實測時間與明列功耗／電價假設計算的能源估計，不是雲端帳單。
