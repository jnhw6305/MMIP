from pathlib import Path
import streamlit as st
from src.mmip_rag.rag import load_rag
from src.mmip_rag.video import load_video
from src.mmip_rag.graph import load_graph

ROOT=Path(__file__).parent
st.set_page_config(page_title="MMIP RAG Lab",page_icon="🔎",layout="wide")
st.title("MMIP RAG Lab")
mode=st.sidebar.radio("系統",["文字 RAG","VideoRAG","GraphRAG","實驗結果"])
if "engines" not in st.session_state: st.session_state.engines={"文字 RAG":load_rag(ROOT),"VideoRAG":load_video(ROOT),"GraphRAG":load_graph(ROOT)}
if mode in ["文字 RAG","VideoRAG"]:
    q=st.chat_input("輸入問題")
    if q:
        out=st.session_state.engines[mode].answer(q); st.chat_message("user").write(q); st.chat_message("assistant").write(out["answer"])
        with st.expander("References / 檢索分數"):
            for r in out["references"]:
                score=f'{r.get("score",0):.4f}'
                timing=""
                if "start" in r and "end" in r:
                    timing=f' · {r["start"]:.1f}s–{r["end"]:.1f}s · {r.get("modality","unknown")}'
                st.markdown(f'**{r.get("id","")} · {r.get("title","")}** · score {score}{timing}')
                st.caption(r.get("text",""))
                st.divider()
elif mode=="GraphRAG":
    q=st.text_input("圖譜問題","Transformer 使用哪些機制？")
    if st.button("查詢"):
        out=st.session_state.engines[mode].query(q); st.write(out["answer"])
        st.code("\n".join(f"{a} --{relation}--> {b}" for a,relation,b in out["paths"]),language="text")
else:
    st.subheader("實驗產物")
    for f in ["rag_summary.json","rag_grid.csv","prompt_ablation.csv","video_eval.csv","graph_eval.csv"]:
        p=ROOT/"outputs"/f
        if p.exists(): st.download_button(f,p.read_bytes(),file_name=f)
    if not any((ROOT/"outputs"/f).exists() for f in ["rag_summary.json","rag_grid.csv","prompt_ablation.csv","video_eval.csv","graph_eval.csv"]):
        st.info("目前沒有可顯示的實驗產物。")

