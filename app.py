import streamlit as st
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import os

CHROMA_PATH = "chroma"

st.set_page_config(page_title="语义搜索助手", layout="wide")
st.title("🔍 本地语义搜索助手")

query = st.text_input("请输入你的问题", "")

if query:
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query, k=3)

    st.markdown(f"### 📄 检索结果 Top {len(results)}")

    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "Unbekannt")
        typ = meta.get("type", "Unbekannt")
        content = doc.page_content.strip()

        with st.expander(f"🔹 结果 {i}: {typ} aus {source} | Score: {score:.4f}"):
            # 显示 Markdown
            st.markdown(content, unsafe_allow_html=False)

            # 可选：查找其中的图片并单独显示（更保险）
            import re
            matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
            for img_path in matches:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
