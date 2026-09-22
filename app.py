import streamlit as st
import requests
import xml.etree.ElementTree as ET
from pinecone import Pinecone
from openai import OpenAI

# 1. 網頁介面標題
st.title("🩺 您的專屬 PubMed 雲端 AI 助理")

# 2. 讓使用者在網頁上輸入自己的 API Keys (安全考量)
with st.sidebar:
    st.header("🔑 雲端金鑰設定")
    openai_key = st.text_input("OpenAI API Key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", type="password")

if openai_key and pinecone_key:
    # 初始化 AI 與資料庫連線
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("pubmed-db")
    client = OpenAI(api_key=openai_key)

    # 頁面分頁：搜尋存儲 vs. 文獻問答
    tab1, tab2 = st.tabs(["🔍 搜尋並存入雲端", "💬 文獻庫問答 (RAG)"])

    with tab1:
        st.subheader("從 PubMed 撈取並同步至雲端庫")
        keyword = st.text_input("輸入搜尋主題 (例如: Lung Cancer Target Therapy)")
        max_results = st.slider("預計撈取篇數", 5, 50, 10)
        
        
        if st.button("開始搜尋並存入雲端"):
            with st.spinner("正在從 PubMed 抓取資料並轉換向量..."):
                # 修正後的正確官方網址格式（加入關鍵字並避免空格出錯）
                import urllib.parse
                safe_keyword = urllib.parse.quote(keyword.strip())
                search_url = f"https://nih.gov{safe_keyword}&retmax={max_results}&retmode=json"
                
                # 執行抓取
                try:
                    id_list = requests.get(search_url).json()["esearchresult"]["idlist"]
                    st.success(f"成功撈取到 {len(id_list)} 篇文獻 ID！")
                except Exception as e:
                    st.error(f"連線失敗，請檢查網址或稍後再試。錯誤原因: {e}")
                
                # B. 根據 ID 抓取標題與摘要
                fetch_url = f"https://nih.gov{','.join(id_list)}&retmode=xml"
                response = requests.get(fetch_url)
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    pmid = article.find('.//PMID').text
                    title = article.find('.//ArticleTitle').text
                    abstract_el = article.find('.//AbstractText')
                    abstract = abstract_el.text if abstract_el is not None else "No abstract available"
                    
                    full_text = f"Title: {title}\nAbstract: {abstract}"
                    
                    # C. 使用 OpenAI 將文字轉為向量
                    emb_res = client.embeddings.create(input=full_text, model="text-embedding-3-small")
                    embedding = emb_res.data[0].embedding
                    
                    # D. 存入 Pinecone 雲端向量庫
                    index.upsert(vectors=[(pmid, embedding, {"title": title, "abstract": abstract})])
                
                st.success(f"成功將 {len(id_list)} 篇文獻永久同步至您的 Pinecone 雲端庫！")

    with tab2:
        st.subheader("對著您的個人文elem庫提問")
        user_question = st.text_input("輸入您的臨床或研究問題")
        
        if st.button("詢問 AI"):
            with st.spinner("正在檢索雲端文獻並生成解答..."):
                # A. 將使用者的提問也轉成向量
                q_emb = client.embeddings.create(input=user_question, model="text-embedding-3-small").data[0].embedding
                
                # B. 去 Pinecone 資料庫搜尋最相關的前 5 篇論文
                res = index.query(vector=q_emb, top_k=5, include_metadata=True)
                
                # C. 組合文獻背景資料
                context = ""
                for match in res["matches"]:
                    meta = match["metadata"]
                    context += f"[PMID: {match['id']}] {meta['title']}\nAbstract: {meta['abstract']}\n\n"
                
                # D. 讓 GPT-4o-mini 根據撈出來的文獻回答問題
                ai_res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "你是一位嚴謹的醫學助手。請完全根據以下提供的文獻內容回答問題，並在結尾註明 PMID 來源。"},
                        {"role": "user", "content": f"文獻背景：\n{context}\n\n問題：{user_question}"}
                    ]
                )
                st.write("### AI 的解答：")
                st.write(ai_res.choices[0].message.content)
else:
    st.info("請在左側欄位輸入您的 OpenAI 與 Pinecone API Key 以啟動系統。")
