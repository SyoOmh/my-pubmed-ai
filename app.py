
import streamlit as st
import requests
import xml.etree.ElementTree as ET
import urllib.parse
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
            if not keyword.strip():
                st.warning("請先輸入搜尋主題！")
            else:
                with st.spinner("正在從 PubMed 抓取資料並轉換向量..."):
                    # 1. 安全地對關鍵字進行網址編碼
                    safe_keyword = urllib.parse.quote(keyword.strip())
 
                    # 2. 正確的官方 PubMed eSearch API 網址
                    # 原本的程式碼寫成 "https://nih.gov{safe_keyword}..."，
                    # 這會把整段查詢字串當成主機名稱，導致 DNS 解析失敗。
                    # 正確的 base URL 是 NCBI eutils 的 esearch.fcgi
                    search_url = (
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                        f"?db=pubmed&term={safe_keyword}&retmax={max_results}&retmode=json"
                    )
 
                    try:
                        # 執行搜尋取得 ID 列表
                        search_res = requests.get(search_url, timeout=15)
                        search_res.raise_for_status()
                        id_list = search_res.json()["esearchresult"]["idlist"]
 
                        if not id_list:
                            st.info("找不到符合該關鍵字的文獻。")
                        else:
                            st.success(f"成功撈取到 {len(id_list)} 篇文獻 ID！")
 
                            # 3. 正確的官方 PubMed eFetch API 網址
                            ids_str = ",".join(id_list)
                            fetch_url = (
                                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                                f"?db=pubmed&id={ids_str}&retmode=xml"
                            )
 
                            # 執行撈取 XML 內容
                            response = requests.get(fetch_url, timeout=30)
                            response.raise_for_status()
                            root = ET.fromstring(response.content)
 
                            saved_count = 0
                            # 開始解析文獻並寫入雲端
                            for article in root.findall(".//PubmedArticle"):
                                pmid_el = article.find(".//PMID")
                                if pmid_el is None:
                                    continue
                                pmid = pmid_el.text
 
                                title_el = article.find(".//ArticleTitle")
                                title = title_el.text if title_el is not None and title_el.text else "No title available"
 
                                # 有些摘要是分段的 (AbstractText 可能出現多次)，這裡把它們合併起來
                                abstract_parts = [
                                    (el.text or "") for el in article.findall(".//AbstractText")
                                ]
                                abstract = " ".join(p for p in abstract_parts if p).strip()
                                if not abstract:
                                    abstract = "No abstract available"
 
                                full_text = f"Title: {title}\nAbstract: {abstract}"
 
                                # C. 使用 OpenAI 將文字轉為向量
                                emb_res = client.embeddings.create(
                                    input=full_text, model="text-embedding-3-small"
                                )
                                embedding = emb_res.data[0].embedding  # data 是 list，要取第一筆
 
                                # D. 存入 Pinecone 雲端向量庫
                                index.upsert(
                                    vectors=[
                                        {
                                            "id": pmid,
                                            "values": embedding,
                                            "metadata": {"title": title, "abstract": abstract},
                                        }
                                    ]
                                )
                                saved_count += 1
 
                            st.success(f"成功將 {saved_count} 篇文獻永久同步至您的 Pinecone 雲端庫！")
 
                    except requests.exceptions.RequestException as e:
                        st.error(f"連線 PubMed 失敗，請稍後再試。錯誤原因: {e}")
                    except Exception as e:
                        st.error(f"系統執行失敗，請檢查 API Key 或稍後再試。錯誤原因: {e}")
 
    with tab2:
        st.subheader("對著您的個人文獻庫提問")
        user_question = st.text_input("輸入您的臨床或研究問題")
 
        if st.button("詢問 AI"):
            if not user_question.strip():
                st.warning("請先輸入您的問題！")
            else:
                with st.spinner("正在檢索雲端文獻並生成解答..."):
                    try:
                        # A. 將使用者的提問也轉成向量
                        q_emb_res = client.embeddings.create(
                            input=user_question, model="text-embedding-3-small"
                        )
                        q_emb = q_emb_res.data[0].embedding  # 同樣要取 data[0]
 
                        # B. 去 Pinecone 資料庫搜尋最相關的前 5 篇論文
                        res = index.query(vector=q_emb, top_k=5, include_metadata=True)
 
                        # C. 組合文獻背景資料
                        context = ""
                        matches = res.get("matches") if isinstance(res, dict) else res.matches
                        if matches:
                            for match in matches:
                                meta = match["metadata"] if isinstance(match, dict) else match.metadata
                                match_id = match["id"] if isinstance(match, dict) else match.id
                                context += f"[PMID: {match_id}] {meta['title']}\nAbstract: {meta['abstract']}\n\n"
 
                            # D. 讓 GPT-4o-mini 根據撈出來的文獻回答問題
                            ai_res = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "你是一位嚴謹的醫學助手。請完全根據以下提供的文獻內容回答問題，並在結尾註明 PMID 來源。",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"文獻背景：\n{context}\n\n問題：{user_question}",
                                    },
                                ],
                            )
                            st.write("### AI 的解答：")
                            st.write(ai_res.choices[0].message.content)  # 要取 choices[0]
                        else:
                            st.warning("您的 Pinecone 雲端庫中目前沒有相關文獻，請先至 Tab 1 撈取文獻。")
 
                    except Exception as e:
                        st.error(f"問答生成失敗，請檢查金鑰或稍後再試。錯誤原因: {e}")
else:
    st.info("請在左側欄位輸入您的 OpenAI 與 Pinecone API Key 以啟動系統。")
