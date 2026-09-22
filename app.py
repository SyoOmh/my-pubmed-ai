import streamlit as st
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import datetime
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
        st.subheader("從 PubMed 搜尋文獻")
        keyword = st.text_input("輸入搜尋主題 (例如: Lung Cancer Target Therapy)")
        max_results = st.slider("預計撈取篇數上限", 5, 50, 10)

        st.markdown("**發表日期區間**（依 PubMed 出版日期篩選）")
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "起始日期", value=datetime.date(2015, 1, 1), key="start_date"
            )
        with date_col2:
            end_date = st.date_input(
                "結束日期", value=datetime.date.today(), key="end_date"
            )

        # 用 session_state 保存搜尋結果，這樣使用者勾選文章、按下存入按鈕造成頁面重新整理時，
        # 搜尋結果不會消失（不需要重新打 PubMed API）
        if "fetched_articles" not in st.session_state:
            st.session_state.fetched_articles = []

        if st.button("開始搜尋"):
            if not keyword.strip():
                st.warning("請先輸入搜尋主題！")
            elif start_date > end_date:
                st.warning("起始日期不能晚於結束日期，請重新選擇！")
            else:
                with st.spinner("正在從 PubMed 抓取資料..."):
                    # 1. 安全地對關鍵字進行網址編碼
                    safe_keyword = urllib.parse.quote(keyword.strip())

                    # 2. 正確的官方 PubMed eSearch API 網址
                    # mindate / maxdate + datetype=pdat 用來依「發表日期」篩選區間
                    search_url = (
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                        f"?db=pubmed&term={safe_keyword}&retmax={max_results}&retmode=json"
                        f"&datetype=pdat"
                        f"&mindate={start_date.strftime('%Y/%m/%d')}"
                        f"&maxdate={end_date.strftime('%Y/%m/%d')}"
                    )

                    try:
                        # 執行搜尋取得 ID 列表
                        search_res = requests.get(search_url, timeout=15)
                        search_res.raise_for_status()
                        id_list = search_res.json()["esearchresult"]["idlist"]

                        if not id_list:
                            st.session_state.fetched_articles = []
                            st.info("在此日期區間內，找不到符合該關鍵字的文獻。")
                        else:
                            # 找到幾篇就是幾篇，不會、也不需要強迫湊到篇數上限
                            if len(id_list) < max_results:
                                st.success(
                                    f"在此日期區間內共找到 {len(id_list)} 篇文獻"
                                    f"（低於您設定的上限 {max_results} 篇，已全數取得）。"
                                )
                            else:
                                st.success(
                                    f"成功撈取到 {len(id_list)} 篇文獻 ID"
                                    f"（已達您設定的上限 {max_results} 篇，可能還有更多符合條件的文獻）。"
                                )

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

                            fetched_articles = []  # 只做搜尋、解析，這裡不呼叫 embedding、也不存 Pinecone

                            # 解析每篇文獻的標題、摘要、年份、研究類型
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

                                # 解析研究年份（優先用 PubDate/Year，找不到時退而求其次用 MedlineDate）
                                year = "年份未知"
                                year_el = article.find(".//JournalIssue/PubDate/Year")
                                if year_el is not None and year_el.text:
                                    year = year_el.text
                                else:
                                    medline_date_el = article.find(".//JournalIssue/PubDate/MedlineDate")
                                    if medline_date_el is not None and medline_date_el.text:
                                        year = medline_date_el.text[:4]  # 取前 4 碼當作年份

                                # 解析研究類型（一篇文獻可能有多個 PublicationType）
                                pub_types = [
                                    el.text
                                    for el in article.findall(".//PublicationTypeList/PublicationType")
                                    if el.text
                                ]
                                if not pub_types:
                                    pub_types = ["未標註類型"]

                                fetched_articles.append(
                                    {
                                        "pmid": pmid,
                                        "title": title,
                                        "abstract": abstract,
                                        "year": year,
                                        "pub_types": pub_types,
                                    }
                                )

                            st.session_state.fetched_articles = fetched_articles

                    except requests.exceptions.RequestException as e:
                        st.error(f"連線 PubMed 失敗，請稍後再試。錯誤原因: {e}")
                    except Exception as e:
                        st.error(f"系統執行失敗，請檢查 API Key 或稍後再試。錯誤原因: {e}")

        # ---- 顯示搜尋結果，讓使用者自己勾選要存入雲端的文章 ----
        if st.session_state.fetched_articles:
            st.markdown("### 📄 搜尋結果（請勾選要存入雲端的文獻）")

            def _apply_select_all():
                # 「全選/全不選」被點擊時，直接覆寫每一篇文章勾選框在 session_state 裡的值，
                # 這樣下一次重新整理時，各個勾選框才會真的跟著變動
                # （單純傳 value= 參數沒用，因為這些勾選框已經有 key，Streamlit 會優先讀 session_state）
                new_value = st.session_state.select_all
                for art in st.session_state.fetched_articles:
                    st.session_state[f"select_{art['pmid']}"] = new_value

            st.checkbox(
                "全選 / 全不選",
                value=True,
                key="select_all",
                on_change=_apply_select_all,
            )

            for art in st.session_state.fetched_articles:
                type_badges = " ".join(f"`{t}`" for t in art["pub_types"])
                col_check, col_info = st.columns([1, 9])
                with col_check:
                    st.checkbox(
                        "選取",
                        value=True,  # 只在該篇文章第一次出現、還沒有 session_state 值時生效
                        key=f"select_{art['pmid']}",
                        label_visibility="collapsed",
                    )
                with col_info:
                    st.markdown(
                        f"**{art['title']}**  \n"
                        f"🗓️ `{art['year']}` &nbsp;&nbsp; 🧪 {type_badges} &nbsp;&nbsp; "
                        f"[PMID: {art['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{art['pmid']}/)"
                    )
                st.divider()

            if st.button("☁️ 將勾選的文獻存入 Pinecone 雲端"):
                selected_articles = [
                    art
                    for art in st.session_state.fetched_articles
                    if st.session_state.get(f"select_{art['pmid']}", False)
                ]

                if not selected_articles:
                    st.warning("您尚未勾選任何文獻，請至少選擇一篇再存入。")
                else:
                    with st.spinner(f"正在將 {len(selected_articles)} 篇文獻轉換向量並存入雲端..."):
                        try:
                            saved_count = 0
                            for art in selected_articles:
                                full_text = f"Title: {art['title']}\nAbstract: {art['abstract']}"

                                # 使用 OpenAI 將文字轉為向量（只針對使用者勾選的文章才呼叫，節省費用）
                                emb_res = client.embeddings.create(
                                    input=full_text, model="text-embedding-3-small"
                                )
                                embedding = emb_res.data[0].embedding

                                # 存入 Pinecone 雲端向量庫
                                index.upsert(
                                    vectors=[
                                        {
                                            "id": art["pmid"],
                                            "values": embedding,
                                            "metadata": {
                                                "title": art["title"],
                                                "abstract": art["abstract"],
                                                "year": art["year"],
                                                "pub_types": ", ".join(art["pub_types"]),
                                            },
                                        }
                                    ]
                                )
                                saved_count += 1

                            st.success(f"成功將 {saved_count} 篇文獻存入您的 Pinecone 雲端庫！")
                        except Exception as e:
                            st.error(f"存入雲端失敗，請檢查 API Key 或稍後再試。錯誤原因: {e}")

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
