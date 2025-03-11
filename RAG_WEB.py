import streamlit as st  # 導入Streamlit庫，用於建立網頁應用
import ollama  # 導入ollama庫，用於自然語言處理
import chromadb  # 導入chromadb庫，用於數據存儲和查詢
from chromadb.errors import InvalidCollectionException  # 導入錯誤處理類
import pandas as pd  # 導入pandas庫，用於數據分析和處理
import os  # 導入os庫，用於操作文件和目錄

# 定義持久化存儲路徑
PERSIST_DIRECTORY = "./chroma_db"

# 定義一個初始化函數，用於設置Streamlit的會話狀態
def initialize():
    # 檢查'session_state'（會話狀態）中是否已有'already_executed'這個變量
    # 這個變量用來檢查是否已經進行過一次資料庫初始化操作
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False  # 若不存在，則設置為False

    # 如果'already_executed'為False，表示還未初始化過資料庫
    if not st.session_state.already_executed:
        setup_database()  # 呼叫setup_database函數來進行資料庫的設置和數據加載
        st.session_state.already_executed = True  # 設置'already_executed'為True，表示已完成初始化

# 定義設置資料庫的函數
def setup_database():
    # 檢查數據庫目錄是否存在，不存在則建立
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    
    # 創建一個持久化的chromadb客戶端
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    try:
        # 嘗試獲取名為"fakenews"的集合
        collection = client.get_collection(name="fakenews")
        st.success("Successfully loaded existing database")
    except InvalidCollectionException:
        # 如果collection不存在，則創建並加載數據
        st.info("Creating new database...")
        file_path = 'QA.xlsx'  # 指定Excel文件的路徑和名稱
        
        # 檢查Excel文件是否存在
        if not os.path.exists(file_path):
            st.error(f"Cannot find {file_path}. Please make sure the file exists in the correct location")
            return
            
        try:
            documents = pd.read_excel(file_path, header=None)  # 使用pandas讀取Excel文件
            # 創建新集合
            collection = client.create_collection(name="fakenews")
            
            # 遍歷從Excel文件中讀取的數據，每一行代表一條記錄
            with st.spinner("Initializing database, please wait..."):
                for index, content in documents.iterrows():
                    if pd.notna(content[0]):  # 確保內容不是NaN
                        try:
                            response = ollama.embeddings(model="mxbai-embed-large", prompt=str(content[0]))  # 通過ollama生成該行文本的嵌入向量
                            collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[str(content[0])])  # 將文本和其嵌入向量添加到集合中
                        except Exception as e:
                            st.warning(f"Error processing row {index}: {str(e)}")
                st.success("Database initialization complete")
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")

# 定義獲取collection的函數
def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        return client.get_collection(name="fakenews")
    except InvalidCollectionException:
        st.error("Cannot find 'fakenews' collection. Please restart the application to initialize the database")
        return None

# 主函數，負責構建UI和處理用戶事件
def main():
    initialize()  # 呼叫初始化函數
    st.title("Fake News Detection System")  # 在網頁應用中設置標題
    st.subheader("Enter news content for verification:")  # 使用與其他副標題相同大小的文字
    user_input = st.text_area("", "")  # 創建一個文本區域供用戶輸入問題，標籤留空

    # 如果用戶點擊"送出"按鈕
    if st.button("Submit"):
        if user_input:
            collection = get_collection()  # 每次需要時獲取collection
            if collection:  # 確保collection不是None
                handle_user_input(user_input, collection)  # 處理用戶輸入，進行查詢和回答
        else:
            st.warning("Please enter news content!")  # 如果用戶沒有輸入，顯示警告消息

# 定義處理用戶輸入的函數
def handle_user_input(user_input, collection):
    try:
        with st.spinner("Analyzing..."):
            response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")  # 生成用戶輸入的嵌入向量
            results = collection.query(query_embeddings=[response["embedding"]], n_results=3)  # 在集合中查詢最相關的三個文檔
            
            if not results['documents'] or len(results['documents'][0]) == 0:
                st.warning("No relevant reference data found. Unable to analyze")
                return
                
            data = results['documents'][0]  # 獲取最相關的文檔
            
            # 修改提示詞，指定特定的輸出格式
            output = ollama.generate(
                model="mistral",
                prompt=f"""Based on this reference data: {data}, analyze the credibility of this news: {user_input}.

Your response MUST follow this exact format:
Label: [Fake or Real]
Reasoning: [Your detailed analysis explaining why the news is fake or real]

The label must be ONLY "Fake" or "Real" without any other text or qualifiers."""
            )
            
            # 解析回應以提取標籤和理由
            response_text = output['response'].strip()
            
            # 檢查是否符合預期格式
            if "Label:" in response_text and "Reasoning:" in response_text:
                # 拆分內容
                parts = response_text.split("Reasoning:", 1)
                label_part = parts[0].strip()
                reasoning_part = parts[1].strip() if len(parts) > 1 else ""
                
                # 提取標籤文字
                label_text = label_part.replace("Label:", "").strip()
                
                # 根據標籤設置顏色
                if "fake" in label_text.lower():
                    label_color = "red"
                    label_display = "Fake"
                elif "real" in label_text.lower():
                    label_color = "green"
                    label_display = "Real"
                else:
                    label_color = "orange"
                    label_display = label_text
                
                # 顯示帶有顏色框的標籤
                st.subheader("Analysis Result")
                st.markdown(
                    f"""
                    <div style="border:2px solid {label_color}; padding:10px; border-radius:5px; background-color:rgba({255 if label_color=='red' else 0}, {255 if label_color=='green' else 0}, 0, 0.1)">
                        <h3 style="color:{label_color}; margin:0">Analyze results: This news is {label_display}.</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # 顯示分析理由
                st.markdown("### Reasoning")
                st.write(reasoning_part)
            else:
                # 如果回應格式不符合預期，則直接顯示完整回應
                st.subheader("Analysis Result")
                st.write(response_text)
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("Please ensure the Ollama service is running and the 'mxbai-embed-large' and 'mistral' models are loaded")
        
if __name__ == "__main__":
    main()  # 如果直接執行此文件，則執行main函數