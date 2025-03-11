import streamlit as st
import ollama
import chromadb
from chromadb.errors import InvalidCollectionException
import pandas as pd
import os
import json
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# 定義持久化存儲路徑
PERSIST_DIRECTORY = "./chroma_db"

# 定義一個初始化函數，用於設置Streamlit的會話狀態
def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False

    if not st.session_state.already_executed:
        setup_database()
        st.session_state.already_executed = True

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

# 定義處理單條新聞的函數
def analyze_news(news_content, collection):
    try:
        # 使用RAG方法
        response = ollama.embeddings(prompt=news_content, model="mxbai-embed-large")  # 生成用戶輸入的嵌入向量
        results = collection.query(query_embeddings=[response["embedding"]], n_results=3)  # 在集合中查詢最相關的三個文檔
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return {"label": "Unknown", "reasoning": "No relevant reference data found"}
            
        data = results['documents'][0]  # 獲取最相關的文檔
        
        # 帶有RAG參考數據的提示詞
        prompt = f"""Based on this reference data: {data}, analyze the credibility of this news: {news_content}.

Your response MUST follow this exact format:
Label: [Fake or Real]
Reasoning: [Your detailed analysis explaining why the news is fake or real]

The label must be ONLY "Fake" or "Real" without any other text or qualifiers."""
        
        # 調用語言模型進行分析
        output = ollama.generate(model="mistral", prompt=prompt)
        
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
            
            # 根據標籤設置結果
            if "fake" in label_text.lower():
                return {"label": "Fake", "reasoning": reasoning_part}
            elif "real" in label_text.lower():
                return {"label": "Real", "reasoning": reasoning_part}
            else:
                return {"label": label_text, "reasoning": reasoning_part}
        else:
            # 如果回應格式不符合預期，則嘗試從文本中提取標籤
            if "fake" in response_text.lower():
                return {"label": "Fake", "reasoning": response_text}
            elif "real" in response_text.lower():
                return {"label": "Real", "reasoning": response_text}
            else:
                return {"label": "Unknown", "reasoning": response_text}
            
    except Exception as e:
        return {"label": "Error", "reasoning": str(e)}

# 定義從JSON文件讀取新聞內容的函數
def read_news_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 假設JSON文件中包含新聞內容的欄位為"text"或"content"
            # 根據您的數據格式調整
            if "text" in data:
                return data["text"]
            elif "content" in data:
                return data["content"]
            else:
                # 如果找不到預期的欄位，則返回整個JSON內容
                return str(data)
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

# 定義批量處理函數
def process_dataset(dataset_dir, is_fake, collection, results):
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.json')]
    total_files = len(files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(files):
        status_text.text(f"Processing {'fake' if is_fake else 'real'} news {i+1}/{total_files}: {file_path}")
        news_content = read_news_from_json(file_path)
        if news_content and news_content != "":
            analysis = analyze_news(news_content, collection)
            result = {
                "file": os.path.basename(file_path),
                "true_label": "Fake" if is_fake else "Real",
                "predicted_label": analysis["label"],
                "reasoning": analysis["reasoning"]
            }
            results.append(result)
        progress_bar.progress((i + 1) / total_files)
        # 加入短暫延遲，避免API限制
        time.sleep(0.5)
    
    return results

# 計算評估指標
def calculate_metrics(true_labels, predicted_labels):
    # 過濾掉非法標籤
    valid_pairs = [(t, p) for t, p in zip(true_labels, predicted_labels) if p in ["Fake", "Real"]]
    
    if not valid_pairs:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "valid_sample_size": 0
        }
    
    filtered_true = [pair[0] for pair in valid_pairs]
    filtered_pred = [pair[1] for pair in valid_pairs]
    
    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(filtered_true, filtered_pred, average='weighted')
    cm = confusion_matrix(filtered_true, filtered_pred, labels=["Fake", "Real"])
    
    # 計算每個類別的精度
    fake_idx = 0  # 假設"Fake"在labels中的索引為0
    real_idx = 1  # 假設"Real"在labels中的索引為1
    
    fake_precision = cm[fake_idx, fake_idx] / (cm[fake_idx, fake_idx] + cm[real_idx, fake_idx]) if (cm[fake_idx, fake_idx] + cm[real_idx, fake_idx]) > 0 else 0
    real_precision = cm[real_idx, real_idx] / (cm[real_idx, real_idx] + cm[fake_idx, real_idx]) if (cm[real_idx, real_idx] + cm[fake_idx, real_idx]) > 0 else 0
    
    fake_recall = cm[fake_idx, fake_idx] / (cm[fake_idx, fake_idx] + cm[fake_idx, real_idx]) if (cm[fake_idx, fake_idx] + cm[fake_idx, real_idx]) > 0 else 0
    real_recall = cm[real_idx, real_idx] / (cm[real_idx, real_idx] + cm[real_idx, fake_idx]) if (cm[real_idx, real_idx] + cm[real_idx, fake_idx]) > 0 else 0
    
    # 計算F1分數
    fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0
    real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "valid_sample_size": len(valid_pairs),
        "per_class": {
            "Fake": {
                "precision": fake_precision,
                "recall": fake_recall,
                "f1": fake_f1
            },
            "Real": {
                "precision": real_precision,
                "recall": real_recall,
                "f1": real_f1
            }
        }
    }

# 生成混淆矩陣視覺化
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    return fig

# 主函數，負責構建UI和處理用戶事件
def main():
    initialize()  # 呼叫初始化函數
    st.title("Fake News Detection System")  # 在網頁應用中設置標題
    
    # 創建選項卡
    tab1, tab2 = st.tabs(["Single News Analysis", "Batch Processing"])
    
    with tab1:
        st.subheader("Enter news content for verification:")
        user_input = st.text_area("", "")
        
        # 如果用戶點擊"送出"按鈕
        if st.button("Submit", key="submit_single"):
            if user_input:
                collection = get_collection()
                if collection:
                    handle_user_input(user_input, collection)
            else:
                st.warning("Please enter news content!")
    
    with tab2:
        st.subheader("Batch Process Dataset")
        
        # 設定默認路徑
        default_fake_path = "/home/server/Dai/RAG_fakenewsdetection/dataset/politifact/pol_fake_news"
        default_real_path = "/home/server/Dai/RAG_fakenewsdetection/dataset/politifact/pol_real_news"
        
        # 輸入目錄路徑
        fake_news_dir = st.text_input("Fake News Directory Path:", default_fake_path)
        real_news_dir = st.text_input("Real News Directory Path:", default_real_path)
        
        # 批量處理按鈕
        if st.button("Start Batch Processing", key="start_batch"):
            if os.path.exists(fake_news_dir) and os.path.exists(real_news_dir):
                collection = get_collection()
                if collection:
                    with st.spinner("Processing datasets..."):
                        results = []
                        
                        # 處理假新聞
                        st.subheader("Processing Fake News")
                        results = process_dataset(fake_news_dir, True, collection, results)
                        
                        # 處理真新聞
                        st.subheader("Processing Real News")
                        results = process_dataset(real_news_dir, False, collection, results)
                        
                        # 計算評估指標
                        all_true_labels = [result["true_label"] for result in results]
                        all_predicted_labels = [result["predicted_label"] for result in results]
                        metrics = calculate_metrics(all_true_labels, all_predicted_labels)
                        
                        # 顯示評估指標
                        st.subheader("Evaluation Metrics")
                        metrics_df = pd.DataFrame({
                            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                            "Value": [
                                f"{metrics['accuracy']:.4f}",
                                f"{metrics['precision']:.4f}",
                                f"{metrics['recall']:.4f}",
                                f"{metrics['f1']:.4f}"
                            ]
                        })
                        st.table(metrics_df)
                        
                        # 顯示每個類別的指標
                        st.subheader("Per-Class Metrics")
                        per_class_df = pd.DataFrame({
                            "Class": ["Fake", "Real"],
                            "Precision": [
                                f"{metrics['per_class']['Fake']['precision']:.4f}",
                                f"{metrics['per_class']['Real']['precision']:.4f}"
                            ],
                            "Recall": [
                                f"{metrics['per_class']['Fake']['recall']:.4f}",
                                f"{metrics['per_class']['Real']['recall']:.4f}"
                            ],
                            "F1 Score": [
                                f"{metrics['per_class']['Fake']['f1']:.4f}",
                                f"{metrics['per_class']['Real']['f1']:.4f}"
                            ]
                        })
                        st.table(per_class_df)
                        
                        # 顯示混淆矩陣
                        st.subheader("Confusion Matrix")
                        cm = metrics["confusion_matrix"]
                        cm_df = pd.DataFrame(cm, columns=["Predicted Fake", "Predicted Real"], index=["Actual Fake", "Actual Real"])
                        st.table(cm_df)
                        
                        # 顯示混淆矩陣視覺化
                        fig = plot_confusion_matrix(cm, title='Confusion Matrix')
                        st.pyplot(fig)
                        
                        # 將結果轉換為DataFrame並顯示
                        results_df = pd.DataFrame(results)
                        st.subheader("Detailed Results")
                        st.dataframe(results_df)
                        
                        # 提供下載結果的選項
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="batch_processing_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("One or both of the specified directories do not exist. Please check the paths.")

# 定義處理用戶輸入的函數 (用於單條新聞分析)
def handle_user_input(user_input, collection):
    try:
        with st.spinner("Analyzing..."):
            analysis = analyze_news(user_input, collection)
            
            # 根據標籤設置顏色
            if analysis["label"] == "Fake":
                label_color = "red"
            elif analysis["label"] == "Real":
                label_color = "green"
            else:
                label_color = "orange"
            
            # 顯示帶有顏色框的標籤
            st.subheader("Analysis Result")
            st.markdown(
                f"""
                <div style="border:2px solid {label_color}; padding:10px; border-radius:5px; background-color:rgba({255 if label_color=='red' else 0}, {255 if label_color=='green' else 0}, 0, 0.1)">
                    <h3 style="color:{label_color}; margin:0">Analyze results: This news is {analysis["label"]}.</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # 顯示分析理由
            st.markdown("### Reasoning")
            st.write(analysis["reasoning"])
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("Please ensure the Ollama service is running and the 'mxbai-embed-large' and 'mistral' models are loaded")
        
if __name__ == "__main__":
    main()  # 如果直接執行此文件，則執行main函數