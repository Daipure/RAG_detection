# Fake News Detection System

一個基於 AI 的新聞真偽檢測系統，能夠分析新聞內容並判斷其可信度。

## 功能特點

- 使用向量資料庫儲存參考資料集
- 利用嵌入模型進行語義相似度比對
- 透過 Mistral 大型語言模型進行內容分析
- 簡潔易用的網頁界面

## 系統架構

該系統結合了以下技術：
- **Streamlit**：用於建立互動式網頁介面
- **Ollama**：用於本地部署和運行 AI 模型
- **ChromaDB**：向量資料庫，用於儲存和檢索參考資料
- **Pandas**：用於資料處理和分析

## 安裝需求

### 前置條件

- Python 3.8 或更高版本
- [Ollama](https://ollama.ai/) 已安裝並運行
- 以下 AI 模型需預先下載到 Ollama：
  - `mxbai-embed-large`（用於生成嵌入向量）
  - `mistral`（用於分析和推理）

### 安裝步驟

1. 複製此專案到本地：
```bash
git clone https://github.com/Daipure/fake-news-detection.git
cd fake-news-detection
```

2. 安裝所需套件：
```
ollama、chromadb、streamlit
```

3. 準備參考資料集：
   - 將包含參考資料的 Excel 文件命名為 `QA.xlsx` 放置於專案根目錄
   - 文件格式應為無標題的單一欄位，每行包含一條參考資料

## 使用方法

1. 確保 Ollama 服務正在運行

2. 啟動應用程式：
```bash
streamlit run app.py
```

3. 在瀏覽器中打開顯示的網址（通常為 http://localhost:8501）

4. 在文字框中輸入要驗證的新聞內容，然後點擊「Submit」

5. 系統將分析內容並顯示結果：
   - 標示為 **Real**（真實）或 **Fake**（虛假）
   - 提供詳細的分析理由

## 運作原理

1. **初始化階段**：
   - 系統首次運行時會檢查並創建向量資料庫
   - 從 QA.xlsx 中加載參考資料並生成向量嵌入

2. **分析階段**：
   - 用戶輸入的新聞內容轉換為向量表示
   - 系統搜索最相似的參考資料
   - 通過 Mistral 模型分析內容並給出判斷

## 故障排除

如遇到問題，請檢查：
- Ollama 服務是否正在運行
- 所需模型是否已下載
- QA.xlsx 文件是否存在且格式正確
- 資料庫目錄 (./chroma_db) 權限是否正確

