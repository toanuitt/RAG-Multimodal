# Unstructured Data Processing Pipeline
This project implements a pipeline for processing unstructured data and generating answers using Large Language Models (LLMs) and vector databases.


**Architecture:**

![User_Centric_RAG_Architecture](https://github.com/toanuitt/RAG-Multimodal/blob/27139513c20424e0649028d14c205975a1294475/Images/Pipeline.png)

**Components**

- Unstructured: A tool or library for processing various unstructured data formats.

- LLM: Large Language Model(s) used for summarization and answer generation.

- Chroma: Vector database for storing and retrieving embeddings.

**Usage**
- Step 1. Clone the repository:
```bash
git clone https://github.com/toanuitt/RAG-Multimodal.git
cd RAG-Multimodal
```
- Step 2. Create API key in [unstructed.io](https://unstructured.io/)
- Step 3. Create API key in [Azure](https://portal.azure.com/#home) for embedding and chating
- Step 4. Install dependencies:
```bash
pip install -r requirements.txt
```
- Step 5. Run chatbot
```bash
streamlit run app.py
```
**References**

https://www.analyticsvidhya.com/blog/2024/09/guide-to-building-multimodal-rag-systems/

https://python.langchain.com/docs/tutorials/rag/
