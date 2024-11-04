## Rag2_Assignment


## **Enhanced Medical RAG Chatbot with Local Knowledge Sources and LLaMA3**
## **Overview**
This project showcases an advanced Retrieval-Augmented Generation (RAG) chatbot designed to respond to medical queries. It combines LLaMA3 and GPT-2 language models with a custom knowledge base derived from local medical books and articles (stored as PDFs) alongside Wikipedia. Built with Streamlit, this chatbot runs locally, offering reliable, contextually accurate responses suitable for medical information queries.


---
## **Features**
- **Enhanced RAG System**: Uses both Wikipedia and locally stored medical sources for more specific and accurate responses.
- **Multiple LLM Options:** Allows switching between LLaMA3 and GPT-2 for generating responses.
- **Flexible Query Mode:** Users can choose between RAG mode (local content + Wikipedia + LLM) and LLM-only mode.
- **User-Friendly UI:** Built with Streamlit for seamless user interaction.
- **Offline Accessibility:** By using local PDFs, the chatbot can answer queries without an internet connection.
- **Efficient Document Retrieval:** Incorporates FAISS indexing for fast access to relevant medical information.
- **Query Caching:** Stores recent queries and responses to minimize re-processing, enhancing speed and efficiency.

---

## **Prerequisites**
- **Python** 3.8+
- **Ollama:** Required for running LLaMA3 models locally.
- **Install Streamlit and Required Packages:**

```bash
pip install streamlit transformers sentence-transformers wikipedia requests faiss-cpu
```

## **Set Up Instructions**

- **Clone the Repository:**

```bash
git clone https://github.com/Ashwinipuranik30/Rag2_Assignment.git
```

- **Set Up a Virtual Environment:**

```bash

python -m venv myenv
source myenv/bin/activate  # On Windows use myenv\Scripts\activate
```

- **Install Dependencies:**

```bash

pip install -r requirements.txt
```

- **Download the LLaMA3 Model:**

```bash
ollama pull llama3
```
- **Start the Ollama Service to ensure LLaMA3 can be used locally.**

- **Run the Streamlit App:**

```bash
streamlit run rag2.py
```


## **Technologies Used**
- Python 3.8+
- Streamlit: Provides an interactive UI for easy-to-use interactions.
- Ollama: Supports running LLaMA3 models locally.
- Transformers Library: Used for GPT-2 text generation.
- Sentence Transformers: Embedding Wikipedia and local documents for similarity search.
- Wikipedia Library: Fetches Wikipedia content as needed.
- FAISS: Optimizes document retrieval through efficient embedding indexing.

  
## **Challenges Faced**

- **Handling Large Medical PDFs:** Developed logic to parse, embed, and retrieve text from large medical documents. FAISS indexing was crucial for optimizing retrieval times from the local knowledge base.

- **Caching for Query Efficiency:** Implemented a query caching mechanism to store recent queries and responses, reducing redundant processing and improving response times.

- **Multi-Turn Conversation Handling:** Designed a mechanism to track conversation history, making it possible to handle multi-turn interactions for more context-aware responses.

- **Performance Optimization:** Achieved a balance between retrieval accuracy and response time by using embedding normalization and FAISS indexing, which helps reduce retrieval latency.

## **Usage Tips**
- **RAG Mode:** Ideal for fact-based queries that require additional context.
- **Standalone LLM Mode:** Suitable for general, conversational, or creative responses.
- **Switching Modes:** Use the Streamlit UI to switch between RAG and non-RAG modes and between LLaMA3 and GPT-2 as needed.


## **Acknowledgments**
This project leveraged open-source tools such as FAISS for efficient document retrieval, Transformers for language model support, and Ollama for LLaMA3 integration. These tools were essential in building a robust, offline-capable RAG system for medical queries.






