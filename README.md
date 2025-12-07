# 🛡️ RAG Poisoning Lab — Educational AI Security Exercise

This repository contains a hands-on lab environment for learning **RAG (Retrieval-Augmented Generation) data poisoning attacks**, detection techniques, and mitigation strategies. It is designed for students, researchers, and security practitioners who want practical experience with adversarial manipulation of AI retrieval systems.

> **⚠️ Educational Purpose Only**  
> This lab is intended *strictly* for learning, research, and training.  
> Do **not** use these techniques on any system you do not own or do not have explicit permission to test.

---

## 📘 Overview

The lab demonstrates how a RAG system can be poisoned by injecting malicious documents into a vector database. You will:

- Build a simple RAG pipeline using FAISS and Ollama  
- Launch a Streamlit interface to query clean documents  
- Execute poisoning attacks by ingesting malicious files  
- Detect suspicious chunks using perplexity + similarity scoring  
- Apply mitigations such as keyword filters and state resets  

This lab mirrors real-world RAG risks seen in enterprise AI applications.

---

## 🚀 Quick Start (Local Setup)

1. Clone the Repository
```bash
git clone https://github.com/r00tb3/RAG-Poisoning-Lab.git
cd rag-poisoning-lab
```

2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt --prefer-binary --no-cache-dir
```

4. Install and Start Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b-instruct-q4_K_M
ollama pull nomic-embed-text
ollama serve    # keep this running
```

5. Run the Streamlit App
```bash
streamlit run rag_app/app_streamlit.py --server.port 8000
```

6. Open your browser and visit:
http://localhost:8000

7. Folder Structure
```bash
rag-poisoning-lab/
│── rag_app/
│   ├── app_streamlit.py      # Main RAG UI
│   ├── knowledge_base.py     # Ingestion + mitigation logic
│   └── detection.py          # Perplexity + similarity detection
│
├── documents/                # Clean knowledge base (15 files)
├── poisoned_docs/            # Malicious payloads (injection, bias, leakage)
├── requirements.txt
└── README.md
```





## 🧪 What You Will Learn ?

🔴 Attack
- Poisoning via malicious document ingestion

- How semantic similarity causes poisoned chunks to be retrieved

🔍 Detect
- Perplexity scoring for unnatural or adversarial text


- Embedding similarity to identify outliers in vector space


🛡️ Mitigate
- Keyword filtering during ingestion


- Rebuilding FAISS index to purge poisoned content



- Validating clean behavior after mitigation






- These techniques are essential for securing real-world AI applications.




## 📜 Disclaimer
_This project is provided for educational, academic, and training purposes only.
Do not use any part of this repository to attack systems without explicit written permission.
The authors assume no liability for misuse._

## ⭐ Contributing Suggestions and improvements are welcome.
#### _You can submit issues or pull requests to expand:_


- Additional poisoning techniques

- New detection modules

- Hardening strategies for RAG pipelines
