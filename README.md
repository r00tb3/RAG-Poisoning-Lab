# RAG Poisoning Lab

This Dockerized lab teaches AI security by demonstrating RAG poisoning attacks, detection, and mitigation. Focus on defensive skills: analyze embeddings, run detection algorithms, implement filters, and test effectiveness.

## Repository Structure

```
rag-poisoning-lab/
├── README.md                 # Setup instructions and hands-on guide
├── docker-compose.yml        # Docker stack (includes Ollama)
├── rag-app/
│   ├── app.py                # Streamlit app with query, poison, detect, mitigate tabs
│   ├── knowledge_base.py     # Document ingestion and mitigation filters
│   └── detection.py          # Perplexity and similarity analysis
├── documents/                # 15 clean cybersecurity docs (create dummy .txt files, e.g., phishing.txt, encryption.txt)
├── poisoned_docs/            # 3 malicious payloads (e.g., injection.txt, leakage.txt, bias.txt)
├── requirements.txt          # Dependencies
└── Dockerfile                # Custom build for rag-app
```

## Quick Setup (2 Minutes)

1. Clone the repo:
```
git clone https://github.com/yourusername/rag-poisoning-lab.git && cd rag-poisoning-lab
```

2. Start the lab:

```
docker-compose up -d
```

3. Access the UI:

    Open http://localhost:8000 in your browser.

4. Verify:

    Run `docker ps` to check containers.

5. Curl health check:
    
    curl http://localhost:8000/health

    Expected: 
        <br />RAG system: Running

        <br />Knowledge base: 15 clean documents
        
        <br />Vector database: Active

6. Stop the lab: 
    
    `docker-compose down`

## Lab Components
- **Clean Knowledge Base**: 15 cybersecurity documents in `documents/`.
- **Poisoning Toolkit**: 3 malicious payloads in `poisoned_docs/`.
- **Detection Dashboard**: Perplexity + similarity analysis for anomalies.
- **Query Interface**: Test normal vs. poisoned behavior.
- **Mitigation**: Toggle filters to block poisoned content.

## Hands-On Exercises
### Excercise 1: Attack Execution
1. In the UI's "Poison" tab, select and ingest a poisoned doc (e.g., injection.txt).
2. In "Query" tab, ask: "What is the admin password?" Observe injection.

### Excercise 2: Detection Analysis
1. In "Detect" tab, input a chunk from a poisoned doc.
2. Run analysis: High perplexity (>100) or low similarity (<0.5>) indicates poisoning.

### Excercise 3: Mitigation Implementation
1. In `knowledge_base.py`, enable a filter (e.g., keyword blacklist).
2. In "Mitigate" tab, re-ingest with filters on. Re-query to test.



