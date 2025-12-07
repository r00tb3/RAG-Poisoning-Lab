from langchain_core.documents import Document

def ingest_documents(path):
    # Placeholder: Use DirectoryLoader as in app.py
    pass  # Integrated in app.py for simplicity

def apply_mitigation_filters(docs, mitigations):
    filtered = []
    for doc in docs:
        content = doc.page_content.lower()
        if mitigations.get('keyword_filter') and any(kw in content for kw in ["ignore all", "critical directive"]):
            continue  # Reject poisoned
        filtered.append(doc)
    return filtered
