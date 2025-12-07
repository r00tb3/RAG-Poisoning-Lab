import streamlit as st
from pypdf import PdfReader
import ollama
import faiss
import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
import sys

# ───── FUTURE-PROOF GPT-2 ─────
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2Tokenizer

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_gpt2()

# ───── KEYWORD FILTER (this one actually stops attacks) ─────
def apply_mitigation_filters(docs, mitigations):
    if not mitigations.get('keyword_filter', False):
        return docs

    BLOCKED_PATTERNS = [
        "ignore all previous", "ignore previous", "disregard previous", "forget previous",
        "override previous", "bypass previous", "new directive", "critical directive",
        "you are now", "now you are", "from now on", "developer mode", "dan mode",
        "jailbreak", "unrestricted", "no restrictions", "do anything now", "never refuse",
        "leak", "secret flag", "admin password", "private key", "secret", "flag",
        "ignore all", "ignore any", "disregard all", "follow only my", "system override"
    ]

    filtered = []
    blocked = 0
    for doc in docs:
        content = doc['content'].lower()
        if any(p in content for p in BLOCKED_PATTERNS):
            blocked += 1
            continue
        filtered.append(doc)

    if blocked:
        st.error(f"MITIGATION ACTIVE: Blocked {blocked} poisoned chunk(s)")
    return filtered

# ───── Rest of functions (unchanged) ─────
def calculate_perplexity(text):
    if len(text.strip()) < 15:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(torch.exp(loss).item())

def get_embedding(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']

def check_similarity(text, avg_clean_emb):
    if avg_clean_emb is None: return None
    emb = get_embedding(text)
    return 1 - cosine(emb, avg_clean_emb)

def get_average_clean_embedding(docs):
    embs = [get_embedding(d['content']) for d in docs]
    return np.mean(embs, axis=0)

# Session state
if 'index' not in st.session_state:
    st.session_state.index = None
    st.session_state.docs = []
    st.session_state.clean_avg_emb = None
    st.session_state.mitigations = {'keyword_filter': False}

# Load documents
@st.cache_resource
def init_db(poison=False, mitigations=None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = "poisoned_docs" if poison else "documents"
    path = os.path.join(base, folder)

    if not os.path.exists(path):
        st.error(f"Folder not found: {path}")
        return

    docs = []
    for file in os.listdir(path):
        fp = os.path.join(path, file)
        try:
            if file.lower().endswith('.pdf'):
                text = ''.join(p.extract_text() or "" for p in PdfReader(fp).pages)
            else:
                with open(fp, 'r', encoding='utf-8') as f:
                    text = f.read()
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
            for c in chunks:
                if c.strip():
                    docs.append({'content': c, 'source': file})
        except: pass

    # APPLY MITIGATIONS HERE (the fix!)
    if mitigations is not None:
        docs = apply_mitigation_filters(docs, mitigations)

    if not docs:
        st.error("No documents loaded after filtering!")
        return

    embeddings = [ollama.embeddings(model='nomic-embed-text', prompt=d['content'])['embedding'] for d in docs]
    emb_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)

    if not poison:
        st.session_state.clean_avg_emb = get_average_clean_embedding(docs)

    st.session_state.index = index
    st.session_state.docs = docs
    st.success(f"Loaded {len(docs)} chunks from {folder}")

# Auto-load clean DB
if st.session_state.index is None:
    with st.spinner("Loading clean KB..."):
        init_db()

st.title("RAG Poisoning Lab: Attack, Detect, Mitigate")

# Defense status
with st.sidebar:
    st.header("Current Defense Status")
    if st.session_state.mitigations.get('keyword_filter', False):
        st.error("DEFENSES ACTIVE")
        st.write("- Keyword filter ON")
    else:
        st.success("DEFENSES OFF")
        st.write("- Poisoning fully enabled")

tab = st.radio("Navigate", ["Query", "Poison", "Detect", "Mitigate"], horizontal=True, label_visibility="collapsed")

if tab == "Query":
    st.header("Query the RAG System")
    prompt = st.text_input("Ask about cybersecurity...")
    if prompt and st.session_state.index:
        q = np.array([ollama.embeddings(model='nomic-embed-text', prompt=prompt)['embedding']]).astype('float32')
        D, I = st.session_state.index.search(q, 6)
        context = "\n\n".join(st.session_state.docs[i]['content'] for i in I[0] if i < len(st.session_state.docs))
        resp = ollama.chat(model='llama3:8b-instruct-q4_K_M', messages=[{'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {prompt}"}])
        st.write(resp['message']['content'])
        with st.expander("Retrieved chunks"):
            for i in I[0]:
                if i < len(st.session_state.docs):
                    d = st.session_state.docs[i]
                    st.caption(d['source'])
                    st.code(d['content'][:500])

elif tab == "Poison":
    st.header("Poison the Knowledge Base")
    
    # Optional reset (still useful to turn off defenses quickly)
    if st.button("RESET ALL MITIGATIONS (to allow poisoning)", type="secondary"):
        st.session_state.mitigations = {'keyword_filter': False}
        st.success("Defenses disabled — poisoning will now succeed!")
        st.rerun()
    
    # Warning if defenses are on
    if st.session_state.mitigations.get('keyword_filter', False):
        st.warning("Defenses are ACTIVE — poisoning will be blocked!")
    else:
        st.info("Defenses OFF — poisoning will succeed!")

    poison_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "poisoned_docs")
    if os.path.exists(poison_dir) and os.listdir(poison_dir):
        file = st.selectbox("Select poison payload", os.listdir(poison_dir))
        if st.button("Ingest Poisoned Document", type="primary"):
            with st.spinner("Ingesting poison..."):
                # ← THE FIX: Always use current state (blocks if defenses on)
                init_db(poison=True, mitigations=st.session_state.mitigations)
            st.success(f"Ingested {file} (subject to current defenses)")
    else:
        st.error("No files in poisoned_docs/")

elif tab == "Detect":
    st.header("Detection Analysis")
    text_input = st.text_area("Input Text Chunk for Analysis", height=150, key="detect_text")

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Running detection..."):
                perp = calculate_perplexity(text_input)
                sim = check_similarity(text_input, st.session_state.clean_avg_emb)

            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Perplexity", f"{perp:.1f}")
                if len(text_input) < 20:
                    st.info("Text too short → perplexity ignored")
                elif perp > 100:
                    st.error("HIGHLY SUSPICIOUS – Likely prompt injection")
                elif perp > 70:
                    st.warning("Suspicious – Elevated perplexity")
                else:
                    st.success("Low – Normal")

            with col2:
                if sim is not None:
                    st.metric("Similarity to Clean KB", f"{sim:.4f}")
                    if sim < 0.50:
                        st.error("HIGHLY POISONED – Very low similarity")
                    elif sim < 0.65:
                        st.warning("Poisoned / Diverged")
                    else:
                        st.success("Clean – Matches legitimate content")
                else:
                    st.info("Clean KB not loaded yet (load clean DB first)")

            st.divider()
            
            # Final combined verdict
            if len(text_input) >= 20 and perp > 100 and sim is not None and sim < 0.50:
                st.error("CRITICAL ALERT: HIGH CONFIDENCE POISONING DETECTED")
            elif len(text_input) >= 20 and (perp > 80 or (sim is not None and sim < 0.60)):
                st.warning("Suspicious – Potential poisoning attack")
            else:
                st.success("All clear – Content appears legitimate")

elif tab == "Mitigate":
    st.header("Mitigation Implementation")
    
    # Checkbox
    st.session_state.mitigations['keyword_filter'] = st.checkbox(
        "Keyword Filter (blocks common jailbreak patterns like 'IGNORE ALL', 'secret flag', etc.)",
        value=st.session_state.mitigations.get('keyword_filter', False)
    )
    
    # Educational note
    st.info("""
    **Note**: This is a basic per-chunk keyword filter.  
    Advanced attackers can split jailbreaks across multiple documents to bypass it.  
    Real-world defense requires:  
    - Input/output guards  
    - Retrieval filtering (e.g., similarity to clean KB)  
    - Prompt hardening  
    """)

    if st.button("Re-Ingest with Mitigations", type="primary"):
        with st.spinner("Applying defenses and re-loading knowledge base..."):
            init_db(mitigations=st.session_state.mitigations)
        
        # BEAUTIFUL SUCCESS MESSAGE WITH APPLIED MITIGATIONS
        st.success("DB Re-Initialized with Mitigations Applied!")
        st.markdown("### Active Defenses")
        
        if st.session_state.mitigations.get('keyword_filter', False):
            st.markdown("Keyword Filter (blocks 30+ jailbreak patterns)")
        else:
            st.markdown("Keyword Filter (currently disabled)")

if len(sys.argv) > 1 and sys.argv[1] == "health":
    print("RAG system: Running")