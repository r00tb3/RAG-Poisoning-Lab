import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial.distance import cosine
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

# ───── FUTURE-PROOF GPT-2 IMPORT (works on transformers ≥4.47) ─────
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2Tokenizer
# ─────────────────────────────────────────────────────────────────────

def calculate_perplexity(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(text, return_tensors="pt")
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def get_embedding(text):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings.embed_query(text)

def check_similarity(text, avg_clean_emb):
    emb = get_embedding(text)
    return 1 - cosine(emb, avg_clean_emb)

def get_average_clean_embedding(texts, embeddings):
    embs = [embeddings.embed_documents([t.page_content])[0] for t in texts]
    return np.mean(embs, axis=0)
