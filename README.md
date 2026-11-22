# RAG Poisoning Lab

A small, runnable RAG pipeline you can poison, then try to detect. Fifteen clean
security documents go into a FAISS index; three payloads go in on top. The app
has a query tab, a poison tab, and a detect tab that scores a pasted chunk.

Built to be poked at rather than read: everything is in one Streamlit file and
the numbers are small enough to check by hand.

> Educational use. Do not point the payloads at systems you do not own.

---

## What is in here

```
rag_app/
  app_streamlit.py   the whole app: ingest, query, poison, detect
  knowledge_base.py  a second, older keyword filter (not imported)
  detection.py       perplexity + similarity helpers (not imported)
documents/           15 clean security documents
poisoned_docs/       3 payloads: bias, injection, leakage
```

The three payloads are deliberately different attacks:

| payload | what it does |
|---|---|
| `injection.txt` | tries to override the system instruction |
| `leakage.txt` | plants a false fact about data handling |
| `bias.txt` | shifts the tone of every answer |

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt --prefer-binary

# in another shell
ollama pull llama3:8b-instruct-q4_K_M
ollama pull nomic-embed-text
ollama serve

streamlit run rag_app/app_streamlit.py
```

Then open <http://localhost:8000>.

If you only want the detection numbers, you do not need Streamlit or the LLM —
just `transformers` and `torch` for perplexity, and a running Ollama for the
embeddings.

## The attack

Open the **Poison** tab, pick a payload, ingest it, then go back to **Query**
and ask something adjacent. Poisoned chunks are retrieved because they are
semantically close to the question — that is the whole mechanism. No prompt
injection trickery is required; the payload only has to look like a document.

## The defences, and what they actually do

There are three checks. They are not equally good, and two of them are
disabled or misconfigured out of the box. Measured against this repo's own
corpus and payloads:

### Keyword filter — blocks 2 of 3 payloads, and 9 of 10 clean documents

`app_streamlit.py` holds a 29-pattern blocklist. It is **off by default**; tick
the box in the **Mitigate** tab to enable it.

It catches `injection` and `leakage`. It also catches almost any real security
document, because the patterns include `secret`, `flag`, `leak`, `private key`
and `admin password`. Nine of ten plausible clean sentences I tested were
rejected. It works by deleting the corpus.

`knowledge_base.py` contains an older two-pattern version. Nothing imports it.

### Perplexity — cannot separate clean from poisoned

GPT-2 perplexity over a chunk. The app uses fixed cut-offs (70 warning, 100
error) rather than anything relative to the corpus.

| | perplexity |
|---|---|
| clean corpus (n=15) | mean 42.6, range 20.4 – 84.7 |
| `bias` | 77.2 |
| `injection` | 96.8 |
| `leakage` | 115.8 |

The clean maximum (84.7) is above the poisoned minimum (77.2). The ranges
overlap, so no threshold admits every clean document and rejects every payload.
`bias` is an ordinary sentence about security, in a corpus of ordinary
sentences about security, and perplexity has nothing to work with.

### Similarity — the one that works, set to a threshold that never fires

Cosine similarity to the mean embedding of the clean corpus:

| | similarity |
|---|---|
| clean corpus (n=15) | 0.7705 – 0.8834 |
| poisoned (n=3) | 0.6857 – 0.7612 |

The populations separate completely — every clean document scores above every
payload. But the shipped thresholds are `0.50` (error) and `0.65` (warning),
and nothing in the repo is below 0.65, so the check never fires. At `0.765` it
catches all three with no false positives.

The margin is 0.0093. That is thin enough that one new document, a different
embedding model, or a slightly better-written payload would close it. Treat it
as a demonstration, not a guarantee.

### Detection is manual

Perplexity and similarity are in the **Detect** tab: paste a chunk, press a
button, read two numbers. Only the keyword filter touches ingestion, and only
when enabled. Nothing scores documents automatically.

## A worked example of getting it wrong

Replaying the app's own combined verdict on the measured values:

| input | perp | sim | verdict |
|---|---|---|---|
| `bias` (attack) | 77.2 | 0.6857 | **All clear** |
| `injection` | 96.8 | 0.7505 | Suspicious |
| `leakage` | 115.8 | 0.7612 | Suspicious |
| `10_social_engineering` (clean) | 84.7 | 0.8165 | **Suspicious** |

The quietest attack is cleared and a legitimate document is flagged, because
both scores reward being unremarkable — which is exactly what a good poisoning
payload is.

## Reproducing the numbers

The keyword-filter check needs nothing but Python:

```python
import re
from pathlib import Path
src = Path("rag_app/app_streamlit.py").read_text()
pats = re.findall(r'"([^"]+)"',
    re.search(r"BLOCKED_PATTERNS = \[(.*?)\]", src, re.S).group(1))
for f in sorted(Path("poisoned_docs").glob("*.txt")):
    body = f.read_text().lower()
    print(f.stem, [p for p in pats if p in body])
```

Perplexity and similarity are in `detection.py`, or inline in
`app_streamlit.py` — the inline copies are the ones that run.

Measured 22 Nov 2026 with `gpt2` and `nomic-embed-text`. A longer writeup with
the tables and the reasoning is at
<https://r00tb3.github.io/breakpoint/rag-poisoning/>.

## Where this could go next

- Anchor the thresholds to the clean distribution instead of hardcoding them.
- Check consistency across documents, not just distance from the centroid:
  `leakage` contradicts a fact stated elsewhere in the corpus.
- Score the retrieved answer, not only the retrieved chunk. `bias` is invisible
  chunk-by-chunk and obvious across ten answers.

Issues and PRs welcome, especially payloads that defeat the similarity check.
