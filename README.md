# Document-QA-Assistant-with-RAG

## Design Decisions

### Why Groq
Groq has a free tier, runs open-weight models (`openai/gpt-oss-120b`) with low latency, and needs no local GPU. It plugs into LangChain via `ChatGroq`, and the model is set through `MODEL_NAME` in `.env`. Trade-offs: it needs internet, queries leave the machine, free-tier rate limits apply, and available models change over time.

### Parent-Child Splitting
- **Child chunks** (~<child_size> chars, overlap <child_overlap>) are small and embedded in ChromaDB, giving precise similarity matches.
- **Parent chunks** (~<parent_size> chars) are the larger sections containing the children. When a child matches, its parent is returned, so the LLM gets full context instead of an isolated sentence.

This avoids the usual trade-off: small chunks retrieve well but lack context, large chunks have context but blur the embedding.

### Why Pickle
ChromaDB stores only the child embeddings, so parent chunks are kept in a key-value docstore serialized to `docstore.pkl`. This persists them to disk (no rebuild on every run), stores LangChain `Document` objects directly, and needs no extra service. Limitation: pickle is Python-specific and unsafe to load from untrusted sources, so only load a `docstore.pkl` created by this project's `ingest.py`.
