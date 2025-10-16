# Bigcon MVP
Streamlit 테스트용 기본 앱입니다.
실행(로컬): `pip install -r requirements.txt && streamlit run app.py`

Diagnostics module removed in 6a2cc72350278cbfd599b419d622a70300825ce8.

## RAG (MVP) — File-based Retrieval
- Expect corpora under `corpus/` (e.g. `corpus/gov/정부24_소상공인_가이드_2025.txt`).
- Embedding indices should live in `indices/<embed_version>/<document_id>/` with `manifest.json`, `chunks.parquet`, and `vectors.npy`.
- The Streamlit app lists available embedded sources in the **Embedded Sources** tab and previews the first few chunks per document.
- Programmatic use:

```python
from rag.retrieval_tool import RetrievalTool

rt = RetrievalTool(root=".", embed_version="embed_v1")
hits = rt.retrieve("정부24 2025 지원 요건", top_k=5)
print(hits["evidence"][:2])
```
