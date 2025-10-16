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

## Data Source Toggles & Fail-Soft Answering
- The **Data Sources** sidebar group lets you enable Weather data, External APIs, and RAG evidence individually. RAG controls include a threshold slider, top-k range, and `auto|always|off` mode selector.
- The consulting view now composes a fail-soft answer that reflects only the active sources and surfaces caveats when data is unavailable. A "Sources Used" footer summarizes structured/weather/external/RAG usage.
- 📎 evidence badges appear beside sentences backed by RAG chunks. When RAG is disabled or below threshold, the UI shows muted badges and caveats instead of hallucinated text.
- Use the "임계값 낮추기/높이기" buttons beneath the analysis to widen or narrow the RAG threshold by ±0.05 and trigger a quick re-query with the latest settings.
