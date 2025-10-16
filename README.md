# Bigcon MVP
Streamlit 테스트용 기본 앱입니다.
실행(로컬): `pip install -r requirements.txt && streamlit run app.py`

Diagnostics module removed in 6a2cc72350278cbfd599b419d622a70300825ce8.

## RAG (MVP) — File-based Retrieval
- RAG data root defaults to `data/rag` (override via `RAG_ROOT` env or Streamlit secret). Originals live under `data/rag/corpus/` and remain untouched.
- Embedding indices belong in `data/rag/indices/<embed_version>/<document_id>/` with `manifest.json`, `chunks.parquet`, and `vectors.npy`.
- The Streamlit sidebar lets you enable RAG, filter documents via a search box, and multi-select which embedded sources to query. If no document is selected, the run proceeds without RAG evidence and surfaces a caveat.
- The **Embedded Sources** tab shows the discovered catalog, including tags/year metadata and quick chunk previews.
- Programmatic use:

```python
from rag.retrieval_tool import RetrievalTool

rt = RetrievalTool(root="data/rag", embed_version="embed_v1")
docs = rt.load_catalog()
hits = rt.retrieve("정부24 2025 지원 요건", top_k=5, doc_ids=["정부24_소상공인_가이드_2025"])
print(hits["evidence"][:2])
```

## Data Source Toggles & Fail-Soft Answering
- The **Data Sources** sidebar group lets you enable Weather data, External APIs, and RAG evidence individually. RAG controls include a threshold slider, top-k range, and `auto|always|off` mode selector.
- The consulting view now composes a fail-soft answer that reflects only the active sources and surfaces caveats when data is unavailable. A "Sources Used" footer summarizes structured/weather/external/RAG usage.
- 📎 evidence badges appear beside sentences backed by RAG chunks. When RAG is disabled or below threshold, the UI shows muted badges and caveats instead of hallucinated text.
- Use the "임계값 낮추기/높이기" buttons beneath the analysis to widen or narrow the RAG threshold by ±0.05 and trigger a quick re-query with the latest settings.

## RAG (MVP) — Data root & document selection
- Data root: `data/rag` (configurable via `RAG_ROOT`). Ensure both `corpus/` and `indices/` subfolders exist.
- Originals: place source docs (txt/pdf) under `data/rag/corpus/...`.
- Embeddings: create `data/rag/indices/<embed_version>/<doc_id>/` with `manifest.json`, `chunks.parquet`, `vectors.npy`.
- Streamlit UI: turn on "Use RAG", filter documents, and select one or more entries before running 분석.

## Organizer Q1/Q2/Q3 answer policy (public mode)
- Public mode is the default experience (toggle via the sidebar App Mode selector or the `APP_MODE` secret) and hides fail-soft banners and internal caveats.
- Agent-2 now answers the organizer’s three evaluation questions (Q1 채널/홍보, Q2 재방문, Q3 요식업 가설) with 3–4 concise ideas, each carrying audience, 채널, 실행, 카피, 측정지표, and at least one evidence item.
- Age cohorts strictly follow Agent-1’s allow-list with deterministic combined-vs-gender merge rules; values are guarded to the 0–100% range and missing cohorts are omitted instead of being synthesized.
- RAG evidence is appended only when the selected documents clear the similarity threshold; otherwise the cards surface structured data alone without “근거 없음” noise in public view.
- Switch to **Debug** mode to inspect fail-soft notes, RAG score gates, and other diagnostics that remain available for reviewers/operators.
