from __future__ import annotations

from typing import Dict

NBSP = "\u00A0"
MID = "·"


def _safe_pct(v):
    try:
        f = float(v)
        if 0 <= f <= 1:
            f *= 100.0
        if f < 0 or f > 100:
            return "—"
        return f"{f:.1f}%"
    except Exception:
        return "—"


def _pull(agent1: Dict):
    k = (agent1.get("kpis") or {})
    mix = (k.get("customer_mix_detail") or {})
    ages = (k.get("age_distribution") or [])
    top_age = None
    if ages:
        top_age = max(ages, key=lambda a: float(a.get("value", -1)))
    return {
        "floating": _safe_pct(mix.get("유동")),
        "workplace": _safe_pct(mix.get("직장")),
        "residential": _safe_pct(mix.get("거주")),
        "new_rate": _safe_pct(k.get("new_rate_avg") or k.get("new_rate")),
        "revisit_rate": _safe_pct(k.get("revisit_rate_avg") or k.get("revisit_rate")),
        "top_age_label": (top_age.get("label") if top_age else None) or "—",
        "top_age_value": _safe_pct((top_age or {}).get("value")),
    }


def build_analyst_summary_text(
    agent1: Dict,
    merchant_mask: str | None = None,
    *,
    use_llm: bool = True,
    llm_max_tokens: int = 900,
) -> str:
    p = _pull(agent1)
    baseline = (
        f"{merchant_mask or '해당 매장'}은 유입 구조에서 "
        f"유동 {p['floating']}{NBSP}{MID}{NBSP}직장 {p['workplace']}{NBSP}{MID}{NBSP}거주 {p['residential']} 비중을 보이며, "
        f"거주 고객 비중은 상대적으로 낮으므로 상권 거점보다는 유입 동선에 맞춰 메시지를 설계해야 합니다. "
        f"유동·직장 층이 핵심 모수로 작동한다는 점을 전제로 동선 인근의 집객 장치를 선제적으로 마련해야 합니다. "
        f"연령대는 {p['top_age_label']} {p['top_age_value']}가 최다로, 학습·커뮤니티 성향이 강한 세그먼트에 맞춘 경험 설계가 요구됩니다. "
        f"신규 {p['new_rate']}{NBSP}{MID}{NBSP}재방문 {p['revisit_rate']} 흐름은 전환 깔때기의 상단·중단에서 동시에 마찰을 줄여야 함을 시사합니다. "
        f"이에 따라 채널 전략은 ‘가시성 확대 + 즉시 전환 유도’의 투 트랙으로 구성합니다. 온라인은 인스타그램 릴스/스토리 기반의 짧은 영상, 네이버 플레이스 상단 노출(대표사진 3장 교체·키워드 보강), 카카오맵 스크랩 유도 메시지로 탐색 의도를 선점합니다. "
        f"오프라인은 {p['top_age_label']} 주 이용 시간대에 타임딜(예: 14–17시 학생증 할인), SNS 팔로우 인증 이벤트, 스탬프 적립을 묶어 방문 직후 재구매 명분을 제공합니다. "
        f"운영팀은 주간 단위로 캠페인별 유입 코드를 기록하고, 2–4주 단위로 도달·클릭·저장수(상단), 쿠폰 사용·14–17시 매출·리뷰 건수(전환) 증분을 베이스라인 대비 비교해 조정안을 마련하십시오. "
        f"보고 체계에는 문의수·예약수와 같은 후속 행동 로그를 추가해 신규와 재방문 간 체류 패턴 차이를 추적하는 것이 좋습니다."
    )

    if use_llm:
        try:
            import app_core.llm as llm  # type: ignore[import-not-found]

            prompt = f"""
당신은 '소상공인 데이터 분석 전문가'입니다. 아래 KPI만 근거로 약 600–1000자 한국어 분석 요약을 작성하세요.
- 톤: 간결·실무형. 목록 대신 문장형.
- 포함: 유입구조(유동/직장/거주), 최다 연령, 신규/재방문 함의, 온라인/오프라인 채널 운용 요지, 2–4주 측정지표.
- 금지: 외부 추정/날씨/RAG 인용. 표나 코드블럭 금지.

[KPI]
유동={p['floating']}, 직장={p['workplace']}, 거주={p['residential']},
최다연령={p['top_age_label']} {p['top_age_value']},
신규={p['new_rate']}, 재방문={p['revisit_rate']}

매장 마스크명: {merchant_mask or '—'}
"""
            text = llm.call_llm(
                system="당신은 데이터 기반 마케팅 분석가입니다.",
                user=prompt,
                max_tokens=llm_max_tokens,
                temperature=0.3,
            ).strip()
            text = " ".join(text.split())
            if 400 <= len(text) <= 1200:
                return text
        except Exception:
            pass

    return " ".join(baseline.split())
