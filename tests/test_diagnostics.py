import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core.diagnostics as diagnostics


def test_baseline_summary_length_and_format():
    sample = {
        "kpis": {
            "customer_mix_detail": {"유동": 0.4, "직장": 0.35, "거주": 0.25},
            "age_distribution": [
                {"label": "20대", "value": 0.45},
                {"label": "30대", "value": 0.35},
            ],
            "new_rate_avg": 0.22,
            "revisit_rate_avg": 0.51,
        }
    }
    text = diagnostics.build_analyst_summary_text(sample, "가*", use_llm=False)
    assert isinstance(text, str)
    assert 600 <= len(text) <= 1000
    assert "```" not in text
    assert "|" not in text


def test_baseline_summary_handles_missing_values():
    text = diagnostics.build_analyst_summary_text({}, None, use_llm=False)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "\n" not in text
