"""실행: python -m tests.test_dashboard_ui  (pytest로도 실행 가능). 대시보드 디자인 토큰 읽기 테스트."""
from src.dashboard.ui import STAGE_COLORS, TOK, UNSTABLE, _tokens


def test_tokens_resolve_var_chains():
    css = ":root {\n  --red: #d70015;\n  --risk-high: var(--red);  --alias: var(--risk-high);\n}\nbody { color: var(--red); }"
    assert _tokens(css) == {"red": "#d70015", "risk-high": "#d70015", "alias": "#d70015"}


def test_theme_stage_colors():
    assert STAGE_COLORS == (TOK["green"], TOK["orange-dot"], TOK["red"], TOK["red"])   # 정상·주의·위험·위험
    assert UNSTABLE == TOK["gray-dot"]
    assert not any(v.startswith("var(") for v in TOK.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
