import maxwell_demon as md


def test_preprocess_and_analyze_smoke():
    text = "Questo e un test semplice."
    tokens = md.preprocess_text(text)
    rows = md.analyze_tokens(tokens, mode="raw", window_size=5, step=2)
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert "mean_entropy" in rows[0]
