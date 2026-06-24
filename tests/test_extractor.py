"""Tests for region normalization and the rule-based event extractor."""
from src import geo
from src.ai_extractor import EVENT_TYPES, _classify_rules, extract_events, load_sample


def test_region_normalization_handles_inflection():
    cases = {
        "Зафіксовано влучання по інфраструктурі в Харкові.": "Kharkivska oblast",
        "Сили ППО збили шахеди над Одещиною.": "Odeska oblast",
        "Вибухи в Запоріжжі.": "Zaporizka oblast",
        "ППО працює над Києвом.": "Kyiv City",
        "Робота ППО на Київщині.": "Kyivska oblast",
        "Влучання зафіксовано в Кривому Розі.": "Dnipropetrovska oblast",
    }
    for text, expected in cases.items():
        assert geo.normalize_region(text) == expected, text


def test_classify_rules_event_and_weapon():
    assert _classify_rules("Сили ППО збили 3 шахеди")[0] == "збиття"
    assert _classify_rules("Зафіксовано пуск ракети з МіГ-31К") == ("пуск", "ракета")
    assert _classify_rules("Група дронів курсом на Дніпро") == ("рух", "дрон/шахед")
    assert _classify_rules("Відбій повітряної тривоги")[0] == "відбій"
    assert _classify_rules("Влучання по інфраструктурі")[0] == "влучання"


def test_extract_events_schema_and_geocoding():
    msgs = load_sample()
    df = extract_events(msgs, prefer_llm=False)  # force offline rules
    assert len(df) == len(msgs)
    assert set(df["event_type"]).issubset(set(EVENT_TYPES))
    assert (df["method"] == "rules").all()
    # most messages mention a place -> should geocode
    assert df["region"].notna().mean() >= 0.7
    # geocoded events get coordinates
    geocoded = df.dropna(subset=["region"])
    assert geocoded["lat"].notna().all()
    assert geocoded["lon"].notna().all()
