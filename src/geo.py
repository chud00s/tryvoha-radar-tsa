"""Geography helpers: canonical regions, centroids, neighbour adjacency, and
name normalization (Ukrainian/English oblast names + major cities -> canonical).

Region names match the Vadimkin dataset exactly (24 oblasts + Kyiv City).
"""
from __future__ import annotations

# Canonical region names (exactly as they appear in the dataset).
REGIONS: list[str] = [
    "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast",
    "Dnipropetrovska oblast", "Donetska oblast", "Ivano-Frankivska oblast",
    "Kharkivska oblast", "Khersonska oblast", "Khmelnytska oblast",
    "Kirovohradska oblast", "Kyiv City", "Kyivska oblast", "Luhanska oblast",
    "Lvivska oblast", "Mykolaivska oblast", "Odeska oblast", "Poltavska oblast",
    "Rivnenska oblast", "Sumska oblast", "Ternopilska oblast", "Vinnytska oblast",
    "Volynska oblast", "Zakarpatska oblast", "Zaporizka oblast", "Zhytomyrska oblast",
]

# Approximate centroid (administrative centre) coordinates: (lat, lon).
COORDS: dict[str, tuple[float, float]] = {
    "Cherkaska oblast": (49.44, 32.06),
    "Chernihivska oblast": (51.30, 31.80),
    "Chernivetska oblast": (48.29, 25.94),
    "Dnipropetrovska oblast": (48.46, 35.04),
    "Donetska oblast": (48.02, 37.80),
    "Ivano-Frankivska oblast": (48.62, 24.80),
    "Kharkivska oblast": (49.80, 36.40),
    "Khersonska oblast": (46.64, 32.90),
    "Khmelnytska oblast": (49.42, 26.99),
    "Kirovohradska oblast": (48.51, 32.26),
    "Kyiv City": (50.45, 30.52),
    "Kyivska oblast": (50.05, 30.20),
    "Luhanska oblast": (48.74, 39.20),
    "Lvivska oblast": (49.55, 23.80),
    "Mykolaivska oblast": (47.20, 31.50),
    "Odeska oblast": (46.48, 30.73),
    "Poltavska oblast": (49.59, 34.00),
    "Rivnenska oblast": (50.80, 26.25),
    "Sumska oblast": (50.91, 34.20),
    "Ternopilska oblast": (49.40, 25.59),
    "Vinnytska oblast": (49.10, 28.47),
    "Volynska oblast": (50.95, 25.00),
    "Zakarpatska oblast": (48.40, 22.80),
    "Zaporizka oblast": (47.50, 35.30),
    "Zhytomyrska oblast": (50.45, 28.40),
}

# Ukrainian display names. Oblast forms match the bundled GeoJSON `name`
# property exactly (data/raw/ua_oblasts.geojson); Kyiv City has no polygon.
UA_NAMES: dict[str, str] = {
    "Cherkaska oblast": "Черкаська область",
    "Chernihivska oblast": "Чернігівська область",
    "Chernivetska oblast": "Чернівецька область",
    "Dnipropetrovska oblast": "Дніпропетровська область",
    "Donetska oblast": "Донецька область",
    "Ivano-Frankivska oblast": "Івано-Франківська область",
    "Kharkivska oblast": "Харківська область",
    "Khersonska oblast": "Херсонська область",
    "Khmelnytska oblast": "Хмельницька область",
    "Kirovohradska oblast": "Кіровоградська область",
    "Kyiv City": "Київ",
    "Kyivska oblast": "Київська область",
    "Luhanska oblast": "Луганська область",
    "Lvivska oblast": "Львівська область",
    "Mykolaivska oblast": "Миколаївська область",
    "Odeska oblast": "Одеська область",
    "Poltavska oblast": "Полтавська область",
    "Rivnenska oblast": "Рівненська область",
    "Sumska oblast": "Сумська область",
    "Ternopilska oblast": "Тернопільська область",
    "Vinnytska oblast": "Вінницька область",
    "Volynska oblast": "Волинська область",
    "Zakarpatska oblast": "Закарпатська область",
    "Zaporizka oblast": "Запорізька область",
    "Zhytomyrska oblast": "Житомирська область",
}


# Temporarily occupied — part of Ukraine, but no alert data is reported there.
# Shown on maps as Ukrainian territory (neutral "no data"), never as foreign.
NODATA_REGIONS = ["Crimea", "Sevastopol"]
UA_NAMES["Crimea"] = "АР Крим"
UA_NAMES["Sevastopol"] = "Севастополь"
COORDS["Crimea"] = (45.3, 34.4)
COORDS["Sevastopol"] = (44.6, 33.5)


def ua_name(region: str | None) -> str:
    """Ukrainian display name for a canonical region; None -> 'Уся Україна'."""
    if region is None:
        return "Уся Україна"
    return UA_NAMES.get(region, region)


# Land-border adjacency (defined one-way; symmetrised below).
_ADJ_RAW: dict[str, list[str]] = {
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Lvivska oblast": ["Volynska oblast", "Rivnenska oblast", "Ternopilska oblast",
                       "Ivano-Frankivska oblast", "Zakarpatska oblast"],
    "Rivnenska oblast": ["Ternopilska oblast", "Khmelnytska oblast", "Zhytomyrska oblast"],
    "Zakarpatska oblast": ["Ivano-Frankivska oblast"],
    "Ivano-Frankivska oblast": ["Chernivetska oblast", "Ternopilska oblast"],
    "Ternopilska oblast": ["Khmelnytska oblast", "Chernivetska oblast"],
    "Chernivetska oblast": ["Khmelnytska oblast", "Vinnytska oblast"],
    "Khmelnytska oblast": ["Vinnytska oblast", "Zhytomyrska oblast"],
    "Zhytomyrska oblast": ["Vinnytska oblast", "Kyivska oblast"],
    "Vinnytska oblast": ["Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast"],
    "Kyivska oblast": ["Cherkaska oblast", "Poltavska oblast", "Chernihivska oblast",
                       "Kyiv City"],
    "Kyiv City": ["Kyivska oblast"],
    "Chernihivska oblast": ["Sumska oblast", "Poltavska oblast"],
    "Sumska oblast": ["Poltavska oblast", "Kharkivska oblast"],
    "Poltavska oblast": ["Kharkivska oblast", "Dnipropetrovska oblast",
                         "Cherkaska oblast", "Kirovohradska oblast"],
    "Kharkivska oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Luhanska oblast"],
    "Luhanska oblast": ["Donetska oblast"],
    "Donetska oblast": ["Dnipropetrovska oblast", "Zaporizka oblast"],
    "Dnipropetrovska oblast": ["Zaporizka oblast", "Khersonska oblast",
                               "Mykolaivska oblast", "Kirovohradska oblast"],
    "Zaporizka oblast": ["Khersonska oblast"],
    "Khersonska oblast": ["Mykolaivska oblast"],
    "Mykolaivska oblast": ["Kirovohradska oblast", "Odeska oblast"],
    "Odeska oblast": ["Kirovohradska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast"],
    "Cherkaska oblast": [],
}


def _build_neighbors() -> dict[str, set[str]]:
    nb: dict[str, set[str]] = {r: set() for r in REGIONS}
    for a, bs in _ADJ_RAW.items():
        for b in bs:
            nb[a].add(b)
            nb[b].add(a)
    return nb


NEIGHBORS: dict[str, set[str]] = _build_neighbors()


# --- Name normalization -----------------------------------------------------
# Alias (lowercased) -> canonical region. Includes Ukrainian oblast names and
# major cities so LLM/rule output ("Харків", "по Одесі") maps to a region.
_ALIASES: dict[str, str] = {}


def _add(canonical: str, *aliases: str) -> None:
    for a in aliases:
        _ALIASES[a.lower()] = canonical


# Canonical + obvious English/Ukrainian forms.
_add("Cherkaska oblast", "cherkaska", "cherkasy", "черкаська", "черкаси", "черкащина")
_add("Chernihivska oblast", "chernihivska", "chernihiv", "чернігівська", "чернігів", "чернігівщина")
_add("Chernivetska oblast", "chernivetska", "chernivtsi", "чернівецька", "чернівці", "буковина")
_add("Dnipropetrovska oblast", "dnipropetrovska", "dnipro", "dnipropetrovsk",
     "дніпропетровська", "дніпро", "дніпропетровщина", "кривий ріг", "kryvyi rih", "нікополь", "павлоград")
_add("Donetska oblast", "donetska", "donetsk", "донецька", "донецьк", "донеччина",
     "маріуполь", "mariupol", "краматорськ", "бахмут", "слов'янськ", "покровськ", "авдіївка")
_add("Ivano-Frankivska oblast", "ivano-frankivska", "ivano-frankivsk", "івано-франківська",
     "івано-франківськ", "прикарпаття", "франківськ")
_add("Kharkivska oblast", "kharkivska", "kharkiv", "харківська", "харків", "харківщина", "куп'янськ", "чугуїв")
_add("Khersonska oblast", "khersonska", "kherson", "херсонська", "херсон", "херсонщина", "нова каховка")
_add("Khmelnytska oblast", "khmelnytska", "khmelnytskyi", "хмельницька", "хмельницький", "хмельниччина", "кам'янець-подільський")
_add("Kirovohradska oblast", "kirovohradska", "kropyvnytskyi", "kirovohrad", "кіровоградська",
     "кропивницький", "кіровоград", "кіровоградщина")
_add("Kyiv City", "kyiv city", "kyiv", "kiev", "м. київ", "місто київ", "київ")
_add("Kyivska oblast", "kyivska", "київська", "київщина", "бровари", "біла церква", "бориспіль", "ірпінь", "буча", "фастів")
_add("Luhanska oblast", "luhanska", "luhansk", "луганська", "луганськ", "луганщина", "сєвєродонецьк")
_add("Lvivska oblast", "lvivska", "lviv", "львівська", "львів", "львівщина", "дрогобич")
_add("Mykolaivska oblast", "mykolaivska", "mykolaiv", "миколаївська", "миколаїв", "миколаївщина", "вознесенськ")
_add("Odeska oblast", "odeska", "odesa", "odessa", "одеська", "одеса", "одещина", "ізмаїл", "чорноморськ")
_add("Poltavska oblast", "poltavska", "poltava", "полтавська", "полтава", "полтавщина", "кременчук", "лубни")
_add("Rivnenska oblast", "rivnenska", "rivne", "рівненська", "рівне", "рівненщина", "сарни")
_add("Sumska oblast", "sumska", "sumy", "сумська", "суми", "сумщина", "конотоп", "шостка", "охтирка")
_add("Ternopilska oblast", "ternopilska", "ternopil", "тернопільська", "тернопіль", "тернопільщина", "кременець")
_add("Vinnytska oblast", "vinnytska", "vinnytsia", "вінницька", "вінниця", "вінниччина", "жмеринка")
_add("Volynska oblast", "volynska", "lutsk", "волинська", "луцьк", "волинь", "ковель")
_add("Zakarpatska oblast", "zakarpatska", "uzhhorod", "закарпатська", "ужгород", "закарпаття", "мукачево")
_add("Zaporizka oblast", "zaporizka", "zaporizhzhia", "zaporizhia", "запорізька", "запоріжжя",
     "запоріжжська", "запорізьк", "мелітополь", "бердянськ", "енергодар")
_add("Zhytomyrska oblast", "zhytomyrska", "zhytomyr", "житомирська", "житомир", "житомирщина", "бердичів", "коростень")

# Case-robust STEMS — Ukrainian declension means "Харкові"/"Одещиною" won't match
# a nominative alias, so we add distinctive prefixes that survive inflection.
_add("Cherkaska oblast", "черка", "черкащ")
_add("Chernihivska oblast", "черніг", "чернігівщ")
_add("Chernivetska oblast", "чернів", "буков")
_add("Dnipropetrovska oblast", "дніпро", "дніпроп", "дніпропетровщ", "криворіж",
     "кривому роз", "кривого рог", "нікопол", "павлогр", "кам'янськ")
_add("Donetska oblast", "донец", "донеч", "маріупол", "краматорськ", "бахмут",
     "покровськ", "авдіїв", "слов'янськ", "костянтинівк")
_add("Ivano-Frankivska oblast", "франків", "прикарп", "івано-франк")
_add("Kharkivska oblast", "харк", "харкі", "харко", "харківщ", "куп'янськ", "чугуїв", "ізюм")
_add("Khersonska oblast", "херсон", "херсонщ", "каховк")
_add("Khmelnytska oblast", "хмельниц", "хмельнич", "хмельниччин", "кам'янець-под")
_add("Kirovohradska oblast", "кропивн", "кіровогр", "кіровоградщ", "олександрі")
_add("Kyiv City", "києв", "київ")
_add("Kyivska oblast", "київськ", "київщ", "бровар", "біла церкв", "бориспіл", "ірпін", "фастів", "буч")
_add("Luhanska oblast", "луган", "луганщ", "сєвєродонецьк", "лисичанськ")
_add("Lvivska oblast", "львів", "львов", "львівщ", "дрогобич", "стрий")
_add("Mykolaivska oblast", "микола", "миколаївщ", "вознесенськ", "первомайськ")
_add("Odeska oblast", "одес", "одещ", "ізмаїл", "чорноморськ", "подільськ")
_add("Poltavska oblast", "полтав", "полтавщ", "кременчу", "лубни", "миргород")
_add("Rivnenska oblast", "рівне", "рівнен", "рівном", "рівног", "рівненщ", "сарни", "дубно")
_add("Sumska oblast", "суми", "сумщ", "сумськ", "сумах", "конотоп", "шостк", "охтирк", "глухів")
_add("Ternopilska oblast", "терноп", "тернопільщ", "кременец", "чортків")
_add("Vinnytska oblast", "вінниц", "вінниччин", "жмеринк", "могилів-под")
_add("Volynska oblast", "волин", "луцьк", "ковел", "нововолинськ")
_add("Zakarpatska oblast", "закарп", "ужгород", "мукачев", "хуст")
_add("Zaporizka oblast", "запор", "запоріж", "мелітопол", "бердянськ", "енергодар", "оріхів")
_add("Zhytomyrska oblast", "житомир", "житомирщ", "бердичів", "коростень", "новоград")

# Also let the canonical name itself resolve.
for _r in REGIONS:
    _ALIASES.setdefault(_r.lower(), _r)

# Longest aliases first, so "івано-франківськ" wins over a shorter accidental match.
_ALIASES_BY_LEN = sorted(_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)


def find_region_mention(text: str | None) -> tuple[str | None, str | None]:
    """Return (canonical_region, matched_alias) for the first place mentioned.

    Tries an exact alias match first, then a substring scan (longest alias wins).
    """
    if not text:
        return None, None
    t = str(text).strip().lower()
    if not t:
        return None, None
    if t in _ALIASES:
        return _ALIASES[t], t
    t_stripped = t.replace(" oblast", "").replace(" область", "").strip()
    if t_stripped in _ALIASES:
        return _ALIASES[t_stripped], t_stripped
    for alias, canonical in _ALIASES_BY_LEN:
        if len(alias) >= 4 and alias in t:
            return canonical, alias
    return None, None


def normalize_region(text: str | None) -> str | None:
    """Map free text (city/oblast, UA or EN) to a canonical region, or None."""
    return find_region_mention(text)[0]


def region_coords(region: str) -> tuple[float, float] | None:
    return COORDS.get(region)
