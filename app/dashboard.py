"""Tryvoha Radar — Streamlit dashboard.

Modes:
  * Споживач  — actionable: current risk + WHAT TO DO + national risk map.
  * Аналітик  — patterns, model quality, mass-attack anomalies, propagation, OSINT.

Visual language: "blueprint control room" (Dovetail design system) — near-black
canvas + faint grid, single cornflower-blue accent, Inter + JetBrains Mono, 8px
radii, flat (no shadows). Danger semantics (red/amber/green) kept for risk only.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src import analysis, config, geo, live
from src.ai_extractor import load_events
from src.transform import load_series

st.set_page_config(page_title="Tryvoha Radar", page_icon=":material/radar:", layout="wide")

# --- design tokens ----------------------------------------------------------
ACCENT = "#6798ff"   # cornflower
INK = "#0a0a0a"
COAL = "#141414"
CARBON = "#1e1e1e"
STEEL = "#313131"
GRAPHITE = "#454545"
FOG = "#7c7c7c"
ASH = "#a7a7a7"
SNOW = "#ffffff"
GOOD, WARN, BAD = "#54a24b", "#f5a623", "#e5484d"   # semantic danger scale

HEAT_SCALE = [[0.0, "#101319"], [0.4, "#21407a"], [0.7, "#3f6fcf"], [1.0, ACCENT]]

GEOJSON_PATH = config.DATA_RAW / "ua_oblasts.geojson"
MODES = [
    ("consumer", "Споживач", ":material/person:"),
    ("analyst", "Аналітик", ":material/analytics:"),
    ("osint", "Загрози зараз", ":material/radar:"),
]
THREAT_TYPES = ["пуск", "рух", "загроза"]
ACTIVE_WINDOW_H = 12

DOW_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
EVENT_COLORS = {
    "пуск": "#f5a623", "рух": ACCENT, "загроза": "#e6c84d",
    "влучання": "#e5484d", "збиття": "#54a24b", "відбій": "#7c7c7c", "інше": "#454545",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Inter',system-ui,sans-serif; color:#ffffff;
}
.stApp{ background-color:#0a0a0a; }

/* blueprint grid motif */
[data-testid="stAppViewContainer"]{
  background-color:#0a0a0a;
  background-image:
    repeating-linear-gradient(0deg, transparent 0 47px, #15161b 47px 48px),
    repeating-linear-gradient(90deg, transparent 0 47px, #15161b 47px 48px);
}
[data-testid="stMainBlockContainer"]{ max-width:1320px; padding-top:2.5rem; }
[data-testid="stHeader"]{ background:transparent; }

[data-testid="stSidebar"]{ background-color:#141414; border-right:1px solid #313131; }

h1,h2,h3,h4{ font-family:'Inter'; font-weight:600; color:#ffffff; letter-spacing:-0.02em; }
h1{ font-size:40px; letter-spacing:-0.03em; }
h2{ font-size:24px; } h3{ font-size:20px; }

.eyebrow{ font-family:'JetBrains Mono',monospace; text-transform:uppercase;
  letter-spacing:1px; font-size:12px; color:#6798ff; margin:0 0 2px 0; }
[data-testid="stCaptionContainer"]{ color:#a7a7a7 !important; }

[data-testid="stMetric"]{ background:#141414; border:1px solid #313131; border-radius:8px; padding:14px 16px; }
[data-testid="stMetricLabel"] p{ font-family:'JetBrains Mono',monospace !important;
  text-transform:uppercase; letter-spacing:0.85px; font-size:11px !important; color:#a7a7a7 !important; }
[data-testid="stMetricValue"]{ font-family:'Inter'; font-weight:600; color:#ffffff; }
[data-testid="stMetricDelta"]{ font-family:'JetBrains Mono',monospace; }

[data-testid="stTabs"] button[role="tab"]{ font-family:'JetBrains Mono',monospace;
  text-transform:uppercase; letter-spacing:0.5px; font-size:12px; color:#a7a7a7; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{ color:#6798ff; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background-color:#6798ff !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]{ background-color:#313131 !important; }

.stButton>button, .stDownloadButton>button{ background:#1e1e1e; color:#ffffff;
  border:1px solid #454545; border-radius:8px; font-weight:500; }
.stButton>button:hover{ border-color:#6798ff; color:#6798ff; }

/* mode switch — two dynamic role buttons (scoped by stable st-key-* class) */
[class*="st-key-mode_btn_"] button{ width:100%; font-family:'JetBrains Mono',monospace;
  letter-spacing:0.2px; font-size:13px; padding:9px 8px; white-space:nowrap; transition:all .15s ease; }
[class*="st-key-mode_btn_"] button p{ font-size:13px; }
[class*="st-key-mode_btn_"] button[data-testid="stBaseButton-secondary"]{ background:#1e1e1e; color:#a7a7a7; border:1px solid #454545; }
[class*="st-key-mode_btn_"] button[data-testid="stBaseButton-secondary"]:hover{ border-color:#6798ff; color:#ffffff; }
[class*="st-key-mode_btn_"] button[data-testid="stBaseButton-primary"]{ background:#6798ff; color:#0a0a0a; border:1px solid #6798ff; font-weight:600; }
[class*="st-key-mode_btn_"] button[data-testid="stBaseButton-primary"]:hover,
[class*="st-key-mode_btn_"] button[data-testid="stBaseButton-primary"]:focus{ background:#6798ff; color:#0a0a0a; border-color:#6798ff; }

/* threat chips (active targets) */
[class*="st-key-th_"] button{ white-space:nowrap; font-size:12px; padding:6px 8px; min-height:0; }
[class*="st-key-th_"] button p{ font-size:12px; overflow:hidden; text-overflow:ellipsis; }

/* freshness chip */
.fresh{ display:inline-flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
  font-size:12px; letter-spacing:.3px; color:#a7a7a7; background:#141414;
  border:1px solid #313131; border-radius:8px; padding:6px 12px; margin:2px 0 8px 0; }
.fresh .dot{ width:8px; height:8px; border-radius:50%; flex:0 0 auto; }
.fresh .lbl{ color:#ffffff; letter-spacing:.4px; }
.fresh .sub{ color:#7c7c7c; }
@keyframes freshpulse{ 0%,100%{opacity:1} 50%{opacity:.4} }
.fresh.live .dot{ animation:freshpulse 1.8s ease-in-out infinite; }

[data-baseweb="select"]>div{ background:#1e1e1e !important; border:1px solid #454545 !important; border-radius:8px !important; }
[data-baseweb="popover"] [role="listbox"]{ background:#1e1e1e !important; }
[data-testid="stDataFrame"]{ border:1px solid #313131; border-radius:8px; }
[data-testid="stAlert"]{ border-radius:8px; border:1px solid #313131; }
hr{ border-color:#313131 !important; }
a, a:visited{ color:#6798ff; }
*{ box-shadow:none !important; }
</style>
"""


def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Завантаження рядів тривог…")
def get_series() -> pd.DataFrame:
    return load_series()


@st.cache_data(ttl=15)
def get_events() -> pd.DataFrame:
    return load_events()


@st.cache_data
def get_risk_now() -> pd.DataFrame:
    p = config.DATA_PROCESSED / "risk_now.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@st.cache_data
def get_metrics() -> dict:
    if config.MODEL_METRICS_JSON.exists():
        return json.loads(config.MODEL_METRICS_JSON.read_text())
    return {}


@st.cache_data(ttl=30, show_spinner=False)
def get_live_regions() -> set[str] | None:
    """Oblasts under an air-raid alert right now (alerts.in.ua), or None if live
    mode is off (no ALERTS_IN_UA_TOKEN / API error). 30s cache respects API limits."""
    return live.get_live_active_regions()


@st.cache_data(show_spinner=False)
def ukraine_geojson() -> dict:
    """Oblast polygons (24 oblasts + Kyiv City) with the canonical region in
    properties.canon. Source: geoBoundaries ADM1 (simplified); Crimea/Sevastopol
    already removed and canon injected when the file was vendored."""
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    feats = [f for f in gj["features"] if f["properties"].get("canon")]
    for f in feats:
        f["properties"]["ua"] = geo.ua_name(f["properties"]["canon"])
    gj["features"] = feats
    return gj


@st.cache_data
def hourly(region: str | None) -> pd.DataFrame:
    return analysis.hourly_pattern(get_series(), [region] if region else None)


@st.cache_data
def monthly(region: str | None) -> pd.DataFrame:
    return analysis.monthly_counts(get_series(), [region] if region else None)


@st.cache_data
def heatmap(region: str | None) -> pd.DataFrame:
    return analysis.hour_dow_heatmap(get_series(), [region] if region else None)


@st.cache_data
def episodes() -> pd.DataFrame:
    return analysis.mass_attack_episodes(get_series())


@st.cache_data
def surprises() -> pd.DataFrame:
    return analysis.surprise_days(get_series())


@st.cache_data
def propagation() -> pd.DataFrame:
    return analysis.propagation_lead_lag(get_series())


def latest_ts_local() -> str:
    ts = get_series()["ts"].max().tz_convert(config.DISPLAY_TZ)
    return ts.strftime("%Y-%m-%d %H:%M")


def risk_band(r: float) -> tuple[str, str]:
    if r < 0.34:
        return "Низький", GOOD
    if r < 0.67:
        return "Підвищений", WARN
    return "Високий", BAD


# --------------------------------------------------------------------------- #
# Freshness + mode switch
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def freshness() -> dict:
    if get_live_regions() is not None:
        return {"is_live": True, "dot": GOOD,
                "label": "Оновлюється в реальному часі", "sub": "поточний стан тривог"}
    return {"is_live": False, "dot": ASH,
            "label": f"Дані станом на {latest_ts_local()} (Київ)", "sub": "оновлюється раз на добу"}


def freshness_chip():
    f = freshness()
    cls = "live" if f["is_live"] else "snap"
    st.markdown(
        f'<span class="fresh {cls}">'
        f'<span class="dot" style="background:{f["dot"]}"></span>'
        f'<span class="lbl">{f["label"]}</span>'
        f'<span class="sub">· {f["sub"]}</span></span>',
        unsafe_allow_html=True,
    )


def _set_mode(m: str):
    st.session_state.mode = m


def mode_switch() -> str:
    """Dynamic role buttons stacked vertically; active = cornflower fill. State in
    session_state. on_click callbacks apply the new mode before the rerun renders —
    no highlight lag, no explicit st.rerun()."""
    if "mode" not in st.session_state:
        st.session_state.mode = "consumer"
    for key, label, icon in MODES:
        active = st.session_state.mode == key
        st.button(label, key=f"mode_btn_{key}", icon=icon,
                  type="primary" if active else "secondary", use_container_width=True,
                  on_click=_set_mode, args=(key,))
    return st.session_state.mode


# --------------------------------------------------------------------------- #
# Consumer action playbook (WHAT TO DO, not analytics)
# --------------------------------------------------------------------------- #
ACTIONS: dict[str, dict] = {
    "Низький": {
        "headline": "Звичайний режим. Тримайте готовність базовою.",
        "steps": [
            "Тримайте телефон зарядженим, а звук сирени — увімкненим.",
            "Знайте найближче укриття біля дому й роботи.",
            "Документи, вода й аптечка — в одному місці.",
        ],
    },
    "Підвищений": {
        "headline": "Будьте напоготові — реагуйте на сирену одразу.",
        "steps": [
            "Зарядіть телефон і павербанк зараз.",
            "Зберіть тривожну валізку: документи, вода, ліки, павербанк.",
            "Сплануйте, куди підете в укриття, і скоротіть час на збори.",
            "Не вимикайте звук сповіщень про тривогу.",
        ],
    },
    "Високий": {
        "headline": "Будьте готові вкритися негайно. Не зволікайте.",
        "steps": [
            "За сиреною — одразу в укриття або «правило двох стін», подалі від вікон.",
            "Тримайте тривожну валізку напоготові: документи, вода, аптечка, павербанк.",
            "Не знімайте й не наближайтеся до уражень та роботи ППО.",
            "Будьте на зв'язку: повідомте рідним, де ви; домовтеся про точку збору.",
        ],
    },
    "—": {
        "headline": "Дані оновлюються.",
        "steps": ["Тримайте сповіщення увімкненими та звіряйтеся з офіційними джерелами."],
    },
}


def render_action_card(band: str, color: str):
    a = ACTIONS.get(band, ACTIONS["—"])
    steps = "".join(f"<li>{s}</li>" for s in a["steps"])
    st.markdown(
        f"""
        <div style="background:{COAL};border:1px solid {STEEL};border-left:3px solid {color};
             border-radius:8px;padding:18px 20px;margin-top:4px;">
          <div style="font-family:'JetBrains Mono',monospace;text-transform:uppercase;
               letter-spacing:1px;font-size:11px;color:{ASH};margin-bottom:6px;">Що робити</div>
          <div style="font-family:'Inter';font-weight:600;font-size:18px;color:{SNOW};
               letter-spacing:-0.01em;margin-bottom:12px;">{a['headline']}</div>
          <ul style="margin:0;padding-left:18px;color:{ASH};font-size:14px;line-height:1.7;">{steps}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Plotly styling + maps
# --------------------------------------------------------------------------- #
def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=ASH, size=13),
        title_font=dict(family="Inter, sans-serif", color=SNOW, size=16),
        legend=dict(font=dict(color=ASH, size=12)),
        colorway=[ACCENT, ASH, GOOD, WARN, BAD, FOG],
        hoverlabel=dict(bgcolor=CARBON, bordercolor=STEEL,
                        font=dict(family="Inter, sans-serif", color=SNOW, size=12)),
    )
    fig.update_xaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
    fig.update_yaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
    return fig


def show(fig: go.Figure):
    st.plotly_chart(_style(fig), use_container_width=True)


def _risk_phrase(r: float) -> str:
    band, _ = risk_band(r)
    return f"{band} · {round(r * 100)}%"


def _nodata_base(gj: dict) -> go.Choroplethmap:
    """Crimea & Sevastopol rendered as Ukrainian territory with no alert data
    (temporarily occupied) — neutral fill so they read as Ukraine, not foreign."""
    feats = [f for f in gj["features"] if f["properties"]["canon"] in geo.NODATA_REGIONS]
    locs = [f["properties"]["canon"] for f in feats]
    return go.Choroplethmap(
        geojson={"type": "FeatureCollection", "features": feats},
        featureidkey="properties.canon", locations=locs, z=[0] * len(locs),
        colorscale=[[0, "#2f3a47"], [1, "#2f3a47"]], showscale=False,
        marker=dict(opacity=0.6, line=dict(color=STEEL, width=0.4)),
        customdata=[geo.ua_name(c) for c in locs],
        hovertemplate="<b>%{customdata}</b><br>тимчасово окупована · немає даних<extra></extra>",
    )


def risk_map(risk_df: pd.DataFrame, active: set[str] | None = None) -> go.Figure:
    """Filled oblast choropleth of alert risk (next 6h); active oblasts (live)
    outlined in cornflower. Ukraine pops on a dark basemap; neighbours muted."""
    gj = ukraine_geojson()
    active = set(active or set())
    df = risk_df.copy()
    canon = {f["properties"]["canon"] for f in gj["features"]}
    df = df[df["region"].isin(canon)]
    is_active = df["region"].isin(active)
    df["name_ua"] = df["region"].map(geo.ua_name)
    df["label"] = ["Тривога зараз" if a else _risk_phrase(r) for r, a in zip(df["risk"], is_active)]

    fig = go.Figure(go.Choroplethmap(
        geojson=gj, featureidkey="properties.canon",
        locations=df["region"], z=df["risk"], zmin=0, zmax=1,
        colorscale=[[0.0, GOOD], [0.34, GOOD], [0.34, WARN], [0.67, WARN], [0.67, BAD], [1.0, BAD]],
        marker=dict(line=dict(color=STEEL, width=0.6), opacity=0.82),
        customdata=df[["name_ua", "label"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        colorbar=dict(title=dict(text="Ризик", font=dict(color=ASH, size=12)),
                      tickvals=[0, 0.5, 1], ticktext=["0%", "50%", "100%"],
                      tickfont=dict(color=FOG, size=11), outlinewidth=0, thickness=10, len=0.7, x=0.99),
    ))
    fig.add_trace(_nodata_base(gj))
    act = df[is_active]
    if not act.empty:
        fig.add_trace(go.Choroplethmap(
            geojson=gj, featureidkey="properties.canon",
            locations=act["region"], z=[1] * len(act),
            colorscale=[[0, ACCENT], [1, ACCENT]], showscale=False,
            marker=dict(line=dict(color=ACCENT, width=2), opacity=0.0), hoverinfo="skip",
        ))
    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=4.3, center={"lat": 48.5, "lon": 31.4}),
        margin=dict(l=0, r=0, t=0, b=0), height=560, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=ASH),
        hoverlabel=dict(bgcolor=CARBON, bordercolor=STEEL, font=dict(family="Inter, sans-serif", color=SNOW)),
    )
    return fig


def hourly_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(df, x="hour", y="alert_rate", title=title)
    fig.update_yaxes(tickformat=".0%", title="Частка годин під тривогою")
    fig.update_xaxes(title="Година (Київ)", dtick=2)
    fig.update_traces(marker_color=ACCENT,
                      hovertemplate="Година %{x}:00<br>%{y:.0%} часу під тривогою<extra></extra>")
    fig.update_layout(height=320, margin=dict(t=40, b=10))
    return fig


def _active_set(risk_df: pd.DataFrame) -> set[str]:
    """Active oblasts only when LIVE (alerts.in.ua). Snapshot data must not claim
    'now', so we return an empty set -> the map shows forecast risk only."""
    live_regions = get_live_regions()
    return live_regions if live_regions is not None else set()


# --------------------------------------------------------------------------- #
# Consumer mode — actionable, minimal
# --------------------------------------------------------------------------- #
def render_consumer(region: str):
    st.subheader(f":material/warning: Ризик для: {geo.ua_name(region)}")
    risk_df = get_risk_now()
    row = risk_df[risk_df["region"] == region]
    risk = float(row["risk"].iloc[0]) if not row.empty else float("nan")
    band, color = risk_band(risk) if risk == risk else ("—", FOG)
    live_regions = get_live_regions()

    c1, c2 = st.columns([1, 1])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100 if risk == risk else 0,
            number={"suffix": "%", "font": {"color": SNOW, "family": "Inter"}},
            title={"text": f"Ризик тривоги в наступні {get_metrics().get('horizon_h', 6)} год",
                   "font": {"color": ASH, "size": 13, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": STEEL, "tickfont": {"color": FOG, "size": 10}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 1, "bordercolor": STEEL,
                "steps": [
                    {"range": [0, 34], "color": "#13211a"},
                    {"range": [34, 67], "color": "#231d12"},
                    {"range": [67, 100], "color": "#231316"},
                ],
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        st.plotly_chart(gauge, use_container_width=True)
    with c2:
        st.metric("Рівень ризику", band)
        if live_regions is not None:   # only claim "now" in live mode
            st.metric("Тривога зараз", "Активна" if region in live_regions else "Немає")
        with st.popover(":material/info: Що означає це число"):
            st.markdown(
                f"Це **оцінка моделі** — ймовірність повітряної тривоги у вашій області в "
                f"наступні {get_metrics().get('horizon_h', config.FORECAST_HORIZON_HOURS)} год за "
                "історичними даними (волонтерський датасет). Це **не офіційне сповіщення** — "
                "завжди реагуйте на реальні сирени.")

    render_action_card(band, color)

    st.divider()
    eyebrow("ПРОГНОЗ · ПО ОБЛАСТЯХ")
    st.markdown("#### :material/map: Поточний ризик по Україні")
    if not risk_df.empty:
        st.plotly_chart(risk_map(risk_df, _active_set(risk_df)), use_container_width=True)
        h = get_metrics().get("horizon_h", config.FORECAST_HORIZON_HOURS)
        cap = f"Колір — ймовірність тривоги в області в наступні {h} год."
        if live_regions is not None:
            cap += " Синім контуром — області з діючою тривогою."
        st.caption(cap)


# --------------------------------------------------------------------------- #
# Analyst mode
# --------------------------------------------------------------------------- #
def render_analyst(region: str | None):
    st.subheader(f":material/analytics: Аналітика — {geo.ua_name(region)}")
    tabs = st.tabs([
        ":material/map: Карта ризику", ":material/insights: Патерни",
        ":material/verified: Якість моделі", ":material/crisis_alert: Масовані атаки",
        ":material/hub: Поширення",
    ])

    with tabs[0]:
        with st.expander("Джерела даних і методологія"):
            st.markdown(
                f"- **Тривоги (історія):** Vadimkin air-raid dataset (GitHub) — погодинні ряди по "
                f"24 областях + Київ; оновлюється приблизно раз на добу, актуальність до "
                f"{latest_ts_local()} (Київ).\n"
                "- **Поточний стан (live):** alerts.in.ua — у реальному часі на деплої.\n"
                "- **OSINT:** повідомлення Telegram → структуровані події (демо-зразок).\n"
                f"- **Прогноз:** модель TSA — ймовірність тривоги в наступні "
                f"{get_metrics().get('horizon_h', 6)} год від останньої години даних."
            )
        risk_df = get_risk_now()
        if risk_df.empty:
            st.warning("Немає прогнозу. Запустіть `python -m src.forecast --predict`.")
        else:
            st.plotly_chart(risk_map(risk_df, _active_set(risk_df)), use_container_width=True)
            d = risk_df.copy()
            d["область"] = d["region"].map(geo.ua_name)
            d["risk"] = (d["risk"] * 100).round(1)
            d["active_now"] = d["active_now"].astype(bool)
            st.dataframe(
                d[["область", "risk", "active_now"]],
                column_config={
                    "область": st.column_config.TextColumn("область"),
                    "risk": st.column_config.NumberColumn(
                        "ризик, %", format="%.1f",
                        help=f"Прогноз: ймовірність тривоги в області в наступні "
                             f"{get_metrics().get('horizon_h', config.FORECAST_HORIZON_HOURS)} год."),
                    "active_now": st.column_config.CheckboxColumn(
                        "тривога на час даних",
                        help="Чи була область під тривогою на момент останнього зрізу даних. "
                             "Історичні дані оновлюються раз на добу, тож це стан станом на дату зрізу, не «зараз»."),
                },
                use_container_width=True, hide_index=True, height=300,
            )

    with tabs[1]:
        hm = heatmap(region)
        pivot = hm.pivot(index="dow", columns="hour", values="alert_rate").reindex(range(7))
        pivot.index = DOW_NAMES
        fig = px.imshow(
            pivot, color_continuous_scale=HEAT_SCALE, aspect="auto",
            labels=dict(x="Година (Київ)", y="День тижня", color="Частка"),
            title="Інтенсивність тривог: день тижня × година",
        )
        fig.update_traces(hovertemplate="%{y}, %{x}:00<br>%{z:.0%} часу під тривогою<extra></extra>")
        fig.update_layout(height=360, margin=dict(t=40))
        show(fig)

        c1, c2 = st.columns(2)
        with c1:
            show(hourly_fig(hourly(region), "За годиною доби"))
        with c2:
            m = monthly(region)
            fig = px.area(m, x="month", y="alerts", title="Кількість тривог за місяцями")
            fig.update_traces(line_color=ACCENT, fillcolor="rgba(103,152,255,0.18)",
                              hovertemplate="%{x}<br>%{y} тривог<extra></extra>")
            fig.update_layout(height=320, margin=dict(t=40), xaxis_title=None, yaxis_title="К-ть тривог")
            show(fig)

    with tabs[2]:
        m = get_metrics()
        if not m:
            st.warning("Немає метрик. Запустіть `python -m src.forecast --backtest`.")
        else:
            st.caption(
                f"Бектест на даних від {m['test_start'][:10]} до {m['test_end'][:10]} "
                f"(строго пізніших за трейн). Ціль: тривога в наступні {m['horizon_h']} год. "
                f"Базова частка позитивів: {m['positive_rate_test']:.0%}."
            )
            with st.popover(":material/info: Як читати цей розділ"):
                st.markdown(
                    "Прогноз перевірено на **пізнішому періоді**, якого модель не бачила (чесний бектест).\n\n"
                    "- **ROC-AUC / PR-AUC** — якість розрізнення «буде / не буде тривога» (більше = краще).\n"
                    "- **Brier** — точність калібрування ймовірностей (менше = краще).\n"
                    "- **Базлайни**: *climatology* (історична норма по годині та області) і *persistence* "
                    "(зараз = далі). Модель цінна, лише якщо їх перевершує.\n\n"
                    "**Як використати.** Якщо модель калібрована й б'є базлайни — числу ризику можна довіряти "
                    "як ймовірності та будувати на ньому сповіщення й графік чергувань."
                )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC (модель)", f"{m['model']['roc_auc']:.3f}",
                      f"{m['model']['roc_auc'] - m['baseline_climatology']['roc_auc']:+.3f} vs baseline",
                      help="Здатність відрізнити годину з тривогою від години без неї. "
                           "0.5 = випадково, 1.0 = ідеально. 0.88 ≈ у 88% пар модель дає вищий ризик "
                           "саме годині з тривогою.")
            c2.metric("PR-AUC (модель)", f"{m['model']['pr_auc']:.3f}",
                      help="Точність на рідких подіях — наскільки добре ловимо саме години з тривогою "
                           "(а не загальну точність). Важлива, коли ціна пропуску висока.")
            c3.metric("Brier (модель)", f"{m['model']['brier']:.3f}", "менше = краще", delta_color="off",
                      help="Похибка калібрування (0 = ідеально). Якщо кажемо «30%», подія має ставатися "
                           "приблизно у 30% таких випадків.")
            c4.metric("Climatology ROC-AUC", f"{m['baseline_climatology']['roc_auc']:.3f}",
                      help="Базлайн «історична норма»: середня частота тривог для цієї області й години доби. "
                           "Модель мусить його перевершувати, інакше не дає нічого понад сезонність.")

            comp = pd.DataFrame({
                "модель": ["Tryvoha Radar", "Climatology", "Persistence"],
                "ROC-AUC": [m["model"]["roc_auc"], m["baseline_climatology"]["roc_auc"], m["baseline_persistence"]["roc_auc"]],
                "PR-AUC": [m["model"]["pr_auc"], m["baseline_climatology"]["pr_auc"], m["baseline_persistence"]["pr_auc"]],
            })
            fig = px.bar(comp.melt(id_vars="модель", var_name="метрика", value_name="значення"),
                         x="метрика", y="значення", color="модель", barmode="group",
                         color_discrete_sequence=[ACCENT, "#5b6b8c", GRAPHITE],
                         title="Модель проти базлайнів (більше = краще)")
            fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}<extra></extra>")
            fig.update_layout(height=340, margin=dict(t=40), yaxis_range=[0.5, 1.0])
            show(fig)

            c1, c2 = st.columns(2)
            with c1:
                with st.popover(":material/info: Калібрування — як читати"):
                    st.markdown(
                        "Чи відповідають прогнозовані ймовірності реальності. Точки **на діагоналі** = "
                        "прогноз збігається з фактичною частотою. Вище діагоналі — модель недооцінює ризик, "
                        "нижче — переоцінює.\n\n**Як використати.** Підтверджує, чи можна довіряти числу % "
                        "як справжній ймовірності."
                    )
                rel = m["reliability"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color=GRAPHITE), name="ідеал",
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["fraction_positive"],
                                         mode="lines+markers", name="модель", line_color=ACCENT,
                                         hovertemplate="Прогноз %{x:.0%} → факт %{y:.0%}<extra></extra>"))
                fig.update_layout(title="Калібрування (reliability)", height=330,
                                  xaxis_title="Прогнозована ймовірність", yaxis_title="Фактична частка")
                show(fig)
            with c2:
                with st.popover(":material/info: ROC-крива — як читати"):
                    st.markdown(
                        "Компроміс між **виявленими тривогами** (TPR, вісь Y) і **хибними тривогами** "
                        "(FPR, вісь X) за різних порогів. Чим ближче крива до лівого-верхнього кута, тим "
                        "краще; діагональ = випадково.\n\n**Як використати.** Обрати поріг сповіщення під "
                        "прийнятну для вас частку хибних тривог."
                    )
                roc = m["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color=GRAPHITE), name="випадково",
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                         name=f"ROC (AUC={m['model']['roc_auc']:.3f})", line_color=ACCENT,
                                         hovertemplate="Хибних тривог %{x:.0%} · виявлено %{y:.0%}<extra></extra>"))
                fig.update_layout(title="ROC-крива", height=330, xaxis_title="FPR", yaxis_title="TPR")
                show(fig)

    with tabs[3]:
        eps = episodes().copy()
        span = get_series()["ts"]
        weeks = max(1.0, (span.max() - span.min()).days / 7)
        eps["start_local"] = eps["start"].dt.tz_convert(config.DISPLAY_TZ)

        m1, m2, m3 = st.columns(3)
        m1.metric("Масованих епізодів", f"{len(eps)}",
                  help="Епізод — проміжок, коли одночасно ≥18 із 24 областей під тривогою "
                       "(сусідні години обʼєднано). Ознака загальнонаціонального удару.")
        m2.metric("У середньому / тиждень", f"{len(eps) / weeks:.1f}",
                  help="Як часто трапляються масовані атаки. У цій війні вони регулярні — "
                       "це не статистична рідкість.")
        biggest = (f"{int(eps['peak_regions'].max())} обл. · {int(eps['duration_h'].max())} год"
                   if not eps.empty else "—")
        m3.metric("Найбільший епізод", biggest,
                  help="Пік одночасно активних областей і найбільша тривалість серед усіх епізодів.")

        with st.popover(":material/info: Що таке масована атака і як це читати"):
            st.markdown(
                "**Що це.** Масована атака — коли одночасно під тривогою щонайменше **18 із 24 областей** "
                "(~70% країни). Сигнатура загальнонаціонального удару (балістика, МіГ-31, великі рої Shahed), "
                "на відміну від локальних прифронтових тривог.\n\n"
                "**Чому їх багато.** У цій війні такі епізоди **часті** (≈2–3 на тиждень), тож це **не аномалія-"
                "рідкість**, а регулярна категорія масштабних загроз.\n\n"
                "**Як трактувати.** Тренд за місяцями показує ескалацію/деескалацію; найбільші епізоди "
                "(за к-тю областей і тривалістю) — це ночі наймасштабніших атак.\n\n"
                "**Як використати.** Планування готовності й чергувань, кореляція з ушкодженнями "
                "інфраструктури, оцінка інтенсивності повітряної кампанії в часі."
            )

        st.divider()
        eyebrow("НЕСПОДІВАНІ СПЛЕСКИ · ФАКТ vs ОЧІКУВАНО")
        with st.popover(":material/info: Що таке несподіваний сплеск і навіщо він"):
            st.markdown(
                "Щодня порівнюємо **фактичну** інтенсивність (область-годин під тривогою) з **очікуваною** — "
                "нормою за останні 4 тижні. **Несподіваний сплеск** — день, коли тривог було значно більше, "
                "ніж передбачає недавній патерн (надійний z ≥ 3).\n\n"
                "**Навіщо це, а не просто великі ночі.** Масштабні за абсолютом атаки видно й так. Цінніше — "
                "зловити **зміну патерну**: початок нової хвилі/кампанії або зсув тактики ворога, який ще не "
                "став «новою нормою».\n\n"
                "**Як використати.** Раннє виявлення ескалації; перевірка гіпотез («чи справді цього тижня "
                "інтенсивніше?»); співставлення сплесків із подіями на фронті та в постачанні засобів."
            )
        sur = surprises().dropna(subset=["expected"]).copy()
        sur["d"] = pd.to_datetime(sur["date"])
        sp = sur[sur["is_surprise"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sur["d"], y=sur["expected"], name="очікувано (норма)", mode="lines",
                                 line=dict(color=FOG, dash="dot"),
                                 hovertemplate="%{x|%d.%m.%Y}<br>очікувано ~%{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sur["d"], y=sur["actual"], name="фактично", mode="lines",
                                 line=dict(color=ACCENT),
                                 hovertemplate="%{x|%d.%m.%Y}<br>фактично %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=sp["d"], y=sp["actual"], name="несподіваний сплеск", mode="markers",
                                 marker=dict(color=BAD, size=7), customdata=sp["expected"],
                                 hovertemplate="%{x|%d.%m.%Y}<br>несподівано: %{y:.0f}"
                                               " (норма ~%{customdata:.0f})<extra></extra>"))
        fig.update_layout(title="Фактична vs очікувана інтенсивність", height=380,
                          margin=dict(t=40, b=70), yaxis_title="область-годин/день",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0, font=dict(size=11)))
        show(fig)
        st.caption(f"Дні, що найбільше вибилися з норми (всього {len(sp)} несподіваних днів за період):")
        top_s = sp.sort_values("ratio", ascending=False).head(10).copy()
        top_s["дата"] = top_s["d"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            top_s[["дата", "expected", "actual", "ratio"]],
            column_config={
                "expected": st.column_config.NumberColumn(
                    "очікувалось", format="%d", help="Норма за останні 4 тижні (медіана), область-годин/день."),
                "actual": st.column_config.NumberColumn(
                    "фактично", format="%d", help="Скільки було насправді цього дня."),
                "ratio": st.column_config.NumberColumn(
                    "× норми", format="%.1f", help="У скільки разів більше за очікуване."),
            },
            use_container_width=True, hide_index=True, height=300,
        )

        st.divider()
        eyebrow("ЕПІЗОДИ ЗА МІСЯЦЯМИ")
        eps["month"] = eps["start_local"].dt.strftime("%Y-%m")
        by_month = eps.groupby("month").size().rename("к-ть").reset_index()
        fig = px.bar(by_month, x="month", y="к-ть", title="Масовані атаки за місяцями")
        fig.update_traces(marker_color=ACCENT, hovertemplate="%{x}<br>%{y} епізодів<extra></extra>")
        fig.update_layout(height=320, margin=dict(t=40), xaxis_title=None, yaxis_title="епізодів/міс")
        show(fig)

        st.caption("Найбільші епізоди (за кількістю областей і тривалістю):")
        top = eps.sort_values(["peak_regions", "duration_h", "start"],
                              ascending=[False, False, False]).head(15).copy()
        top["дата (Київ)"] = top["start_local"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            top[["дата (Київ)", "peak_regions", "duration_h"]],
            column_config={
                "peak_regions": st.column_config.NumberColumn(
                    "пік областей", help="Максимум областей одночасно під тривогою в епізоді (з 24)."),
                "duration_h": st.column_config.NumberColumn(
                    "тривалість, год", help="Скільки тривав епізод масштабної тривоги."),
            },
            use_container_width=True, hide_index=True, height=360,
        )

    with tabs[4]:
        prop = propagation().head(15).copy()
        with st.popover(":material/info: Як читати поширення"):
            st.markdown(
                "Чи **перетікає** загроза між сусідніми областями. Для пари **A → B** дивимось, як часто "
                "нова тривога в A супроводжується новою тривогою в B упродовж **6 год**, і порівнюємо з "
                "випадковим збігом.\n\n"
                "- **lift > 1** — реальний «коридор» поширення (загроза справді йде далі).\n"
                "- **lift ≈ 1** — просто збіг.\n\n"
                "**Як використати.** Якщо в A почалася тривога — куди ймовірно піде далі й за який час "
                "(«запас, год»). Це основа прогнозу наступної області на вкладці «Загрози зараз»."
            )
        prop["з області"] = prop["from"].map(geo.ua_name)
        prop["в область"] = prop["to"].map(geo.ua_name)
        prop_disp = prop.assign(
            **{"follow_rate": (prop["follow_rate"] * 100).round(0),
               "base_rate": (prop["base_rate"] * 100).round(0),
               "lift": prop["lift"].round(2), "lead_h": prop["lead_h"].round(0)}
        )
        st.dataframe(
            prop_disp[["з області", "в область", "n_from", "follow_rate", "base_rate", "lift", "lead_h"]],
            column_config={
                "з області": st.column_config.TextColumn(
                    "з області", help="Область, де почалася тривога (джерело)."),
                "в область": st.column_config.TextColumn(
                    "→ в область", help="Сусідня область, куди загроза може поширитись."),
                "n_from": st.column_config.NumberColumn(
                    "n подій", help="Скільки разів у джерелі починалася нова тривога — база статистики."),
                "follow_rate": st.column_config.NumberColumn(
                    "слідує, %", format="%d",
                    help="У якій частці цих випадків у сусідній області теж почалася тривога впродовж 6 год."),
                "base_rate": st.column_config.NumberColumn(
                    "база, %", format="%d",
                    help="Фонова частота: скільки було б випадково (нова тривога в сусіда за будь-яке 6-год вікно)."),
                "lift": st.column_config.NumberColumn(
                    "lift ×", format="%.2f",
                    help="У скільки разів «слідує» перевищує випадковість. >1 = реальний коридор поширення."),
                "lead_h": st.column_config.NumberColumn(
                    "запас, год", format="%d",
                    help="Типовий час від тривоги в джерелі до тривоги в сусіда (медіана)."),
            },
            use_container_width=True, hide_index=True, height=420,
        )
        fig = px.bar(prop, x="lift", y=prop["з області"] + " → " + prop["в область"], orientation="h",
                     title="Топ-коридори поширення (lift)", color="lift", color_continuous_scale=HEAT_SCALE)
        fig.update_traces(hovertemplate="%{y}<br>підсилення ×%{x:.2f}<extra></extra>")
        fig.update_layout(height=460, margin=dict(t=40), yaxis_title=None, xaxis_title="lift ×")
        fig.update_yaxes(autorange="reversed")
        show(fig)


# --------------------------------------------------------------------------- #
# OSINT mode — live threat layer (its own top-level section)
# --------------------------------------------------------------------------- #
def _recent_events(ev: pd.DataFrame) -> tuple[pd.DataFrame, "pd.Timestamp | None"]:
    if ev.empty:
        return ev, None
    now = ev["timestamp"].max()
    rec = ev[ev["timestamp"] >= now - pd.Timedelta(hours=ACTIVE_WINDOW_H)].copy()
    return rec, now


ROCKET_SPEED_KMH = 800  # generic cruise-missile speed for schematic ETA


def _haversine_km(a: tuple, b: tuple) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = (math.sin(math.radians(la2 - la1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _rocket_origin(region: str) -> tuple:
    """Schematic launch point east of the target, inside Russian territory."""
    lat, lon = geo.COORDS.get(region, (49.0, 36.0))
    return (lat, min(max(lon + 4.0, 38.5), 41.0))


def _drone_color(p: float) -> str:
    return ACCENT if p >= 0.7 else "#3f6fcf" if p >= 0.45 else "#28406e" if p >= 0.25 else "#1a2740"


def _eta_color(mins: float) -> str:
    return (BAD if mins < 12 else "#f5712f" if mins < 22 else WARN if mins < 35
            else "#6f5f2a" if mins < 50 else "#1b2230")


def _osint_shading(gj, selected):
    """Per-oblast fill color + centroid label (ETA min / probability %) for selection."""
    canons = [f["properties"]["canon"] for f in gj["features"]]
    colors = {c: "#1b2230" for c in canons}
    labels = {}
    if selected and selected[0] == "drone":
        sub = propagation()
        sub = sub[sub["from"] == selected[1]["region"]]
        probs = dict(zip(sub["to"], sub["follow_rate"]))
        colors = {c: "#10141c" for c in canons}
        for c, p in probs.items():
            if c in colors:
                colors[c], labels[c] = _drone_color(p), f"{int(round(p * 100))}%"
    elif selected and selected[0] == "rocket":
        origin = _rocket_origin(selected[1]["region"])
        colors = {c: "#10141c" for c in canons}
        for c in canons:
            if c in geo.COORDS and c not in geo.NODATA_REGIONS:
                mins = _haversine_km(origin, geo.COORDS[c]) / ROCKET_SPEED_KMH * 60
                colors[c], labels[c] = _eta_color(mins), str(int(round(mins)))
    for nd in geo.NODATA_REGIONS:
        colors[nd] = "#2f3a47"
    return colors, labels


_PULSE_CSS = (
    "<style>@keyframes trpulse{0%{transform:scale(.5);opacity:.85}"
    "70%{transform:scale(2.6);opacity:0}100%{opacity:0}}"
    ".trping{position:relative;width:14px;height:14px;pointer-events:none}"
    ".trping .c{position:absolute;left:1px;top:1px;width:12px;height:12px;border-radius:50%}"
    ".trping .r{position:absolute;left:1px;top:1px;width:12px;height:12px;border-radius:50%;"
    "animation:trpulse 1.6s ease-out infinite}</style>"
)


def _osint_folium(gj, drones, rockets, impacts, intercepts, selected, show_air, show_hits):
    """Folium live-threat map. Only the drone/rocket/hit MARKERS are interactive;
    oblast polygons are non-clickable. Impacts/intercepts pulse (CSS animation)."""
    colors, labels = _osint_shading(gj, selected)
    m = folium.Map(location=[48.6, 33.3], zoom_start=5, tiles="CartoDB dark_matter", zoom_control=True)
    m.get_root().header.add_child(folium.Element(_PULSE_CSS))
    folium.GeoJson(
        gj, name="oblasts", interactive=False,
        style_function=lambda f: {"fillColor": colors.get(f["properties"]["canon"], "#1b2230"),
                                  "color": "#313131", "weight": 0.6, "fillOpacity": 0.6},
    ).add_to(m)

    for canon, txt in labels.items():
        c = geo.COORDS.get(canon)
        if c:
            folium.Marker(list(c), interactive=False, icon=folium.DivIcon(
                html=f'<div style="color:#fff;font:600 11px Inter,sans-serif;text-align:center;'
                     f'text-shadow:0 0 3px #000;pointer-events:none;">{txt}</div>',
                icon_size=(44, 16), icon_anchor=(22, 8))).add_to(m)

    if show_air:
        for _, d in drones.iterrows():
            c = geo.COORDS.get(d["region"])
            if c:
                t = d["timestamp"].tz_convert(config.DISPLAY_TZ).strftime("%H:%M")
                folium.CircleMarker([c[0], c[1]], radius=8, color="#f5a623", weight=1, fill=True,
                                    fill_color="#f5a623", fill_opacity=0.9,
                                    tooltip=f"Дрон (Shahed) · {geo.ua_name(d['region'])} · {t}").add_to(m)
        for _, r in rockets.iterrows():
            o = _rocket_origin(r["region"])
            t = r["timestamp"].tz_convert(config.DISPLAY_TZ).strftime("%H:%M")
            folium.CircleMarker([o[0], o[1]], radius=9, color="#c77dff", weight=1, fill=True,
                                fill_color="#c77dff", fill_opacity=0.9,
                                tooltip=f"Ракета (схематично, рф) · ціль {geo.ua_name(r['region'])} · {t}").add_to(m)
    if show_hits:
        def ping(lat, lon, color, tip):
            folium.Marker([lat, lon], tooltip=tip, icon=folium.DivIcon(
                html=f'<div class="trping"><span class="r" style="background:{color}"></span>'
                     f'<span class="c" style="background:{color}"></span></div>',
                icon_size=(14, 14), icon_anchor=(7, 7))).add_to(m)
        for _, e in intercepts.iterrows():
            t = e["timestamp"].tz_convert(config.DISPLAY_TZ).strftime("%H:%M")
            ping(e["lat"], e["lon"], "#54a24b", f"Збито · {geo.ua_name(e['region'])} · {t}")
        for _, e in impacts.iterrows():
            t = e["timestamp"].tz_convert(config.DISPLAY_TZ).strftime("%H:%M")
            ping(e["lat"], e["lon"], "#e5484d", f"Влучання · {geo.ua_name(e['region'])} · {t}")
    return m


def _resolve_folium(state, drones, rockets):
    """Map a folium marker click (last_object_clicked) to the nearest drone/rocket."""
    c = (state or {}).get("last_object_clicked")
    if not c:
        return None
    lat, lon = c.get("lat"), c.get("lng")
    if lat is None or lon is None:
        return None
    best, bestd = None, 0.6 ** 2
    for kind, df, loc in [("drone", drones, lambda r: geo.COORDS.get(r["region"])),
                          ("rocket", rockets, lambda r: _rocket_origin(r["region"]))]:
        for _, row in df.iterrows():
            p = loc(row)
            if not p:
                continue
            dd = (p[0] - lat) ** 2 + (p[1] - lon) ** 2
            if dd < bestd:
                bestd, best = dd, (kind, row)
    return best


def render_osint():
    st.subheader(":material/radar: Загрози зараз — жива карта")
    ev = get_events()
    if ev.empty:
        st.info("Кеш подій порожній. Запустіть `python -m src.ai_extractor` "
                "або live-колектор `python -m src.collector`.")
        return
    rec, _now = _recent_events(ev)
    gj = ukraine_geojson()
    inflight = rec[rec["event_type"].isin(["пуск", "рух"]) & rec["region"].notna()].sort_values(
        "timestamp", ascending=False)
    drones = inflight[inflight["weapon"] == "дрон/шахед"]
    rockets = inflight[inflight["weapon"] == "ракета"]
    impacts = rec[rec["event_type"] == "влучання"].dropna(subset=["lat", "lon"])
    intercepts = rec[rec["event_type"] == "збиття"].dropna(subset=["lat", "lon"])
    active = not drones.empty or not rockets.empty

    col_map, col_feed = st.columns([3, 2])
    with col_map:
        if not active:
            rn = get_risk_now()
            if rn.empty:
                st.warning("Немає прогнозу. Запустіть `python -m src.forecast --predict`.")
            else:
                st.success("Наразі активних пусків немає — показано прогноз ризику тривог по областях.")
                st.plotly_chart(risk_map(rn, set()), use_container_width=True)
        else:
            ctrl = st.columns(2)
            show_air = ctrl[0].toggle("Ракети та дрони", value=True, key="osint_air")
            show_hits = ctrl[1].toggle("Прильоти та збиття", value=True, key="osint_hits")
            sel = _resolve_folium(st.session_state.get("osint_map"), drones, rockets)
            m = _osint_folium(gj, drones, rockets, impacts, intercepts, sel, show_air, show_hits)
            st_folium(m, use_container_width=True, height=560,
                      returned_objects=["last_object_clicked"], key="osint_map")
            if sel and sel[0] == "rocket":
                st.caption(f"Ракета на {geo.ua_name(sel[1]['region'])}: числа на областях = "
                           "орієнтовний час підльоту, хв (від часу повідомлення, ~800 км/год, схематично).")
            elif sel and sel[0] == "drone":
                st.caption(f"Дрон ({geo.ua_name(sel[1]['region'])}): % — ймовірність, що тривога "
                           "пошириться в цю область (історична аналітика). Без даних Telegram про курс "
                           "точний напрямок не показуємо.")
            else:
                st.caption("Натисніть на маркер дрона або ракети, щоб побачити деталі. "
                           "Прильоти — червоні, збиття — зелені.")

    with col_feed:
        fh = st.columns([2, 1])
        with fh[0]:
            eyebrow("СТРІЧКА СПОВІЩЕНЬ")
        as_table = fh[1].toggle("Таблиця", value=False, key="osint_feed_table")
        feed = ev.sort_values("timestamp", ascending=False).head(14)
        if as_table:
            f = feed.copy()
            f["час"] = f["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%m-%d %H:%M")
            f["область"] = f["region"].map(lambda r: geo.ua_name(r) if r else "")
            f["weapon"] = f["weapon"].where(f["weapon"] != "невідомо", "")
            st.dataframe(f[["час", "event_type", "weapon", "область"]]
                         .rename(columns={"event_type": "подія", "weapon": "засіб"}),
                         use_container_width=True, hide_index=True, height=560)
        else:
            cards = []
            for _, e in feed.iterrows():
                color = EVENT_COLORS.get(e["event_type"], "#454545")
                t = e["timestamp"].tz_convert(config.DISPLAY_TZ).strftime("%d.%m %H:%M")
                reg = geo.ua_name(e["region"]) if e["region"] else ""
                meta = " · ".join(x for x in [reg, e["weapon"]] if x and x != "невідомо")
                text = (e["text"] or "").replace("<", "&lt;").replace(">", "&gt;")
                cards.append(
                    f'<div style="background:#141414;border:1px solid #313131;'
                    f'border-left:3px solid {color};border-radius:8px;padding:9px 12px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:JetBrains Mono,monospace;font-size:11px;color:#7c7c7c;">'
                    f'<span>{t}</span><span style="color:{color};text-transform:uppercase;'
                    f'letter-spacing:.5px;">{e["event_type"]}</span></div>'
                    f'<div style="color:#e9edf5;font-size:13px;line-height:1.45;margin-top:5px;">{text}</div>'
                    + (f'<div style="color:#7c7c7c;font-size:11px;margin-top:4px;">{meta}</div>' if meta else "")
                    + "</div>")
            st.markdown(
                f'<div style="max-height:600px;overflow-y:auto;padding-right:4px;">{"".join(cards)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inject_theme()
    eyebrow("ПОВІТРЯНІ ТРИВОГИ · УКРАЇНА · АНАЛІТИКА ТА ЖИВА КАРТА")
    st.title("Tryvoha Radar")
    freshness_chip()

    with st.sidebar:
        eyebrow("РЕЖИМ")
        mode = mode_switch()
        st.divider()
        regions = sorted(geo.REGIONS, key=geo.ua_name)
        if mode == "consumer":
            region = st.selectbox("Ваша область", regions,
                                  index=regions.index("Kyiv City"), format_func=geo.ua_name)
        elif mode == "analyst":
            region = st.selectbox("Фокус (необов'язково)", [None] + regions, format_func=geo.ua_name)
        else:
            region = None

    if mode == "consumer":
        render_consumer(region)
    elif mode == "analyst":
        render_analyst(region)
    else:
        render_osint()


main()
