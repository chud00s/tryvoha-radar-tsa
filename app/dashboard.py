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
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    ("osint", "OSINT · Telegram", ":material/radar:"),
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
    gj["features"] = [f for f in gj["features"] if f["properties"].get("canon")]
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
def intensity() -> pd.DataFrame:
    return analysis.detect_mass_attacks(get_series())


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
    )
    fig.update_xaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
    fig.update_yaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
    return fig


def show(fig: go.Figure):
    st.plotly_chart(_style(fig), use_container_width=True)


def _risk_phrase(r: float) -> str:
    band, _ = risk_band(r)
    return f"{band} · {round(r * 100)}%"


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
        colorscale=[[0.0, GOOD], [0.5, WARN], [1.0, BAD]],
        marker=dict(line=dict(color=STEEL, width=0.6), opacity=0.82),
        customdata=df[["name_ua", "label"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
        colorbar=dict(title=dict(text="Ризик", font=dict(color=ASH, size=12)),
                      tickvals=[0, 0.5, 1], ticktext=["0%", "50%", "100%"],
                      tickfont=dict(color=FOG, size=11), outlinewidth=0, thickness=10, len=0.7, x=0.99),
    ))
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
    fig.update_traces(marker_color=ACCENT)
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

    render_action_card(band, color)

    st.divider()
    eyebrow("ПРОГНОЗ · ПО ОБЛАСТЯХ")
    st.markdown("#### :material/map: Поточний ризик по Україні")
    if not risk_df.empty:
        st.plotly_chart(risk_map(risk_df, _active_set(risk_df)), use_container_width=True)
        cap = "Колір — ймовірність тривоги в області в наступні 6 год."
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
            st.dataframe(
                d[["область", "risk", "active_now"]]
                .rename(columns={"risk": "ризик, %", "active_now": "активна (ост. год.)"}),
                use_container_width=True, hide_index=True, height=300,
            )

    with tabs[1]:
        hm = heatmap(region)
        pivot = hm.pivot(index="dow", columns="hour", values="alert_rate").reindex(range(7))
        fig = px.imshow(
            pivot, color_continuous_scale=HEAT_SCALE, aspect="auto",
            labels=dict(x="Година (Київ)", y="День тижня", color="Частка"),
            title="Інтенсивність тривог: день тижня × година",
        )
        fig.update_yaxes(tickvals=list(range(7)), ticktext=DOW_NAMES)
        fig.update_layout(height=360, margin=dict(t=40))
        show(fig)

        c1, c2 = st.columns(2)
        with c1:
            show(hourly_fig(hourly(region), "За годиною доби"))
        with c2:
            m = monthly(region)
            fig = px.area(m, x="month", y="alerts", title="Кількість тривог за місяцями")
            fig.update_traces(line_color=ACCENT, fillcolor="rgba(103,152,255,0.18)")
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC (модель)", f"{m['model']['roc_auc']:.3f}",
                      f"{m['model']['roc_auc'] - m['baseline_climatology']['roc_auc']:+.3f} vs baseline")
            c2.metric("PR-AUC (модель)", f"{m['model']['pr_auc']:.3f}")
            c3.metric("Brier (модель)", f"{m['model']['brier']:.3f}", "менше = краще", delta_color="off")
            c4.metric("Climatology ROC-AUC", f"{m['baseline_climatology']['roc_auc']:.3f}")

            comp = pd.DataFrame({
                "модель": ["Tryvoha Radar", "Climatology", "Persistence"],
                "ROC-AUC": [m["model"]["roc_auc"], m["baseline_climatology"]["roc_auc"], m["baseline_persistence"]["roc_auc"]],
                "PR-AUC": [m["model"]["pr_auc"], m["baseline_climatology"]["pr_auc"], m["baseline_persistence"]["pr_auc"]],
            })
            fig = px.bar(comp.melt(id_vars="модель", var_name="метрика", value_name="значення"),
                         x="метрика", y="значення", color="модель", barmode="group",
                         color_discrete_sequence=[ACCENT, "#5b6b8c", GRAPHITE],
                         title="Модель проти базлайнів (більше = краще)")
            fig.update_layout(height=340, margin=dict(t=40), yaxis_range=[0.5, 1.0])
            show(fig)

            c1, c2 = st.columns(2)
            with c1:
                rel = m["reliability"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color=GRAPHITE), name="ідеал"))
                fig.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["fraction_positive"],
                                         mode="lines+markers", name="модель", line_color=ACCENT))
                fig.update_layout(title="Калібрування (reliability)", height=330,
                                  xaxis_title="Прогнозована ймовірність", yaxis_title="Фактична частка")
                show(fig)
            with c2:
                roc = m["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color=GRAPHITE), name="випадково"))
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                         name=f"ROC (AUC={m['model']['roc_auc']:.3f})", line_color=ACCENT))
                fig.update_layout(title="ROC-крива", height=330, xaxis_title="FPR", yaxis_title="TPR")
                show(fig)

    with tabs[3]:
        inten = intensity().copy()
        inten["ts_local"] = inten["ts"].dt.tz_convert(config.DISPLAY_TZ)
        daily = (
            inten.set_index("ts_local").resample("D")
            .agg(regions_active=("regions_active", "max"), anomaly=("is_anomaly", "max")).reset_index()
        )
        n_anom = int(inten["is_anomaly"].sum())
        st.caption(
            f"Аномалії = години, коли кількість одночасно активних областей різко перевищує "
            f"ковзний baseline (z ≥ 3). Виявлено **{n_anom}** таких годин — сигнатура масованих атак."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["ts_local"], y=daily["regions_active"],
                                 mode="lines", name="макс. областей/день", line_color=ACCENT))
        anom = daily[daily["anomaly"] == 1]
        fig.add_trace(go.Scatter(x=anom["ts_local"], y=anom["regions_active"], mode="markers",
                                 name="день з аномалією", marker=dict(color=BAD, size=6)))
        fig.update_layout(title="Загальнонаціональна інтенсивність тривог (денний максимум)",
                          height=420, margin=dict(t=40), yaxis_title="Областей одночасно")
        show(fig)

    with tabs[4]:
        prop = propagation().head(15).copy()
        st.caption(
            "Наскільки нова тривога в області A передвіщає нову тривогу в сусідній B упродовж 6 год. "
            "**lift** — у скільки разів частіше за випадковий збіг (база B). lift > 1 = реальний «коридор» загрози."
        )
        prop["з області"] = prop["from"].map(geo.ua_name)
        prop["в область"] = prop["to"].map(geo.ua_name)
        prop_disp = prop.assign(
            **{"follow_rate": (prop["follow_rate"] * 100).round(0),
               "base_rate": (prop["base_rate"] * 100).round(0), "lift": prop["lift"].round(2)}
        )
        st.dataframe(
            prop_disp[["з області", "в область", "n_from", "follow_rate", "base_rate", "lift"]]
            .rename(columns={"n_from": "n подій", "follow_rate": "слідує, %", "base_rate": "база, %", "lift": "lift ×"}),
            use_container_width=True, hide_index=True, height=420,
        )
        fig = px.bar(prop, x="lift", y=prop["з області"] + " → " + prop["в область"], orientation="h",
                     title="Топ-коридори поширення (lift)", color="lift", color_continuous_scale=HEAT_SCALE)
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


def predicted_vectors(active_regions: list[str]) -> list[dict]:
    """For each active threat oblast, the most likely NEXT oblast (highest lift)
    with its empirical lead time — the data-backed 'куди далі'."""
    prop = propagation()
    vecs, seen = [], set()
    for r in active_regions:
        if r in seen or r not in geo.COORDS:
            continue
        seen.add(r)
        cand = prop[prop["from"] == r]
        if cand.empty:
            continue
        top = cand.sort_values("lift", ascending=False).iloc[0]
        to = top["to"]
        if to not in geo.COORDS:
            continue
        lead = top.get("lead_h", float("nan"))
        lead_lbl = ("<1 год" if lead == lead and lead < 1
                    else f"~{int(round(lead))} год" if lead == lead else "")
        c0, c1 = geo.COORDS[r], geo.COORDS[to]
        vecs.append({"lat0": c0[0], "lon0": c0[1], "lat1": c1[0], "lon1": c1[1],
                     "to_ua": geo.ua_name(to), "from_ua": geo.ua_name(r),
                     "label": f"{geo.ua_name(to)} · {top['lift']:.1f}× · {lead_lbl}"})
    return vecs[:5]


def threat_map(risk_df: pd.DataFrame, rec: pd.DataFrame, vectors: list[dict]) -> go.Figure:
    gj = ukraine_geojson()
    canon = {f["properties"]["canon"] for f in gj["features"]}
    base = risk_df[risk_df["region"].isin(canon)]
    fig = go.Figure(go.Choroplethmap(
        geojson=gj, featureidkey="properties.canon", locations=base["region"],
        z=base["risk"], zmin=0, zmax=1, colorscale=[[0.0, GOOD], [0.5, WARN], [1.0, BAD]],
        marker=dict(opacity=0.30, line=dict(color=STEEL, width=0.4)),
        showscale=False, hoverinfo="skip",
    ))
    for v in vectors:  # predicted-next propagation vectors
        fig.add_trace(go.Scattermap(lat=[v["lat0"], v["lat1"]], lon=[v["lon0"], v["lon1"]],
                                    mode="lines", line=dict(color=ACCENT, width=2),
                                    hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattermap(lat=[v["lat1"]], lon=[v["lon1"]], mode="markers+text",
                                    marker=dict(size=8, color=ACCENT), text=[v["label"]],
                                    textposition="top center", textfont=dict(color=ACCENT, size=10),
                                    hoverinfo="skip", showlegend=False))
    if not rec.empty:
        m = rec.dropna(subset=["lat", "lon"]).copy()
        m["k"] = m.groupby("region").cumcount()
        m["lat"] += (m["k"] % 3 - 1) * 0.09
        m["lon"] += (m["k"] // 3) * 0.11
        m["name_ua"] = m["region"].map(geo.ua_name)
        m["t"] = m["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%H:%M")
        for et, color in EVENT_COLORS.items():
            sub = m[m["event_type"] == et]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermap(
                lat=sub["lat"], lon=sub["lon"], mode="markers", name=et,
                marker=dict(size=13, color=color),
                customdata=sub[["name_ua", "weapon", "t"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>" + et +
                              " · %{customdata[1]} · %{customdata[2]}<extra></extra>",
            ))
    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=4.4, center={"lat": 48.6, "lon": 31.4}),
        margin=dict(l=0, r=0, t=0, b=0), height=580, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=ASH, size=11), bgcolor="rgba(20,20,20,0.6)", title_text=""),
        hoverlabel=dict(bgcolor=CARBON, bordercolor=STEEL, font=dict(family="Inter", color=SNOW)),
    )
    return fig


@st.fragment(run_every="30s")
def render_osint():
    st.subheader(":material/radar: OSINT — жива карта загроз")
    method = ""
    ev = get_events()
    if ev.empty:
        st.info("Кеш подій порожній. Запустіть `python -m src.ai_extractor` "
                "або live-колектор `python -m src.collector`.")
        return
    method = ev["method"].iloc[0] if "method" in ev else "—"
    rec, _now = _recent_events(ev)
    threat_regions = [r for r in
                      dict.fromkeys(rec.sort_values("timestamp", ascending=False)["region"].tolist())
                      if r and r in set(rec[rec["event_type"].isin(THREAT_TYPES)]["region"])]
    vectors = predicted_vectors(threat_regions)
    risk_df = get_risk_now()

    col_map, col_feed = st.columns([3, 2])
    with col_map:
        st.plotly_chart(threat_map(risk_df, rec, vectors), use_container_width=True)
        st.caption(
            f"Маркери — OSINT-події за останні {ACTIVE_WINDOW_H} год "
            f"({'LLM (Claude)' if method == 'llm' else 'rule-based'}). "
            "Сині вектори — ймовірно наступна область за аналітикою поширення (ETA орієнтовно)."
        )
    with col_feed:
        eyebrow("СТРІЧКА ПОДІЙ")
        feed = ev.sort_values("timestamp", ascending=False).head(10).copy()
        feed["час"] = feed["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%m-%d %H:%M")
        feed["область"] = feed["region"].map(geo.ua_name)
        st.dataframe(
            feed[["час", "event_type", "weapon", "область"]]
            .rename(columns={"event_type": "подія", "weapon": "засіб"}),
            use_container_width=True, hide_index=True, height=300,
        )
        if vectors:
            eyebrow("ПРОГНОЗ ПОШИРЕННЯ")
            for v in vectors[:4]:
                st.markdown(f"- **{v['from_ua']}** → {v['label']}")

    st.divider()
    c1, c2 = st.columns([2, 3])
    with c1:
        counts = ev["event_type"].value_counts().reset_index()
        counts.columns = ["подія", "к-ть"]
        fig = px.bar(counts, x="к-ть", y="подія", orientation="h", color="подія",
                     color_discrete_map=EVENT_COLORS, title="Події за типом")
        fig.update_layout(height=300, margin=dict(t=40), showlegend=False, yaxis_title=None)
        show(fig)
    with c2:
        eyebrow("СИНТЕЗ · OSINT × ТРИВОГИ")
        fusion_view(ev)


def fusion_view(ev: pd.DataFrame):
    series = get_series()
    inten = analysis.national_intensity(series)
    t0, t1 = ev["timestamp"].min(), ev["timestamp"].max()
    win = inten[(inten["ts"] >= t0 - pd.Timedelta(hours=2)) & (inten["ts"] <= t1 + pd.Timedelta(hours=2))]
    if win.empty:
        st.caption("Вікно подій поза межами наявних даних тривог — синтез недоступний для цього зразка.")
        return
    ev_hourly = ev.assign(h=ev["timestamp"].dt.floor("h")).groupby("h").size().rename("events").reset_index()
    win = win.copy()
    win["ts_local"] = win["ts"].dt.tz_convert(config.DISPLAY_TZ)
    ev_hourly["ts_local"] = ev_hourly["h"].dt.tz_convert(config.DISPLAY_TZ)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win["ts_local"], y=win["regions_active"], name="областей під тривогою",
                             mode="lines", line_color=ACCENT, fill="tozeroy", fillcolor="rgba(103,152,255,0.15)"))
    fig.add_trace(go.Bar(x=ev_hourly["ts_local"], y=ev_hourly["events"], name="OSINT-подій/год",
                         marker_color=ASH, yaxis="y2", opacity=0.85))
    fig.update_layout(
        height=360, margin=dict(t=30), yaxis=dict(title="Областей під тривогою"),
        yaxis2=dict(title="OSINT-подій", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12),
    )
    show(fig)
    st.caption(
        "Сплески OSINT-повідомлень синхронні зі зростанням кількості областей під тривогою — "
        "OSINT-моніторинг дає ранній контекст до/під час офіційних тривог."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inject_theme()
    eyebrow("ПОВІТРЯНІ ТРИВОГИ · УКРАЇНА · TSA + OSINT")
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
