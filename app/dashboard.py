"""Tryvoha Radar — Streamlit dashboard.

Two modes:
  * Споживач  — "what is the risk in my oblast now / today", simple.
  * Аналітик  — patterns, model quality, mass-attack anomalies, cross-border
                propagation, and the OSINT (Telegram) event layer.

Visual language: "blueprint control room" (Dovetail design system) — near-black
canvas with a faint grid, a single cornflower-blue accent, Inter + JetBrains Mono,
8px radii, flat surfaces (no shadows). Danger semantics (red/amber/green) are kept
for the risk gauge/map since they encode meaning, not decoration.
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

st.set_page_config(page_title="Tryvoha Radar", page_icon="🛰️", layout="wide")

# --- design tokens ----------------------------------------------------------
ACCENT = "#6798ff"   # cornflower
INK = "#0a0a0a"      # page
COAL = "#141414"     # section / card
CARBON = "#1e1e1e"   # surface / button
STEEL = "#313131"    # hairline border
GRAPHITE = "#454545"
FOG = "#7c7c7c"
ASH = "#a7a7a7"
SNOW = "#ffffff"
GOOD, WARN, BAD = "#54a24b", "#f5a623", "#e5484d"   # semantic danger scale

HEAT_SCALE = [[0.0, "#101319"], [0.4, "#21407a"], [0.7, "#3f6fcf"], [1.0, ACCENT]]
BLUE_SCALE = [[0.0, "#1a2336"], [1.0, ACCENT]]

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

/* blueprint grid motif on the canvas */
[data-testid="stAppViewContainer"]{
  background-color:#0a0a0a;
  background-image:
    repeating-linear-gradient(0deg, transparent 0 47px, #15161b 47px 48px),
    repeating-linear-gradient(90deg, transparent 0 47px, #15161b 47px 48px);
}
[data-testid="stMainBlockContainer"]{ max-width:1320px; padding-top:2.5rem; }
[data-testid="stHeader"]{ background:transparent; }

/* sidebar */
[data-testid="stSidebar"]{ background-color:#141414; border-right:1px solid #313131; }

/* headings — engineered, tight tracking */
h1,h2,h3,h4{ font-family:'Inter'; font-weight:600; color:#ffffff; letter-spacing:-0.02em; }
h1{ font-size:40px; letter-spacing:-0.03em; }
h2{ font-size:24px; } h3{ font-size:20px; }

/* eyebrow + captions */
.eyebrow{ font-family:'JetBrains Mono',monospace; text-transform:uppercase;
  letter-spacing:1px; font-size:12px; color:#6798ff; margin:0 0 2px 0; }
[data-testid="stCaptionContainer"]{ color:#a7a7a7 !important; }

/* metric = instrument readout card */
[data-testid="stMetric"]{ background:#141414; border:1px solid #313131;
  border-radius:8px; padding:14px 16px; }
[data-testid="stMetricLabel"] p{ font-family:'JetBrains Mono',monospace !important;
  text-transform:uppercase; letter-spacing:0.85px; font-size:11px !important; color:#a7a7a7 !important; }
[data-testid="stMetricValue"]{ font-family:'Inter'; font-weight:600; color:#ffffff; }
[data-testid="stMetricDelta"]{ font-family:'JetBrains Mono',monospace; }

/* tabs — technical mono, blue active */
[data-testid="stTabs"] button[role="tab"]{ font-family:'JetBrains Mono',monospace;
  text-transform:uppercase; letter-spacing:0.5px; font-size:12px; color:#a7a7a7; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{ color:#6798ff; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background-color:#6798ff !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]{ background-color:#313131 !important; }

/* buttons */
.stButton>button, .stDownloadButton>button{ background:#1e1e1e; color:#ffffff;
  border:1px solid #454545; border-radius:8px; font-weight:500; }
.stButton>button:hover{ border-color:#6798ff; color:#6798ff; }

/* select / inputs */
[data-baseweb="select"]>div{ background:#1e1e1e !important; border:1px solid #454545 !important; border-radius:8px !important; }
[data-baseweb="popover"] [role="listbox"]{ background:#1e1e1e !important; }

/* dataframe + alerts */
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


@st.cache_data
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


@st.cache_data
def hourly(region: str | None) -> pd.DataFrame:
    return analysis.hourly_pattern(get_series(), [region] if region else None)


@st.cache_data
def dow(region: str | None) -> pd.DataFrame:
    return analysis.dow_pattern(get_series(), [region] if region else None)


@st.cache_data
def monthly(region: str | None) -> pd.DataFrame:
    return analysis.monthly_counts(get_series(), [region] if region else None)


@st.cache_data
def heatmap(region: str | None) -> pd.DataFrame:
    return analysis.hour_dow_heatmap(get_series(), [region] if region else None)


@st.cache_data
def region_summary() -> pd.DataFrame:
    return analysis.region_summary(get_series())


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
# Plotly styling
# --------------------------------------------------------------------------- #
def _style(fig: go.Figure, cartesian: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=ASH, size=13),
        title_font=dict(family="Inter, sans-serif", color=SNOW, size=16),
        legend=dict(font=dict(color=ASH, size=12)),
        colorway=[ACCENT, ASH, GOOD, WARN, BAD, FOG],
    )
    if cartesian:
        fig.update_xaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
        fig.update_yaxes(gridcolor=CARBON, linecolor=STEEL, zerolinecolor=STEEL, tickfont=dict(color=FOG))
    return fig


def show(fig: go.Figure, cartesian: bool = True):
    st.plotly_chart(_style(fig, cartesian), use_container_width=True)


def map_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=ASH),
    )
    return fig


# --------------------------------------------------------------------------- #
# Shared figures
# --------------------------------------------------------------------------- #
def risk_map(risk_df: pd.DataFrame) -> go.Figure:
    df = risk_df.copy()
    df["lat"] = df["region"].map(lambda r: geo.COORDS.get(r, (None, None))[0])
    df["lon"] = df["region"].map(lambda r: geo.COORDS.get(r, (None, None))[1])
    df = df.dropna(subset=["lat", "lon"])
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="risk", size=df["risk"] * 40 + 6,
        color_continuous_scale="RdYlGn_r", range_color=(0, 1),
        hover_name="region", hover_data={"risk": ":.0%", "lat": False, "lon": False},
        zoom=4.3, center={"lat": 48.4, "lon": 31.2}, height=560,
    )
    return map_layout(fig)


def hourly_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(df, x="hour", y="alert_rate", title=title)
    fig.update_yaxes(tickformat=".0%", title="Частка годин під тривогою")
    fig.update_xaxes(title="Година (Київ)", dtick=2)
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(height=320, margin=dict(t=40, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Consumer mode
# --------------------------------------------------------------------------- #
def render_consumer(region: str):
    st.subheader(f"🚦 Ризик для: {region}")
    risk_df = get_risk_now()
    series = get_series()

    row = risk_df[risk_df["region"] == region]
    risk = float(row["risk"].iloc[0]) if not row.empty else float("nan")
    band, color = risk_band(risk) if risk == risk else ("—", FOG)

    # "Alert now": prefer real-time alerts.in.ua, fall back to latest snapshot hour.
    live_regions = get_live_regions()
    if live_regions is not None:
        active_now = region in live_regions
        live_badge = "🔴 LIVE · alerts.in.ua"
    else:
        active_now = bool(series[series["region"] == region]["active"].iloc[-1])
        live_badge = "🕒 snapshot · історичний датасет"

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
        st.metric("Тривога зараз", "🔴 Активна" if active_now else "🟢 Немає")
        st.caption(live_badge)
        rs = region_summary().set_index("region").loc[region]
        st.metric("Частка часу під тривогою (весь період)", f"{rs['alert_rate']:.0%}")
        if "avg_duration_min" in rs and pd.notna(rs["avg_duration_min"]):
            st.metric("Середня тривалість тривоги", f"{rs['avg_duration_min']:.0f} хв")

    st.markdown(f"**Висновок:** ризик зараз — :{'red' if band=='Високий' else 'orange' if band=='Підвищений' else 'green'}[{band}].")

    st.divider()
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        show(hourly_fig(hourly(region), "Найнебезпечніші години доби"))
    with cc2:
        d = dow(region)
        fig = px.bar(d, x="day", y="alert_rate", title="За днями тижня")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title=None)
        fig.update_traces(marker_color=ASH)
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        show(fig)

    st.divider()
    eyebrow("OSINT · TELEGRAM")
    st.markdown("#### 🛰️ Останні події")
    ev = get_events()
    if ev.empty:
        st.info("Немає подій. Згенеруйте кеш: `python -m src.ai_extractor`.")
    else:
        reg_ev = ev[ev["region"] == region]
        show_ev = reg_ev if not reg_ev.empty else ev
        if reg_ev.empty:
            st.caption("Подій саме для цієї області немає — показано всі останні події.")
        disp = show_ev.copy()
        disp["час (Київ)"] = disp["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            disp[["час (Київ)", "event_type", "weapon", "region", "text"]]
            .rename(columns={"event_type": "подія", "weapon": "засіб", "region": "область", "text": "повідомлення"})
            .tail(12), use_container_width=True, hide_index=True,
        )

    st.divider()
    eyebrow("ПРОГНОЗ · ПО ОБЛАСТЯХ")
    st.markdown("#### 🗺️ Поточний ризик по Україні")
    if not risk_df.empty:
        st.plotly_chart(risk_map(risk_df), use_container_width=True)


# --------------------------------------------------------------------------- #
# Analyst mode
# --------------------------------------------------------------------------- #
def render_analyst(region: str | None):
    scope = region or "Уся Україна"
    st.subheader(f"🛠️ Аналітика — {scope}")
    tabs = st.tabs([
        "🗺️ Карта ризику", "📈 Патерни", "🎯 Якість моделі",
        "🚨 Масовані атаки", "🔗 Поширення", "🛰️ OSINT-шар",
    ])

    # --- risk map ---
    with tabs[0]:
        risk_df = get_risk_now()
        if risk_df.empty:
            st.warning("Немає прогнозу. Запустіть `python -m src.forecast --predict`.")
        else:
            st.plotly_chart(risk_map(risk_df), use_container_width=True)
            d = risk_df.copy()
            d["risk"] = (d["risk"] * 100).round(1)
            st.dataframe(
                d[["region", "risk", "active_now"]]
                .rename(columns={"region": "область", "risk": "ризик, %", "active_now": "активна зараз"}),
                use_container_width=True, hide_index=True, height=300,
            )

    # --- patterns ---
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

    # --- model quality ---
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

    # --- mass attacks ---
    with tabs[3]:
        inten = intensity().copy()
        inten["ts_local"] = inten["ts"].dt.tz_convert(config.DISPLAY_TZ)
        daily = (
            inten.set_index("ts_local")
            .resample("D")
            .agg(regions_active=("regions_active", "max"), anomaly=("is_anomaly", "max"))
            .reset_index()
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
        fig.add_trace(go.Scatter(x=anom["ts_local"], y=anom["regions_active"],
                                 mode="markers", name="день з аномалією",
                                 marker=dict(color=BAD, size=6)))
        fig.update_layout(title="Загальнонаціональна інтенсивність тривог (денний максимум)",
                          height=420, margin=dict(t=40), yaxis_title="Областей одночасно")
        show(fig)

    # --- propagation ---
    with tabs[4]:
        prop = propagation().head(15).copy()
        st.caption(
            "Наскільки нова тривога в області A передвіщає нову тривогу в сусідній B упродовж 6 год. "
            "**lift** — у скільки разів частіше за випадковий збіг (база B). lift > 1 = реальний «коридор» загрози."
        )
        prop_disp = prop.assign(
            **{"follow_rate": (prop["follow_rate"] * 100).round(0),
               "base_rate": (prop["base_rate"] * 100).round(0),
               "lift": prop["lift"].round(2)}
        )
        st.dataframe(
            prop_disp[["from", "to", "n_from", "follow_rate", "base_rate", "lift"]]
            .rename(columns={"from": "з області", "to": "→ в область", "n_from": "n подій",
                             "follow_rate": "слідує, %", "base_rate": "база, %", "lift": "lift ×"}),
            use_container_width=True, hide_index=True, height=420,
        )
        fig = px.bar(prop, x="lift", y=prop["from"] + " → " + prop["to"], orientation="h",
                     title="Топ-коридори поширення (lift)", color="lift",
                     color_continuous_scale=HEAT_SCALE)
        fig.update_layout(height=460, margin=dict(t=40), yaxis_title=None, xaxis_title="lift ×")
        fig.update_yaxes(autorange="reversed")
        show(fig)

    # --- OSINT layer ---
    with tabs[5]:
        render_osint_tab()


def render_osint_tab():
    ev = get_events()
    if ev.empty:
        st.info("Кеш подій порожній. Запустіть `python -m src.ai_extractor`.")
        return
    method = ev["method"].iloc[0] if "method" in ev else "—"
    st.caption(
        f"OSINT-шар: повідомлення з Telegram → структуровані події через "
        f"{'LLM (Claude)' if method == 'llm' else 'rule-based fallback'}. "
        "Дані — демонстраційний зразок; з live-Telegram оновлюється у реальному часі."
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        geo_ev = ev.dropna(subset=["lat", "lon"]).copy()
        geo_ev["k"] = geo_ev.groupby("region").cumcount()
        geo_ev["lat"] += (geo_ev["k"] % 3 - 1) * 0.10
        geo_ev["lon"] += (geo_ev["k"] // 3) * 0.12
        geo_ev["час"] = geo_ev["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%m-%d %H:%M")
        fig = px.scatter_mapbox(
            geo_ev, lat="lat", lon="lon", color="event_type",
            color_discrete_map=EVENT_COLORS, hover_name="region",
            hover_data={"час": True, "weapon": True, "lat": False, "lon": False, "k": False},
            zoom=4.3, center={"lat": 48.4, "lon": 31.2}, height=520,
        )
        fig.update_traces(marker=dict(size=13))
        fig.update_layout(legend_title="Тип події")
        st.plotly_chart(map_layout(fig), use_container_width=True)
    with c2:
        counts = ev["event_type"].value_counts().reset_index()
        counts.columns = ["подія", "к-ть"]
        fig = px.bar(counts, x="к-ть", y="подія", orientation="h", color="подія",
                     color_discrete_map=EVENT_COLORS, title="Події за типом")
        fig.update_layout(height=300, margin=dict(t=40), showlegend=False, yaxis_title=None)
        show(fig)
        wc = ev["weapon"].value_counts().reset_index()
        wc.columns = ["засіб", "к-ть"]
        st.dataframe(wc, use_container_width=True, hide_index=True)

    st.divider()
    eyebrow("СИНТЕЗ · OSINT × ТРИВОГИ")
    st.markdown("##### 🔬 OSINT-події vs офіційні тривоги")
    fusion_view(ev)


def fusion_view(ev: pd.DataFrame):
    """Overlay OSINT events-per-hour with #regions under official alert."""
    series = get_series()
    inten = analysis.national_intensity(series)
    t0, t1 = ev["timestamp"].min(), ev["timestamp"].max()
    win = inten[(inten["ts"] >= t0 - pd.Timedelta(hours=2)) & (inten["ts"] <= t1 + pd.Timedelta(hours=2))]
    if win.empty:
        st.caption("Вікно подій поза межами наявних даних тривог — синтез недоступний для цього зразка.")
        return
    ev_hourly = (
        ev.assign(h=ev["timestamp"].dt.floor("h")).groupby("h").size().rename("events").reset_index()
    )
    win = win.copy()
    win["ts_local"] = win["ts"].dt.tz_convert(config.DISPLAY_TZ)
    ev_hourly["ts_local"] = ev_hourly["h"].dt.tz_convert(config.DISPLAY_TZ)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win["ts_local"], y=win["regions_active"], name="областей під тривогою",
                             mode="lines", line_color=ACCENT, fill="tozeroy",
                             fillcolor="rgba(103,152,255,0.15)"))
    fig.add_trace(go.Bar(x=ev_hourly["ts_local"], y=ev_hourly["events"], name="OSINT-подій/год",
                         marker_color=ASH, yaxis="y2", opacity=0.85))
    fig.update_layout(
        height=360, margin=dict(t=30),
        yaxis=dict(title="Областей під тривогою"),
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
    st.caption(
        f"Аналіз часових рядів повітряних тривог + OSINT-збагачення з Telegram. "
        f"Дані станом на {latest_ts_local()} (Київ). Виключно для захисного раннього попередження."
    )

    with st.sidebar:
        eyebrow("РЕЖИМ")
        mode = st.radio("Хто ви?", ["👤 Споживач", "🛠️ Аналітик"], label_visibility="collapsed")
        st.divider()
        regions = geo.REGIONS
        if mode.startswith("👤"):
            region = st.selectbox("Ваша область", regions, index=regions.index("Kyiv City"))
        else:
            region = st.selectbox("Фокус (необов'язково)", ["Уся Україна"] + regions)
            region = None if region == "Уся Україна" else region
        st.divider()
        if get_live_regions() is not None:
            st.success("🔴 Live: поточний стан з alerts.in.ua")
        else:
            st.info("🕒 Snapshot-режим. Live вмикається змінною `ALERTS_IN_UA_TOKEN` на деплої.")
        st.caption("Джерело тривог: Vadimkin air-raid dataset. OSINT: Telegram (демо-зразок).")

    if mode.startswith("👤"):
        render_consumer(region)
    else:
        render_analyst(region)


main()
