"""Tryvoha Radar — Streamlit dashboard.

Two modes:
  * Споживач  — "what is the risk in my oblast now / today", simple.
  * Аналітик  — patterns, model quality, mass-attack anomalies, cross-border
                propagation, and the OSINT (Telegram) event layer.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis, config, geo, live
from src.ai_extractor import load_events
from src.transform import load_series

st.set_page_config(page_title="Tryvoha Radar", page_icon="🛰️", layout="wide")

DOW_NAMES = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
EVENT_COLORS = {
    "пуск": "#e45756", "рух": "#f58518", "загроза": "#eeca3b",
    "влучання": "#b3122b", "збиття": "#54a24b", "відбій": "#4c78a8", "інше": "#9d9d9d",
}


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
        return "Низький", "#54a24b"
    if r < 0.67:
        return "Підвищений", "#f58518"
    return "Високий", "#e45756"


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
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
    return fig


def hourly_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(df, x="hour", y="alert_rate", title=title)
    fig.update_yaxes(tickformat=".0%", title="Частка годин під тривогою")
    fig.update_xaxes(title="Година (Київ)", dtick=2)
    fig.update_traces(marker_color="#e45756")
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
    band, color = risk_band(risk) if risk == risk else ("—", "#9d9d9d")

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
            number={"suffix": "%"},
            title={"text": f"Ризик тривоги в наступні {get_metrics().get('horizon_h', 6)} год"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 34], "color": "#e8f5e9"},
                    {"range": [34, 67], "color": "#fff3e0"},
                    {"range": [67, 100], "color": "#ffebee"},
                ],
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=10))
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
        st.plotly_chart(hourly_fig(hourly(region), "Найнебезпечніші години доби"), use_container_width=True)
    with cc2:
        d = dow(region)
        fig = px.bar(d, x="day", y="alert_rate", title="За днями тижня")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title=None)
        fig.update_traces(marker_color="#4c78a8")
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 🛰️ Останні OSINT-події (з Telegram)")
    ev = get_events()
    if ev.empty:
        st.info("Немає подій. Згенеруйте кеш: `python -m src.ai_extractor`.")
    else:
        reg_ev = ev[ev["region"] == region]
        show = reg_ev if not reg_ev.empty else ev
        if reg_ev.empty:
            st.caption("Подій саме для цієї області немає — показано всі останні події.")
        disp = show.copy()
        disp["час (Київ)"] = disp["timestamp"].dt.tz_convert(config.DISPLAY_TZ).dt.strftime("%m-%d %H:%M")
        st.dataframe(
            disp[["час (Київ)", "event_type", "weapon", "region", "text"]]
            .rename(columns={"event_type": "подія", "weapon": "засіб", "region": "область", "text": "повідомлення"})
            .tail(12), use_container_width=True, hide_index=True,
        )

    st.divider()
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
            pivot, color_continuous_scale="Reds", aspect="auto",
            labels=dict(x="Година (Київ)", y="День тижня", color="Частка"),
            title="Інтенсивність тривог: день тижня × година",
        )
        fig.update_yaxes(tickvals=list(range(7)), ticktext=DOW_NAMES)
        fig.update_layout(height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(hourly_fig(hourly(region), "За годиною доби"), use_container_width=True)
        with c2:
            m = monthly(region)
            fig = px.area(m, x="month", y="alerts", title="Кількість тривог за місяцями")
            fig.update_traces(line_color="#e45756")
            fig.update_layout(height=320, margin=dict(t=40), xaxis_title=None, yaxis_title="К-ть тривог")
            st.plotly_chart(fig, use_container_width=True)

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
                         title="Модель проти базлайнів (більше = краще)")
            fig.update_layout(height=340, margin=dict(t=40), yaxis_range=[0.5, 1.0])
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                rel = m["reliability"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color="gray"), name="ідеал"))
                fig.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["fraction_positive"],
                                         mode="lines+markers", name="модель", line_color="#e45756"))
                fig.update_layout(title="Калібрування (reliability)", height=330,
                                  xaxis_title="Прогнозована ймовірність", yaxis_title="Фактична частка")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                roc = m["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash", color="gray"), name="випадково"))
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                         name=f"ROC (AUC={m['model']['roc_auc']:.3f})", line_color="#4c78a8"))
                fig.update_layout(title="ROC-крива", height=330,
                                  xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)

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
                                 mode="lines", name="макс. областей/день", line_color="#4c78a8"))
        anom = daily[daily["anomaly"] == 1]
        fig.add_trace(go.Scatter(x=anom["ts_local"], y=anom["regions_active"],
                                 mode="markers", name="день з аномалією",
                                 marker=dict(color="#e45756", size=6)))
        fig.update_layout(title="Загальнонаціональна інтенсивність тривог (денний максимум)",
                          height=420, margin=dict(t=40), yaxis_title="Областей одночасно")
        st.plotly_chart(fig, use_container_width=True)

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
                     color_continuous_scale="Reds")
        fig.update_layout(height=460, margin=dict(t=40), yaxis_title=None, xaxis_title="lift ×")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

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
        # deterministic jitter so co-located events don't overlap
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
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0),
                          legend_title="Тип події")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        counts = ev["event_type"].value_counts().reset_index()
        counts.columns = ["подія", "к-ть"]
        fig = px.bar(counts, x="к-ть", y="подія", orientation="h", color="подія",
                     color_discrete_map=EVENT_COLORS, title="Події за типом")
        fig.update_layout(height=300, margin=dict(t=40), showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        wc = ev["weapon"].value_counts().reset_index()
        wc.columns = ["засіб", "к-ть"]
        st.dataframe(wc, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### 🔬 Синтез: OSINT-події vs офіційні тривоги")
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
                             mode="lines", line_color="#4c78a8", fill="tozeroy"))
    fig.add_trace(go.Bar(x=ev_hourly["ts_local"], y=ev_hourly["events"], name="OSINT-подій/год",
                         marker_color="#e45756", yaxis="y2", opacity=0.7))
    fig.update_layout(
        height=360, margin=dict(t=30),
        yaxis=dict(title="Областей під тривогою"),
        yaxis2=dict(title="OSINT-подій", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Сплески OSINT-повідомлень синхронні зі зростанням кількості областей під тривогою — "
        "OSINT-моніторинг дає ранній контекст до/під час офіційних тривог."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("🛰️ Tryvoha Radar")
    st.caption(
        "Аналіз часових рядів повітряних тривог України + OSINT-збагачення з Telegram. "
        f"Дані станом на {latest_ts_local()} (Київ). Виключно для захисного раннього попередження."
    )

    with st.sidebar:
        st.header("Режим")
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
