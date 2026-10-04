"""Nifty Global Tracker (Streamlit)
Colab tool के सारे हिस्से एक app में: global असर, मदद/दबाव signal, Layer 1 (आज),
Layer 2 (भविष्य), CPR/VWAP, option chain, news, signal log और hit-rate.
Token: Streamlit Secrets में UPSTOX_ACCESS_TOKEN = "..." या app के Setup बॉक्स में डालें.
"""
import os
import re
import time
from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Nifty Global Tracker", page_icon="📈", layout="centered")

IST = timezone(timedelta(hours=5, minutes=30))
UA = {"User-Agent": "Mozilla/5.0"}
LOG = "signal_log.csv"


def now():
    return datetime.now(IST)


# ===================== CONFIG =====================
NIFTY = ("^NSEI", "NSE_INDEX|Nifty 50")
STOCKS = {"Reliance": ("RELIANCE.NS", "NSE_EQ|INE002A01018"),
          "Infosys": ("INFY.NS", "NSE_EQ|INE009A01021"),
          "HDFC Bank": ("HDFCBANK.NS", "NSE_EQ|INE040A01034")}
TICK = {"Nifty": NIFTY[0], **{k: v[0] for k, v in STOCKS.items()}}
KEY = {"Nifty": NIFTY[1], **{k: v[1] for k, v in STOCKS.items()}}
GLOBAL = {"S&P Fut": ("ES=F", "US"), "Nasdaq Fut": ("NQ=F", "US"),
          "Nikkei": ("^N225", "ASIA"), "Hang Seng": ("^HSI", "ASIA"),
          "FTSE": ("^FTSE", "EU"), "DAX": ("^GDAXI", "EU"),
          "Crude Brent": ("BZ=F", "OIL"), "DXY": ("DX-Y.NYB", "USD"),
          "US 10Y": ("^TNX", "RATE"), "India VIX": ("^INDIAVIX", "VIX")}


def gn(q, d=1):
    return ("https://news.google.com/rss/search?q=" + requests.utils.quote(q)
            + f"+when:{d}d&hl=en-IN&gl=IN&ceid=IN:en")


SRC_GLOBAL = [("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
              ("BBC", "https://feeds.bbci.co.uk/news/business/rss.xml"),
              ("Reuters", gn("site:reuters.com markets stocks")),
              ("Bloomberg", gn("site:bloomberg.com markets"))]
SRC_INDIA = [("ET", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
             ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
             ("Mint", "https://www.livemint.com/rss/markets"),
             ("BS", "https://www.business-standard.com/rss/markets-106.rss"),
             ("Reuters India", gn("site:reuters.com India stocks Nifty"))]
FUTURE = {"GDP": "India GDP growth", "महंगाई": "India CPI inflation", "RBI": "RBI repo rate policy",
          "Govt capex": "India government capex", "Manufacturing": "India manufacturing PMI",
          "FII trend": "FII flows India equities", "Earnings": "India quarterly earnings Nifty companies",
          "US trade": "India US trade tariff", "Crude": "Brent crude oil outlook",
          "Geopolitics": "geopolitical risk global markets"}
STOCK_Q = {"Reliance": "Reliance Industries share", "Infosys": "Infosys share", "HDFC Bank": "HDFC Bank share"}
MUST = {"gift": ["gift nifty", "nifty"],
        "fii": ["fii", "dii", "fpi", "foreign investor", "foreign institutional"],
        "Reliance": ["reliance", "ril", "jio"], "Infosys": ["infosys", "infy"], "HDFC Bank": ["hdfc"]}
POS = set("rise rises gain gains surge rally jump record upgrade beat beats growth strong profit rebound soar".split())
NEG = set("fall falls drop slump crash plunge war tariff downgrade miss weak loss selloff sell-off hike inflation fear concern ban probe".split())
MKT = set("stock stocks share shares market markets nifty sensex oil crude fed rate rates rbi inflation gdp tariff tariffs dollar bond bonds yield yields gold earnings economy economic trade bank banks fii fpi dii ipo sebi rupee growth currency futures index treasury".split())


# ===================== HELPERS =====================
def market_open():
    t = now()
    o = t.replace(hour=9, minute=15, second=0, microsecond=0)
    c = t.replace(hour=15, minute=30, second=0, microsecond=0)
    return t.weekday() < 5 and o <= t <= c


def get_token():
    if st.session_state.get("tok"):
        return st.session_state["tok"].strip()
    try:
        return str(st.secrets.get("UPSTOX_ACCESS_TOKEN", "")).strip()
    except Exception:
        return ""


def flat(h):
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    return h


def tag(t):
    w = set(re.findall(r"[a-z\-]+", t.lower()))
    s = len(w & POS) - len(w & NEG)
    return "🟢 मदद" if s > 0 else "🔴 दबाव" if s < 0 else "⚪"


def tw(g):
    h = now().hour + now().minute / 60
    if h < 9.25:
        m = {"US": 1.5, "ASIA": 1.0, "EU": 0.6}
    elif h < 13.5:
        m = {"US": 1.0, "ASIA": 1.5, "EU": 0.8}
    else:
        m = {"US": 1.2, "ASIA": 0.5, "EU": 1.5}
    return m.get(g, 1.0)


def strength(sc):
    a = abs(sc)
    return "कमज़ोर" if a < 0.2 else "मध्यम" if a < 0.5 else "मज़बूत"


# ===================== DATA =====================
@st.cache_data(ttl=300, show_spinner="बाज़ार का data आ रहा है…")
def load_hist():
    tk = sorted({v[0] for v in GLOBAL.values()} | set(TICK.values()))
    d = yf.download(tk, period="9mo", interval="1d", auto_adjust=True, progress=False)["Close"]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d


def prep(d):
    names = [n for n, v in GLOBAL.items() if v[0] in d.columns and d[v[0]].notna().sum() > 70]
    r = d.ffill().pct_change().dropna(how="all")
    gr = pd.DataFrame({n: (r[GLOBAL[n][0]].shift(1) if GLOBAL[n][1] == "US" else r[GLOBAL[n][0]])
                       for n in names})
    return r, gr, names


def analyse(tk, r, gr, names):
    corr = gr.iloc[-90:].corrwith(r[tk].iloc[-90:]).fillna(0)
    today = pd.Series({n: r[GLOBAL[n][0]].iloc[-1] for n in names})
    sd = pd.Series({n: r[GLOBAL[n][0]].iloc[-60:].std() for n in names})
    z = (today / sd).replace([np.inf, -np.inf], 0).clip(-3, 3).fillna(0)
    w = corr * pd.Series({n: tw(GLOBAL[n][1]) for n in names})
    contrib = w * z
    sc = float(contrib.sum() / max(w.abs().sum(), 1e-9))
    zz = (gr / gr.rolling(60).std()).clip(-3, 3)
    bt = pd.concat([(zz * corr).sum(axis=1), r[tk]], axis=1).dropna().iloc[-60:]
    hit = int(round((np.sign(bt.iloc[:, 0]) == np.sign(bt.iloc[:, 1])).mean() * 100)) if len(bt) > 10 else None
    return sc, contrib, hit, today * 100


@st.cache_data(ttl=15, show_spinner=False)
def upstox_quotes(token):
    keys = {v: k for k, v in KEY.items()}  # instrument_key -> नाम
    resp = requests.get("https://api.upstox.com/v2/market-quote/quotes",
                        params={"instrument_key": ",".join(keys)},
                        headers={"Authorization": "Bearer " + token, "Accept": "application/json"}, timeout=10)
    resp.raise_for_status()
    out = {}
    for v in resp.json().get("data", {}).values():
        nm = keys.get(v.get("instrument_token"))
        if nm:
            p = float(v["last_price"])
            nc = float(v.get("net_change") or 0)
            out[nm] = {"price": p, "pct": nc / (p - nc) * 100 if (p - nc) else 0.0,
                       "vwap": v.get("average_price") or None}
    return out


def get_quotes(d):
    out = {}
    for nm, tk in TICK.items():
        s = d[tk].dropna()
        out[nm] = {"price": float(s.iloc[-1]), "pct": float(s.pct_change().iloc[-1] * 100), "vwap": None}
    src = "yfinance (थोड़ा देर से)"
    tok = get_token()
    if tok:
        try:
            q = upstox_quotes(tok)
            if q:
                for nm, v in q.items():
                    if abs(v["pct"]) < 1e-9 and market_open():
                        v = {**v, "pct": out[nm]["pct"]}
                    out[nm].update({k: x for k, x in v.items() if x is not None})
                src = "Upstox LIVE"
            else:
                src = "Upstox से data खाली → yfinance"
        except Exception as e:
            src = f"Upstox नहीं चला ({str(e)[:40]}) → yfinance"
    return out, src


@st.cache_data(ttl=3600, show_spinner=False)
def cpr(tk, day):
    h = flat(yf.download(tk, period="10d", interval="1d", progress=False, auto_adjust=True)).dropna()
    if h.index[-1].date() == now().date() and market_open():
        h = h.iloc[:-1]
    p = h.iloc[-1]
    P = (p.High + p.Low + p.Close) / 3
    BC = (p.High + p.Low) / 2
    TC = 2 * P - BC
    return round(float(min(BC, TC)), 1), round(float(max(BC, TC)), 1)


@st.cache_data(ttl=120, show_spinner=False)
def intraday(tk):
    h = flat(yf.download(tk, period="5d", interval="5m", progress=False, auto_adjust=True)).dropna()
    return h[h.index.date == h.index[-1].date()]


def yf_vwap(tk):
    try:
        h = intraday(tk)
        v = float(((h.High + h.Low + h.Close) / 3 * h.Volume).sum() / h.Volume.sum())
        return v if v == v and v > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def option_view(token):
    H = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    c = requests.get("https://api.upstox.com/v2/option/contract",
                     params={"instrument_key": NIFTY[1]}, headers=H, timeout=10).json()["data"]
    today = f"{now():%Y-%m-%d}"
    exp = sorted({str(x["expiry"])[:10] for x in c if str(x["expiry"])[:10] >= today})[0]
    ch = requests.get("https://api.upstox.com/v2/option/chain",
                      params={"instrument_key": NIFTY[1], "expiry_date": exp}, headers=H, timeout=10).json()["data"]
    ce = sum(x["call_options"]["market_data"]["oi"] for x in ch)
    pe = sum(x["put_options"]["market_data"]["oi"] for x in ch)
    res = max(ch, key=lambda x: x["call_options"]["market_data"]["oi"])["strike_price"]
    sup = max(ch, key=lambda x: x["put_options"]["market_data"]["oi"])["strike_price"]
    return {"exp": exp, "pcr": pe / ce, "sup": sup, "res": res}


# ===================== NEWS =====================
def fetch(url, n=3, hrs=30, must=None):
    try:
        f = feedparser.parse(requests.get(url, headers=UA, timeout=8).content)
        out = []
        for e in f.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            if not ts:
                continue
            t = e.title
            tl = t.lower()
            if "profile and biography" in tl:
                continue
            if must and not any(m in tl for m in must):
                continue
            dt = datetime.fromtimestamp(timegm(ts), timezone.utc).astimezone(IST)
            if now() - dt <= timedelta(hours=hrs):
                out.append((dt, t, e.get("link", "")))
        return sorted(out, reverse=True)[:n]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner="खबरें आ रही हैं…")
def news_bundle():
    jobs = {}
    for s, u in SRC_GLOBAL:
        jobs[f"g:{s}"] = (u, 8, 30, None)
    for s, u in SRC_INDIA:
        jobs[f"i:{s}"] = (u, 8, 30, None)
    jobs["gift"] = (gn("GIFT Nifty", 3), 3, 72, MUST["gift"])
    jobs["fii"] = (gn("FII DII data today", 3), 3, 72, MUST["fii"])
    for nm, q in STOCK_Q.items():
        jobs[f"s:{nm}"] = (gn(q, 3), 3, 72, MUST[nm])
    for t, q in FUTURE.items():
        jobs[f"f:{t}"] = (gn(q, 3), 1, 72, None)
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda j: fetch(*j), jobs.values())))

    def merge(prefix):
        seen, items = set(), []
        for k, v in res.items():
            if k.startswith(prefix):
                for dt, t, link in v:
                    if t[:40] not in seen and set(re.findall(r"[a-z]+", t.lower())) & MKT:
                        seen.add(t[:40])
                        items.append((dt, t, link, k.split(":", 1)[1]))
        return sorted(items, reverse=True)[:10]

    return {"global": merge("g:"), "india": merge("i:"), "gift": res["gift"], "fii": res["fii"],
            "stocks": {nm: res[f"s:{nm}"] for nm in STOCK_Q},
            "future": {t: res[f"f:{t}"] for t in FUTURE}}


def show_news(items):
    if not items:
        st.caption("अभी कोई ताज़ा खबर नहीं मिली")
        return
    for it in items:
        dt, t, link = it[0], it[1], it[2]
        src = f" · {it[3]}" if len(it) > 3 else ""
        t = t.replace("[", "(").replace("]", ")")[:110]
        st.markdown(f"{tag(t)} `{dt:%d-%b %H:%M}` [{t}]({link}){src}")


# ===================== SIGNAL LOG / CONFIRM =====================
def confirm(nm, sig, pc):
    if not market_open():
        return "बाज़ार बंद — पुष्टि बाज़ार खुलने पर"
    h = st.session_state.setdefault("sig_hist", {})
    ts, s, cnt = h.get(nm, (0, None, 0))
    if s != sig:
        cnt = 1
        h[nm] = (time.time(), sig, 1)
    elif time.time() - ts >= 25:
        cnt += 1
        h[nm] = (time.time(), sig, cnt)
    agree = abs(pc) >= 0.05 and ((pc > 0) == (sig == "मदद"))
    return "✔✔ पक्का" if agree and cnt >= 2 else "✔ पुष्टि" if agree else "✘ पुष्टि नहीं"


def log_signals(rows):
    if not market_open() or time.time() - st.session_state.get("last_log", 0) < 300:
        return
    st.session_state["last_log"] = time.time()
    try:
        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    except Exception:
        pass


# ===================== PAGE =====================
st.title("📈 Nifty Global Tracker")

with st.expander("⚙️ Setup (token / refresh)"):
    st.text_input("Upstox token (आज का)", type="password", key="tok",
                  help="खाली छोड़ने पर Streamlit Secrets वाला token लगेगा। Token रोज़ बदलता है।")
    auto = st.checkbox("बाज़ार खुला हो तब अपने-आप refresh", value=True)
    every = st.selectbox("कितने सेकंड में", [30, 60, 120], index=1)
    if st.button("🔄 अभी refresh करें"):
        st.cache_data.clear()
        st.rerun()
if auto and market_open():
    st_autorefresh(interval=every * 1000, key="ar")

try:
    hist = load_hist()
    R, GR, NAMES = prep(hist)
    Q, SRC = get_quotes(hist)
    RES = {nm: analyse(tk, R, GR, NAMES) for nm, tk in TICK.items()}
except Exception as e:
    st.error(f"Data नहीं आ पाया: {str(e)[:150]}")
    st.stop()

st.caption(f"🕒 {now():%d-%b-%Y %I:%M:%S %p} IST · "
           f"{'🟢 बाज़ार खुला' if market_open() else '🔴 बाज़ार बंद — पिछले trading दिन का data'} · Data: {SRC}")

rows = []
for nm in TICK:
    sc = RES[nm][0]
    rows.append({"time": f"{now():%Y-%m-%d %H:%M}", "symbol": nm, "signal": "मदद" if sc > 0 else "दबाव",
                 "score": round(sc, 3), "price": round(Q[nm]["price"], 2)})
log_signals(rows)

tabs = st.tabs(["📊 आज", "🌍 Global", "🔵 Layer 1", "🟣 Layer 2", "📰 खबरें", "📈 Log"])

# ---------- आज ----------
with tabs[0]:
    for nm in TICK:
        sc, _, hit, _t = RES[nm]
        q = Q[nm]
        sig = "मदद" if sc > 0 else "दबाव"
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            c1.metric(nm, f"{q['price']:,.1f}", f"{q['pct']:+.2f}%")
            c2.markdown(f"### {'🟢' if sc > 0 else '🔴'} {sig}")
            c2.caption(f"ताक़त: {strength(sc)} (score {sc:+.2f})")
            hit_txt = f"{hit}%" if hit is not None else "—"
            st.caption(f"{confirm(nm, sig, q['pct'])} · पिछले 60 दिन hit-rate {hit_txt} (अंदाज़ा)")
    st.caption("⚠️ सिर्फ़ जानकारी के लिए है, trade की सलाह नहीं।")

# ---------- Global ----------
with tabs[1]:
    nift = RES["Nifty"]
    gdf = pd.DataFrame({"बाज़ार": NAMES, "बदलाव %": [round(float(nift[3][n]), 2) for n in NAMES],
                        "Nifty पर असर": ["मदद" if nift[1][n] > 0 else "दबाव" if nift[1][n] < 0 else "—"
                                          for n in NAMES]})
    st.dataframe(gdf, hide_index=True, use_container_width=True)
    cs = nift[1].sort_values()
    fig = px.bar(x=cs.values, y=cs.index, orientation="h", color=np.where(cs.values > 0, "मदद", "दबाव"),
                 color_discrete_map={"मदद": "#2e9e5b", "दबाव": "#d64545"})
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      xaxis_title=None, yaxis_title=None)
    st.caption("Nifty के score में हर बाज़ार का हिस्सा")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Layer 1 ----------
with tabs[2]:
    top = RES["Nifty"][1].reindex(RES["Nifty"][1].abs().sort_values(ascending=False).index)[:3]
    st.markdown("**Drivers:** " + ", ".join(f"{n} ({'मदद' if v > 0 else 'दबाव'})" for n, v in top.items()))

    st.markdown("**Option chain (Nifty)**")
    tok = get_token()
    if not tok:
        st.caption("Token डालें तो option chain दिखेगा")
    else:
        try:
            o = option_view(tok)
            a, b, c = st.columns(3)
            a.metric("PCR", f"{o['pcr']:.2f}", "मदद" if o["pcr"] > 1 else "दबाव" if o["pcr"] < 0.8 else "तटस्थ",
                     delta_color="off")
            b.metric("Support", f"{o['sup']:.0f}")
            c.metric("Resistance", f"{o['res']:.0f}")
            st.caption(f"Expiry {o['exp']}")
        except Exception as e:
            st.caption(f"Option chain नहीं मिला ({str(e)[:50]})")

    st.markdown("**CPR और VWAP**")
    crow = []
    for nm, tk in TICK.items():
        try:
            lo, hi = cpr(tk, str(now().date()))
        except Exception:
            continue
        p = Q[nm]["price"]
        pos = "मदद" if p > hi else "दबाव" if p < lo else "तटस्थ"
        v = None if nm == "Nifty" else (Q[nm].get("vwap") or yf_vwap(tk))
        crow.append({"नाम": nm, "भाव": round(p, 1), "CPR": f"{lo}–{hi}", "CPR असर": pos,
                     "VWAP": round(v, 1) if v else None,
                     "VWAP असर": ("मदद" if p > v else "दबाव") if v else "—"})
    if crow:
        st.dataframe(pd.DataFrame(crow), hide_index=True, use_container_width=True)

    sel = st.selectbox("Intraday chart", list(TICK))
    try:
        ih = intraday(TICK[sel])
        if len(ih):
            f2 = px.line(ih, y="Close")
            f2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(f2, use_container_width=True)
    except Exception:
        st.caption("Chart का data नहीं मिला")

    NB = news_bundle()
    st.markdown("**GIFT Nifty (ताज़ा खबर)**")
    show_news(NB["gift"])
    st.markdown("**FII / DII (ताज़ा खबर)**")
    show_news(NB["fii"])

# ---------- Layer 2 ----------
with tabs[3]:
    NB = news_bundle()
    for t in FUTURE:
        it = NB["future"][t]
        if it:
            dt, title, link = it[0]
            title = title.replace("[", "(").replace("]", ")")[:100]
            st.markdown(f"**{t}:** {tag(title)} [{title}]({link})")
        else:
            st.markdown(f"**{t}:** —")

# ---------- खबरें ----------
with tabs[4]:
    NB = news_bundle()
    st.markdown("#### 🌐 Global (CNBC, BBC, Reuters, Bloomberg)")
    show_news(NB["global"])
    st.markdown("#### 🇮🇳 Nifty / भारत (ET, Moneycontrol, Mint, BS, Reuters)")
    show_news(NB["india"])
    for nm in STOCK_Q:
        st.markdown(f"#### 📰 {nm}")
        show_news(NB["stocks"][nm])

# ---------- Log ----------
with tabs[5]:
    if not os.path.exists(LOG):
        st.caption("Log बाज़ार खुलने पर बनना शुरू होगा (हर 5 मिनट में एक बार)।")
    else:
        df = pd.read_csv(LOG)
        df["date"] = df["time"].str[:10]
        last_p = df.groupby(["date", "symbol"])["price"].transform("last")
        ok = np.where(df["signal"] == "मदद", last_p > df["price"], last_p < df["price"])
        m = last_p != df["price"]
        if m.sum():
            st.markdown("**दिन के आख़िरी भाव के मुक़ाबले signal कितना सही रहा**")
            s = pd.DataFrame({"symbol": df["symbol"][m], "सही": ok[m]}).groupby("symbol")["सही"].agg(["mean", "count"])
            s["hit-rate %"] = (s["mean"] * 100).round(0)
            st.dataframe(s[["hit-rate %", "count"]].rename(columns={"count": "signals"}), use_container_width=True)
        st.dataframe(df.drop(columns="date").tail(60).iloc[::-1], hide_index=True, use_container_width=True)
        st.download_button("⬇️ Log download (CSV)", df.drop(columns="date").to_csv(index=False),
                           "signal_log.csv", "text/csv")
        st.caption("Streamlit Cloud restart पर यह फ़ाइल मिट सकती है, इसलिए CSV बीच-बीच में download कर लें।")
