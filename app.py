"""Nifty Global Tracker v2 (Streamlit)
- 10 Nifty heavyweights + Nifty को global बाज़ार/शेयरों से मिलाना (correlation + जोड़ी की accuracy)
- दो अलग मॉडल: Gap (सुबह के खुलने का रुख) और दिन (दिन भर का रुख)
- ईमानदार walk-forward scoreboard: baseline, 95% range, ताक़त के हिसाब से hit-rate
Token: Streamlit Secrets में UPSTOX_ACCESS_TOKEN = "..." या app के Setup बॉक्स में।
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
WK = 120  # walk-forward जाँच के दिन


def now():
    return datetime.now(IST)


# ===================== CONFIG =====================
NIFTY = ("^NSEI", "NSE_INDEX|Nifty 50")
# नाम: (yfinance, Upstox key, मिलाने वाले global शेयर)
STOCKS = {
    "Reliance": ("RELIANCE.NS", "NSE_EQ|INE002A01018", ["XOM", "CVX", "INDA"]),
    "HDFC Bank": ("HDFCBANK.NS", "NSE_EQ|INE040A01034", ["HDB", "JPM", "BAC", "INDA"]),
    "ICICI Bank": ("ICICIBANK.NS", "NSE_EQ|INE090A01021", ["IBN", "JPM", "BAC", "INDA"]),
    "Infosys": ("INFY.NS", "NSE_EQ|INE009A01021", ["INFY", "ACN", "CTSH", "IBM"]),
    "TCS": ("TCS.NS", "NSE_EQ|INE467B01029", ["ACN", "CTSH", "IBM", "INFY"]),
    "Bharti Airtel": ("BHARTIARTL.NS", "NSE_EQ|INE397D01024", ["VZ", "TMUS", "EEM"]),
    "ITC": ("ITC.NS", "NSE_EQ|INE154A01025", ["PM", "MO", "EEM"]),
    "L&T": ("LT.NS", "NSE_EQ|INE018A01030", ["CAT", "HG=F", "EEM"]),
    "SBI": ("SBIN.NS", "NSE_EQ|INE062A01020", ["JPM", "C", "EEM"]),
    "Axis Bank": ("AXISBANK.NS", "NSE_EQ|INE238A01034", ["IBN", "JPM", "BAC", "INDA"]),
}
PEERS_OF = {"Nifty": ["INDA", "HDB", "IBN", "INFY", "EEM"], **{k: v[2] for k, v in STOCKS.items()}}
TICK = {"Nifty": NIFTY[0], **{k: v[0] for k, v in STOCKS.items()}}
KEY = {"Nifty": NIFTY[1], **{k: v[1] for k, v in STOCKS.items()}}
GLOBAL = {"S&P Fut": ("ES=F", "US"), "Nasdaq Fut": ("NQ=F", "US"),
          "Nikkei": ("^N225", "ASIA"), "Hang Seng": ("^HSI", "ASIA"),
          "FTSE": ("^FTSE", "EU"), "DAX": ("^GDAXI", "EU"),
          "Crude Brent": ("BZ=F", "OIL"), "DXY": ("DX-Y.NYB", "USD"),
          "US 10Y": ("^TNX", "RATE"), "India VIX": ("^INDIAVIX", "VIX")}
PEER = {"XOM": "Exxon", "CVX": "Chevron", "JPM": "JPMorgan", "BAC": "BofA", "C": "Citi",
        "HDB": "HDFC Bank ADR", "IBN": "ICICI Bank ADR", "INFY": "Infosys ADR", "ACN": "Accenture",
        "CTSH": "Cognizant", "IBM": "IBM", "PM": "Philip Morris", "MO": "Altria", "CAT": "Caterpillar",
        "HG=F": "Copper", "EEM": "EM ETF", "INDA": "India ETF (US)", "VZ": "Verizon", "TMUS": "T-Mobile"}
FACT = dict(GLOBAL)  # नाम -> (ticker, group)
for _t, _l in PEER.items():
    FACT[_l] = (_t, "US")
SYM_F = {s: list(GLOBAL) + [PEER[t] for t in PEERS_OF[s]] for s in TICK}


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
STOCK_MUST = {"Reliance": ["reliance", "ril", "jio"], "HDFC Bank": ["hdfc"], "ICICI Bank": ["icici"],
              "Infosys": ["infosys", "infy"], "TCS": ["tcs", "tata consultancy"],
              "Bharti Airtel": ["airtel", "bharti"], "ITC": ["itc"], "L&T": ["l&t", "larsen"],
              "SBI": ["sbi", "state bank"], "Axis Bank": ["axis"]}
MUST = {"gift": ["gift nifty", "nifty"],
        "fii": ["fii", "dii", "fpi", "foreign investor", "foreign institutional"]}
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


def strength(sc):
    a = abs(sc)
    return "कमज़ोर" if a < 0.2 else "मध्यम" if a < 0.5 else "मज़बूत"


def sig(sc):
    return "मदद" if sc > 0 else "दबाव"


def ico(sc):
    return ("🟢 " if sc > 0 else "🔴 ") + sig(sc)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


# ===================== MODEL =====================
@st.cache_data(ttl=300, show_spinner=False)
def load():
    tk = sorted({v[0] for v in FACT.values()} | set(TICK.values()))
    raw = yf.download(tk, period="1y", interval="1d", auto_adjust=True, progress=False)
    c, o = raw["Close"], raw["Open"]
    for x in (c, o):
        x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    return c, o


def factors(r, labels, mode):
    cols = {}
    for n in labels:
        tk, g = FACT[n]
        if tk not in r.columns:
            continue
        s = r[tk]
        cols[n] = s.shift(1) if (mode == "gap" or g == "US") else s
    return pd.DataFrame(cols)


def target(c, o, tk, mode):
    cc = c[tk].dropna()
    if mode == "day":
        return cc.pct_change().dropna()
    oo = o[tk].dropna()
    return (oo / cc.shift(1) - 1).dropna()


def run_model(tk, labels, mode, r, c, o):
    """mode='gap': सुबह का gap (पिछले close तक की जानकारी से)
       mode='day': दिन का close-to-close (US पिछली रात का, एशिया/यूरोप आज का)"""
    try:
        y = target(c, o, tk, mode)
        F = factors(r, labels, mode).reindex(y.index)
        F = F.loc[:, F.notna().sum() > 100].fillna(0)
        if len(y) < 170 or F.shape[1] == 0:
            return None
        corr = F.rolling(90).corr(y)
        cp = corr.shift(1)                       # सिर्फ़ पिछले दिनों का correlation (look-ahead नहीं)
        sdf = F.rolling(60).std()
        z = (F / sdf.shift(1)).replace([np.inf, -np.inf], 0).clip(-3, 3)
        den = cp.abs().sum(axis=1).replace(0, np.nan)
        s = (cp * z).sum(axis=1) / den
        df = pd.concat([s.rename("s"), y.rename("y")], axis=1).dropna().iloc[-WK:]
        if len(df) < 40:
            return None
        ok = np.sign(df["s"]) == np.sign(df["y"])
        n, k = len(df), int(ok.sum())
        up = (df["y"] > 0).mean()
        lo, hi = wilson(k, n)
        med = df["s"].abs().median()
        big = df["s"].abs() >= med
        stt = {"n": n, "hit": k / n, "base": max(up, 1 - up), "lo": lo, "hi": hi,
               "strong": float(ok[big].mean()), "weak": float(ok[~big].mean())}
        # आज का live score (नवीनतम correlation और नवीनतम global बदलाव)
        labs = list(F.columns)
        cl = corr.iloc[-1].fillna(0)
        fl = pd.Series({n_: r[FACT[n_][0]].iloc[-1] for n_ in labs})
        zl = (fl / sdf.iloc[-1]).replace([np.inf, -np.inf], 0).clip(-3, 3).fillna(0)
        contrib = cl * zl
        live = float(contrib.sum() / max(cl.abs().sum(), 1e-9))
        # हर जोड़ी की अलग walk-forward accuracy
        pf = np.sign(cp * F).fillna(0).iloc[-WK:]
        yy = np.sign(y.reindex(pf.index))
        valid = pf.ne(0)
        pair = (pf.eq(yy, axis=0) & valid).sum() / valid.sum().replace(0, np.nan)
        return {"live": live, "contrib": contrib, "corr": cl, "pair": pair, "st": stt}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="बाज़ार का data और model तैयार हो रहे हैं…")
def build_all():
    c, o = load()
    r = c.ffill().pct_change().iloc[1:]
    M = {s: {md: run_model(TICK[s], SYM_F[s], md, r, c, o) for md in ("gap", "day")} for s in TICK}
    last = {}
    for s, tk in TICK.items():
        x = c[tk].dropna()
        last[s] = (float(x.iloc[-1]), float(x.pct_change().iloc[-1] * 100))
    gnow = {n: float(r[v[0]].iloc[-1] * 100) for n, v in FACT.items() if v[0] in r.columns}
    return M, last, gnow


@st.cache_data(ttl=15, show_spinner=False)
def upstox_quotes(token):
    keys = {v: k for k, v in KEY.items()}
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


def get_quotes(last):
    out = {s: {"price": p, "pct": pc, "vwap": None} for s, (p, pc) in last.items()}
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
def cpr_table(day):
    h = yf.download(list(TICK.values()), period="10d", interval="1d", auto_adjust=True, progress=False)
    rows = {}
    for sym, tk in TICK.items():
        try:
            d = pd.concat([h["High"][tk], h["Low"][tk], h["Close"][tk]], axis=1, keys=["H", "L", "C"]).dropna()
            d.index = pd.to_datetime(d.index).tz_localize(None)
            if d.index[-1].date() == now().date() and market_open():
                d = d.iloc[:-1]
            p = d.iloc[-1]
            P = (p.H + p.L + p.C) / 3
            BC = (p.H + p.L) / 2
            TC = 2 * P - BC
            rows[sym] = (round(float(min(BC, TC)), 1), round(float(max(BC, TC)), 1))
        except Exception:
            continue
    return rows


@st.cache_data(ttl=120, show_spinner=False)
def intraday(tk):
    h = flat(yf.download(tk, period="5d", interval="5m", progress=False, auto_adjust=True)).dropna()
    h = h[h.index.date == h.index[-1].date()].copy()
    if "Volume" in h and h["Volume"].sum() > 0:
        tp = (h["High"] + h["Low"] + h["Close"]) / 3
        h["VWAP"] = (tp * h["Volume"]).cumsum() / h["Volume"].cumsum()
    return h


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
            "future": {t: res[f"f:{t}"] for t in FUTURE}}


@st.cache_data(ttl=300, show_spinner=False)
def stock_news(nm):
    return fetch(gn(f"{nm} share", 3), 5, 72, STOCK_MUST[nm])


def show_news(items):
    if not items:
        st.caption("अभी कोई ताज़ा खबर नहीं मिली")
        return
    for it in items:
        dt, t, link = it[0], it[1], it[2]
        src = f" · {it[3]}" if len(it) > 3 else ""
        t = t.replace("[", "(").replace("]", ")")[:110]
        st.markdown(f"{tag(t)} `{dt:%d-%b %H:%M}` [{t}]({link}){src}")


# ===================== CONFIRM / LOG =====================
def confirm(nm, s, pc):
    if not market_open():
        return "बाज़ार बंद"
    h = st.session_state.setdefault("sig_hist", {})
    ts, prev, cnt = h.get(nm, (0, None, 0))
    if prev != s:
        cnt = 1
        h[nm] = (time.time(), s, 1)
    elif time.time() - ts >= 25:
        cnt += 1
        h[nm] = (time.time(), s, cnt)
    agree = abs(pc) >= 0.05 and ((pc > 0) == (s == "मदद"))
    return "✔✔ पक्का" if agree and cnt >= 2 else "✔ पुष्टि" if agree else "✘ पुष्टि नहीं"


def log_signals(rows):
    if not market_open() or time.time() - st.session_state.get("last_log", 0) < 300:
        return
    st.session_state["last_log"] = time.time()
    try:
        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    except Exception:
        pass


def verdict(s):
    if not s:
        return "—"
    return "✅ असर दिखा" if (s["lo"] > 0.5 and s["hit"] > s["base"]) else "⚠️ तुक्के जैसा"


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
    M, LAST, GNOW = build_all()
    Q, SRC = get_quotes(LAST)
    if not M["Nifty"]["day"]:
        raise ValueError("Nifty का model नहीं बन पाया (data कम है)")
except Exception as e:
    st.error(f"Data नहीं आ पाया: {str(e)[:150]}")
    st.stop()

st.caption(f"🕒 {now():%d-%b-%Y %I:%M:%S %p} IST · "
           f"{'🟢 बाज़ार खुला' if market_open() else '🔴 बाज़ार बंद — पिछले trading दिन का data'} · Data: {SRC}")

rows = []
for s in TICK:
    m = M[s]["day"]
    if m:
        rows.append({"time": f"{now():%Y-%m-%d %H:%M}", "symbol": s, "signal": sig(m["live"]),
                     "score": round(m["live"], 3), "price": round(Q[s]["price"], 2)})
log_signals(rows)

tabs = st.tabs(["📊 आज", "🎯 Accuracy", "🔗 Global मेल", "🔵 Layer 1", "🟣 Layer 2", "📰 खबरें", "📈 Log"])

# ---------- आज ----------
with tabs[0]:
    nd, ng = M["Nifty"]["day"], M["Nifty"]["gap"]
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        c1.metric("Nifty", f"{Q['Nifty']['price']:,.1f}", f"{Q['Nifty']['pct']:+.2f}%")
        c2.markdown(f"### {ico(nd['live'])}")
        c2.caption(f"दिन का रुख · {strength(nd['live'])} (score {nd['live']:+.2f})")
        if ng:
            st.markdown(f"**Gap संकेत (सुबह का खुलना):** {ico(ng['live'])} · {strength(ng['live'])}")
        st.caption(f"{confirm('Nifty', sig(nd['live']), Q['Nifty']['pct'])} · दिन-मॉडल: {verdict(nd['st'])}"
                   f" (hit {nd['st']['hit']*100:.0f}% बनाम baseline {nd['st']['base']*100:.0f}%)")
    trows = []
    for s in STOCKS:
        d, g = M[s]["day"], M[s]["gap"]
        if not d:
            continue
        trows.append({"नाम": s, "भाव": round(Q[s]["price"], 1), "बदलाव %": round(Q[s]["pct"], 2),
                      "Gap": ico(g["live"]) if g else "—", "दिन": ico(d["live"]),
                      "ताक़त": strength(d["live"]), "पुष्टि": confirm(s, sig(d["live"]), Q[s]["pct"]),
                      "भरोसा": verdict(d["st"])})
    st.dataframe(pd.DataFrame(trows), hide_index=True, use_container_width=True)
    st.caption("**भरोसा** कॉलम 🎯 Accuracy वाले ईमानदार test से आता है। ⚠️ तुक्के जैसा = इस शेयर का signal "
               "अभी सांख्यिकीय रूप से भरोसे लायक साबित नहीं हुआ। ⚠️ सिर्फ़ जानकारी है, trade की सलाह नहीं।")

# ---------- Accuracy ----------
with tabs[1]:
    st.markdown(f"**Walk-forward test (पिछले {WK} trading दिन):** हर दिन का signal सिर्फ़ उससे पहले के data से "
                "बना, इसलिए आगे की जानकारी का फ़ायदा नहीं मिला। इसलिए यह पुराने 'अंदाज़ा' वाले hit-rate से "
                "ज़्यादा ईमानदार है।")
    arows = []
    for s in TICK:
        for md, nm_ in (("gap", "Gap"), ("day", "दिन")):
            m = M[s][md]
            if m:
                x = m["st"]
                arows.append({"नाम": s, "मॉडल": nm_, "tests": x["n"], "hit %": round(x["hit"] * 100),
                              "baseline %": round(x["base"] * 100),
                              "95% range": f"{x['lo']*100:.0f}–{x['hi']*100:.0f}",
                              "फैसला": verdict(x)})
    st.dataframe(pd.DataFrame(arows), hide_index=True, use_container_width=True)
    st.caption("**baseline** = हमेशा एक ही तरफ़ (जो ज़्यादा चला) कहने वाले की accuracy। **95% range** = असली "
               "accuracy इसी दायरे में होने की संभावना। ✅ तब, जब पूरा दायरा 50% से ऊपर हो और hit baseline से ज़्यादा हो।")
    pick = st.selectbox("ताक़त के हिसाब से देखें", list(TICK), key="acc_sel")
    brow = []
    for md, nm_ in (("gap", "Gap"), ("day", "दिन")):
        m = M[pick][md]
        if m:
            brow.append({"मॉडल": nm_, "मज़बूत signal (ऊपर का आधा) hit %": round(m["st"]["strong"] * 100),
                         "कमज़ोर signal (नीचे का आधा) hit %": round(m["st"]["weak"] * 100)})
    if brow:
        st.dataframe(pd.DataFrame(brow), hide_index=True, use_container_width=True)
        st.caption("अगर मज़बूत signal का hit-rate कमज़ोर से ज़्यादा नहीं है, तो 'ताक़त' पर भरोसा न करें।")
    st.info("समय के हिसाब से अलग वज़न (सुबह US ज़्यादा, दोपहर यूरोप ज़्यादा) इस version में हटा दिया है, क्योंकि "
            "रोज़ के data पर उसे परखा नहीं जा सकता। Live signal और test एक ही तरीके से बनते हैं।")

# ---------- Global मेल ----------
with tabs[2]:
    sel = st.selectbox("Nifty / शेयर चुनें", list(TICK), key="imp_sel")
    mdsel = st.radio("मॉडल", ["दिन", "Gap"], horizontal=True)
    m = M[sel]["day" if mdsel == "दिन" else "gap"]
    if not m:
        st.caption("इस मॉडल के लिए data कम है")
    else:
        st.markdown(f"**{sel} — आज का कुल असर:** {ico(m['live'])} ({strength(m['live'])}, score {m['live']:+.2f})")
        irows = []
        for n in m["corr"].index:
            cv = float(m["corr"][n])
            today = GNOW.get(n)
            ef = "—" if today is None or cv * today == 0 else ("मदद" if cv * today > 0 else "दबाव")
            ph = m["pair"].get(n)
            irows.append({"Global जोड़ी": n + (" ★" if n in [PEER[t] for t in PEERS_OF[sel]] else ""),
                          "आज %": None if today is None else round(today, 2),
                          "असर": ef, "correlation": round(cv, 2),
                          "जोड़ी accuracy %": None if ph is None or ph != ph else round(float(ph) * 100),
                          "_a": abs(cv)})
        idf = pd.DataFrame(irows).sort_values("_a", ascending=False).drop(columns="_a")
        st.dataframe(idf, hide_index=True, use_container_width=True)
        st.caption("★ = उसी शेयर/क्षेत्र से सीधा जुड़ा global शेयर। **correlation** + का मतलब साथ-साथ चलना, − का "
                   "मतलब उल्टा। **जोड़ी accuracy** = सिर्फ़ उस एक जोड़ी से दिशा सही निकलने का walk-forward प्रतिशत "
                   "(50% = तुक्का)।")
        cs = m["contrib"].sort_values()
        fig = px.bar(x=cs.values, y=cs.index, orientation="h",
                     color=np.where(cs.values > 0, "मदद", "दबाव"),
                     color_discrete_map={"मदद": "#2e9e5b", "दबाव": "#d64545"})
        fig.update_layout(height=max(300, 22 * len(cs)), margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                          xaxis_title=None, yaxis_title=None)
        st.caption("आज किस global जोड़ी ने कितना ज़ोर लगाया")
        st.plotly_chart(fig, use_container_width=True)
    heat = pd.DataFrame({s: M[s]["day"]["corr"].reindex(list(GLOBAL)) for s in TICK if M[s]["day"]}).T
    hf = px.imshow(heat.round(2), color_continuous_scale="RdYlGn", zmin=-0.6, zmax=0.6, text_auto=True, aspect="auto")
    hf.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.markdown("**सबका correlation एक नज़र में (दिन-मॉडल, 90 दिन)**")
    st.plotly_chart(hf, use_container_width=True)

# ---------- Layer 1 ----------
with tabs[3]:
    d0 = M["Nifty"]["day"]
    top = d0["contrib"].reindex(d0["contrib"].abs().sort_values(ascending=False).index)[:3]
    st.markdown("**Drivers:** " + ", ".join(f"{n} ({sig(v)})" for n, v in top.items()))
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
    CP = cpr_table(str(now().date()))
    crow = []
    for s in TICK:
        if s not in CP:
            continue
        lo, hi = CP[s]
        p = Q[s]["price"]
        v = Q[s].get("vwap")
        crow.append({"नाम": s, "भाव": round(p, 1), "CPR": f"{lo}–{hi}",
                     "CPR असर": "मदद" if p > hi else "दबाव" if p < lo else "तटस्थ",
                     "VWAP": round(v, 1) if v else None,
                     "VWAP असर": ("मदद" if p > v else "दबाव") if v else "—"})
    if crow:
        st.dataframe(pd.DataFrame(crow), hide_index=True, use_container_width=True)
    pick2 = st.selectbox("Intraday chart", list(TICK), key="chart_sel")
    try:
        ih = intraday(TICK[pick2])
        cols_ = [c_ for c_ in ("Close", "VWAP") if c_ in ih.columns]
        if len(ih):
            f2 = px.line(ih[cols_])
            f2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None,
                             yaxis_title=None, legend_title=None)
            st.plotly_chart(f2, use_container_width=True)
    except Exception:
        st.caption("Chart का data नहीं मिला")
    NB = news_bundle()
    st.markdown("**GIFT Nifty (ताज़ा खबर)**")
    show_news(NB["gift"])
    st.markdown("**FII / DII (ताज़ा खबर)**")
    show_news(NB["fii"])

# ---------- Layer 2 ----------
with tabs[4]:
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
with tabs[5]:
    NB = news_bundle()
    st.markdown("#### 🌐 Global (CNBC, BBC, Reuters, Bloomberg)")
    show_news(NB["global"])
    st.markdown("#### 🇮🇳 Nifty / भारत (ET, Moneycontrol, Mint, BS, Reuters)")
    show_news(NB["india"])
    nsel = st.selectbox("किस शेयर की खबरें", list(STOCKS), key="news_sel")
    st.markdown(f"#### 📰 {nsel}")
    show_news(stock_news(nsel))

# ---------- Log ----------
with tabs[6]:
    if not os.path.exists(LOG):
        st.caption("Log बाज़ार खुलने पर बनना शुरू होगा (हर 5 मिनट में एक बार, 'दिन' signal का)।")
    else:
        df = pd.read_csv(LOG)
        df["date"] = df["time"].str[:10]
        last_p = df.groupby(["date", "symbol"])["price"].transform("last")
        okk = np.where(df["signal"] == "मदद", last_p > df["price"], last_p < df["price"])
        mk = last_p != df["price"]
        if mk.sum():
            st.markdown("**दिन के आख़िरी भाव के मुक़ाबले signal कितना सही रहा**")
            s_ = pd.DataFrame({"symbol": df["symbol"][mk], "सही": okk[mk]}).groupby("symbol")["सही"].agg(["mean", "count"])
            s_["hit-rate %"] = (s_["mean"] * 100).round(0)
            st.dataframe(s_[["hit-rate %", "count"]].rename(columns={"count": "signals"}), use_container_width=True)
        st.dataframe(df.drop(columns="date").tail(60).iloc[::-1], hide_index=True, use_container_width=True)
        st.download_button("⬇️ Log download (CSV)", df.drop(columns="date").to_csv(index=False),
                           "signal_log.csv", "text/csv")
        st.caption("Streamlit Cloud restart पर यह फ़ाइल मिट सकती है, इसलिए CSV बीच-बीच में download कर लें।")
