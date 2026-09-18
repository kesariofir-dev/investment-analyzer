"""Investment Analyzer V1.1 — free, rule-based equity research in Hebrew."""
from datetime import datetime, timezone
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

NA = "נתון לא זמין"
DISCLAIMER = "המערכת מיועדת למחקר ולמידה בלבד ואינה מהווה ייעוץ השקעות, המלצה לביצוע פעולה או תחליף לבדיקת מקורות פיננסיים רשמיים."


def number(value):
    """Never turn missing values into zero, or expose NaN/inf to the UI."""
    try:
        if value is None or isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def safe_get(mapping, key):
    return number(mapping.get(key)) if isinstance(mapping, dict) else None


def first(*values):
    return next((v for v in values if v is not None), None)


def ratio(a, b):
    a, b = number(a), number(b)
    return a / b if a is not None and b is not None and b > 0 else None


def fmt(value, kind="num"):
    value = number(value)
    if value is None:
        return NA
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "money":
        for size, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
            if abs(value) >= size:
                return f"{value / size:,.2f}{suffix}"
    return f"{value:,.2f}"


def normalize_symbol(value):
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.\-^=]{0,19}", symbol):
        return None
    return symbol


def statement_series(frame, labels):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=float)
    for label in labels:
        if label in frame.index:
            series = frame.loc[label]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[0]
            series = pd.to_numeric(series, errors="coerce")
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series.loc[~series.index.isna()].dropna().sort_index()
            return series[~series.index.duplicated(keep="last")]
    return pd.Series(dtype=float)


def latest_statement(frame, fields):
    """Use one common fiscal date; never subtract values from different years."""
    series = {key: statement_series(frame, labels) for key, labels in fields.items()}
    dates = [s.index.max() for s in series.values() if not s.empty]
    date = max(dates) if dates else None
    return {key: number(s.get(date)) for key, s in series.items()}, date


@st.cache_data(ttl=900, max_entries=60, show_spinner=False)
def load_data(symbol, detailed=True):
    ticker = yf.Ticker(symbol)
    warnings = []
    try:
        info = ticker.get_info()
        info = info if isinstance(info, dict) else {}
    except Exception:
        info = {}
        warnings.append("מידע החברה לא התקבל מ־Yahoo Finance. ייתכן עומס זמני.")
    history = pd.DataFrame()
    if detailed:
        try:
            start = (pd.Timestamp.now() - pd.DateOffset(years=6, days=15)).date().isoformat()
            history = ticker.history(start=start, interval="1d", auto_adjust=False,
                                     actions=False, timeout=15, raise_errors=True)
            if not history.empty:
                history.index = pd.to_datetime(history.index)
                if history.index.tz is not None:
                    history.index = history.index.tz_localize(None)
                history = history.sort_index()
                history = history.loc[~history.index.duplicated(keep="last")]
                for column in history:
                    history[column] = pd.to_numeric(history[column], errors="coerce")
        except Exception:
            warnings.append("היסטוריית המחירים לא התקבלה. גרפים ותשואות עשויים להיות חסרים.")
    frames = {}
    is_fund = info.get("quoteType") in {"ETF", "MUTUALFUND"}
    for key, getter in [("income", ticker.get_income_stmt),
                        ("cashflow", ticker.get_cashflow),
                        ("balance", ticker.get_balance_sheet)]:
        frames[key] = pd.DataFrame()
        if detailed and not is_fund:
            try:
                result = getter(freq="yearly")
                if isinstance(result, pd.DataFrame):
                    frames[key] = result
            except Exception:
                warnings.append({"income": "דוח רווח והפסד שנתי לא זמין.",
                                 "cashflow": "דוח תזרים שנתי לא זמין.",
                                 "balance": "מאזן שנתי לא זמין."}[key])
    return dict(symbol=symbol, info=info, history=history, **frames,
                warnings=warnings, fetched=datetime.now(timezone.utc).isoformat())


def wilder_rsi(close, period=14):
    """Wilder's original SMA seed, followed by recursive smoothing."""
    close = pd.to_numeric(close, errors="coerce").dropna()
    result = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= period:
        return result
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    gain, loss = gains.iloc[1:period + 1].mean(), losses.iloc[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            gain = (gain * (period - 1) + gains.iloc[i]) / period
            loss = (loss * (period - 1) + losses.iloc[i]) / period
        result.iloc[i] = 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))
    return result


def performance(series):
    """As-of calendar returns; no invented history for recent IPOs."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0].sort_index()
    keys = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]
    values = {key: None for key in keys}
    if series.empty:
        return values
    end = series.index[-1]
    targets = {"1M": end - pd.DateOffset(months=1), "3M": end - pd.DateOffset(months=3),
               "6M": end - pd.DateOffset(months=6), "YTD": pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1),
               "1Y": end - pd.DateOffset(years=1), "3Y": end - pd.DateOffset(years=3),
               "5Y": end - pd.DateOffset(years=5)}
    for key, target in targets.items():
        before = series.loc[:target]
        if not before.empty and (target - before.index[-1]).days <= 7:
            values[key] = float(series.iloc[-1] / before.iloc[-1] - 1)
    return values


def derive(data):
    info, history = data["info"], data["history"]
    close = history.get("Close", pd.Series(dtype=float)).dropna()
    close = close[close > 0]
    technical = pd.DataFrame({"Close": close})
    for window in [50, 150, 200]:
        technical[f"SMA{window}"] = close.rolling(window, min_periods=window).mean()
    technical["RSI14"] = wilder_rsi(close)
    adjusted = history.get("Adj Close", pd.Series(dtype=float))
    returns = performance(adjusted)
    get = lambda key: safe_get(info, key)
    live_price = first(get("currentPrice"), get("regularMarketPrice"))
    price = first(live_price, number(close.iloc[-1]) if not close.empty else None)
    previous = first(get("regularMarketPreviousClose"), get("previousClose"))
    if live_price is None and len(close) >= 2:
        previous = number(close.iloc[-2])
    daily_ratio = ratio(price, previous)
    cash, cash_date = latest_statement(data["cashflow"], {
        "ocf": ["Operating Cash Flow", "Total Cash From Operating Activities"],
        "capex": ["Capital Expenditure", "Capital Expenditures"],
        "fcf": ["Free Cash Flow"]})
    if cash["fcf"] is None and cash["ocf"] is not None and cash["capex"] is not None:
        cash["fcf"] = cash["ocf"] - abs(cash["capex"])
    balance, balance_date = latest_statement(data["balance"], {
        "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
        "debt": ["Total Debt"], "equity": ["Stockholders Equity", "Common Stock Equity"],
        "assets": ["Current Assets"], "liabilities": ["Current Liabilities"]})
    # Fall back as a group to keep dates/bases internally consistent.
    if balance_date is None:
        balance.update(cash=get("totalCash"), debt=get("totalDebt"))
        debt_equity = get("debtToEquity")  # Yahoo supplies percent, e.g. 150 = 1.5x.
        debt_equity = debt_equity / 100 if debt_equity is not None and debt_equity >= 0 else None
        current_ratio = get("currentRatio")
    else:
        debt_equity = ratio(balance["debt"], balance["equity"])
        current_ratio = ratio(balance["assets"], balance["liabilities"])
    net_cash = balance["cash"] - balance["debt"] if balance["cash"] is not None and balance["debt"] is not None else None
    revenues = statement_series(data["income"], ["Total Revenue", "Operating Revenue"])
    eps = statement_series(data["income"], ["Diluted EPS", "Basic EPS"])
    eps_growth = None
    if len(eps) >= 2 and 300 <= (eps.index[-1] - eps.index[-2]).days <= 430 and eps.iloc[-2] > 0:
        eps_growth = float(eps.iloc[-1] / eps.iloc[-2] - 1)
    high = get("fiftyTwoWeekHigh")
    trailing_eps = get("trailingEps")
    pe = get("trailingPE")
    if pe is not None and (pe <= 0 or (trailing_eps is not None and trailing_eps <= 0)):
        pe = None
    forward_pe = get("forwardPE")
    if forward_pe is not None and forward_pe <= 0:
        forward_pe = None
    metrics = dict(price=price, daily=daily_ratio - 1 if daily_ratio is not None else None,
                   market_cap=get("marketCap"), high=high, low=get("fiftyTwoWeekLow"),
                   distance=(price / high - 1) if price is not None and high is not None and high > 0 else None,
                   pe=pe, forward_pe=forward_pe, peg=first(get("pegRatio"), get("trailingPegRatio")),
                   ps=get("priceToSalesTrailing12Months"), pb=get("priceToBook"), ev=get("enterpriseValue"),
                   ev_ebitda=get("enterpriseToEbitda"), revenue=get("totalRevenue"),
                   revenue_growth=get("revenueGrowth"), earnings_growth=get("earningsGrowth"),
                   eps=trailing_eps, forward_eps=get("forwardEps"), eps_growth=eps_growth,
                   gross=get("grossMargins"), operating=get("operatingMargins"), profit=get("profitMargins"),
                   ebitda=get("ebitdaMargins"), roe=get("returnOnEquity"), roa=get("returnOnAssets"),
                   **cash, cash=balance["cash"], debt=balance["debt"], net_cash=net_cash,
                   debt_equity=debt_equity, current_ratio=current_ratio)
    for key in ["SMA50", "SMA150", "SMA200", "RSI14"]:
        metrics[key] = number(technical[key].iloc[-1]) if not technical.empty else None
    metrics["technical_price"] = number(close.iloc[-1]) if not close.empty else None
    return dict(metrics=metrics, technical=technical, returns=returns, revenues=revenues,
                cash_date=cash_date, balance_date=balance_date, live_price=live_price,
                equity=balance["equity"])


def ramp(value, low, high, reverse=False):
    value = number(value)
    if value is None:
        return None
    result = float(np.clip((value - low) / (high - low), 0, 1))
    return 1 - result if reverse else result


def score_model(m, returns, sector, equity=None):
    """Transparent heuristic, not peer percentiles or a return forecast."""
    growth = m["revenue_growth"]
    allowance = 1 + 2 * min(max(growth or 0, 0), .4)
    pe_base = 28 if sector in {"Technology", "Communication Services", "Healthcare"} else 22
    ps_base = 6 if sector in {"Technology", "Communication Services"} else 3
    def multiple(value, base):
        return ramp(value / (base * allowance), .5, 2.5, True) if value is not None and value > 0 else None
    def feature(label, value, score, kind="num"):
        return (label, value, score, kind)
    f = feature
    price = m["technical_price"]
    p50, p200 = ratio(price, m["SMA50"]), ratio(price, m["SMA200"])
    rsi = m["RSI14"]
    rsi_score = None if rsi is None else float(np.interp(rsi, [0, 30, 50, 65, 80, 100], [0, .2, .65, 1, .65, .2]))
    cash_debt = ratio(m["cash"], m["debt"])
    if m["debt"] == 0 and m["cash"] is not None:
        cash_debt = 3.0
    fcf_score = None if m["fcf"] is None else (1.0 if m["fcf"] > 0 else 0.0)
    roe_score = ramp(m["roe"], 0, .25) if equity is None or equity > 0 else None
    groups = {
        "Quality · איכות": [f("רווח גולמי", m["gross"], ramp(m["gross"], .1, .6), "pct"),
            f("רווח תפעולי", m["operating"], ramp(m["operating"], 0, .3), "pct"),
            f("רווח נקי", m["profit"], ramp(m["profit"], 0, .25), "pct"),
            f("ROE", m["roe"], roe_score, "pct"), f("תזרים חופשי שנתי", m["fcf"], fcf_score, "money")],
        "Growth · צמיחה": [f("צמיחת הכנסות", growth, ramp(growth, -.1, .3), "pct"),
            f("צמיחת רווחים", m["earnings_growth"], ramp(m["earnings_growth"], -.1, .4), "pct"),
            f("צמיחת EPS שנתית", m["eps_growth"], ramp(m["eps_growth"], -.1, .4), "pct")],
        "Valuation · תמחור": [f("P/E", m["pe"], multiple(m["pe"], pe_base)),
            f("Forward P/E", m["forward_pe"], multiple(m["forward_pe"], pe_base)),
            f("PEG", m["peg"], ramp(m["peg"], .5, 3, True) if m["peg"] is not None and m["peg"] > 0 else None),
            f("Price/Sales", m["ps"], multiple(m["ps"], ps_base))],
        "Financial Strength · חוסן": [f("מזומן / חוב (תקרה 3 בניקוד)", cash_debt, ramp(cash_debt, 0, 2)),
            f("חוב / הון", m["debt_equity"], ramp(m["debt_equity"], 0, 2, True)),
            f("יחס שוטף", m["current_ratio"], ramp(m["current_ratio"], .5, 2)),
            f("תזרים חופשי שנתי", m["fcf"], fcf_score, "money")],
        "Momentum · מומנטום": [f("מחיר / SMA50", p50, ramp(p50, .85, 1.15)),
            f("מחיר / SMA200", p200, ramp(p200, .8, 1.2)), f("RSI14", rsi, rsi_score),
            f("תשואת 6 חודשים", returns["6M"], ramp(returns["6M"], -.2, .3), "pct"),
            f("תשואת שנה", returns["1Y"], ramp(returns["1Y"], -.3, .5), "pct")]
    }
    results = []
    for title, features in groups.items():
        available = [x[2] for x in features if x[2] is not None]
        coverage = len(available) / len(features)
        score = round(20 * float(np.mean(available)), 1) if coverage >= .6 else None
        results.append(dict(title=title, score=score, coverage=coverage, features=features))
    total = round(sum(x["score"] for x in results), 1) if all(x["score"] is not None and x["coverage"] >= .75 for x in results) else None
    return results, total, allowance, pe_base, ps_base


def observations(m):
    simple, strengths, risks = [], [], []
    growth = m["revenue_growth"]
    if growth is not None:
        if growth > .15:
            strengths.append(f"צמיחת הכנסות של {fmt(growth, 'pct')} לפי הנתון האחרון.")
            simple.append("החברה מצליחה להגדיל את המכירות שלה בקצב גבוה.")
        elif growth < 0:
            risks.append(f"ההכנסות התכווצו ב־{fmt(abs(growth), 'pct')} ביחס לתקופה המקבילה.")
            simple.append("המכירות בתקופה המדווחת נמוכות מהתקופה המקבילה אשתקד.")
        else:
            simple.append(f"ההכנסות גדלו ב־{fmt(growth, 'pct')} ביחס לתקופה המקבילה.")
    if m["gross"] is not None and m["gross"] > .4:
        strengths.append(f"שיעור רווח גולמי של {fmt(m['gross'], 'pct')}.")
        simple.append("חלק משמעותי מהמכירות נשאר לאחר עלות המוצר, לפני הוצאות תפעול, מימון ומסים.")
    if m["fcf"] is not None:
        if m["fcf"] > 0:
            strengths.append("תזרים חופשי חיובי בשנת הדיווח האחרונה.")
            simple.append("לאחר הפעילות השוטפת וההשקעות בציוד ותשתיות נשאר לחברה מזומן בתקופה המדווחת.")
        elif m["fcf"] < 0:
            risks.append("תזרים חופשי שלילי בשנת הדיווח האחרונה.")
            simple.append("התזרים מהפעילות לא כיסה את ההשקעות בציוד ותשתיות. ייתכן שזו תקופת השקעה, אך הנתון לבדו אינו מסביר את הסיבה.")
    if m["pe"] is not None and m["pe"] > 35:
        risks.append(f"מכפיל רווח של {fmt(m['pe'])}; רגישות לציפיות הצמיחה.")
        simple.append("מחיר המניה גבוה יחסית לרווח הנוכחי. נדרשת בחינה האם הצמיחה העתידית תצדיק אותו.")
    if m["profit"] is not None and m["profit"] < 0:
        risks.append("שיעור הרווח הנקי שלילי.")
    if m["net_cash"] is not None:
        if m["net_cash"] > 0:
            strengths.append("יתרת המזומן וההשקעות הנזילות המדווחות עולה על החוב.")
        elif m["net_cash"] < 0:
            simple.append("החוב גבוה מהמזומן המדווח; חשוב לבחון גם את יכולת ההחזר ואת מועדי הפירעון.")
    if m["debt_equity"] is not None and m["debt_equity"] > 1.5:
        risks.append(f"חוב / הון של {fmt(m['debt_equity'])} פעמים; יש לבחון בהקשר הענפי.")
    if m["current_ratio"] is not None and m["current_ratio"] < 1:
        risks.append("ההתחייבויות השוטפות גבוהות מהנכסים השוטפים.")
    if m["RSI14"] is not None and m["RSI14"] > 70:
        risks.append("RSI מעל 70: אזור קניית יתר טכני, שאינו מנבא בהכרח ירידה.")
    return simple, strengths, risks



def price_scenario(metrics, change_pct):
    """Sensitivity only: no forecast, target, or change to business results."""
    factor = 1 + change_pct / 100
    result = {"price": None, "pe": None, "forward_pe": None, "ps": None, "pb": None}
    for key in result:
        value = number(metrics.get(key))
        if value is not None and value > 0:
            result[key] = value * factor
    return result


def historical_investment(history, years, amount):
    """Use adjusted closes and disclose the actual calendar baseline."""
    series = history.get("Adj Close", pd.Series(dtype=float))
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series = series[series > 0]
    if series.empty:
        return None
    end = series.index[-1]
    target = end - pd.DateOffset(years=int(years))
    before = series.loc[:target]
    if before.empty or (target - before.index[-1]).days > 7:
        return None
    start = before.index[-1]
    path = series.loc[start:] / before.iloc[-1] * float(amount)
    total_return = float(path.iloc[-1] / amount - 1)
    annualized = float((1 + total_return) ** (365.25 / (end - start).days) - 1)
    return dict(start=start, end=end, path=path, final=float(path.iloc[-1]),
                profit=float(path.iloc[-1] - amount), total_return=total_return,
                annualized=annualized)


def contextual_explanation(label, value):
    value = number(value)
    if value is None:
        return ""
    if label in {"P/E", "Forward P/E"} and value > 0:
        period = "השנתי המדווח" if label == "P/E" else "העתידי הצפוי"
        return f"משלמים {value:,.2f} יחידות מחיר על כל יחידה של רווח {period} למניה."
    if label == "Price / Sales" and value > 0:
        return f"שווי החברה הוא פי {value:,.2f} מהכנסותיה השנתיות."
    if label == "Price / Book":
        return f"שווי השוק הוא פי {value:,.2f} מההון החשבונאי המדווח; הון שלילי מגביל את משמעות היחס."
    if label in {"Gross Margin", "Operating Margin", "Profit Margin", "EBITDA Margin"}:
        stage = {"Gross Margin": "לאחר עלות המוצר", "Operating Margin": "לאחר הוצאות התפעול",
                 "Profit Margin": "בשורה התחתונה, לאחר כלל ההוצאות", "EBITDA Margin": "לפני ריבית, מסים, פחת והפחתות"}[label]
        outcome = "רווח" if value >= 0 else "הפסד"
        return f"מכל 100 יחידות הכנסה, נרשם {outcome} של {abs(value * 100):,.2f} יחידות {stage}."
    if label in {"Revenue Growth", "Earnings Growth", "EPS Growth · שנתי"}:
        subject = {"Revenue Growth": "ההכנסות", "Earnings Growth": "הרווחים", "EPS Growth · שנתי": "הרווח למניה"}[label]
        verb = "עלה" if label.startswith("EPS") and value >= 0 else "ירד" if label.startswith("EPS") else "עלו" if value >= 0 else "ירדו"
        period = "השנה הקודמת" if label.startswith("EPS") else "התקופה המקבילה"
        return f"{subject} {verb} ב־{abs(value * 100):,.2f}% לעומת {period}, לפי הנתונים הזמינים."
    if label == "Current Ratio":
        return f"מול כל יחידה של התחייבויות שוטפות יש {value:,.2f} יחידות של נכסים שוטפים."
    if label == "Debt / Equity · פעמים":
        return f"מול כל יחידה של הון עצמי יש {value:,.2f} יחידות חוב."
    if label in {"EPS · TTM", "Forward EPS"}:
        period = "ב־12 החודשים האחרונים" if label == "EPS · TTM" else "לפי התחזית העתידית"
        return f"{'רווח' if value >= 0 else 'הפסד'} של {abs(value):,.2f} לכל מניה {period}."
    if label == "Free Cash Flow":
        return f"{'נותר עודף' if value >= 0 else 'נוצר פער שלילי'} של {fmt(abs(value), 'money')} בין התזרים השוטף להשקעות בתקופה."
    if label in {"ROE", "ROA"}:
        denominator = "הון עצמי" if label == "ROE" else "נכסים"
        return f"יחס הרווח ל{denominator} המדווח הוא {value * 100:,.2f} לכל 100. זהו יחס חשבונאי, לא תשואה על המניה."
    return ""


def export_report(data, model):
    """Portable UTF-8 Markdown snapshot, no extra libraries or external calls."""
    m, info = model["metrics"], data["info"]
    def clean(value):
        return str(value).replace("|", "/").replace("\n", " ").replace("\r", " ")
    lines = [f"# Investment Analyzer — {data['symbol']}", "",
             clean(info.get("longName") or info.get("shortName") or data["symbol"]), "",
             f"שליפה (UTC): {data['fetched']}",
             f"מטבע מסחר: {info.get('currency') or NA} | מטבע דוחות: {info.get('financialCurrency') or NA}",
             "מקור: Yahoo Finance באמצעות yfinance. הנתונים עשויים להיות מושהים.", "",
             "הדוח הוא צילום מצב של הנתונים שנשלפו, ואינו כולל תרחישים שהוזנו במחשבונים.", "",
             "## מידע ונתונים", "", "| מדד | ערך |", "|---|---|"]
    fields = [
        ("price", "מחיר אחרון", "num"), ("daily", "שינוי יומי", "pct"),
        ("market_cap", "שווי שוק", "money"), ("high", "שיא 52 שבועות", "num"),
        ("low", "שפל 52 שבועות", "num"), ("distance", "מרחק מהשיא", "pct"),
        ("pe", "P/E", "num"), ("forward_pe", "Forward P/E", "num"), ("peg", "PEG", "num"),
        ("ps", "P/S", "num"), ("pb", "P/B", "num"), ("ev", "EV", "money"),
        ("ev_ebitda", "EV/EBITDA", "num"), ("revenue", "הכנסות TTM", "money"),
        ("revenue_growth", "צמיחת הכנסות", "pct"), ("earnings_growth", "צמיחת רווחים", "pct"),
        ("eps", "EPS", "num"), ("forward_eps", "Forward EPS", "num"), ("eps_growth", "צמיחת EPS שנתית", "pct"),
        ("gross", "רווחיות גולמית", "pct"), ("operating", "רווחיות תפעולית", "pct"),
        ("profit", "רווחיות נקייה", "pct"), ("ebitda", "EBITDA Margin", "pct"),
        ("roe", "ROE", "pct"), ("roa", "ROA", "pct"), ("ocf", "תזרים שוטף שנתי", "money"),
        ("fcf", "תזרים חופשי שנתי", "money"), ("capex", "CAPEX בסימן המקור", "money"),
        ("cash", "מזומן", "money"), ("debt", "חוב", "money"), ("net_cash", "מזומן פחות חוב", "money"),
        ("debt_equity", "חוב/הון בפעמים", "num"), ("current_ratio", "יחס שוטף", "num"),
        ("technical_price", "מחיר בחישוב הטכני", "num"), ("SMA50", "SMA50", "num"),
        ("SMA150", "SMA150", "num"), ("SMA200", "SMA200", "num"), ("RSI14", "RSI14", "num")]
    for key, label, kind in fields:
        lines.append(f"| {label} | {fmt(m[key], kind)} |")
    for label, key in [("תאריך תזרים שנתי", "cash_date"), ("תאריך מאזן", "balance_date")]:
        lines.append(f"\n{label}: {model[key].date() if model[key] is not None else NA}")
    if not model["technical"].empty:
        lines.append(f"\nתאריך מחיר טכני: {model['technical'].index[-1].date()}")
    lines.extend(["", "הכנסות ורווחיות בסיכום הן בדרך כלל TTM; צמיחת הכנסות ורווחים מול הרבעון המקביל. תזרים ומאזן שנתיים עשויים להתייחס לתקופה שונה.",
                  "", "## הכנסות שנתיות", "", "| סיום שנת דיווח | הכנסות |", "|---|---|"])
    for date, value in model["revenues"].items():
        lines.append(f"| {date.date()} | {fmt(value, 'money')} |")
    if model["revenues"].empty:
        lines.append(f"| — | {NA} |")
    lines.extend(["", "## תשואות מצטברות", "", "| תקופה | תשואה |", "|---|---|"])
    for key, value in model["returns"].items():
        lines.append(f"| {key} | {fmt(value, 'pct')} |")
    lines.extend(["", "תשואות על בסיס Adj Close המותאם לפיצולים ולדיבידנדים; לא תשואה שנתית ממוצעת. החישובים הטכניים משתמשים ב־Close ללא התאמת דיבידנדים."])
    for title, texts in zip(["במילים פשוטות", "חוזקות", "נקודות לתשומת לב"], observations(m)):
        lines.extend(["", f"## {title}", ""] + ["- " + clean(t) for t in texts or ["אין מספיק ממצאים לפי הכללים והנתונים הזמינים."]])
    if info.get("quoteType") == "EQUITY" and info.get("sector") not in {"Financial Services", "Real Estate"}:
        results, total, allowance, pe_base, ps_base = score_model(m, model["returns"], info.get("sector"), model["equity"])
        lines.extend(["", "## ניקוד מחקר", "", f"ציון כולל: {fmt(total)} / 100",
                      "מודל כללים פנימי שאינו תחזית או המלצה. ציון קטגוריה דורש 60% כיסוי; ציון כולל דורש 75% בכל קטגוריה. המדדים הזמינים שווים במשקלם בתוך הקטגוריה.",
                      f"בסיס P/E: {pe_base}; בסיס P/S: {ps_base}; מקדם התאמת צמיחה: {allowance:.2f}."])
        for result in results:
            lines.extend(["", f"### {result['title']}: {fmt(result['score'])}/20 — כיסוי {result['coverage']:.0%}", "",
                          "| מדד | ערך | תרומה יחסית 0–100 |", "|---|---|---|"])
            for label, value, score, kind in result["features"]:
                lines.append(f"| {label} | {fmt(value, kind)} | {fmt(score * 100) if score is not None else NA} |")
    else:
        lines.extend(["", "ניקוד עסקי אינו מוצג לסוג נכס או ענף זה."])
    lines.extend(["", "K אלפים, M מיליונים, B מיליארדים, T טריליונים. אין המרת מטבע.", "", DISCLAIMER])
    return "\n".join(lines)


def tools_ui(data, model):
    m, info = model["metrics"], data["info"]
    currency = info.get("currency") or NA
    st.markdown("### מה יקרה אם המחיר ישתנה?")
    st.write("הזז את הסליידר כדי לבדוק איך שינוי במחיר משפיע על המכפילים.")
    change = st.slider("שינוי במחיר (%)", min_value=-80, max_value=100, value=0, step=5,
                       key="scenario_" + data["symbol"])
    scenario = price_scenario(m, change)
    st.caption("תרחיש חישובי בלבד: הרווח למניה, תחזיות הרווח, ההכנסות, ההון ומספר המניות נשארים קבועים. אין כאן תחזית מחיר או הערכת שווי הוגן.")
    if scenario["price"] is None:
        st.info("אין מחיר זמין לחישוב תרחיש.")
    else:
        a, b = st.columns(2)
        a.metric(f"מחיר מקור · {currency}", fmt(m["price"]))
        b.metric(f"מחיר בתרחיש · {currency}", fmt(scenario["price"]), f"{change:+d}%")
        rows = [{"מדד": label, "כעת": fmt(m[key]) if scenario[key] is not None else NA,
                 "בתרחיש": fmt(scenario[key])} for key, label in
                [("pe", "P/E"), ("forward_pe", "Forward P/E"), ("ps", "Price / Sales"), ("pb", "Price / Book")]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if scenario["pe"] is not None:
            st.write(f"בשינוי של {change:+d}% במחיר, מכפיל הרווח משתנה מ־{fmt(m['pe'])} ל־{fmt(scenario['pe'])}, רק אם הרווח נשאר קבוע.")
        else:
            st.caption("אין מכפיל רווח חיובי תקף; לא מוצג תרחיש P/E לחברה הפסדית או כשהנתון חסר.")
    st.divider()
    st.markdown("### כמה הייתה שווה השקעה בעבר?")
    st.caption(f"הסכום במטבע המסחר: {currency}. אין המרה לשקלים או לדולרים אם זה אינו מטבע המסחר.")
    amount = st.number_input("סכום השקעה התחלתי", min_value=100.0, max_value=100000000.0,
                             value=1000.0, step=100.0, key="investment_" + data["symbol"])
    years = st.selectbox("לפני כמה שנים?", [1, 3, 5], index=2, key="years_" + data["symbol"])
    result = historical_investment(data["history"], years, amount)
    if result is None:
        st.info("אין היסטוריה מותאמת מספקת לתקופה שבחרת. אפשר לבחור תקופה קצרה יותר.")
    else:
        a, b, c = st.columns(3)
        a.metric(f"שווי בתאריך האחרון · {currency}", fmt(result["final"]))
        b.metric(f"רווח / הפסד · {currency}", fmt(result["profit"]), fmt(result["total_return"], "pct"))
        c.metric("תשואה שנתית ממוצעת מצטברת (CAGR)", fmt(result["annualized"], "pct"))
        st.caption(f"תאריכי החישוב בפועל: {result['start'].date()} עד {result['end'].date()}. הבסיס הוא יום המסחר האחרון ביום היעד או לפניו, עד 7 ימים קודם.")
        fig = go.Figure(go.Scatter(x=result["path"].index, y=result["path"].values,
                                  name="שווי ההשקעה", line=dict(color="#26a69a", width=2)))
        fig.update_layout(height=320, hovermode="x unified", yaxis_title=currency,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("הדמיה היסטורית לפי Adj Close עם התאמות לפיצולים ולדיבידנדים, כקירוב להשקעה מחדש. מניחה אפשרות לשברי מניות, ללא מסים, עמלות או שינויי מטבע. ביצועי עבר אינם מבטיחים תשואה עתידית.")


def metric_grid(items, columns=4):
    for offset in range(0, len(items), columns):
        for col, item in zip(st.columns(columns), items[offset:offset + columns]):
            label, value, kind, help_text = item
            col.metric(label, fmt(value, kind), help=help_text)
            explanation = contextual_explanation(label, value)
            if explanation:
                col.caption(explanation)


def score_ui(m, model, info):
    if info.get("quoteType") != "EQUITY":
        st.info("ניקוד עסקי מוצג רק לנכס ש־Yahoo זיהה כמניה. ל־ETF, קרן או סוג לא מזוהה מוצגים נתוני השוק הזמינים.")
        return
    if info.get("sector") in {"Financial Services", "Real Estate"}:
        st.info("הניקוד הכללי אינו מתאים לבנקים, חברות פיננסיות ונדל״ן, ולכן אינו מוצג. הנתונים עצמם זמינים בלשוניות.")
        return
    results, total, allowance, pe_base, ps_base = score_model(m, model["returns"], info.get("sector"), model["equity"])
    st.metric("ציון מחקר כולל / 100", fmt(total))
    st.caption("מודל כללים פנימי, ללא בדיקה היסטורית או תחזית תשואה. ציון גבוה אינו מעיד שהמניה זולה או מתאימה לך.")
    if total is None:
        st.info("אין כיסוי מספיק לציון כולל: נדרשים לפחות 75% מהמדדים בכל קטגוריה. נתון חסר אינו מקבל אפס.")
    for column, result in zip(st.columns(5), results):
        column.metric(result["title"], f"{result['score']:.1f}/20" if result["score"] is not None else NA)
        column.caption(f"כיסוי נתונים: {result['coverage']:.0%}")
    with st.expander("למה התקבל כל ציון? פירוט מלא של המדדים והמשקלים"):
        st.write("לכל קטגוריה משקל של 20 נקודות; בתוך הקטגוריה המדדים הזמינים שווים במשקלם. ציון חלקי מוצג מכיסוי של 60%. היעדר נתונים עלול להטות השוואות.")
        for result in results:
            st.markdown(f"**{result['title']}**")
            rows = [{"מדד": label, "ערך": fmt(value, kind),
                     "תרומה יחסית (0–100)": fmt(score * 100) if score is not None else NA}
                    for label, value, score, kind in result["features"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            known = [x for x in result["features"] if x[2] is not None]
            if known:
                best, worst = max(known, key=lambda x: x[2]), min(known, key=lambda x: x[2])
                st.write(f"התרומה הגבוהה ביותר: {best[0]} ({fmt(best[1], best[3])}); הנמוכה ביותר: {worst[0]} ({fmt(worst[1], worst[3])}).")
        st.write(f"בתמחור: בסיס P/E = {pe_base}, בסיס P/S = {ps_base}; מקדם צמיחת הכנסות = {allowance:.2f}. זו התאמה גסה לענף ולצמיחה, לא השוואה לחברות מתחרות.")
        st.write("ציונים רציפים בין ספים: איכות — רווח גולמי 10%–60%, תפעולי 0%–30%, נקי 0%–25%, ROE 0%–25%, תזרים חיובי 100 ושלילי/אפס 0. צמיחה — הכנסות ‎−10% עד 30%, רווחים ו־EPS ‎−10% עד 40%. חוסן — מזומן/חוב 0–2, חוב/הון 2–0, יחס שוטף 0.5–2, ותזרים חיובי.")
        st.write("תמחור — מכפיל חלקי (בסיס × מקדם צמיחה): ציון יורד מ־100 ל־0 בין 0.5 ל־2.5; PEG יורד בין 0.5 ל־3. בסיס P/E הוא 28 בטכנולוגיה/תקשורת/בריאות ו־22 באחרים; P/S הוא 6 בטכנולוגיה/תקשורת ו־3 באחרים. מקדם הצמיחה: 1 ועוד פעמיים צמיחת ההכנסות, מוגבלת לטווח 0%–40%.")
        st.write("מומנטום — מחיר/SMA50 בטווח 0.85–1.15, מחיר/SMA200 בטווח 0.8–1.2, תשואת חצי שנה ‎−20% עד 30%, שנה ‎−30% עד 50%. RSI מקבל ציונים 0,20,65,100,65,20 בנקודות 0,30,50,65,80,100 בהתאמה, וביניהן אינטרפולציה. אין כאן תחזית כיוון.")
        st.caption("ROE וחוב/הון אינם מדורגים אם ידוע שההון אינו חיובי. מכפיל רווח אינו מדורג לחברה הפסדית. תזרים שנתי ונתוני TTM אינם אותה תקופה; הניקוד הוא תמונת מחקר משולבת.")


def render_analysis(data):
    info = data["info"]
    model = derive(data)
    m = model["metrics"]
    if not info and data["history"].empty:
        st.error("לא התקבלו נתונים. בדוק שהטיקר נכון; ייתכן גם כשל חיבור או הגבלת בקשות ב־Yahoo. ניתן לנסות שוב בעוד מספר דקות.")
        return
    for warning in data["warnings"]:
        st.warning(warning)
    st.subheader(f"{info.get('longName') or info.get('shortName') or data['symbol']} · {data['symbol']}")
    currency = info.get("currency") or NA
    financial_currency = info.get("financialCurrency") or NA
    st.caption(f"מטבע מסחר: {currency} | מטבע דוחות: {financial_currency} | סוג נכס: {info.get('quoteType', NA)} | שליפה (UTC): {data['fetched'][:19].replace('T', ' ')}")
    stamp = safe_get(info, "regularMarketTime")
    if stamp is not None:
        try:
            st.caption("זמן ציטוט (UTC): " + datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d %H:%M"))
        except (ValueError, OSError, OverflowError):
            pass
    st.caption("מקור: Yahoo Finance באמצעות yfinance · מטמון 15 דקות · הציטוט עשוי להיות מושהה ואינו בהכרח בזמן אמת. סכומים: K אלפים, M מיליונים, B מיליארדים, T טריליונים.")
    if model["live_price"] is None and m["price"] is not None:
        st.info("המחיר המוצג הוא מחיר ההיסטוריה האחרון הזמין; ציטוט נוכחי לא התקבל.")
    if info.get("quoteType") in {"ETF", "MUTUALFUND"}:
        st.info("זהו נכס מסוג קרן. מכפילים שמופיעים עשויים לתאר את התיק; מדדי חברה ודוחות חסרים אינם מעידים על בעיה בקרן.")
    a, b, c, d = st.columns(4)
    a.metric(f"מחיר אחרון · {currency}", fmt(m["price"]), fmt(m["daily"], "pct") if m["daily"] is not None else None)
    b.metric(f"שווי שוק · {currency}", fmt(m["market_cap"], "money"))
    c.metric("מרחק משיא 52 שבועות", fmt(m["distance"], "pct"), help="מחיר חלקי שיא פחות 1; ערך שלילי מציין מרחק מתחת לשיא.")
    d.metric("P/E · מכפיל רווח", fmt(m["pe"]))
    st.write(f"ענף: {info.get('sector') or NA} | תעשייה: {info.get('industry') or NA}")
    st.caption(f"גבוה 52 שבועות: {fmt(m['high'])} | נמוך 52 שבועות: {fmt(m['low'])}")
    st.download_button("הורד דוח מחקר", data=export_report(data, model).encode("utf-8"),
                       file_name=f"{data['symbol']}-report-{data['fetched'][:10]}.md",
                       mime="text/markdown", on_click="ignore", key="report_" + data["symbol"])
    st.caption("הורדה כקובץ טקסט מעוצב (Markdown) עם נתונים, הסברים וניקוד. אפשר לפתוח בעורך טקסט; הגרפים והתרחישים אינם כלולים.")
    overview, fundamentals, tech, scoring, calculators = st.tabs(["במילים פשוטות", "נתונים פיננסיים", "גרף ותשואות", "ניקוד מחקר", "כלים ומחשבונים"])
    with calculators:
        tools_ui(data, model)
    with overview:
        simple, strengths, risks = observations(m)
        st.markdown("### במילים פשוטות")
        for text in simple or ["אין מספיק נתונים ליצירת הסבר פיננסי מבוסס."]:
            st.write("• " + text)
        left, right = st.columns(2)
        with left:
            st.markdown("#### חוזקות")
            for text in strengths:
                st.success(text)
            if not strengths:
                st.caption("לא זוהו חוזקות לפי הכללים והנתונים הזמינים.")
        with right:
            st.markdown("#### נקודות שדורשות תשומת לב")
            for text in risks:
                st.warning(text)
            if not risks:
                st.caption("לא זוהו דגלים לפי הכללים; אין בכך הוכחה להיעדר סיכון.")
        st.caption("הסברים אוטומטיים מבוססי תנאים בלבד. לא נבדקו חדשות, הנהלה, יתרון תחרותי או מגמות במרווחים לאורך זמן.")
    with fundamentals:
        st.markdown("### תמחור · Valuation")
        metric_grid([
            ("P/E", m["pe"], "num", "כמה משלמים עבור יחידת רווח ב־12 החודשים האחרונים. לחברה הפסדית לא מוצג מכפיל."),
            ("Forward P/E", m["forward_pe"], "num", "מחיר ביחס לרווח העתידי שמעריכים האנליסטים; זו תחזית."),
            ("PEG", m["peg"], "num", "מכפיל רווח ביחס לצמיחה. הגדרות התקופה עשויות להשתנות אצל הספק."),
            ("Price / Sales", m["ps"], "num", "מחיר החברה ביחס להכנסותיה, ללא התחשבות בהוצאות."),
            ("Price / Book", m["pb"], "num", "שווי שוק ביחס להון החשבונאי; לא מתאים לכל ענף."),
            (f"Enterprise Value · {currency}", m["ev"], "money", "שווי הפעילות לפי ספק הנתונים, הכולל התאמות לחוב ולמזומן."),
            ("EV / EBITDA", m["ev_ebitda"], "num", "שווי פעילות ביחס לרווח לפני ריבית, מסים, פחת והפחתות.")])
        st.markdown("### צמיחה · Growth")
        st.caption(f"נתוני ההכנסות והרווחיות בדרך כלל ל־12 חודשים (TTM); צמיחת הכנסות ורווחים מתייחסת לרבעון האחרון מול המקביל לפי Yahoo. מטבע דוחות: {financial_currency}; EPS במטבע שמדווח הספק.")
        metric_grid([
            ("Revenue · TTM", m["revenue"], "money", "סך המכירות בתקופה."),
            ("Revenue Growth", m["revenue_growth"], "pct", "השינוי בהכנסות מול התקופה המקבילה."),
            ("Earnings Growth", m["earnings_growth"], "pct", "השינוי ברווח מול התקופה המקבילה; בסיס קטן עלול לעוות אחוזים."),
            ("EPS · TTM", m["eps"], "num", "הרווח למניה ב־12 החודשים האחרונים."),
            ("Forward EPS", m["forward_eps"], "num", "רווח עתידי למניה לפי תחזית האנליסטים."),
            ("EPS Growth · שנתי", m["eps_growth"], "pct", "שינוי בין שתי שנות הדיווח האחרונות; לא מחושב מבסיס אפס או שלילי.")])
        rev = model["revenues"].tail(5)
        if not rev.empty:
            fig = go.Figure(go.Bar(x=rev.index.strftime("%Y-%m-%d"), y=rev.values, marker_color="#26a69a"))
            fig.update_layout(title="Revenue History · הכנסות שנתיות", xaxis_title="סיום שנת דיווח", yaxis_title=financial_currency, height=340)
            st.plotly_chart(fig, use_container_width=True)
            differences = rev.diff().dropna()
            trend = "עולות" if len(differences) >= 2 and (differences > 0).all() else "יורדות" if len(differences) >= 2 and (differences < 0).all() else "לא מציגות מגמה ברורה"
            st.write(f"לפי {len(rev)} שנות הדיווח הזמינות: ההכנסות {trend}.")
            if len(rev) < 4:
                st.caption("המקור לא סיפק ארבע שנות הכנסות מלאות.")
        else:
            st.info("היסטוריית הכנסות: " + NA)
        st.markdown("### רווחיות · Profitability")
        metric_grid([
            ("Gross Margin", m["gross"], "pct", "מכל 100 יחידות מכירות, כמה נשאר לאחר עלות המוצר."),
            ("Operating Margin", m["operating"], "pct", "כמה נשאר אחרי עלות המוצר והוצאות התפעול."),
            ("Profit Margin", m["profit"], "pct", "כמה נשאר כרווח נקי מכל 100 יחידות הכנסה."),
            ("EBITDA Margin", m["ebitda"], "pct", "שיעור הרווח לפני ריבית, מסים, פחת והפחתות."),
            ("ROE", m["roe"], "pct", "רווח ביחס להון העצמי; מינוף והון קטן יכולים להעלות את היחס."),
            ("ROA", m["roa"], "pct", "רווח ביחס לנכסים של החברה.")])
        st.markdown("### תזרים מזומנים · Cash Flow")
        st.caption(f"נתונים שנתיים · סיום תקופה: {model['cash_date'].date() if model['cash_date'] is not None else NA} · מטבע: {financial_currency}")
        metric_grid([
            ("Operating Cash Flow", m["ocf"], "money", "מזומן שנוצר מהפעילות השוטפת בתקופה."),
            ("Free Cash Flow", m["fcf"], "money", "הכסף שנותר מפעילות שוטפת לאחר השקעות בציוד ותשתיות."),
            ("CAPEX · הוצאה", abs(m["capex"]) if m["capex"] is not None else None, "money", "השקעות בציוד, מפעלים, תשתיות וטכנולוגיה; כאן ההוצאה מוצגת כמספר חיובי.")], 3)
        if m["fcf"] is not None:
            if m["fcf"] > 0:
                st.success("תזרים חופשי חיובי בתקופה המדווחת.")
            elif m["fcf"] < 0:
                st.warning("תזרים חופשי שלילי בתקופה המדווחת.")
        st.markdown("### מאזן · Balance Sheet")
        st.caption(f"תאריך מאזן: {model['balance_date'].date() if model['balance_date'] is not None else 'נתוני הסיכום האחרונים של Yahoo, ללא תאריך מאזן מאומת'} · מטבע: {financial_currency}")
        metric_grid([
            ("Total Cash", m["cash"], "money", "מזומן ושווי מזומן, כולל השקעות קצרות טווח אם המקור כולל אותן."),
            ("Total Debt", m["debt"], "money", "סך החוב הפיננסי, לא כל ההתחייבויות."),
            ("Net Cash / Debt", m["net_cash"], "money", "מזומן פחות חוב. שלילי פירושו חוב נטו."),
            ("Debt / Equity · פעמים", m["debt_equity"], "num", "חוב חלקי הון. 1.5 פירושו 150%; הון אפס/שלילי אינו בסיס תקף."),
            ("Current Ratio", m["current_ratio"], "num", "נכסים שוטפים חלקי התחייבויות שוטפות.")])
        if m["net_cash"] is not None:
            st.write("החברה מחזיקה יותר מזומן מחוב." if m["net_cash"] > 0 else "החברה מחזיקה יותר חוב ממזומן." if m["net_cash"] < 0 else "המזומן והחוב שווים.")
    with tech:
        technical = model["technical"]
        st.caption("החישובים הטכניים מבוססים על Close יומי מותאם לפיצולים, ללא התאמת דיבידנדים. הנר האחרון עשוי להיות יום מסחר שטרם הסתיים; מחיר הכותרת עשוי להיות מאוחר יותר.")
        if not technical.empty:
            st.caption(f"תאריך המחיר הטכני האחרון: {technical.index[-1].date()}")
        metric_grid([(label, m[key], "num", help_text) for label, key, help_text in [
            ("מחיר בחישוב הטכני", "technical_price", "מחיר אחרון בהיסטוריה היומית."),
            ("SMA50", "SMA50", "ממוצע 50 ימי מסחר."), ("SMA150", "SMA150", "ממוצע 150 ימי מסחר."),
            ("SMA200", "SMA200", "ממוצע 200 ימי מסחר."), ("RSI14", "RSI14", "מדד עוצמה לפי Wilder, ב־14 ימי מסחר.")]])
        if m["RSI14"] is not None:
            st.write("RSI: " + ("מכירת יתר" if m["RSI14"] < 30 else "קניית יתר" if m["RSI14"] > 70 else "אזור ניטרלי"))
        if m["technical_price"] is not None and m["SMA200"] is not None:
            st.write("מגמה ארוכת טווח חיובית — המחיר מעל SMA200." if m["technical_price"] > m["SMA200"] else "מגמה ארוכת טווח חלשה — המחיר מתחת ל־SMA200." if m["technical_price"] < m["SMA200"] else "המחיר שווה ל־SMA200.")
        st.caption("תיאור טכני בלבד; קניית יתר ומכירת יתר אינן הוראות פעולה.")
        period = st.radio("טווח גרף", ["6 חודשים", "1 שנה", "3 שנים", "5 שנים"], horizontal=True, key="chart_period")
        if not technical.empty:
            months = {"6 חודשים": 6, "1 שנה": 12, "3 שנים": 36, "5 שנים": 60}[period]
            plotted = technical.loc[technical.index[-1] - pd.DateOffset(months=months):]
            fig = go.Figure()
            for key, label, color in [("Close", "Price", "#26a69a"), ("SMA50", "SMA50", "#6a9de8"), ("SMA200", "SMA200", "#d9aa55")]:
                fig.add_trace(go.Scatter(x=plotted.index, y=plotted[key], name=label, line=dict(color=color, width=2)))
            fig.update_layout(height=440, hovermode="x unified", yaxis_title=currency, legend=dict(orientation="h"), margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("גרף: " + NA)
        st.markdown("### ביצועים · Performance")
        st.caption("תשואות מצטברות, לא שנתיות, לפי Adj Close המותאם לפיצולים ודיבידנדים. בסיס: יום המסחר האחרון ביום היעד או לפניו (עד 7 ימים קודם). YTD מתחיל בסגירת השנה הקודמת. אין השלמה מלאכותית להיסטוריה קצרה.")
        metric_grid([(label, model["returns"][key], "pct", "תשואה מצטברת לתאריך האחרון בהיסטוריה; אינה מבטיחה תשואה עתידית.") for key, label in [("1M", "חודש"), ("3M", "3 חודשים"), ("6M", "6 חודשים"), ("YTD", "מתחילת השנה"), ("1Y", "שנה"), ("3Y", "3 שנים"), ("5Y", "5 שנים")]])
    with scoring:
        score_ui(m, model, info)


def compare_ui():
    with st.form("compare_form"):
        raw = st.text_input("טיקרים להשוואה (עד 5, מופרדים ברווח או פסיק)", "TSLA NVDA GOOGL")
        submitted = st.form_submit_button("השווה מניות", width="stretch")
    if submitted:
        symbols = list(dict.fromkeys(re.split(r"[\s,;]+", raw.strip().upper())))
        if not symbols or len(symbols) > 5 or any(normalize_symbol(x) is None for x in symbols):
            st.error("הזן בין טיקר אחד לחמישה טיקרים תקינים.")
        else:
            rows = []
            with st.spinner("טוען נתונים להשוואה…"):
                for symbol in symbols:
                    data = load_data(symbol, detailed=False)
                    info = data["info"]
                    row = {"Ticker": symbol, "מטבע מסחר": info.get("currency", NA), "מטבע דוחות": info.get("financialCurrency", NA)}
                    fields = [("Price", "regularMarketPrice", "num"), ("Market Cap", "marketCap", "money"),
                              ("P/E", "trailingPE", "num"), ("Forward P/E", "forwardPE", "num"),
                              ("Revenue Growth", "revenueGrowth", "pct"), ("Gross Margin", "grossMargins", "pct"),
                              ("Profit Margin", "profitMargins", "pct"), ("FCF · Yahoo snapshot", "freeCashflow", "money"),
                              ("Debt/Equity · x", "debtToEquity", "num"), ("ROE", "returnOnEquity", "pct")]
                    for label, key, kind in fields:
                        value = safe_get(info, key)
                        if key == "regularMarketPrice":
                            value = first(safe_get(info, "currentPrice"), value)
                        if key == "debtToEquity" and value is not None:
                            value = value / 100 if value >= 0 else None
                        if key == "trailingPE" and safe_get(info, "trailingEps") is not None and safe_get(info, "trailingEps") <= 0:
                            value = None
                        if key in {"trailingPE", "forwardPE"} and value is not None and value <= 0:
                            value = None
                        row[label] = fmt(value, kind)
                    row["מצב"] = "התקבל מידע" if info else "לא התקבל מידע; בדוק טיקר או נסה מאוחר יותר"
                    row["שליפה UTC"] = data["fetched"][:19].replace("T", " ")
                    rows.append(row)
            st.session_state["comparison"] = rows
    if st.session_state.get("comparison"):
        st.dataframe(pd.DataFrame(st.session_state["comparison"]), hide_index=True, width="stretch")
        st.caption("FCF בהשוואה הוא נתון הסיכום של Yahoo, ועשוי להיות שונה מהתזרים השנתי בדוח הראשי. סכומים במטבעות שונים אינם בני השוואה ישירה. x = יחס בפעמים, לא באחוזים.")


def main():
    st.set_page_config(page_title="Investment Analyzer", page_icon="📈", layout="wide")
    st.markdown('''<style>
    .stApp {background:#0c1421; color:#edf2f7;}
    .block-container {max-width:1450px; padding-top:2.2rem;}
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stAlert"] {direction:rtl; text-align:right;}
    [data-testid="stMetric"] {background:#142033; border:1px solid #25334a; border-radius:12px; padding:16px;}
    [data-testid="stMetricValue"] {direction:ltr; text-align:left; font-size:1.7rem;}
    input {direction:ltr;}
    h1 {letter-spacing:0.035em;}
    .hero {direction:ltr; text-align:left; margin-bottom:1.6rem; border-bottom:1px solid #25334a; padding-bottom:1.5rem;}
    .hero p {color:#a1b0c3; direction:rtl; text-align:left;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="hero"><small>RESEARCH WORKSPACE / V1.1</small><h1>INVESTMENT ANALYZER</h1><p>נתונים. הקשר. הבנה.</p></div>', unsafe_allow_html=True)
    st.session_state.setdefault("watchlist", [])
    with st.sidebar:
        st.subheader("רשימת מעקב זמנית")
        with st.form("watch_form", clear_on_submit=True):
            watch = st.text_input("הוסף טיקר", placeholder="AMZN")
            add = st.form_submit_button("הוסף לרשימה")
        if add:
            symbol = normalize_symbol(watch)
            if symbol is None:
                st.error("הזן טיקר תקין.")
            elif symbol not in st.session_state.watchlist:
                if len(st.session_state.watchlist) < 20:
                    st.session_state.watchlist.append(symbol)
                else:
                    st.warning("ניתן לשמור עד 20 טיקרים במושב.")
        if st.session_state.watchlist:
            chosen = st.selectbox("בחר טיקר", st.session_state.watchlist)
            if st.button("נתח מהרשימה", width="stretch"):
                st.session_state.active_symbol = chosen
                st.session_state.ticker_input = chosen
            if st.button("הסר מהרשימה", width="stretch"):
                st.session_state.watchlist.remove(chosen)
                st.rerun()
        st.caption("הרשימה נשמרת במושב הנוכחי בלבד, ללא חשבון או מסד נתונים. רענון או ניתוק עשויים למחוק אותה.")
        if st.button("רענן נתונים מהמקור", width="stretch"):
            # Clear only this session's symbols; the underlying market cache is shared.
            symbols = {st.session_state.get("active_symbol")}
            symbols.update(row["Ticker"] for row in st.session_state.get("comparison", []))
            for symbol in symbols - {None}:
                load_data.clear(symbol)
                load_data.clear(symbol, detailed=False)
            st.session_state.pop("comparison", None)
            st.rerun()
    analysis, compare = st.tabs(["ניתוח מניה", "Compare Stocks · השוואה"])
    with analysis:
        st.caption("ניתוח מהיר בלחיצה")
        for column, (label, symbol) in zip(st.columns(5), [("Apple", "AAPL"), ("Nvidia", "NVDA"), ("Tesla", "TSLA"), ("Google", "GOOGL"), ("Amazon", "AMZN")]):
            if column.button(label, key="quick_" + symbol, width="stretch"):
                st.session_state.active_symbol = symbol
                st.session_state.ticker_input = symbol
        st.session_state.setdefault("ticker_input", st.session_state.get("active_symbol", "TSLA"))
        with st.form("analyzer"):
            raw = st.text_input("הכנס טיקר של מניה", key="ticker_input", placeholder="TSLA / NVDA / AAPL / GOOGL / AMZN")
            submitted = st.form_submit_button("נתח מניה", type="primary", width="stretch")
        if submitted:
            symbol = normalize_symbol(raw)
            if symbol is None:
                st.error("הטיקר אינו תקין. השתמש בסימול כמו TSLA או BRK-B.")
            else:
                st.session_state.active_symbol = symbol
        if st.session_state.get("active_symbol"):
            with st.spinner("אוסף נתונים ומחשב את הדוח…"):
                data = load_data(st.session_state.active_symbol)
            render_analysis(data)
        else:
            st.info("הכנס טיקר ולחץ על ״נתח מניה״. הטעינה הראשונה עשויה להימשך יותר זמן בהתאם לזמינות Yahoo.")
    with compare:
        compare_ui()
    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
