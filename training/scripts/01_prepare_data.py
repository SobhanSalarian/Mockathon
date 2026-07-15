#!/usr/bin/env python3
"""
Convert Australian financial datasets (AFR news + ASX prices + RBA rates)
into NeMo-ready instruction-following JSONL training pairs.

Data sources:
  - AFR Jasonl/          : Australian Financial Review articles (HEADLINE, TEXT, PUBLICATIONDATE)
  - ASX-18-companies/    : 18 ASX stocks OHLCV 2015-2021 (ticker, date, open/high/low/close/volume)
  - RBA-Rates-2010-2026/ : RBA cash rate decisions (Effective Date, Cash rate target%)

Usage:
    python 01_prepare_data.py \
        --afr_dir  "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/AFR Jasonl" \
        --asx_dir  "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/ASX-18-companies-2015-2021-Jasonl" \
        --rba_file "/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/RBA-Rates-2010-2026/RBA-rates.jsonl" \
        --out_dir  /home/cognitivo/deploy/hackathon-finagent/data
"""

import json, os, random, glob, re, argparse
from datetime import datetime, timedelta
from collections import defaultdict

# Map company name fragments → ASX ticker
COMPANY_TICKER = {
    "agl": "AGL.AX", "amp": "AMP.AX", "anz": "ANZ.AX",
    "aurizon": "AQR.AX", "bhp": "BHP.AX", "cba": "CBA.AX",
    "commonwealth bank": "CBA.AX", "cromwell": "CMW.AX",
    "gpt": "GPT.AX", "iag": "IAG.AX", "nab": "NAB.AX",
    "national australia": "NAB.AX", "qantas": "QAN.AX",
    "qbe": "QBE.AX", "rio tinto": "RIO.AX", "rio": "RIO.AX",
    "stockland": "SGP.AX", "suncorp": "SUN.AX",
    "tabcorp": "TAH.AX", "tpg": "TPG.AX", "transurban": "TCL.AX",
    "westpac": "WBC.AX", "westfield": "WFD.AX",
}

ALL_TICKERS = [
    "AGL.AX","AMP.AX","ANZ.AX","AQR.AX","BHP.AX","CBA.AX",
    "CMW.AX","GPT.AX","IAG.AX","NAB.AX","QAN.AX","QBE.AX",
    "RIO.AX","SGP.AX","SUN.AX","TAH.AX","TPG.AX","TCL.AX",
]


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_jsonl(path, encoding="utf-8-sig"):
    records = []
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def parse_afr_date(s):
    try:
        return datetime.strptime(str(s).strip(), "%Y%m%d").date()
    except Exception:
        return None


def parse_rba_date(s):
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except Exception:
            continue
    return None


# ─── Indexers ────────────────────────────────────────────────────────────────

def load_asx(asx_dir):
    """Returns prices[ticker][date] = {open,high,low,close,volume}"""
    prices = defaultdict(dict)
    for path in glob.glob(os.path.join(asx_dir, "*.jsonl")):
        for r in load_jsonl(path):
            ticker = r.get("ticker", "").upper()
            try:
                date = datetime.strptime(str(r["date"]), "%Y-%m-%d").date()
            except Exception:
                continue
            prices[ticker][date] = {
                "open":   float(r.get("open",   0)),
                "high":   float(r.get("high",   0)),
                "low":    float(r.get("low",    0)),
                "close":  float(r.get("close",  0)),
                "volume": float(r.get("volume", 0)),
            }
    return prices


def load_rba(rba_file):
    """Returns sorted list of (date, rate_pct) tuples"""
    rates = []
    for r in load_jsonl(rba_file):
        date = parse_rba_date(r.get("Effective Date", ""))
        rate = r.get("Cash rate target%", "")
        try:
            if date:
                rates.append((date, float(rate)))
        except Exception:
            continue
    return sorted(rates)


def get_rate_at(rates, date):
    best = None
    for d, r in rates:
        if d <= date:
            best = r
        else:
            break
    return best


def price_change(prices, ticker, from_date, days):
    """Return % price change from from_date to from_date + days"""
    target = from_date + timedelta(days=days)
    base = prices.get(ticker, {}).get(from_date)
    if not base or base["close"] == 0:
        return None
    for delta in range(0, 10):
        future = prices.get(ticker, {}).get(target + timedelta(days=delta))
        if future:
            return round((future["close"] - base["close"]) / base["close"] * 100, 2)
    return None


def find_ticker(text):
    """Detect if AFR article mentions a known company"""
    lower = text.lower()
    for kw, ticker in COMPANY_TICKER.items():
        if kw in lower:
            return ticker
    return None


# ─── Sample builders ─────────────────────────────────────────────────────────

def news_sentiment_sample(article, rate, prices):
    headline = article.get("HEADLINE", "").strip()
    text = (article.get("INTRO") or article.get("TEXT") or "").strip()[:600]
    date = parse_afr_date(article.get("PUBLICATIONDATE", ""))
    if not headline or not date:
        return None

    ticker = find_ticker(headline + " " + text)
    rate_str = f"{rate:.2f}%" if rate else "N/A"
    pnl_str = ""
    if ticker and date in prices.get(ticker, {}):
        pnl = price_change(prices, ticker, date, 30)
        if pnl is not None:
            pnl_str = f" {ticker} moved {pnl:+.1f}% over the next 30 days."

    prompt = (
        f"Date: {date}\n"
        f"RBA cash rate: {rate_str}\n"
        f"AFR Headline: {headline}\n"
        + (f"Article: {text}\n" if text else "") +
        f"\nAs an Australian financial analyst, assess the market sentiment and likely ASX impact."
    )
    direction = "cautious" if not pnl_str or "-" in pnl_str else "positive"
    response = (
        f"Sentiment Assessment ({date}):\n\n"
        f"The headline signals a {direction} market tone. "
        f"With the RBA cash rate at {rate_str}, "
        f"{'elevated rates add pressure on rate-sensitive sectors (banks, REITs, utilities).' if rate and rate > 3.0 else 'low rates support equity valuations and growth stocks.'}"
        f"{pnl_str}\n\n"
        f"Key risks to monitor: RBA guidance shifts, earnings revisions, global risk sentiment, and AUD moves."
    )
    return {"input": prompt, "output": response}


def price_analysis_sample(ticker, date, row, rate):
    close, high, low = row["close"], row["high"], row["low"]
    vol = row["volume"]
    spread = round((high - low) / close * 100, 2) if close else 0
    rate_str = f"{rate:.2f}%" if rate else "N/A"
    company = ticker.replace(".AX", "")

    prompt = (
        f"ASX daily data for {ticker} on {date}:\n"
        f"  Close: ${close:.2f} | High: ${high:.2f} | Low: ${low:.2f}\n"
        f"  Volume: {vol:,.0f} | Daily range: {spread}%\n"
        f"  RBA cash rate: {rate_str}\n\n"
        f"Provide a technical and macro assessment for {company}."
    )
    vol_comment = "above-average institutional activity" if vol > 2_000_000 else "normal retail volume"
    response = (
        f"Technical Assessment — {ticker} ({date}):\n\n"
        f"Daily range of {spread}% indicates {'elevated' if spread > 3 else 'contained'} volatility. "
        f"Volume of {vol:,.0f} suggests {vol_comment}.\n\n"
        f"Support: ${low:.2f} | Resistance: ${high:.2f} | Close: ${close:.2f}\n\n"
        f"Macro context: RBA at {rate_str}. "
        f"{'Rate headwinds compress margins for banks and REITs; watch NIM guidance.' if rate and rate > 3 else 'Accommodative policy favours yield stocks and leveraged balance sheets.'}"
    )
    return {"input": prompt, "output": response}


def rba_decision_sample(date, rate, prev_rate, next_rate=None):
    change = round(rate - prev_rate, 2) if prev_rate is not None else 0
    bps = round(abs(change) * 100)
    direction = "increased" if change > 0 else "decreased" if change < 0 else "held steady"
    move = f"{bps}bps {'hike' if change > 0 else 'cut'}" if change != 0 else "on hold"

    prompt = (
        f"The RBA {direction} the cash rate to {rate:.2f}% on {date} ({move}). "
        f"Analyse the implications for Australian equities, the AUD, and fixed income."
    )
    response = (
        f"RBA Decision Analysis — {date}:\n\n"
        f"**Decision:** Cash rate {direction} to {rate:.2f}% ({move})\n\n"
        f"**ASX impact:** "
        + ("Banks benefit from NIM expansion; growth stocks de-rate on higher discount rates. "
           "REITs and utilities face multiple compression." if change > 0 else
           "Equity re-rating tailwind. Banks face NIM compression; growth and REIT sectors benefit. "
           "Defensive yield stocks attract inflows." if change < 0 else
           "Markets interpret hold as neutral-to-dovish. Sector rotation toward quality yield.") +
        f"\n\n**AUD:** "
        + ("Strength likely on positive rate differential vs USD/EUR." if change > 0 else
           "Downward pressure; watch 0.70 USD support." if change < 0 else
           "Range-bound near-term.") +
        f"\n\n**Fixed income:** "
        + ("Bond yields rise; duration exposure should be reduced." if change > 0 else
           "Bond rally; extend duration to lock in yields." if change < 0 else
           "Yield curve stable; credit spreads unchanged.")
    )
    return {"input": prompt, "output": response}


def portfolio_sample(prices, rates, date):
    """Multi-stock portfolio question"""
    rate = get_rate_at(rates, date)
    tickers_today = [t for t in ALL_TICKERS if date in prices.get(t, {})]
    if len(tickers_today) < 3:
        return None
    selected = random.sample(tickers_today, 3)
    rows = {t: prices[t][date] for t in selected}
    rate_str = f"{rate:.2f}%" if rate else "N/A"

    holdings = "\n".join(
        f"  {t}: close ${rows[t]['close']:.2f}, range {round((rows[t]['high']-rows[t]['low'])/rows[t]['close']*100,1)}%"
        for t in selected
    )
    prompt = (
        f"Portfolio snapshot — {date} | RBA rate: {rate_str}\n{holdings}\n\n"
        f"Assess the risk profile of this 3-stock ASX portfolio and suggest a rebalancing action."
    )
    changes = {}
    for t in selected:
        pnl = price_change(prices, t, date, 21)
        changes[t] = pnl
    best = max(changes, key=lambda x: changes[x] or -99)
    response = (
        f"Portfolio Risk Assessment ({date}):\n\n"
        f"With RBA at {rate_str}, {'rate sensitivity is elevated — overweight defensive sectors' if rate and rate > 3 else 'low-rate environment favours growth and yield plays'}.\n\n"
        + "\n".join(f"- {t}: {f'{changes[t]:+.1f}% 21-day forward return' if changes[t] is not None else 'forward data unavailable'}" for t in selected) +
        f"\n\n**Rebalancing suggestion:** Trim {best} on strength, maintain diversification across sectors. "
        f"Consider adding fixed income allocation given current RBA posture."
    )
    return {"input": prompt, "output": response}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afr_dir",  default="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/AFR Jasonl")
    parser.add_argument("--asx_dir",  default="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/ASX-18-companies-2015-2021-Jasonl")
    parser.add_argument("--rba_file", default="/home/cognitivo/Desktop/HackathonDataset/Jsonl format DataSets/RBA-Rates-2010-2026/RBA-rates.jsonl")
    parser.add_argument("--out_dir",  default="/home/cognitivo/deploy/hackathon-finagent/data")
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading ASX price data...")
    prices = load_asx(args.asx_dir)
    print(f"  {len(prices)} tickers loaded")

    print("Loading RBA rates...")
    rates = load_rba(args.rba_file)
    print(f"  {len(rates)} rate decisions loaded")

    print("Loading AFR news...")
    afr_articles = []
    for path in sorted(glob.glob(os.path.join(args.afr_dir, "*.jsonl"))):
        afr_articles.extend(load_jsonl(path, encoding="utf-8"))
    print(f"  {len(afr_articles):,} articles loaded")

    samples = []

    # 1. News sentiment samples
    print("Building news sentiment samples...")
    for article in afr_articles:
        date = parse_afr_date(article.get("PUBLICATIONDATE", ""))
        rate = get_rate_at(rates, date) if date else None
        s = news_sentiment_sample(article, rate, prices)
        if s:
            samples.append(s)

    # 2. Price analysis samples (every 3rd row per ticker)
    print("Building price analysis samples...")
    for ticker, days in prices.items():
        day_list = sorted(days.items())
        for date, row in day_list[::3]:
            rate = get_rate_at(rates, date)
            samples.append(price_analysis_sample(ticker, date, row, rate))

    # 3. RBA decision samples
    print("Building RBA macro samples...")
    for i, (date, rate) in enumerate(rates):
        prev = rates[i-1][1] if i > 0 else None
        nxt  = rates[i+1][1] if i < len(rates)-1 else None
        samples.append(rba_decision_sample(date, rate, prev, nxt))

    # 4. Portfolio samples
    print("Building portfolio samples...")
    all_dates = sorted({d for t in prices.values() for d in t.keys()})
    for date in all_dates[::10]:
        s = portfolio_sample(prices, rates, date)
        if s:
            samples.append(s)

    # Shuffle and cap
    random.shuffle(samples)
    samples = [s for s in samples if s][:args.max_samples]
    n = len(samples)

    # 80/10/10 split
    splits = {
        "train": samples[:int(n * 0.8)],
        "val":   samples[int(n * 0.8):int(n * 0.9)],
        "test":  samples[int(n * 0.9):]
    }

    for split, data in splits.items():
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {split:6s}: {len(data):,} samples → {size_mb:.1f} MB")

    print(f"\nDone. Total: {n:,} training samples.")


if __name__ == "__main__":
    main()
