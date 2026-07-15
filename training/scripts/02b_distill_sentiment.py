#!/usr/bin/env python3
"""Distill AFR sentiment labels from the live brain (agent-brain via LiteLLM) into
SFT pairs for the domain-ft Nemotron LoRA.

  trained INPUT  = the exact prompt src/tools/sentiment_assess.py sends the model
  trained OUTPUT = a structured sentiment read produced by the 35B Qwen brain

Only finance-relevant AFR articles are labelled. Sentiment is one of
positive|negative|mixed, grounded in the article + the RBA rate in force.
"""
import argparse, glob, json, os, random, re, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

LITELLM_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "EMPTY")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "agent-brain")

FINANCE_KW = re.compile(
    r"\b(shares?|stocks?|asx|market|investors?|invest|interest rate|rates?|rba|banks?|"
    r"profit|earnings|dividend|economy|economic|inflation|bond|yield|equit|sector|listed|"
    r"ipo|takeover|merger|index|dollar|aud|mining|energy|retail|property|reit|qantas|bhp|"
    r"rio tinto|commonwealth bank|westpac|\bnab\b|\banz\b|\bamp\b|suncorp|insurance|"
    r"transurban|aurizon|stockland|cromwell)\b", re.I)

SYS = ("/no_think\n"
       "You are an Australian financial markets analyst. Read the AFR article and the RBA "
       "cash rate in force. Respond in EXACTLY this format and nothing else:\n"
       "Sentiment: <positive|negative|mixed>\n"
       "Direction: <one short phrase on the likely ASX impact>\n"
       "Assessment: <2 sentences grounded in the article and the rate; no invented numbers>")

_SENT_RE = re.compile(r"sentiment:\s*(positive|negative|mixed)", re.I)


def load_rba(path):
    rates = []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = str(r.get("Effective Date", "")).strip()
        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                rates.append((datetime.strptime(d, fmt).date(), float(r.get("Cash rate target%", ""))))
                break
            except Exception:
                pass
    return sorted(rates)


def rate_at(rates, d):
    best = None
    for dt, v in rates:
        if dt <= d:
            best = v
        else:
            break
    return best


def parse_afr_date(s):
    try:
        return datetime.strptime(str(s).strip(), "%Y%m%d").date()
    except Exception:
        return None


def agent_prompt(date, rate, headline, article):
    return (f"Date: {date}\nRBA cash rate: {rate:.2f}%\nAFR Headline: {headline}\n"
            f"Article: {article[:800]}\n\n"
            "As an Australian financial analyst, assess the market sentiment and likely ASX impact.")


def call_brain(user, max_tokens=200, retries=3):
    body = json.dumps({
        "model": BRAIN_MODEL, "temperature": 0.2, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        LITELLM_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"content-type": "application/json", "Authorization": f"Bearer {LITELLM_KEY}"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"].strip()
        except Exception:
            if i == retries - 1:
                return None
    return None


def label_article(art, rates):
    date = parse_afr_date(art.get("PUBLICATIONDATE", ""))
    if not date:
        return None
    rate = rate_at(rates, date)
    if rate is None:
        return None
    headline = (art.get("HEADLINE") or "").strip()
    article = (art.get("INTRO") or art.get("TEXT") or "").strip()
    if not headline:
        return None
    prompt = agent_prompt(date, rate, headline, article)
    out = call_brain(prompt)
    if not out:
        return None
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    m = _SENT_RE.search(out)
    if not m:
        return None
    return {"input": prompt, "output": out, "sentiment": m.group(1).lower(), "date": str(date)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afr_dir", required=True)
    ap.add_argument("--rba_file", required=True)
    ap.add_argument("--out", default="data/sentiment_distilled.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    rates = load_rba(a.rba_file)
    print(f"RBA rates loaded: {len(rates)}", flush=True)

    arts = []
    for p in sorted(glob.glob(os.path.join(a.afr_dir, "*.jsonl"))):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("HEADLINE", "") + " " + (r.get("INTRO") or ""))
            if parse_afr_date(r.get("PUBLICATIONDATE", "")) and FINANCE_KW.search(text):
                arts.append(r)
    print(f"finance-relevant articles: {len(arts):,}", flush=True)

    random.shuffle(arts)
    arts = arts[:a.n]
    print(f"labelling {len(arts)} articles with {a.workers} workers...", flush=True)

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(label_article, art, rates) for art in arts]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r:
                results.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(arts)} ({len(results)} ok)", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} pairs -> {a.out}", flush=True)
    print(f"sentiment distribution: {dict(Counter(r['sentiment'] for r in results))}", flush=True)


if __name__ == "__main__":
    main()
