"""
finance_engine.py
-----------------
Intelligence behind the Personal Finance Management Agent (Option 3).

* Expense categorisation  - TF-IDF + Logistic Regression text classifier trained on seed merchant phrases,
                            retrained instantly when the user corrects a category (online learning).
* Pattern recognition     - weekday profile, recurring payments, small-purchase leakage, spending trend,
                            robust outlier / duplicate detection.
* Excess detection        - category budgets from the 50/30/20 guideline compared with the user's income.
* Prediction              - Ridge regression on trend + day-of-week for variable spending, plus recurring
                            monthly bills; the better of model and moving average is chosen by back-testing.
* Recommendations         - ranked, quantified saving strategies and a savings-goal planner.

The seed phrases include Kenyan merchants and services (matatu, boda boda, KPLC tokens, M-Pesa, Naivas ...)
but the amounts are currency-agnostic.
"""
from __future__ import annotations

import calendar
import datetime as dt
import html as _html
import io
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

SAV = "Savings & Investments"
GROUP = {  # 50/30/20 guideline groups
    "Food & Groceries": "Needs", "Transport": "Needs", "Rent & Housing": "Needs", "Utilities": "Needs",
    "Airtime & Data": "Needs", "Health": "Needs", "Education": "Needs", "Family & Giving": "Needs",
    "Eating Out": "Wants", "Entertainment": "Wants", "Subscriptions": "Wants", "Shopping": "Wants",
    "Personal Care": "Wants", "Other": "Wants", SAV: "Savings",
}
CATEGORIES = list(GROUP)
CAPS = {  # sensible ceilings as % of monthly income
    "Food & Groceries": 20, "Eating Out": 6, "Transport": 10, "Rent & Housing": 30, "Utilities": 6, "Airtime & Data": 3,
    "Health": 5, "Education": 12, "Family & Giving": 10, "Entertainment": 5, "Subscriptions": 3, "Shopping": 7,
    "Personal Care": 3, "Other": 5,
}
LUMPY = {"Rent & Housing", "Education", "Subscriptions"}      # forecast as monthly bills, not daily noise
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH = 30.4

SEED = {
    "Food & Groceries": ["naivas supermarket", "quickmart groceries", "carrefour shopping food", "mama mboga vegetables",
                         "maize flour and sugar", "milk and bread", "sukuma wiki and tomatoes", "eggs and rice", "market vegetables",
                         "groceries", "butchery meat", "cooking oil", "fruits at the market", "supermarket", "unga"],
    "Eating Out": ["kfc", "java house coffee", "chicken inn", "lunch at cafeteria", "pizza inn", "coffee and snack",
                   "mandazi and tea", "restaurant dinner", "glovo food order", "uber eats", "burger and fries",
                   "lunch", "dinner with friends", "breakfast cafe", "chips and sausage"],
    "Transport": ["matatu fare", "uber ride", "bolt ride", "boda boda", "fuel petrol", "parking fee", "sgr train ticket",
                  "bus fare to nairobi", "little cab", "taxi", "diesel", "fare to town", "car service and tyres", "bus ticket"],
    "Rent & Housing": ["house rent", "monthly rent", "rent payment", "landlord payment", "hostel rent", "rent for november",
                       "apartment rent", "room rent", "housing deposit", "service charge for the house"],
    "Utilities": ["kplc tokens", "electricity bill", "water bill", "nairobi water", "gas refill", "cooking gas",
                  "garbage collection", "wifi bill", "zuku internet", "safaricom fibre", "prepaid electricity", "power tokens"],
    "Airtime & Data": ["airtime", "safaricom airtime", "data bundles", "airtel airtime", "buy bundles", "mpesa airtime",
                       "internet bundles", "weekly data bundle", "okoa jahazi", "top up airtime"],
    "Health": ["pharmacy", "hospital bill", "clinic visit", "doctor consultation", "medicine", "sha contribution",
               "dental check up", "lab test", "prescription drugs", "eye test", "health insurance"],
    "Education": ["school fees", "textbooks", "stationery", "tuition", "exam fee", "printing and photocopy", "course fee",
                  "udemy course", "university fees", "notebooks and pens", "project binding", "student fees"],
    "Family & Giving": ["sent to mum", "church offering", "tithe", "harambee contribution", "gift for a friend",
                        "support to parents", "wedding contribution", "donation", "sent to brother", "funeral contribution",
                        "pocket money for sister", "birthday gift"],
    "Entertainment": ["cinema ticket", "movie night", "concert ticket", "sportybet stake", "betting", "club entry",
                      "game credits", "pool and drinks", "outing with friends", "bar tab", "betway", "playstation games"],
    "Subscriptions": ["netflix", "showmax", "spotify", "youtube premium", "dstv subscription", "icloud storage",
                      "gym membership", "canva pro", "monthly subscription", "amazon prime", "apple music", "software subscription"],
    "Shopping": ["clothes", "shoes", "jumia order", "amazon order", "electronics", "phone case", "mall shopping",
                 "furniture", "kitenge fabric", "jeans and shirt", "headphones", "online shopping", "bag", "watch"],
    "Personal Care": ["salon", "barber haircut", "haircut", "cosmetics", "body lotion", "deodorant", "spa", "manicure",
                      "toiletries", "soap and toothpaste", "braids", "perfume"],
    SAV: ["sacco contribution", "savings deposit", "money market fund", "mmf deposit", "investment", "chama contribution",
          "emergency fund", "mshwari savings", "treasury bills", "buy stocks", "pension top up", "savings"],
    "Other": ["miscellaneous", "mpesa withdrawal charges", "bank charges", "unknown payment", "misc", "other expenses",
              "ledger fee", "penalty", "lost item replacement"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(s).lower())).strip()


# --------------------------------------------------------------------------
# Expense categorisation (text classifier that can learn)
# --------------------------------------------------------------------------
class Categoriser:
    """db: remembers corrections between visits. gemini: optional, resolves descriptions the model is unsure about."""
    ASSIST_BELOW = 0.45          # ask Gemini only when the model's confidence is under this
    ASSIST_BUDGET = 20           # at most this many Gemini lookups per batch

    def __init__(self, db=None, gemini=None):
        self.db = db
        self.gemini = gemini
        self.assist_left = self.ASSIST_BUDGET
        self.assisted = 0
        self.extra: list[tuple[str, str]] = []
        if db:
            for desc, c in db.fin_corrections():
                self.extra.extend([(desc, c)] * 3)
        self.train()

    def _pipe(self):
        f = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
        return Pipeline([("f", f), ("c", LogisticRegression(C=30, max_iter=2000))])

    def _data(self):
        X, y = [], []
        for cat, items in SEED.items():
            for it in items:
                X.append(_norm(it)); y.append(cat)
        for t, c in self.extra:
            X.append(_norm(t)); y.append(c)
        return X, y

    def train(self):
        X, y = self._data()
        self.model = self._pipe().fit(X, y)

    def predict(self, desc: str) -> tuple[str, float]:
        p = self.model.predict_proba([_norm(desc)])[0]
        i = int(np.argmax(p))
        cat, conf = str(self.model.classes_[i]), float(p[i])
        if conf < self.ASSIST_BELOW and self.gemini is not None and self.assist_left > 0 and self.gemini.available():
            got = self._ask_gemini(desc)
            if got:
                return got, 0.6
        return cat, conf

    def _ask_gemini(self, desc: str):
        """Gemini picks one of OUR categories for a description the model does not recognise. Anything else is ignored."""
        cats = list(self.model.classes_)
        self.assist_left -= 1
        r = self.gemini.generate(
            "Classify this Kenyan personal-finance transaction description into exactly one of these categories: "
            + ", ".join(cats) + f".\nDescription: {desc}\nReply with the category name only.",
            temperature=0.0, cache_key="fincat:" + desc.lower())
        if r["ok"]:
            ans = r["text"].strip().strip(".").strip()
            for c in cats:
                if ans.lower() == str(c).lower():
                    self.assisted += 1
                    return str(c)
        return None

    def predict_many(self, descs) -> list[tuple[str, float]]:
        return [self.predict(d) for d in descs]

    def learn(self, desc: str, category: str):
        if self.db:
            self.db.fin_add_correction(desc, category)
        self.extra.append((desc, category))
        self.extra.extend([(desc, category)] * 2)          # corrections count more than seed data
        self.train()

    def evaluate(self) -> float:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        X, y = self._data()
        return float(cross_val_score(self._pipe(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1)).mean())


COLS = ["date", "description", "amount", "category", "confidence", "manual"]


def empty_tx() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "description": pd.Series(dtype="object"),
                         "amount": pd.Series(dtype="float"), "category": pd.Series(dtype="object"),
                         "confidence": pd.Series(dtype="float"), "manual": pd.Series(dtype="bool")})


def add_transactions(df: pd.DataFrame, rows: list[tuple], cat: Categoriser) -> pd.DataFrame:
    """rows: (date, description, amount[, category]). Missing categories are predicted automatically."""
    cat.assist_left = cat.ASSIST_BUDGET
    new = []
    for r in rows:
        d, desc, amt = pd.Timestamp(r[0]).normalize(), str(r[1]).strip(), float(r[2])
        if len(r) > 3 and r[3]:
            c, conf, manual = r[3], 1.0, True
        else:
            c, conf = cat.predict(desc); manual = False
        new.append({"date": d, "description": desc, "amount": abs(amt), "category": c, "confidence": conf, "manual": manual})
    out = pd.concat([df, pd.DataFrame(new)], ignore_index=True) if new else df
    return out.sort_values("date").reset_index(drop=True)


def parse_text(text: str, today: dt.date | None = None) -> tuple[list[tuple], list[str]]:
    """Parse pasted lines like '2026-11-03, Naivas, 1450' or 'Matatu fare, 100'."""
    today = today or dt.date.today()
    rows, errors = [], []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.lower().startswith("date"):
            continue
        parts = [p.strip() for p in line.split(",")]
        while len(parts) >= 2 and re.fullmatch(r"\d{3}(\.\d+)?", parts[-1]) and re.fullmatch(r"\d{1,3}", parts[-2]):
            parts[-2:] = [parts[-2] + parts[-1]]            # '1,450' -> '1450'
        try:
            amt = float(re.sub(r"[^\d.]", "", parts[-1]))
            first = pd.to_datetime(parts[0], errors="coerce") if len(parts) >= 3 else pd.NaT
            if pd.isna(first):
                d, desc = today, ", ".join(parts[:-1])
            else:
                d, desc = first, ", ".join(parts[1:-1])
            if not desc or amt <= 0:
                raise ValueError
            rows.append((d, desc, amt))
        except Exception:
            errors.append(f"Line {ln}: could not read '{line}'")
    return rows, errors


# --------------------------------------------------------------------------
# Income helpers
# --------------------------------------------------------------------------
def monthly_income(sources: list[dict]) -> float:
    f = {"Monthly": 1.0, "Weekly": 52 / 12, "Yearly": 1 / 12}
    return float(sum(s["amount"] * f.get(s["frequency"], 1.0) for s in sources))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _daily(spend: pd.DataFrame, first, last) -> pd.Series:
    idx = pd.date_range(first, last)
    return spend.groupby("date")["amount"].sum().reindex(idx, fill_value=0.0)


def patterns(df: pd.DataFrame) -> dict:
    spend = df[df["category"] != SAV]
    first, last = df["date"].min(), df["date"].max()
    days = (last - first).days + 1
    var = spend[~spend["category"].isin(LUMPY)]
    daily = _daily(var, first, last)
    dow = daily.groupby(daily.index.dayofweek).mean().reindex(range(7), fill_value=0.0)
    wk_end = dow.iloc[5:].mean(); wk_day = dow.iloc[:5].mean()

    # recurring payments
    rec = []
    tmp = spend.copy()
    tmp["key"] = tmp["description"].map(lambda s: " ".join(re.findall(r"[a-z]+", s.lower())[:2]))
    for key, g in tmp.groupby("key"):
        if len(g) < 3 or not key:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        med = float(gaps.median())
        cv = g["amount"].std() / g["amount"].mean() if g["amount"].mean() else 1
        if 26 <= med <= 35 and cv < 0.35:
            freq = "Monthly"
        elif 6 <= med <= 8 and cv < 0.35:
            freq = "Weekly"
        elif med <= 4 and len(g) >= 6 and cv < 0.5:
            freq = "Every few days"
        else:
            continue
        typical = float(g["amount"].median())
        rec.append({"Item": g["description"].iloc[-1], "Frequency": freq, "Typical amount": typical,
                    "Per month": float(g["amount"].sum() * MONTH / days), "Category": g["category"].mode().iat[0]})
    rec = pd.DataFrame(rec).sort_values("Per month", ascending=False) if rec else pd.DataFrame(
        columns=["Item", "Frequency", "Typical amount", "Per month", "Category"])

    # small purchases that add up
    leak = {"n": 0, "share": 0.0, "monthly": 0.0}
    if len(var) >= 20:
        thr = float(var["amount"].quantile(0.35))
        small = var[var["amount"] <= thr]
        leak = {"n": len(small), "threshold": thr, "share": float(small["amount"].sum() / max(spend["amount"].sum(), 1)),
                "monthly": float(small["amount"].sum() * MONTH / days)}

    # trend: later half vs earlier half of variable spending
    trend = 0.0
    if days >= 28:
        mid = first + (last - first) / 2
        a = daily[daily.index <= mid].mean(); b = daily[daily.index > mid].mean()
        trend = float((b - a) / a) if a > 0 else 0.0

    # outliers: compare each purchase with the same merchant's usual amount (robust median / MAD)
    out = []
    tmp["merchant"] = tmp["key"]
    for key, g in tmp.groupby("merchant"):
        if len(g) < 5 or not key:
            continue
        med = g["amount"].median(); mad = np.median(np.abs(g["amount"] - med)) * 1.4826
        mad = mad if mad > 0 else max(g["amount"].std(), 0.05 * med, 1e-9)
        z = (g["amount"] - med) / mad
        for _, r in g[(z > 3.5) & (g["amount"] > 1.5 * med)].iterrows():
            out.append({"Date": r["date"].date().isoformat(), "Description": r["description"], "Amount": r["amount"],
                        "Category": r["category"], "Why": f"{r['amount'] / med:.1f}x your usual {key} amount"})
    # one-off merchants that are far above the rest of their category
    counts = tmp["merchant"].value_counts()
    for cat, g in spend.groupby("category"):
        if len(g) < 6:
            continue
        q75 = g["amount"].quantile(0.75)
        for _, r in tmp[(tmp["category"] == cat) & (tmp["amount"] > 2 * q75)].iterrows():
            if counts[r["merchant"]] <= 2 and not any(o["Description"] == r["description"] and o["Date"] == r["date"].date().isoformat() for o in out):
                out.append({"Date": r["date"].date().isoformat(), "Description": r["description"], "Amount": r["amount"],
                            "Category": cat, "Why": f"much larger than most {cat.lower()} spending"})
    dup = spend[spend.duplicated(["date", "description", "amount"], keep="first") & (spend["amount"] > 0)]
    for _, r in dup.iterrows():
        out.append({"Date": r["date"].date().isoformat(), "Description": r["description"], "Amount": r["amount"],
                    "Category": r["category"], "Why": "possible duplicate charge (same day, name and amount)"})
    return {"dow": dow, "weekend_ratio": float(wk_end / wk_day) if wk_day else 1.0, "peak_day": DAYS[int(dow.idxmax())],
            "recurring": rec, "leak": leak, "trend": trend, "unusual": pd.DataFrame(out), "days": days}


def analyse(df: pd.DataFrame, income_m: float, current_savings: float = 0.0) -> dict:
    spend = df[df["category"] != SAV]
    saved_tx = df[df["category"] == SAV]["amount"].sum()
    first, last = df["date"].min(), df["date"].max()
    days = (last - first).days + 1
    k = MONTH / days
    by_cat = spend.groupby("category")["amount"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    by_cat["monthly"] = by_cat["sum"] * k
    by_cat["share"] = by_cat["sum"] / by_cat["sum"].sum()
    by_cat["group"] = [GROUP[c] for c in by_cat.index]
    spend_m = float(spend["amount"].sum() * k)
    saved_m = float(saved_tx * k)
    surplus_m = income_m - spend_m
    groups = by_cat.groupby("group")["monthly"].sum().to_dict()
    groups[ "Savings"] = max(surplus_m, 0) + saved_m if income_m else saved_m
    pct = {g: (v / income_m * 100 if income_m else 0) for g, v in groups.items()}

    excess = []
    for c, r in by_cat.iterrows():
        cap = CAPS.get(c)
        if cap and income_m:
            cap_amt = cap / 100 * income_m
            if r["monthly"] > cap_amt * 1.05:
                excess.append({"Category": c, "Type": GROUP[c], "Monthly": r["monthly"], "Guideline": cap_amt,
                               "Over by": r["monthly"] - cap_amt, "pct_income": r["monthly"] / income_m * 100})
    excess = pd.DataFrame(excess).sort_values("Over by", ascending=False) if excess else pd.DataFrame(
        columns=["Category", "Type", "Monthly", "Guideline", "Over by", "pct_income"])

    m = spend.copy(); m["month"] = m["date"].dt.to_period("M")
    monthly = m.groupby("month")["amount"].sum()
    return {"spend": spend, "by_cat": by_cat, "spend_m": spend_m, "saved_m": saved_m, "income_m": income_m,
            "surplus_m": surplus_m, "savings_rate": surplus_m / income_m if income_m else 0.0, "groups": groups,
            "pct": pct, "excess": excess, "days": days, "monthly": monthly, "first": first, "last": last,
            "current_savings": current_savings, "pat": patterns(df), "k": k}


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
def _features(dates: pd.DatetimeIndex, first, scale: float) -> np.ndarray:
    t = ((dates - first).days.values / scale)[:, None]
    dow = np.eye(7)[dates.dayofweek]
    return np.hstack([t, dow])


def forecast(df: pd.DataFrame, horizon: int = 30) -> dict:
    spend = df[df["category"] != SAV]
    first, last = df["date"].min(), df["date"].max()
    days = (last - first).days + 1
    var = spend[~spend["category"].isin(LUMPY)]
    daily = _daily(var, first, last)
    n = len(daily)
    scale = max(n, 1)

    def fit_predict(train: pd.Series, dates):
        m = Ridge(alpha=2.0).fit(_features(train.index, first, scale), train.values)
        return np.clip(m.predict(_features(dates, first, scale)), 0, None), m

    model_name, mae_m, mae_b = "Moving average", None, None
    if n >= 28:
        tr, te = daily.iloc[:-7], daily.iloc[-7:]
        pm, _ = fit_predict(tr, te.index)
        pb = np.full(7, tr.iloc[-14:].mean())
        mae_m, mae_b = float(np.mean(np.abs(pm - te.values))), float(np.mean(np.abs(pb - te.values)))
        if mae_m < mae_b:
            model_name = "Trend + weekday regression"
    use_model = model_name != "Moving average"

    fdates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    if use_model:
        pred, mdl = fit_predict(daily, fdates)
        resid = daily.values - np.clip(mdl.predict(_features(daily.index, first, scale)), 0, None)
    else:
        base = float(daily.iloc[-14:].mean()) if n else 0.0
        pred = np.full(horizon, base)
        resid = daily.values - base
    sd = float(np.std(resid)) if n > 2 else 0.0
    lower, upper = np.clip(pred - 1.28 * sd, 0, None), pred + 1.28 * sd

    lump_cat = spend[spend["category"].isin(LUMPY)].groupby("category")["amount"].sum() * MONTH / days
    lump_total = float(lump_cat.sum() * horizon / MONTH)
    var_total = float(pred.sum())
    total = var_total + lump_total
    margin = 1.28 * sd * np.sqrt(horizon)

    # per-category forecast, scaled to match the model total
    ew = {}
    for c, g in var.groupby("category"):
        s = _daily(g, first, last).ewm(halflife=14).mean().iloc[-1] * horizon
        ew[c] = float(s)
    scale_f = var_total / sum(ew.values()) if sum(ew.values()) else 1.0
    per_cat = {c: v * scale_f for c, v in ew.items()}
    for c, v in lump_cat.items():
        per_cat[c] = float(v * horizon / MONTH)

    hist = daily.rolling(7, min_periods=1).mean()
    joined = pd.concat([daily, pd.Series(pred, index=fdates)]).rolling(7, min_periods=1).mean()
    smooth_fc = joined.iloc[-horizon:].values              # forecast line continuous with the history
    band = 1.28 * sd / np.sqrt(7)                          # uncertainty of a 7-day average
    return {"smooth_fc": smooth_fc, "smooth_low": np.clip(smooth_fc - band, 0, None), "smooth_high": smooth_fc + band,
            "dates": fdates, "pred": pred, "lower": lower, "upper": upper, "total": total,
            "low": max(total - margin, 0), "high": total + margin, "model": model_name, "mae_model": mae_m,
            "mae_base": mae_b, "per_cat": per_cat, "hist": hist, "horizon": horizon, "lump": lump_total,
            "enough_data": days >= 14}


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
def recommend(a: dict, f: dict) -> list[dict]:
    recs: list[dict] = []
    inc, pat = a["income_m"], a["pat"]
    spend_m, surplus = a["spend_m"], a["surplus_m"]

    def add(title, detail, saving, priority):
        recs.append({"title": title, "detail": detail, "saving": float(max(saving, 0)), "priority": priority})

    if inc and surplus < 0:
        add("You are spending more than you earn",
            f"Spending is about {spend_m:,.0f} a month against an income of {inc:,.0f}. Start with the largest "
            "discretionary categories below until the gap closes.", -surplus, "High")
    elif inc and a["savings_rate"] < 0.20:
        gap = 0.20 * inc - max(surplus, 0)
        add("Raise your savings rate toward 20%",
            f"You currently keep {a['savings_rate'] * 100:.0f}% of your income. Moving {gap:,.0f} a month into savings "
            "reaches the 20% guideline.", gap, "High" if a["savings_rate"] < 0.10 else "Medium")

    for _, r in a["excess"].iterrows():
        discretionary = r["Type"] == "Wants"
        cut = r["Over by"] * (1.0 if discretionary else 0.5)
        add(f"Bring {r['Category']} back toward its guideline",
            f"{r['Category']} takes {r['pct_income']:.0f}% of income ({r['Monthly']:,.0f} a month) against a guideline of "
            f"about {CAPS[r['Category']]}%. " + ("Set a monthly limit and track it weekly." if discretionary else
                                                  "This is an essential, so aim to trim part of the excess (for example by comparing prices or bulk buying)."),
            cut, "High" if discretionary and r["Over by"] > 0.03 * inc else "Medium")

    rec = pat["recurring"]
    subs = rec[rec["Category"] == "Subscriptions"] if len(rec) else rec
    if len(subs):
        add("Review recurring subscriptions",
            "Recurring charges found: " + ", ".join(f"{r['Item']} ({r['Per month']:,.0f}/month)" for _, r in subs.iterrows()) +
            ". Cancel any you did not use in the last month.", float(subs["Per month"].sum()) * 0.5, "Medium")

    lk = pat["leak"]
    if lk["n"] >= 20 and lk["share"] >= 0.10:
        add("Small purchases are adding up",
            f"{lk['n']} purchases of {lk['threshold']:,.0f} or less make up {lk['share'] * 100:.0f}% of your spending "
            f"(about {lk['monthly']:,.0f} a month). Plan these in a weekly cash or wallet limit.", lk["monthly"] * 0.25, "Medium")

    if pat["weekend_ratio"] >= 1.4:
        add(f"Weekend spending is {pat['weekend_ratio']:.1f}x your weekday level",
            "Set a separate weekend budget and decide in advance what you will spend on outings and food.",
            0.1 * spend_m * 0.3, "Low")

    if pat["trend"] > 0.10:
        add("Your daily spending is rising",
            f"Variable spending is up {pat['trend'] * 100:.0f}% in the second half of your records. Find which category is "
            "growing and cap it before it becomes the new normal.", 0.5 * pat["trend"] * spend_m * 0.3, "Medium")

    if len(pat["unusual"]):
        add("Check unusual transactions",
            f"{len(pat['unusual'])} transaction(s) look unusually large or duplicated. See the Patterns tab.",
            float(pat["unusual"]["Amount"].sum()) * 0.1, "Low")

    essentials = float(a["by_cat"][a["by_cat"]["group"] == "Needs"]["monthly"].sum())
    if essentials:
        target = essentials * 3
        have = a["current_savings"]
        months = (target - have) / surplus if surplus > 0 and target > have else None
        txt = (f"Aim for three months of essentials, about {target:,.0f}. You have {have:,.0f}. "
               + ("You have reached this target." if have >= target else
                  f"At your current surplus that takes about {months:.0f} months." if months else
                  "Free up a monthly surplus first."))
        add("Build an emergency fund", txt, 0.0, "Medium" if have < target else "Low")

    add("Pay yourself first",
        "On payday move your savings amount to a separate account or money market fund before spending anything. "
        "Automating it is more reliable than saving what is left.", 0.0, "Low")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda r: (order[r["priority"]], -r["saving"]))


def goal_plan(goal: float, months: int, saved: float, surplus_m: float, excess: pd.DataFrame) -> dict:
    need = max(goal - saved, 0)
    per_month = need / months if months else need
    gap = per_month - max(surplus_m, 0)
    cuts = []
    if gap > 0 and len(excess):
        left = gap
        for _, r in excess[excess["Type"] == "Wants"].iterrows():
            take = min(r["Over by"], left)
            if take > 0:
                cuts.append((r["Category"], take)); left -= take
            if left <= 0:
                break
    covered = sum(c[1] for c in cuts)
    return {"need": need, "per_month": per_month, "feasible": gap <= 0, "gap": max(gap, 0), "cuts": cuts,
            "remaining_gap": max(gap - covered, 0), "months_at_surplus": (need / surplus_m) if surplus_m > 0 else None}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report_html(owner: str, cur: str, a: dict, f: dict, recs: list[dict]) -> str:
    e = _html.escape
    m = lambda v: f"{cur} {v:,.0f}"
    today = dt.date.today().strftime("%d %B %Y")
    cat_rows = "".join(f"<tr><td>{e(c)}</td><td>{e(r['group'])}</td><td class=n>{m(r['monthly'])}</td>"
                       f"<td class=n>{r['share'] * 100:.0f}%</td></tr>" for c, r in a["by_cat"].iterrows())
    fc_rows = "".join(f"<tr><td>{e(c)}</td><td class=n>{m(v)}</td></tr>"
                      for c, v in sorted(f["per_cat"].items(), key=lambda x: -x[1]))
    rec_html = "".join(f"<div class=card><b>{e(r['title'])}</b> <small>({r['priority']} priority"
                       + (f", up to {m(r['saving'])} a month" if r["saving"] else "") + f")</small><p>{e(r['detail'])}</p></div>"
                       for r in recs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Financial summary</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:30px auto;padding:0 18px;color:#0b0b0b}}
h1{{color:#2a78d6;margin-bottom:0}} h2{{border-bottom:2px solid #d8d7d2;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}} th,td{{border:1px solid #d8d7d2;padding:7px 10px;text-align:left}}
th{{background:#f1f1ee}} td.n{{text-align:right}} .kpi{{display:flex;gap:12px;margin:14px 0}}
.kpi div{{flex:1;background:#2a78d6;color:#fff;padding:12px;border-radius:6px;text-align:center}} .kpi b{{display:block;font-size:22px}}
.card{{border:1px solid #d8d7d2;border-left:5px solid #2a78d6;padding:2px 14px;margin:10px 0;border-radius:6px}} small{{color:#52514e}}
</style></head><body><h1>Personal Financial Summary</h1><small>{e(owner or 'Prepared for you')} | generated {today} |
records from {a['first'].date()} to {a['last'].date()}</small>
<div class=kpi><div><b>{m(a['income_m'])}</b>Monthly income</div><div><b>{m(a['spend_m'])}</b>Monthly spending</div>
<div><b>{a['savings_rate'] * 100:.0f}%</b>Savings rate</div><div><b>{m(f['total'])}</b>Predicted next 30 days</div></div>
<h2>Spending by category (monthly equivalent)</h2><table><tr><th>Category</th><th>Type</th><th>Per month</th><th>Share</th></tr>{cat_rows}</table>
<h2>Prediction</h2><p>Model: {e(f['model'])}. Expected spending over the next {f['horizon']} days is <b>{m(f['total'])}</b>
(likely {m(f['low'])} to {m(f['high'])}).</p><table><tr><th>Category</th><th>Predicted</th></tr>{fc_rows}</table>
<h2>Recommendations</h2>{rec_html}
<p><small>Figures are estimates from the records you entered. They are guidance, not financial advice.</small></p></body></html>"""


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
def sample_data(cat: Categoriser, days: int = 90, today: dt.date | None = None) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(7)
    end = pd.Timestamp(today or dt.date.today()).normalize()
    start = end - pd.Timedelta(days=days - 1)
    rows = []
    for d in pd.date_range(start, end):
        dow = d.dayofweek
        rows.append((d, "Matatu fare", float(rng.choice([100, 120, 150]))) if dow < 5 else None)
        if dow < 5 and rng.random() < 0.8:
            rows.append((d, "Lunch at cafeteria", float(rng.integers(150, 220))))
        if rng.random() < 0.55:
            rows.append((d, rng.choice(["Mandazi and tea", "Coffee and snack", "Airtime", "Bottled water and snack"]),
                         float(rng.integers(30, 120))))
        if dow == 5:
            rows.append((d, str(rng.choice(["Naivas supermarket", "Quickmart groceries"])), float(rng.integers(1800, 3200))))
            if rng.random() < 0.65:
                rows.append((d, "Outing with friends", float(rng.integers(1000, 2400))))
            if rng.random() < 0.4:
                rows.append((d, "Uber ride", float(rng.integers(400, 800))))
        if dow == 2 and rng.random() < 0.7:
            rows.append((d, "Mama mboga vegetables", float(rng.integers(250, 500))))
        if dow == 6 and rng.random() < 0.35:
            rows.append((d, "Dinner at restaurant", float(rng.integers(900, 1800))))
        if d.day == 1:
            rows += [(d, "House rent", 15000.0), (d, "Netflix subscription", 1100.0), (d, "Gym membership", 2500.0)]
        if d.day == 5:
            rows.append((d, "KPLC tokens", float(rng.integers(1800, 2600))))
        if d.day == 8:
            rows.append((d, "Wifi bill", 3000.0))
        if d.day in (10, 25):
            rows.append((d, "Safaricom data bundles", float(rng.integers(800, 1200))))
        if d.day == 15 and d.month % 2 == 0:
            rows.append((d, "Pharmacy", float(rng.integers(900, 2200))))
        if d.day == 28:
            rows += [(d, "Sacco contribution", 5000.0), (d, "Sent to mum", 4000.0)]
        if d.day in (12, 22) and rng.random() < 0.8:
            rows.append((d, str(rng.choice(["Jumia order shoes", "Clothes at mall", "Headphones"])), float(rng.integers(1500, 4500))))
        if d.day == 18:
            rows.append((d, "Salon", float(rng.integers(1200, 2000))))
    rows = [r for r in rows if r]
    rows.append((end - pd.Timedelta(days=20), "Sportybet stake", 6500.0))        # an unusual spike
    rows.append((end - pd.Timedelta(days=9), "Naivas supermarket", 2450.0))
    rows.append((end - pd.Timedelta(days=9), "Naivas supermarket", 2450.0))      # duplicate charge
    df = add_transactions(empty_tx(), rows, cat)
    return df, [{"source": "Salary", "amount": 80000.0, "frequency": "Monthly"}]


# --------------------------------------------------------------------------
# AUTONOMOUS BEHAVIOUR  (budget, watch the month, alert, schedule saving, dispute)
# --------------------------------------------------------------------------
PACED = {"Food & Groceries", "Eating Out", "Transport", "Entertainment", "Shopping"}
HELD_OUT = [
    ("Tuskys shopping", "Food & Groceries"), ("Rongai matatu", "Transport"), ("KPLC prepaid units", "Utilities"),
    ("Netflix renewal", "Subscriptions"), ("Safaricom 1GB bundle", "Airtime & Data"), ("Chemist panadol", "Health"),
    ("Textbook purchase", "Education"), ("Harambee for neighbour", "Family & Giving"), ("Cinema with friends", "Entertainment"),
    ("Barber shop", "Personal Care"), ("Jumia shoes", "Shopping"), ("Rent for December", "Rent & Housing"),
    ("Pizza Inn lunch", "Eating Out"), ("Sacco deposit", SAV), ("Bank ledger fee", "Other"), ("Uber to airport", "Transport"),
    ("Gas cylinder refill", "Utilities"), ("Showmax", "Subscriptions"), ("Mama mboga sukuma", "Food & Groceries"),
]


def categoriser_heldout(cat: "Categoriser") -> dict:
    wrong = []
    for text, exp in HELD_OUT:
        got = cat.predict(text)[0]
        if got != exp:
            wrong.append((text, exp, got))
    return {"accuracy": 1 - len(wrong) / len(HELD_OUT), "n": len(HELD_OUT), "wrong": wrong}


def autopilot(df: pd.DataFrame, a: dict, f: dict, payday: int = 25) -> dict:
    """Decide budgets without being asked, watch the current month against them, and schedule the saving."""
    inc = a["income_m"]
    rows = []
    for c, r in a["by_cat"].iterrows():
        typical, cap = r["monthly"], CAPS.get(c)
        cap_amt = cap / 100 * inc if (cap and inc) else None
        if cap_amt and GROUP[c] == "Wants" and typical > cap_amt:
            budget, why = cap_amt, "discretionary, capped at the guideline"
        elif cap_amt and GROUP[c] == "Needs" and typical > cap_amt * 1.05:
            budget, why = typical * 0.95, "essential, trimmed by 5%"
        else:
            budget, why = typical * 1.05, "within guideline, kept with 5% room"
        rows.append({"Category": c, "Typical": typical, "Budget": budget, "Change": budget - typical, "Why": why})
    budget_df = pd.DataFrame(rows)

    # watch the latest month against the budgets
    last = a["last"]
    month_start = last.replace(day=1)
    dim = calendar.monthrange(last.year, last.month)[1]
    elapsed = (last - month_start).days + 1
    spent = df[(df["date"] >= month_start) & (df["category"] != SAV)].groupby("category")["amount"].sum()
    alerts = []
    if elapsed >= 7:
        for _, r in budget_df.iterrows():
            c, b = r["Category"], r["Budget"]
            s_now = float(spent.get(c, 0.0))
            if s_now <= 0:
                continue
            rate = s_now / elapsed
            # everyday categories are projected from the daily pace; bills and one-off categories keep their usual size
            projected = rate * dim if c in PACED else max(s_now, float(r["Typical"]))
            if projected > b * 1.10:
                if s_now >= b:
                    when = "the budget is already used up"
                elif c not in PACED:
                    when = "this is already above the budget"
                else:
                    day = int(np.ceil(b / rate))
                    when = f"the limit will be reached around {(month_start + pd.Timedelta(days=day - 1)):%d %b}"
                alerts.append({"Category": c, "Spent so far": s_now, "Budget": b, "Projected": projected,
                               "Message": f"{c} is on pace for {projected:,.0f} against a budget of {b:,.0f}; {when}."})

    cuts = float(-budget_df[budget_df["Change"] < 0]["Change"].sum())
    achievable = max(a["surplus_m"], 0) + cuts
    amount = float(np.floor(min(0.2 * inc, achievable) / 100) * 100) if inc else 0.0
    today = last
    dates = []
    for k in range(3):
        y, m = today.year + (today.month - 1 + k + (1 if today.day >= payday else 0)) // 12, \
               (today.month - 1 + k + (1 if today.day >= payday else 0)) % 12 + 1
        dates.append(dt.date(y, m, min(payday, calendar.monthrange(y, m)[1])))
    disputes = []
    for _, r in a["pat"]["unusual"].iterrows() if len(a["pat"]["unusual"]) else []:
        if "duplicate" in r["Why"]:
            disputes.append(f"{r['Date']}: {r['Description']} charged twice ({r['Amount']:,.0f}). Draft dispute prepared.")
    return {"budget": budget_df, "alerts": alerts, "transfer_amount": amount, "transfer_dates": dates, "payday": payday,
            "disputes": disputes, "elapsed": elapsed, "month": f"{last:%B %Y}", "cuts": cuts}


def draft_dispute(owner: str, cur: str, row: pd.Series) -> str:
    return (f"Subject: Duplicate charge on {row['date']:%d %B %Y}\n\nHello,\n\nI see two identical charges of {cur} "
            f"{row['amount']:,.0f} for '{row['description']}' on {row['date']:%d %B %Y}. Please review the second charge and "
            f"reverse it if it was made in error.\n\nThank you,\n{owner or 'Customer'}")


def agent_run(a: dict, f: dict, recs: list, ap: dict, cur: str = "KES") -> dict:
    m = lambda v: f"{cur} {v:,.0f}"
    pat = a["pat"]
    perceive = [f"Read {len(a['spend']):,} spending records over {a['days']} days ({a['first']:%d %b} to {a['last']:%d %b}).",
                f"Income on record: {m(a['income_m'])} a month." if a["income_m"] else "No income entered yet."]
    analyse = [f"Spending is about {m(a['spend_m'])} a month; savings rate {a['savings_rate']:.0%}.",
               f"Highest category: {a['by_cat'].index[0]} ({a['by_cat'].iloc[0]['share']:.0%} of spending).",
               f"Spending peaks on {pat['peak_day']}; weekends run {pat['weekend_ratio']:.1f}x weekdays."]
    if len(a["excess"]):
        analyse.append(f"{len(a['excess'])} categories exceed their guideline, led by {a['excess'].iloc[0]['Category']}.")
    analyse.append(f"Forecast for the next {f['horizon']} days: {m(f['total'])} (model: {f['model']}).")
    decide = [f"Set {len(ap['budget'])} category budgets; trimmed {m(ap['cuts'])} a month from over-guideline categories."]
    decide.append(f"Chose a monthly saving transfer of {m(ap['transfer_amount'])}." if ap["transfer_amount"] else
                  "Cannot choose a transfer until income is entered.")
    decide += [x["Message"] for x in ap["alerts"][:3]]
    recommend = [r["title"] + (f" (up to {m(r['saving'])} a month)" if r["saving"] else "") for r in recs[:4]]
    act = [f"Created a {ap['month']} budget for every category.",
           f"Scheduled saving transfers of {m(ap['transfer_amount'])} on " + ", ".join(f"{d:%d %b}" for d in ap["transfer_dates"]) + " (assuming the new budgets are followed)."
           if ap["transfer_amount"] else "No transfer scheduled."]
    act.append(f"Raised {len(ap['alerts'])} overspending alert(s) before the month ends.")
    if ap["disputes"]:
        act.append(f"Flagged {len(ap['disputes'])} duplicate charge(s) and drafted a dispute message.")
    return {"perceive": perceive, "analyse": analyse, "decide": decide, "recommend": recommend, "act": act}


def gemini_plan(gemini, a: dict, f: dict, recs: list) -> dict | None:
    """Ask Gemini to turn the agent's own numbers into a short, plain-language plan. It may not invent figures."""
    if gemini is None or not gemini.available():
        return None
    top = "; ".join(f"{r['title']} (up to {r['saving']:,.0f}/month)" for r in recs[:5])
    facts = (f"Monthly income {a['income_m']:,.0f}; monthly spending {a['spend_m']:,.0f}; savings rate {a['savings_rate']:.0%}; "
             f"forecast next 30 days spending {f['total']:,.0f}. Agent's recommendations: {top}.")
    r = gemini.generate(
        "You are a practical personal-finance coach in Kenya. Using ONLY these numbers, write a plan in under 140 words with "
        "three concrete steps for this month. Do not invent figures.\n" + facts, temperature=0.4)
    return {"text": r["text"], "sources": []} if r["ok"] else None
