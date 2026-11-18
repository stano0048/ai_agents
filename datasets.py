"""Sample datasets the agents are demonstrated with (downloadable from the Testing and data page)."""
from __future__ import annotations

import advisor_engine as adv
import finance_engine as fin
from support_engine import HELD_OUT, SAMPLE_INBOX


def advisor_marks_csv() -> str:
    return adv.sample_marks().to_csv(index=False)


def finance_transactions_csv() -> str:
    df, _ = fin.sample_data(fin.Categoriser())
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out[["date", "description", "amount", "category"]].to_csv(index=False)


def support_inbox_txt() -> str:
    return "\n".join(SAMPLE_INBOX) + "\n"


def support_heldout_csv() -> str:
    return "phrase,expected_topic\n" + "\n".join(f'"{t}",{i}' for t, i in HELD_OUT) + "\n"


def products_csv(db) -> str:
    import pandas as pd
    return pd.DataFrame(db.products()).to_csv(index=False)
