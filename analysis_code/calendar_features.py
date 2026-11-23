"""Calendar columns (contest notice section 4, last line: weekday, weekend, statutory / substitute / temporary holidays and election days may be used as inputs).

Why: in spring 2024/2025 the fleet produced 5-9 % LESS than the weather-only model on clear weekend / holiday middays (11-16 h), concentrated in a group of plants in the
north-west.  That is the signature of grid-operator output reductions on low-demand days (the metered values contain them, see the notice).  The weather features cannot know it;
the calendar can.

cal_off   1 if the day is a Saturday, Sunday or public holiday (holidays_kr.csv: statutory + substitute + temporary holidays and election days, 2021-2026)
cal_dow   day of week, 0 = Monday ... 6 = Sunday
cal_run   number of consecutive days off in the block the day belongs to (1 for a weekday)
cal_pos   position of the day inside that block (1 = first day off), 0 for a weekday
cal_prev  1 if the previous day was a day off, cal_next 1 if the next day is a day off (bridge / long weekends)

The day of an hour label ts (right-closed hour, KST) is the date of (ts - 1 h), as everywhere else in the code base.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HOLIDAYS = os.environ.get("OIBC_HOLIDAYS", os.path.join(HERE, "external_data", "holidays_kr.csv"))
COLUMNS = ["cal_off", "cal_dow", "cal_run", "cal_pos", "cal_prev", "cal_next"]


def day_table(start: str = "2020-12-25", end: str = "2027-01-10") -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    hol = set(pd.to_datetime(pd.read_csv(HOLIDAYS).date).dt.date)
    off = np.array([(d.weekday() >= 5) or (d.date() in hol) for d in days], dtype=bool)
    run = np.zeros(len(days), dtype="int16"); pos = np.zeros(len(days), dtype="int16")
    i = 0
    while i < len(days):
        if off[i]:
            j = i
            while j + 1 < len(days) and off[j + 1]:
                j += 1
            run[i:j + 1] = j - i + 1
            pos[i:j + 1] = np.arange(1, j - i + 2)
            i = j + 1
        else:
            i += 1
    prev = np.r_[False, off[:-1]]; nxt = np.r_[off[1:], False]
    return pd.DataFrame(dict(cal_off=off.astype("int8"), cal_dow=np.array([d.weekday() for d in days], dtype="int8"), cal_run=run, cal_pos=pos,
                             cal_prev=prev.astype("int8"), cal_next=nxt.astype("int8")), index=days)


def calendar_columns(ts) -> pd.DataFrame:
    """columns COLUMNS for a (tz-aware, KST) hour-label index or Series; returned as a DataFrame with a default RangeIndex aligned with the input order."""
    ts = pd.DatetimeIndex(ts)
    day = (ts - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    tab = day_table()
    out = tab.reindex(day)
    if out.isna().any().any():
        raise ValueError("calendar table does not cover all dates")
    out = out.reset_index(drop=True)
    return out.astype({"cal_off": "int8", "cal_dow": "int8", "cal_run": "int16", "cal_pos": "int16", "cal_prev": "int8", "cal_next": "int8"})
