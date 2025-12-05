# FIX: include polars
import pandas as pd

from ..Operators.Seekers import MultiColumnOverlap
from ..Plan import Plan


def MultiColumnJoinSearch(query_dataset: pd.DataFrame, k: int = 10) -> Plan:
    plan = Plan()
    plan.add("multi_column_overlap", MultiColumnOverlap(query_dataset, k))
    return plan
