# FIX: include polars
import pandas as pd

from ..Operators import Combiners, Seekers
from ..Plan import Plan


def UnionSearch(dataset: pd.DataFrame, k: int = 10) -> Plan:
    plan = Plan()
    for clm_name in dataset.columns:
        plan.add(clm_name, Seekers.SC(dataset[clm_name], k * 10))

    plan.add("union", Combiners.Union(k=k), inputs=dataset.columns)

    return plan
