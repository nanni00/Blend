# FIX: include polars
from typing import List

from ..Operators.Seekers import Keyword
from ..Plan import Plan


def KeywordSearch(query_values: List[str], k: int = 10) -> Plan:
    plan = Plan()
    plan.add("keyword", Keyword(query_values, k))
    return plan
