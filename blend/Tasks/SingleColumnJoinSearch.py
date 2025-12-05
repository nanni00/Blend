from typing import Iterable

from ..Operators.Seekers import SingleColumnOverlap
from ..Plan import Plan


def SingleColumnJoinSearch(query_values: Iterable[any], k: int = 10) -> Plan:
    plan = Plan()
    plan.add("single_column_overlap", SingleColumnOverlap(query_values, k))
    return plan
