from numbers import Number
from typing import List, Literal

from ..Operators.Seekers import Correlation
from ..Plan import Plan


def CorrelationSearch(
    source_column: List[str],
    target_column: List[Number],
    k: int = 10,
    hash_size: Literal[256, 512, 1024] = 256,
) -> Plan:
    """
    Args:
        source_column: The key values.
        target_column: The numeric values.
        k: Number of results to return.
    Returns:
        [TODO:description]
    """
    plan = Plan()
    plan.add(
        "correlation",
        Correlation(source_column, target_column, k, hash_size),
    )
    return plan
