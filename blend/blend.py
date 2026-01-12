from numbers import Number
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import polars as pl

from .DBHandler import DBHandler
from .Operators import Combiners, Seekers
from .Plan import Plan
from .utils import clean

__all__ = ["BLEND"]


class BLEND:
    def __init__(
        self,
        db_path: Path,
        clean_function: Optional[Callable] = None,
        clean_function_args: Optional[dict] = None,
        xash_size: int = 128,
        disable_xash: bool = False,
    ) -> None:
        self._db_path = db_path
        self.db_handler: DBHandler = DBHandler(self._db_path)

        # Clean function and relative parameters
        self._clean_function = clean_function if clean_function else clean
        self._clean_function_args = clean_function_args if clean_function_args else {}

        self.xash_size = xash_size
        self.disable_xash = disable_xash

    def remove_table(self, table_id: str):
        self.db_handler.remove_table_from_index(table_id)

    def get_table(self, table_id: str) -> pl.DataFrame | pd.DataFrame:
        return self.db_handler.get_table_from_index(table_id)

    def close(self):
        self.db_handler.close()

    def keyword_search(self, values: list[str], k: int):
        """
        Execute a keyword search on the given query values.

        :param values: a list of string keywords.
            These values are assumed to be already cleaned and formatted.
        :param k: The number of results to return.
        :return: A list of tuples <table id, overlap size (distinct)>.
        """
        plan = Plan(self.db_handler)
        plan.add("keyword", Seekers.K(values, k))

        return plan.run()

    def single_column_join_search(
        self, column: list[str], k: int
    ) -> list[tuple[str, int, int]]:
        """
        Execute a single-column join search on the given column values.

        :param column: a list of strings representing the query column.
            These values are assumed to be already cleaned and formatted.
        :param k: The number of results to return.
        :return: A list of tuples <table id, column number, overlap size (distinct)>.
        """
        plan = Plan(self.db_handler)
        plan.add("single_column_join", Seekers.SC(column, k))

        return plan.run()

    def multi_column_join_search(
        self, table: list[list[str]], k: int, verbose: bool = False
    ) -> list[tuple[str, list, float]]:
        """
        Execute a multi-column join search on the given table.
        This method is built on top of the MATE discovery algorithm.

        :param table: A list-of-rows representing a table. The values is assumed are already cleaned.
        :param k: The number of results to return.
        :param verbose:
        :return: A list of tuples <table id, column numbers, joinability score>
        """
        df = pd.DataFrame(table)

        plan = Plan(self.db_handler)
        plan.add("multi_column_join", Seekers.MC(df, k, self.xash_size, verbose))
        return plan.run()

    def correlation_search(
        self,
        keys: list[str],
        targets: list[Number],
        k: int = 10,
        hash_size: int = 256,
        verbose: bool = False,
    ):
        """
        Execute a join-correlation search on the given key and target columns.
        This method is built on top of the QCR Join-Correlation search schema.

        :param keys: A list of strings representing a key column.
        :param targets: A list of numbers representing a target column.
        :param k: The number of results to return.
        :param hash_size: The dimension of the hash size used by the QCR approach.
        :param verbose:
        """
        plan = Plan(self.db_handler)
        plan.add("correlation", Seekers.C(keys, targets, k, hash_size))

        return plan.run()

    def union_search(self, table: list[list[str]], k: int):
        """
        Execute a union search on the given table.
        This method exeutes a union of the results given by a single-column search
        on all the table columns.

        :param table: A list-of-rows representing a table. The values is assumed that are already cleaned.
        :param k: The number of results to return.
        """

        df = pd.DataFrame(table)

        plan = Plan(self.db_handler)
        for clm_name in df.columns:
            plan.add(clm_name, Seekers.SC(df[clm_name], k * 10))

        plan.add("union", Combiners.Counter(k=k), inputs=df.columns)

        return plan.run()
