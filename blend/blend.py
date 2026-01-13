from numbers import Number
from pathlib import Path
from typing import Callable, Optional, Any

import pandas as pd
import polars as pl

from .db import DBHandler
from .operators import combiners, seekers
from .plan import Plan
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
        """
        Instantiate a BLEND indexer and retriever.

        :param db_path: A Path object leading to a duckdb file position.
        :param clean_function: The clean function for any cell value. It accepts any type and returns a string.
        :param clean_function_args: The clean function arguments, passed to it with any call.
        :param xash_size: The XASH size used for the super key.
        :param disable_xash: If true, the super key is replaced with empty values.
        """
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

    def keyword_search(self, values: list[Any], k: int, clean: bool = True):
        """
        Execute a keyword search on the given query values.

        :param values: A list of string keywords.
        :param k: The number of results to return.
        :param clean: If True, apply the default clean function on the input values.
        :return: A list of tuples <table id, overlap size (distinct)>.
        """
        if clean:
            values = [
                self._clean_function(v, **self._clean_function_args) for v in values
            ]
        plan = Plan(self.db_handler)
        plan.add("keyword", seekers.K(values, k))

        return plan.run()

    def single_column_join_search(
        self, column: list[Any], k: int, clean: bool = True
    ) -> list[tuple[str, int, int]]:
        """
        Execute a single-column join search on the given column values.

        :param column: a list of strings representing the query column.
        :param k: The number of results to return.
        :param clean: If True, apply the default clean function on the input values.
        :return: A list of tuples <table id, column number, overlap size (distinct)>.
        """
        if clean:
            column = [
                self._clean_function(cell, **self._clean_function_args)
                for cell in column
            ]
        plan = Plan(self.db_handler)
        plan.add("single_column_join", seekers.SC(column, k))

        return plan.run()

    def multi_column_join_search(
        self, table: list[list[Any]], k: int, clean: bool = True, verbose: bool = False
    ) -> list[tuple[str, list, float]]:
        """
        Execute a multi-column join search on the given table.
        This method is built on top of the MATE discovery algorithm.

        :param table: A list-of-rows representing a table.
        :param k: The number of results to return.
        :param clean: If True, apply the default clean function on the input values.
        :param verbose:
        :return: A list of tuples <table id, column numbers, joinability score>
        """
        if clean:
            table = [
                [
                    self._clean_function(cell, **self._clean_function_args)
                    for cell in row
                ]
                for row in table
            ]
        df = pd.DataFrame(table)

        plan = Plan(self.db_handler)
        plan.add("multi_column_join", seekers.MC(df, k, self.xash_size, verbose))
        return plan.run()

    def correlation_search(
        self,
        keys: list[Any],
        targets: list[Number],
        k: int = 10,
        hash_size: int = 256,
        clean: bool = True,
        verbose: bool = False,
    ):
        """
        Execute a join-correlation search on the given key and target columns.
        This method is built on top of the QCR Join-Correlation search schema.

        :param keys: A list of strings representing a key column.
        :param targets: A list of numbers representing a target column.
        :param k: The number of results to return.
        :param hash_size: The dimension of the hash size used by the QCR approach.
        :param clean: If True, apply the default clean function on the input values.
        :param verbose:
        """
        if clean:
            keys = [self._clean_function(k, **self._clean_function_args) for k in keys]

        plan = Plan(self.db_handler)
        plan.add("correlation", seekers.C(keys, targets, k, hash_size))

        return plan.run()

    def union_search(self, table: list[list[Any]], k: int, clean: bool = True):
        """
        Execute a union search on the given table.
        This method exeutes a union of the results given by a single-column search
        on all the table columns.

        :param table: A list-of-rows representing a table.
        :param k: The number of results to return.
        :param clean: If True, apply the default clean function on the input values.
        """
        if clean:
            table = [
                [
                    self._clean_function(cell, **self._clean_function_args)
                    for cell in row
                ]
                for row in table
            ]

        # switch to a list-of-columns view of the table
        table = list(zip(*table))

        plan = Plan(self.db_handler)
        for n_column, column in enumerate(table):
            plan.add(str(n_column), seekers.SC(column, k * 10))

        plan.add(
            "union", combiners.Counter(k=k), inputs=list(map(str, range(len(table))))
        )

        return plan.run()
