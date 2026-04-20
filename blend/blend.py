from __future__ import annotations

from collections.abc import Iterable
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import polars as pl

from .db import DBHandler
from .operators import combiners, seekers
from .plan import Plan
from .utils import _truncate, clean

__all__ = ["BLEND"]

if TYPE_CHECKING:
    import pandas as pd

try:
    import pandas as _pd
except ImportError:  # pragma: no cover - optional dependency
    _pd = None


class BLEND:
    def __init__(
        self,
        db_path: Path,
        index_table: str = "all_tables",
        clean_function: Optional[Callable] = None,
        clean_args: Optional[dict] = None,
        xash_size: int = 128,
        max_cell_length: Optional[int] = 128,
        use_ml_optimizer: bool = False,
        freq_dict_path: Optional[Path] = None,
    ) -> None:
        """Instantiate a BLEND indexer and retriever.

        Args:
            db_path: Path to the DuckDB database file.
            index_table: Name of the database table used to store the BLEND index.
            clean_function: Function applied to cell values before indexing and querying.
                It accepts any type and returns a string.
            clean_args: Keyword arguments passed to clean_function at each call.
            xash_size: Size of the XASH super key in bits.
            max_cell_length: Maximum number of characters stored for each cell value.
                If positive, only the first max_cell_length characters are kept.
                If negative, only the last abs(max_cell_length) characters are kept.
                If None, cell values are not truncated.
            use_ml_optimizer: Whether to enable the ML-based optimizer.
            freq_dict_path: Path to the token-frequency CSV used by the ML optimizer.
        """
        self._db_path = db_path
        self.db_handler: DBHandler = DBHandler(
            self._db_path, index_table, use_ml_optimizer, freq_dict_path
        )

        # Clean function and relative parameters
        self._clean_function = clean_function if clean_function else clean
        self._clean_args = clean_args if clean_args else {}

        self.xash_size = xash_size
        self.max_cell_length = max_cell_length

    def remove_table(self, table_id: str):
        self.db_handler.remove_table_from_index(table_id)

    def get_table(self, table_id: str) -> pl.DataFrame:
        return self.db_handler.get_table_from_index(table_id)

    def extract_token_frequencies_from_db(self) -> pl.DataFrame:
        return self.db_handler.extract_token_frequencies_from_db()

    def close(self):
        self.db_handler.close()

    @staticmethod
    def _is_pandas_dataframe(value: Any) -> bool:
        return _pd is not None and isinstance(value, _pd.DataFrame)

    def _to_polars_dataframe(self, value: Any, arg_name: str) -> pl.DataFrame:
        if isinstance(value, pl.DataFrame):
            return value
        if self._is_pandas_dataframe(value):
            return pl.from_pandas(value)
        if isinstance(value, list):
            return pl.DataFrame(value, orient="row")

        raise TypeError(
            f"{arg_name} must be a row-oriented list, a Polars DataFrame, or a pandas DataFrame."
        )

    def _extract_values(
        self,
        value: Iterable[Any] | pl.DataFrame | pd.DataFrame,
        *,
        arg_name: str,
        flatten_dataframe: bool = False,
    ) -> list[Any]:
        if isinstance(value, pl.DataFrame) or self._is_pandas_dataframe(value):
            frame = self._to_polars_dataframe(value, arg_name)
            if flatten_dataframe:
                return [cell for row in frame.iter_rows() for cell in row]
            if frame.width != 1:
                raise ValueError(
                    f"{arg_name} dataframe input must contain exactly one column."
                )
            return frame.to_series(0).to_list()

        return list(value)

    def _prepare_text_values(self, values: Iterable[Any], clean_input: bool) -> list[str]:
        if clean_input:
            cleaned_values = [
                self._clean_function(value, **self._clean_args) for value in values
            ]
        else:
            cleaned_values = [str(value) for value in values]

        return [_truncate(value, self.max_cell_length) for value in cleaned_values]

    def _prepare_table(
        self,
        table: list[list[Any]] | pl.DataFrame | pd.DataFrame,
        clean_input: bool,
    ) -> pl.DataFrame:
        table = self._to_polars_dataframe(table, "table")

        if clean_input:
            table = table.with_columns(
                [
                    pl.col(column).map_elements(
                        lambda value: self._clean_function(value, **self._clean_args),
                        pl.String,
                    )
                    for column in table.columns
                ]
            )
        else:
            table = table.with_columns(
                [pl.col(column).cast(pl.String) for column in table.columns]
            )

        if isinstance(self.max_cell_length, int):
            if self.max_cell_length > 0:
                table = table.with_columns(
                    [pl.col(column).str.head(self.max_cell_length) for column in table.columns]
                )
            elif self.max_cell_length < 0:
                table = table.with_columns(
                    [pl.col(column).str.tail(self.max_cell_length) for column in table.columns]
                )

        return table

    def keyword_search(
        self,
        values: Iterable[Any] | pl.DataFrame | pd.DataFrame,
        k: int,
        clean: bool = True,
    ) -> list[tuple[str, int]]:
        """Execute a keyword search on the given query values.

        Args:
            values: An iterable of keywords or a pandas/polars DataFrame.
                DataFrame inputs are flattened across all cells.
            k: The number of results to return.
            clean: If True, apply the configured clean function to the input values.

        Returns:
            A list of tuples <table id, overlap size (distinct)>.
        """
        values = self._extract_values(values, arg_name="values", flatten_dataframe=True)
        values = self._prepare_text_values(values, clean)
        plan = Plan(self.db_handler)
        plan.add("keyword", seekers.K(values, k))

        return plan.run()

    def single_column_join_search(
        self,
        column: Iterable[Any] | pl.DataFrame | pd.DataFrame,
        k: int,
        clean: bool = True,
    ) -> list[tuple[str, int, int]]:
        """Execute a single-column join search on the given column values.

        Args:
            column: An iterable of values or a single-column pandas/polars DataFrame.
            k: The number of results to return.
            clean: If True, apply the configured clean function to the input values.

        Returns:
            A list of tuples <table id, column number, overlap size (distinct)>.
        """
        column = self._extract_values(column, arg_name="column")
        column = self._prepare_text_values(column, clean)
        plan = Plan(self.db_handler)
        plan.add("single_column_join", seekers.SC(column, k))

        return plan.run()

    def multi_column_join_search(
        self,
        table: list[list[Any]] | pl.DataFrame | pd.DataFrame,
        k: int,
        clean: bool = True,
        verbose: bool = False,
    ) -> list[tuple[str, list[int], float]]:
        """Execute a multi-column join search on the given table.

        This method is built on top of the MATE discovery algorithm.

        Args:
            table: A row-oriented nested list or a pandas/polars DataFrame.
            k: The number of results to return.
            clean: If True, apply the configured clean function to the input values.
            verbose: If True, print verbose output.

        Returns:
            A list of tuples <table id, column numbers, joinability score>.
        """
        table = self._prepare_table(table, clean)

        plan = Plan(self.db_handler)
        plan.add("multi_column_join", seekers.MC(table, k, self.xash_size, verbose))
        return plan.run()

    def correlation_search(
        self,
        keys: Iterable[Any] | pl.DataFrame | pd.DataFrame,
        targets: Iterable[Number] | pl.DataFrame | pd.DataFrame | None = None,
        k: int = 10,
        hash_size: int = 256,
        clean: bool = True,
    ) -> list[tuple[str, int, int, float]]:
        """Execute a join-correlation search on the given key and target columns.

        This method is built on top of the QCR Join-Correlation search schema.

        Args:
            keys: An iterable of key values, a single-column pandas/polars DataFrame,
                or a two-column pandas/polars DataFrame containing both keys and targets.
            targets: An iterable of numeric target values or a single-column
                pandas/polars DataFrame. Omit it when keys is a two-column DataFrame.
            k: The number of results to return.
            hash_size: The dimension of the hash size used by the QCR approach.
            clean: If True, apply the configured clean function to the key values.

        Returns:
            A list of tuples <table id, join column number, target column number, QCR>.
        """
        if isinstance(keys, pl.DataFrame) or self._is_pandas_dataframe(keys):
            if isinstance(targets, int):
                if k != 10:
                    raise TypeError(
                        "k was provided both positionally and by keyword."
                    )
                k = targets
                targets = None

            keys_frame = self._to_polars_dataframe(keys, "keys")
            if targets is None:
                if keys_frame.width != 2:
                    raise ValueError(
                        "keys dataframe input must contain exactly two columns when targets is omitted."
                    )
                keys = keys_frame.to_series(0).to_list()
                targets = keys_frame.to_series(1).to_list()
            else:
                if keys_frame.width != 1:
                    raise ValueError(
                        "keys dataframe input must contain exactly one column when targets is provided separately."
                    )
                keys = keys_frame.to_series(0).to_list()
                targets = self._extract_values(targets, arg_name="targets")
        else:
            if targets is None or isinstance(targets, int):
                raise TypeError(
                    "targets must be provided when keys is not a two-column dataframe."
                )
            keys = list(keys)
            targets = self._extract_values(targets, arg_name="targets")

        keys = self._prepare_text_values(keys, clean)

        plan = Plan(self.db_handler)
        plan.add("correlation", seekers.C(keys, targets, k, hash_size))

        return plan.run()

    def union_search(
        self,
        table: list[list[Any]] | pl.DataFrame | pd.DataFrame,
        k: int,
        clean: bool = True,
    ) -> list[tuple[str]]:
        """Execute a union search on the given table.

        This method executes a union of the results given by a single-column search
        on all the table columns.

        Args:
            table: A row-oriented nested list or a pandas/polars DataFrame.
            k: The number of results to return.
            clean: If True, apply the configured clean function to the input values.

        Returns:
            A list of tuples <table id>.
        """
        table = self._prepare_table(table, clean)

        plan = Plan(self.db_handler)
        for n_column, column in enumerate(table.columns):
            plan.add(str(n_column), seekers.SC(table.get_column(column), k * 10))

        plan.add(
            "union",
            combiners.Counter(k=k),
            inputs=list(map(str, range(len(table.columns)))),
        )

        return plan.run()
