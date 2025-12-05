import multiprocessing
import os
from _thread import LockType
from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import cleandoc
from numbers import Number
from pathlib import Path
from time import time
from typing import Callable, Optional

import pandas as pd
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from .DBHandler import DBHandler
from .Operators import Combiners, Seekers
from .Plan import Plan
from .utils import calculate_xash, clean

__all__ = ["BLEND"]


def parse_table(
    table_path: Path,
    scan_table_opts: dict,
    clean_function: Callable,
    clean_function_args: dict,
    xash_size: int,
    disable_xash: bool,
):
    table_id = table_path.stem
    format = table_path.suffix.replace(".", "")

    try:
        match format:
            case "csv":
                table_df = pl.scan_csv(table_path, **scan_table_opts)

                # ignore_errors=True,
                # n_rows=limit_rows_per_table,
                # infer_schema_length=10_000,
            case "parquet":
                table_df = pl.scan_parquet(table_path, **scan_table_opts)
            case _:
                raise ValueError(f"Unknown table format in {table_path}: {format}")

        # we need to keep track of the real row index of each record
        # even after dropping nulls, thus we create a new column to this aim
        table_df = table_df.with_row_index(name="blend_row_index")

        # in this way we drop only those rows that have all values nulls,
        # except for the configured row index
        table_df = table_df.filter(
            ~pl.all_horizontal(pl.all().exclude("blend_row_index").is_null())
        ).collect()
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, ValueError) as e:
        return table_id, False, e

    if table_df.shape[0] == 0 or table_df.shape[1] == 0:
        return table_id, False, "Empty table"

    # identify the numeric columns for the correlation part
    numeric_cols = set(table_df.select(cs.numeric()).columns)
    columns_df = []

    for col_counter, col_name in enumerate(table_df.columns[1:]):
        # to insert this column values, select also the row index
        column = table_df.select("blend_row_index", col_name)

        if hasattr(clean_function, "__vectorized__"):
            cleaned = column.select(
                pl.col(col_name)
                .map_elements(
                    lambda x: clean_function(x, **clean_function_args),
                    return_dtype=pl.String,
                )
                .alias("CellValue")
            )
        else:
            cleaned = pl.Series(
                "CellValue",
                [
                    clean_function(item, **clean_function_args)
                    for row_counter, item in column.rows()
                ],
            )

        result_df = column.with_columns(
            [
                cleaned,
                pl.lit(table_id).alias("TableId"),
                pl.lit(col_counter).alias("ColumnId"),
                pl.col("blend_row_index").alias("RowId"),
            ]
        )

        is_numeric = col_name in numeric_cols
        if is_numeric:
            mean = column.select(col_name).to_series().mean()
            result_df = result_df.with_columns(
                pl.when(pl.col(col_name).is_not_null())
                .then(pl.col(col_name) >= mean)
                .otherwise(None)
                .alias("Quadrant")
            )
        else:
            result_df = result_df.with_columns(
                pl.lit(None).cast(pl.Boolean).alias("Quadrant")
            )

        columns_df.append(result_df.drop("blend_row_index", col_name))

    all_data = pl.concat(columns_df)

    if disable_xash:
        final_data = all_data.with_columns(pl.lit(int(0).to_bytes(), pl.Binary))
    else:
        superkey_data = all_data.group_by("RowId").agg(
            pl.map_groups(
                ["CellValue"],
                lambda values: calculate_superkey_for_row(
                    values[0].to_list(), xash_size
                ),
                return_dtype=pl.Binary,
                returns_scalar=True,
            ).alias("SuperKey")
        )

        final_data = all_data.join(superkey_data, on="RowId")

    return table_id, True, final_data


def calculate_superkey_for_row(cell_values: list, xash_size: int) -> bytes:
    superkey = 0
    for value in cell_values:
        superkey |= calculate_xash(value, xash_size)
    return bytes(f"{superkey:0128b}".encode())


def _process_task(
    table_path: Path,
    scan_table_opts: Optional[dict],
    clean_function: Callable,
    clean_function_args: dict,
    xash_size: int,
    disable_xash: bool,
    db_handler: DBHandler,
    lock: LockType,
):
    table_id, success, df_or_error = parse_table(
        table_path,
        scan_table_opts or {},
        clean_function,
        clean_function_args,
        xash_size,
        disable_xash,
    )

    if success:
        # FIX: this sequentialization can be improved
        with lock:
            try:
                db_handler.save_data_to_duckdb(df_or_error)
                return table_id, True
            except Exception as e:
                print(f"Failed insert for table={table_id}, e={e}")
                return table_id, False
    return table_id, True


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

        # TODO: DBHandler is instantiated only when it is actually accessed,
        # not sure if this is the best option for this workflow
        self._db_handler: Optional[DBHandler] = None

        # assert not clean_function or isinstance(clean_function, function), (
        #     "Clean function must be None or Callable"
        # )

        # Clean function and relative parameters
        self._clean_function = clean_function if clean_function else clean
        self._clean_function_args = clean_function_args if clean_function_args else {}

        self.xash_size = xash_size
        self.disable_xash = disable_xash

    @property
    def db_handler(self):
        if self._db_handler is None:
            self._db_handler = DBHandler(self._db_path)
        return self._db_handler

    def create_index(
        self,
        tables_path: str | Path,
        max_workers: Optional[int] = None,
        scan_table_opts: Optional[dict] = None,
        verbose: bool = False,
    ) -> tuple:
        if isinstance(tables_path, str):
            tables_path = Path(tables_path)

        if not tables_path.exists():
            raise FileNotFoundError(f"tables path doesn't exist: {tables_path}")

        # get IDs of the effective tables
        table_ids = os.listdir(tables_path)

        # drop the main index if already exists
        self.db_handler.drop_table()

        # create the main index
        self.db_handler.create_table()

        start_t = time()

        # FIX: sometimes this doesn't work at all (check on polars with multiproc
        # setting), even by changhing with different mp_context
        with ProcessPoolExecutor(max_workers) as executor:
            clean_function = self._clean_function
            clean_function_args = self._clean_function_args
            xash_size = self.xash_size
            db_handler = self.db_handler
            manager = multiprocessing.Manager()
            lock = manager.Lock()

            futures = {
                executor.submit(
                    _process_task,
                    tables_path.joinpath(table_id),
                    scan_table_opts,
                    clean_function,
                    clean_function_args,
                    xash_size,
                    self.disable_xash,
                    db_handler,
                    lock,
                )
                for table_id in list(table_ids)
            }

            non_empty_tables = 0

            for future in tqdm(
                as_completed(futures),
                desc="Parsing and storing tables: ",
                total=len(table_ids),
                disable=False,
            ):
                try:
                    table_id, success = future.result()
                    non_empty_tables += success
                except TimeoutError:
                    continue

        end_ins_t = time()

        # create indexes
        if verbose:
            s = f"""
            Tables ingestion completed.
            Correctly parsed {non_empty_tables}.
            Creating indexes...
            """
            print(cleandoc(s))

        self.db_handler.create_column_indexes()
        end_idx_t = time()

        if verbose:
            print("Index creation completed.")

        self.db_handler.close()
        return (end_ins_t - start_t, end_idx_t - end_ins_t, end_idx_t - start_t)

    def remove_table(self, table_id: str):
        self.db_handler.remove_table_from_index(table_id)

    def get_table(self, table_id: str) -> pl.DataFrame | pd.DataFrame:
        return self.db_handler.get_table_from_index(table_id)

    def close(self):
        self.db_handler.close()

    def keyword_search(self, values: list[str], k: int):
        plan = Plan(self.db_handler)
        plan.add("keyword", Seekers.K(values, k))

        return plan.run()

    def single_column_join_search(self, values: list[str], k: int):
        plan = Plan(self.db_handler)
        plan.add("single_column_join", Seekers.SC(values, k))

        return plan.run()

    def multi_column_join_search(
        self, columns: list[list[str]], k: int, verbose: bool = False
    ) -> list[tuple[str, list, float]]:
        columns_df = pd.DataFrame(columns)

        plan = Plan(self.db_handler)
        plan.add(
            "multi_column_join", Seekers.MC(columns_df, k, self.xash_size, verbose)
        )
        return plan.run()

    def correlation_search(
        self,
        keys: list[str],
        targets: list[Number],
        k: int = 10,
        hash_size: int = 256,
        verbose: bool = False,
    ):
        plan = Plan(self.db_handler)
        plan.add("correlation", Seekers.C(keys, targets, k, hash_size))

        return plan.run()

    def union_search(self, table: pd.DataFrame, k: int):
        table = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)

        plan = Plan(self.db_handler)
        for clm_name in table.columns:
            plan.add(clm_name, Seekers.SC(table[clm_name], k * 10))

        plan.add("union", Combiners.Union(k=k), inputs=table.columns)

        return plan.run()
