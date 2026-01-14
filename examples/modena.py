import os
import sys
from pathlib import Path
import polars as pl
from tabulate import tabulate

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from blend import BLEND
from blend.indexing import index_tables

data_path = Path(__file__).parent.parent.joinpath("examples", "example-data", "modena")

data_lake_path = data_path.joinpath("data-lake")
index_db_path = data_path.joinpath("modena.db")
logdir_path = data_path.joinpath("log")
queries_path = data_path.joinpath("queries")

data_path.exists()

indexer = BLEND(
    index_db_path, clean_function_args={"lowercase": True, "filter_bad_tokens": True}
)

scan_table_opts = {"ignore_errors": True}
index_tables(indexer, data_lake_path, True, None, 4, scan_table_opts)
