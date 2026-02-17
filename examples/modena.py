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

if not data_path.exists():
    print(f"Data path {data_path} does not exist.")
    sys.exit(1)

print("--- Instantiating BLEND index ---")
indexer = BLEND(
    index_db_path,
    clean_args={"lowercase": True},
    max_cell_length=512,
)

load_opts = {"ignore_errors": True}
if not index_db_path.exists():
    print("--- Indexing tables ---")
    index_tables(indexer, data_lake_path, True, None, 4, load_opts)
else:
    print(f"--- Index {index_db_path} already exists, skipping indexing ---")

print("\n--- Loading query dataset ---")
queries = sorted(os.listdir(queries_path))
query_table_idx = 1
query_table_name = queries[query_table_idx]
qdf = pl.read_csv(queries_path.joinpath(query_table_name))
print(f"Query table: {query_table_name}")
# print(qdf.head())

print("\n--- Keyword Search ---")
# Flatten query dataframe values to a set
values = list({cell for row in qdf.rows() for cell in row})
results = indexer.keyword_search(values, k=20)
print(tabulate(results, headers=["dataset", "overlap"]))

print("\n--- Unionable Table Search ---")
table = [list(row) for row in qdf.rows()]
results = indexer.union_search(table, 10)
print(tabulate(results, headers=["dataset"]))

print("\n--- Single Column JOIN Search ---")
# Extract key values from THE_PK_KEY column if it exists, else use the first column
if "THE_PK_KEY" in qdf.columns:
    column = qdf.get_column("THE_PK_KEY").drop_nulls().to_list()
else:
    column = qdf.get_column(qdf.columns[0]).drop_nulls().to_list()

results = indexer.single_column_join_search(column, k=20)
print(
    tabulate(
        results,
        headers=["dataset", "column idx", "overlap (distinct)", "overlap (general)"],
    )
)

print("\n--- Multi-Column JOIN Search (MATE) ---")
# Drop THE_PK_KEY if it exists for multi-column search
if "THE_PK_KEY" in qdf.columns:
    qdf_mate = qdf.drop("THE_PK_KEY")
else:
    qdf_mate = qdf

# Select a subset of columns for the search
search_cols = [c for c in ["SECTION", "DISTRICT", "ELECTION"] if c in qdf_mate.columns]
if not search_cols:
    search_cols = qdf_mate.columns[:3]

mate_table = qdf_mate.select(search_cols)
print(f"Searching for joins on columns: {search_cols}")

# The multi-column task now accepts a polars DataFrame instead of a list of lists.
mc_results = indexer.multi_column_join_search(mate_table, 10, verbose=True)
print(tabulate(mc_results, headers=["dataset", "columns", "join_score"]))

indexer.close()
