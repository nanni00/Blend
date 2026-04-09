import os
import sys
from pathlib import Path

import polars as pl
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from blend import BLEND
from blend.indexing import index_tables

data_path = Path(__file__).parent.parent.joinpath("examples", "example-data", "undata")

data_lake_path = data_path.joinpath("data-lake")
index_db_path = data_path.joinpath("undata.db")
freq_dict_path = data_path.joinpath("freqs_dict.csv")
logdir_path = data_path.joinpath("log")
queries_path = data_path.joinpath("queries")

if not data_path.exists():
    print(f"Data path {data_path} does not exist.")
    sys.exit(1)

# use_ml_optimizer = freq_dict_path.exists() # FIX: optimizer support not completed yet
use_ml_optimizer = False

print("--- Instantiating BLEND index ---")
indexer = BLEND(
    index_db_path,
    clean_args={"lowercase": True},
    max_cell_length=512,
    use_ml_optimizer=use_ml_optimizer,
    freq_dict_path=freq_dict_path,
)

load_opts = {"ignore_errors": True}
if not index_db_path.exists():
    print("--- Indexing tables ---")
    index_tables(indexer, data_lake_path, True, None, 4, load_opts)
else:
    print(f"--- Index {index_db_path} already exists, skipping indexing ---")


print("--- Extracting token frequencies ---")
freqs = indexer.extract_token_frequencies_from_db()
print(freqs)
freqs.write_csv(freq_dict_path)


print("\n--- Loading query dataset ---")
queries = sorted(os.listdir(queries_path))
query_table_idx = 1
query_table_name = queries[query_table_idx]
qdf = pl.read_csv(queries_path.joinpath(query_table_name))
print(f"Query dataset: {query_table_name}")

print("\n--- Union Search ---")
# union_search expects list of lists
table = [list(row) for row in qdf.rows()]
results = indexer.union_search(table, 10)
results_df = pl.DataFrame(results, orient="row")
print(results_df)

print("\n--- Join-Correlation Search (QCR) ---")
# Group by key column and mean on target column
# Using logic from notebook:
# query_table_idx = 0 (Adult literacy rate)
query_table_idx_corr = 0
query_table_name_corr = queries[query_table_idx_corr]
qdf_corr = pl.read_csv(queries_path.joinpath(query_table_name_corr))
print(f"Query dataset for correlation: {query_table_name_corr}")

target_column_name = "Value"
key_column_name = "Sub-region Name"

grouped_qdf = qdf_corr.group_by(key_column_name).agg(pl.col(target_column_name).mean())
grouped_qdf = grouped_qdf.rename({target_column_name: "Value_left"})

keys = grouped_qdf.get_column(key_column_name).to_list()
targets = grouped_qdf.get_column("Value_left").to_list()

results = indexer.correlation_search(keys, targets, 20)
results_df = pl.DataFrame(
    results, schema=["dataset", "join_col_idx", "target_col_idx", "QCR"], orient="row"
)
print(results_df)

print("\n--- Comparison with actual Pearson correlation ---")


def compare_with_pearson(results_list: list) -> list:
    results_with_pearson = []
    for table_id, join_col_idx, target_col_idx, qcr in results_list:
        try:
            r_df = pl.scan_csv(data_lake_path.joinpath(f"{table_id}.csv"))
            r_df = (
                r_df.group_by(pl.nth(join_col_idx))
                .agg(pl.nth(target_col_idx).mean())
                .collect()
            )

            target_col_name = r_df.columns[1]
            r_df = r_df.rename(
                {r_df.columns[0]: key_column_name, r_df.columns[1]: "Value_right"}
            )

            join = grouped_qdf.join(r_df, on=key_column_name)
            value_left = join.get_column("Value_left")
            value_right = join.get_column("Value_right")

            statistics = stats.pearsonr(value_left, value_right)
            pearson = statistics.correlation
            p_value = statistics.pvalue

            results_with_pearson.append(
                [
                    table_id,
                    join_col_idx,
                    target_col_idx,
                    target_col_name,
                    qcr,
                    pearson,
                    p_value,
                ]
            )
        except Exception as e:
            # print(f"Error computing Pearson for {table_id}: {e}")
            pass

    return results_with_pearson


results_with_pearson = compare_with_pearson(results)
results_with_pearson_df = pl.DataFrame(
    results_with_pearson,
    schema=[
        "dataset",
        "join_col_idx",
        "target_col_idx",
        "target_col_name",
        "QCR",
        "pearson",
        "p_value",
    ],
    orient="row",
).with_row_index("rank")

print(results_with_pearson_df)

indexer.close()
