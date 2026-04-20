import argparse
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

# use_ml_optimizer = freq_dict_path.exists() # FIX: optimizer support not completed yet
use_ml_optimizer = False

load_opts = {"ignore_errors": True}


def build_indexer() -> BLEND:
    print("--- Instantiating BLEND index ---")
    return BLEND(
        index_db_path,
        clean_args={"lowercase": True},
        max_cell_length=512,
        use_ml_optimizer=use_ml_optimizer,
        freq_dict_path=freq_dict_path,
    )


def run_indexing(indexer: BLEND) -> None:
    print("--- Indexing tables ---")
    index_tables(indexer, data_lake_path, True, None, 4, load_opts)

    print("--- Extracting token frequencies ---")
    freqs = indexer.extract_token_frequencies_from_db()
    print(freqs)
    freqs.write_csv(freq_dict_path)


def refresh_token_frequencies(indexer: BLEND) -> None:
    print("--- Extracting token frequencies ---")
    freqs = indexer.extract_token_frequencies_from_db()
    print(freqs)
    freqs.write_csv(freq_dict_path)


def compare_with_pearson(
    results_list: list,
    grouped_qdf: pl.DataFrame,
    key_column_name: str,
) -> list:
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
        except Exception:
            # print(f"Error computing Pearson for {table_id}: {e}")
            pass

    return results_with_pearson


def run_query_examples(indexer: BLEND) -> None:
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

    grouped_qdf = qdf_corr.group_by(key_column_name).agg(
        pl.col(target_column_name).mean()
    )
    grouped_qdf = grouped_qdf.rename({target_column_name: "Value_left"})

    keys = grouped_qdf.get_column(key_column_name).to_list()
    targets = grouped_qdf.get_column("Value_left").to_list()

    results = indexer.correlation_search(keys, targets, 20)
    results_df = pl.DataFrame(
        results,
        schema=["dataset", "join_col_idx", "target_col_idx", "QCR"],
        orient="row",
    )
    print(results_df)

    print("\n--- Comparison with actual Pearson correlation ---")
    results_with_pearson = compare_with_pearson(
        results,
        grouped_qdf,
        key_column_name,
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UNData BLEND example.")
    parser.add_argument(
        "-i",
        "--index",
        action="store_true",
        help="Rebuild the index and refresh token frequencies.",
    )
    parser.add_argument(
        "-q",
        "--queries",
        action="store_true",
        help="Run the example queries.",
    )
    return parser.parse_args()


def resolve_requested_stages(
    args: argparse.Namespace,
) -> tuple[bool, bool, bool]:
    explicit_selection = args.index or args.queries
    run_queries = args.queries or not explicit_selection
    run_indexing_stage = args.index
    refresh_frequencies = args.index or not explicit_selection

    if not index_db_path.exists():
        run_indexing_stage = True
        refresh_frequencies = True

    return run_indexing_stage, refresh_frequencies, run_queries


def main() -> int:
    args = parse_args()

    if not data_path.exists():
        print(f"Data path {data_path} does not exist.")
        return 1

    run_indexing_stage, refresh_frequencies, run_queries = resolve_requested_stages(
        args
    )

    indexer = build_indexer()
    try:
        if run_indexing_stage:
            if args.index and index_db_path.exists():
                print(f"--- Rebuilding index at {index_db_path} ---")
            elif not index_db_path.exists():
                print("--- Index not found, building it now ---")
            run_indexing(indexer)
        else:
            print(f"--- Index {index_db_path} already exists, skipping indexing ---")
            if refresh_frequencies:
                refresh_token_frequencies(indexer)

        if run_queries:
            run_query_examples(indexer)
    finally:
        indexer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
