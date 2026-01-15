import logging
import string
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
import polars.selectors as cs

whitespace_translator = str.maketrans(string.whitespace, " " * len(string.whitespace))

LRU_CACHE_SIZE = 1024


@lru_cache(maxsize=LRU_CACHE_SIZE)
def calculate_xash(token: str, hash_size: int = 128) -> int:
    """
    Calculates the XASH hash of a token.
    Setting is the same as provided by XASH/MATE authors.
    """

    number_of_ones = 5
    char = list(string.ascii_lowercase + string.digits + " ")

    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]

    n_bits_for_chars = 37 * segment_size
    length_bit_start = n_bits_for_chars
    n_bits_for_length = hash_size - length_bit_start
    token_size = len(token)

    # - Character position encoding
    result = 0
    # Pick the 5 most infrequent characters
    counts = Counter(token).items()
    sorted_counts = sorted(counts, key=lambda char_occurances: char_occurances[::-1])
    selected_chars = [char for char, _ in sorted_counts[:number_of_ones]]
    # Encode the position of the 5 most infrequent characters
    for c in selected_chars:
        if c not in char:
            continue
        # Calculate the mean position of the character and set the one bit in the corresponding segment
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = sum(indices) / len(indices)
        normalized_mean_index = mean_index / token_size
        segment = max(int(normalized_mean_index * segment_size - 1e-6), 0)  # Legacy fix
        location = char.index(c) * segment_size + segment
        result = result | 2**location

    # Rotate position encoding
    shift_distance = (
        length_bit_start
        * (token_size % (hash_size - length_bit_start))
        // (hash_size - length_bit_start)
    )
    left_bits = result << shift_distance
    wrapped_bits = result >> (n_bits_for_chars - shift_distance)
    cut_overlapping_bits = 2**n_bits_for_chars

    result = (left_bits | wrapped_bits) % cut_overlapping_bits

    # - Add length bit
    length_bit = 1 << (length_bit_start + token_size % n_bits_for_length)
    result = result | length_bit

    return result


def init_logger(logfile: Optional[Path] = None, stdout: bool = False):
    logger = logging.getLogger("BLEND")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if logfile and not any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)  # Set minimum level for file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout and not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Set minimum level for console
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


@lru_cache(maxsize=LRU_CACHE_SIZE)
def clean(
    s: Any,
    lowercase: bool = False,
    replace_whitespaces: bool = False,
    replace_custom: Optional[dict] = None,
    filter_bad_tokens: bool = False,
    bad_tokens: Optional[list[str]] = None,
):
    if not bad_tokens:
        bad_tokens = ["nan", "null", "none"]

    s = str(s)
    if lowercase:
        s = s.lower()
    if replace_whitespaces:
        s = s.translate(whitespace_translator)
    if replace_custom:
        s = s.translate(replace_custom)
    if filter_bad_tokens and s in bad_tokens:
        return ""
    return s.strip()


def _clean(
    column_name: str,
    lowercase: bool = True,
    replace_whitespaces: bool = True,
    filter_bad_tokens: bool = True,
    bad_tokens: Optional[list[str]] = None,
) -> pl.Expr:
    if not bad_tokens:
        bad_tokens = ["nan", "null", "none"]

    e = pl.col(column_name).cast(pl.String)

    if lowercase:
        e = e.str.to_lowercase()
    if replace_whitespaces:
        e = e.str.replace_all(r"[\t\n\r]", " ")
        e = e.str.strip_chars()
    if filter_bad_tokens:
        e = pl.when(e.is_in(set(bad_tokens))).then(pl.lit("")).otherwise(e)

    return e


def remove_null_rows(df: pl.DataFrame, *exclude_columns) -> pl.DataFrame:
    return df.filter(~pl.all_horizontal(pl.all().exclude(*exclude_columns).is_null()))


def remove_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df[[s.name for s in df if not (s.null_count() == df.height)]]


def parse_table(
    table_path: Path,
    scan_table_opts: dict,
    clean_function: Callable,
    clean_function_args: dict,
    xash_size: int,
    disable_xash: bool,
) -> tuple[str, pl.DataFrame | str]:
    table_id = table_path.stem
    format_ = table_path.suffix.replace(".", "")

    try:
        match format_:
            case "csv":
                table_df = pl.scan_csv(table_path, **scan_table_opts)
            case "parquet":
                table_df = pl.scan_parquet(table_path, **scan_table_opts)
            case _:
                raise ValueError(f"Unknown table format in {table_path}: {format_}")

        # BUG: here lazy-mode seems to be the worst choice:
        # filtering the all-nulls rows on the already collected
        # dataframe is much faster than doing this on a LazyFrame
        #
        # we need to keep track of the real row index of each record
        # even after dropping nulls, thus we create a new column to this aim
        table_df = (
            table_df.collect()
            .with_row_index(name="blend_row_index")
            .pipe(remove_null_rows, "blend_row_index")
            .pipe(remove_null_columns)
        )

        if table_df.shape[0] * table_df.shape[1] == 0:
            raise pl.exceptions.NoDataError("Empty table.")

    except (
        pl.exceptions.ComputeError,
        pl.exceptions.SchemaError,
        pl.exceptions.NoDataError,
        ValueError,
    ) as e:
        return table_id, f"{type(e)}::{str(e)}"

    # identify the numeric columns for the correlation part
    numeric_cols = set(table_df.select(cs.numeric()).columns)

    exprs = []
    for col_counter, col_name in enumerate(
        c for c in table_df.columns if c != "blend_row_index"
    ):
        is_numeric = col_name in numeric_cols
        if is_numeric:
            quadrant_expr = (
                pl.when(pl.col(col_name).is_not_null())
                .then(pl.col(col_name) >= pl.col(col_name).mean())
                .otherwise(None)
            )
        else:
            quadrant_expr = pl.lit(None, pl.Boolean)

        clean_expr = _clean(col_name, **clean_function_args)

        exprs.append(
            pl.struct(
                [
                    clean_expr.alias("cell_value"),
                    quadrant_expr.alias("quadrant"),
                    pl.lit(col_counter).alias("column_id"),
                ]
            ).alias(col_name)
        )

    all_data = (
        table_df.lazy()
        .select(
            [
                pl.lit(table_id).alias("table_id"),
                pl.col("blend_row_index").alias("row_id"),
                *exprs,
            ]
        )
        # Unpivot the table to go from Wide to Long format
        .unpivot(
            index=["table_id", "row_id"],
            variable_name="original_col_name",
            value_name="packed_data",
        )
        # Expand the struct back into individual columns
        .unnest("packed_data")
        .filter(pl.col("cell_value") != "")
        .select("table_id", "column_id", "row_id", "quadrant", "cell_value")
        .collect()
    )

    if disable_xash:
        final_data = all_data.with_columns(pl.lit(int(0).to_bytes(), pl.Binary))
    else:
        superkey_data = all_data.group_by("row_id").agg(
            pl.map_groups(
                ["cell_value"],
                lambda values: calculate_superkey_for_row(
                    values[0].to_list(), xash_size
                ),
                return_dtype=pl.Binary,
                returns_scalar=True,
            ).alias("super_key")
        )

        final_data = all_data.join(superkey_data, on="row_id", coalesce=True)

    return table_id, final_data


def calculate_superkey_for_row(cell_values: list, xash_size: int) -> bytes:
    superkey = 0
    for value in cell_values:
        if value is None:
            print(cell_values)
        superkey |= calculate_xash(value, xash_size)
    return superkey.to_bytes(16, byteorder="big")
    # return bytes(f"{superkey:0128b}".encode())
