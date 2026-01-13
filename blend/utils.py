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


@lru_cache(maxsize=1_000)
def clean(
    s: Any,
    lowercase: bool = False,
    replace_whitespaces: bool = False,
    replace_custom: Optional[dict] = None,
    filter_bad_tokens: bool = False,
    bad_tokens: Optional[set] = {"nan", "null", "none"},
):
    s = str(s)
    if lowercase:
        s = s.lower()
    if replace_whitespaces:
        s = s.translate(whitespace_translator)
    if replace_custom:
        s = s.translate(replace_custom)
    if bad_tokens and s in bad_tokens:
        return ""
    return s.strip()


@lru_cache(maxsize=1_000)
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
    logger = logging.getLogger("JOSIE")
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


def parse_table(
    table_path: Path,
    scan_table_opts: dict,
    clean_function: Callable,
    clean_function_args: dict,
    xash_size: int,
    disable_xash: bool,
) -> tuple[str, pl.DataFrame | str]:
    table_id = table_path.stem
    format = table_path.suffix.replace(".", "")

    try:
        match format:
            case "csv":
                table_df = pl.scan_csv(table_path, **scan_table_opts)
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
                .alias("cell_value")
            )
        else:
            cleaned = pl.Series(
                "cell_value",
                [
                    clean_function(item, **clean_function_args)
                    for row_counter, item in column.rows()
                ],
            )

        result_df = column.with_columns(
            [
                cleaned,
                pl.lit(table_id).alias("table_id"),
                pl.lit(col_counter).alias("column_id"),
                pl.col("blend_row_index").alias("row_id"),
            ]
        )

        is_numeric = col_name in numeric_cols
        if is_numeric:
            mean = column.select(col_name).to_series().mean()
            result_df = result_df.with_columns(
                pl.when(pl.col(col_name).is_not_null())
                .then(pl.col(col_name) >= mean)
                .otherwise(None)
                .alias("quadrant")
            )
        else:
            result_df = result_df.with_columns(
                pl.lit(None).cast(pl.Boolean).alias("quadrant")
            )

        columns_df.append(result_df.drop("blend_row_index", col_name))

    all_data = pl.concat(columns_df)

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

        final_data = all_data.join(superkey_data, on="row_id")

    return table_id, final_data


def calculate_superkey_for_row(cell_values: list, xash_size: int) -> bytes:
    superkey = 0
    for value in cell_values:
        superkey |= calculate_xash(value, xash_size)
    return bytes(f"{superkey:0128b}".encode())
