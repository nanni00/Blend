import logging
import multiprocessing
import os
from _thread import LockType
from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import cleandoc
from pathlib import Path
from time import time
from typing import Callable, Optional

import polars as pl
from tqdm import tqdm

from .blend import BLEND
from .db import DBHandler
from .utils import init_logger, parse_table


def _process_task(
    table_path: Path,
    scan_table_opts: dict,
    clean_function: Callable,
    clean_function_args: dict,
    xash_size: int,
    disable_xash: bool,
    db_handler: DBHandler,
    lock: LockType,
):
    table_id, df_or_error = parse_table(
        table_path,
        scan_table_opts,
        clean_function,
        clean_function_args,
        xash_size,
        disable_xash,
    )

    if isinstance(df_or_error, pl.DataFrame):
        # FIX: this sequentialization can be improved
        with lock:
            db_handler.save_data_to_duckdb(df_or_error)
        return table_id, True
    return table_id, False


def index_tables_old(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    scan_table_opts: dict = {},
) -> tuple:
    """
    Index all the tables stored under the given tables path, considering
    it as a flat folder with only tables.

    :param indexer: A BLEND indexer instance.
    :param tables_path: The path to the folder containing the tables to index.
    :param logfile_path: The path to a logfile.
    :param max_workers: Maximum number of processes to instantiate.
    :param scan_table_opts: A dictionary with Polars scan csv/parquet/... configuration options.
    :return: A tuple with timing for the tables parse and insertion time, support indexes creation time and total time.
    """
    if not tables_path.exists():
        raise FileNotFoundError(f"tables path doesn't exist: {tables_path}")

    init_logger(logfile_path, log_stdout)

    logger = logging.getLogger(f"blend_logger_{os.getpid()}")

    # get IDs of the effective tables
    table_ids = os.listdir(tables_path)

    # drop the main index if already exists
    indexer.db_handler.drop_index_table()

    # create the main index
    indexer.db_handler.create_index_table()

    start_t = time()

    # FIX: sometimes this doesn't work at all
    # (check on polars with multiproc setting),
    # even by changhing with different mp_context
    with ProcessPoolExecutor(max_workers) as executor:
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        futures = {
            executor.submit(
                _process_task,
                tables_path.joinpath(table_id),
                scan_table_opts,
                indexer._clean_function,
                indexer._clean_function_args,
                indexer.xash_size,
                indexer.disable_xash,
                indexer.db_handler,
                lock,
            )
            for table_id in list(table_ids)
        }

        non_empty_tables = 0

        for future in tqdm(
            as_completed(futures),
            desc="Parsing and storing tables: ",
            total=len(table_ids),
            disable=not log_stdout,
        ):
            try:
                table_id, success = future.result()
                non_empty_tables += success
            except Exception as e:
                logger.error(f"[error:{type(e)}][msg:{e}]")

    end_ins_t = time()

    # create indexes
    s = f"""
        Tables ingestion completed.
        Correctly parsed {non_empty_tables}.
        Creating indexes...
        """
    logger.info(cleandoc(s))

    indexer.db_handler.create_column_indexes()
    end_idx_t = time()

    logger.info("Index creation completed.")

    indexer.db_handler.close()
    return (end_ins_t - start_t, end_idx_t - end_ins_t, end_idx_t - start_t)


def index_tables(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    scan_table_opts: dict = {},
    batch_size: Optional[int] = None,
) -> tuple:
    """
    Index all the tables stored under the given tables path, considering
    it as a flat folder with only tables.

    :param indexer: A BLEND indexer instance.
    :param tables_path: The path to the folder containing the tables to index.
    :param logfile_path: The path to a logfile.
    :param max_workers: Maximum number of processes to instantiate. (not used - single process)
    :param scan_table_opts: A dictionary with Polars scan csv/parquet/... configuration options.
    :param batch_size: Batch size as total number of rows inserted at a time into the underlying database (default, one table at a time).
    :return: A tuple with timing for the tables parse and insertion time, support indexes creation time and total time.
    """
    if not tables_path.exists():
        raise FileNotFoundError(f"tables path doesn't exist: {tables_path}")

    if not batch_size:
        batch_size = 0

    init_logger(logfile_path, log_stdout)

    logger = logging.getLogger(f"blend_logger_{os.getpid()}")

    # get IDs of the effective tables
    table_ids = os.listdir(tables_path)

    # drop the main index if already exists
    indexer.db_handler.drop_index_table()

    # create the main index
    indexer.db_handler.create_index_table()

    parsed_tables = 0

    start_t = time()

    data_to_store = []

    for table_id in tqdm(
        table_ids, desc="Parsing and storing tables", disable=not log_stdout
    ):
        table_path = tables_path.joinpath(table_id)
        table_id, df = parse_table(
            table_path,
            scan_table_opts,
            indexer._clean_function,
            indexer._clean_function_args,
            indexer.xash_size,
            indexer.disable_xash,
        )

        if not isinstance(df, pl.DataFrame):
            continue

        data_to_store.append(df)

        if sum(_df.height for _df in data_to_store) >= batch_size:
            # print("Large insert!")
            t = time()
            indexer.db_handler.save_data_to_duckdb(data_to_store)
            data_to_store.clear()
            t = time() - t
            # print(f"Insert time: {round(t, 3)} s")
        parsed_tables += 1

    indexer.db_handler.save_data_to_duckdb(data_to_store)
    end_ins_t = time()

    # create indexes
    s = f"""
        Tables ingestion completed.
        Correctly parsed {parsed_tables}.
        Creating indexes...
        """
    logger.info(cleandoc(s))

    indexer.db_handler.create_column_indexes()
    end_idx_t = time()

    logger.info("Index creation completed.")

    indexer.db_handler.close()
    return (end_ins_t - start_t, end_idx_t - end_ins_t, end_idx_t - start_t)
