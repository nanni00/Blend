import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Optional, Protocol

import polars as pl
from tqdm import tqdm

from .blend import BLEND
from .db import DBHandler
from .utils import init_logger, parse_table


class _DataFrameQueue(Protocol):
    def get(self) -> pl.DataFrame | None: ...

    def put(self, item: pl.DataFrame | None, timeout: Optional[float] = None) -> None: ...


def _db_worker(
    queue: Optional[_DataFrameQueue],
    batch_rows: int,
    db_handler: DBHandler,
) -> None:
    """Consume parsed tables from the queue and write them to DuckDB.

    Args:
        queue: The queue containing parsed tables to write.
        batch_rows: The number of rows to accumulate before writing.
        db_handler: The DBHandler instance.
    """
    if queue is None:
        raise ValueError("queue must be provided")

    dataframes: list[pl.DataFrame] = []
    while True:
        df = queue.get()
        if df is None:  # Poison pill to stop the process
            break

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected polars.DataFrame, found: {type(df)}")
        dataframes.append(df)
        if sum(_df.height for _df in dataframes) < batch_rows:
            continue

        try:
            db_handler.save_data_to_duckdb(dataframes)
            dataframes.clear()
        except Exception as e:
            print(f"DB Write Error: {e}")

    if dataframes:
        try:
            db_handler.save_data_to_duckdb(dataframes)
        except Exception as e:
            print(f"DB Write Error: {e}")


def _table_parsing_worker(
    table_path: Path,
    load_opts: Optional[dict],
    clean_args: Optional[dict],
    xash_size: int,
    max_cell_length: Optional[int],
    queue: Optional[_DataFrameQueue],
) -> tuple[str, bool]:
    """Parse a table and send the result to the DB worker.

    Args:
        table_path: Path to the table file.
        load_opts: Options for loading the table.
        clean_args: Options for cleaning the table.
        xash_size: Size of the XASH hash.
        max_cell_length: Maximum length of cell values.
        queue: Queue used to hand parsed dataframes to the DB worker.

    Returns:
        The parsed table ID and whether parsing produced a non-empty dataframe.
    """
    table_id, df = parse_table(
        table_path,
        load_opts,
        clean_args,
        xash_size,
        max_cell_length,
    )

    if isinstance(df, pl.DataFrame):
        if queue is None:
            raise ValueError("queue must be provided")
        queue.put(df, timeout=20)
        return table_id, True
    return table_id, False


def index_tables(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    load_opts: Optional[dict] = None,
    max_queue_size: Optional[int] = None,
    batch_rows: Optional[int] = None,
) -> tuple[float, float, float]:
    """Index all the tables stored under the given tables path.

    It considers the path as a flat folder with only tables.

    Args:
        indexer: A BLEND indexer instance.
        tables_path: The path to the folder containing the tables to index.
        log_stdout: Whether to log to stdout.
        logfile_path: The path to a logfile.
        max_workers: Maximum number of processes to instantiate.
        load_opts: A dictionary with Polars scan csv/parquet/... configuration options. See blend.utils.load_table.
        max_queue_size: Maximum size of the queue used between parsers and the DB writer.
        batch_rows: Number of rows to accumulate before each DB write.

    Returns:
        A tuple containing insertion time, index creation time, and total runtime.
    """
    if not tables_path.exists():
        raise FileNotFoundError(f"tables path doesn't exist: {tables_path}")

    if max_queue_size is None:
        max_queue_size = 100
    if batch_rows is None:
        batch_rows = 1_000_000

    init_logger(logfile_path, log_stdout)

    logger = logging.getLogger(f"blend_logger_{os.getpid()}")

    # get IDs of the effective tables
    table_ids = os.listdir(tables_path)

    # drop the main index if already exists
    logger.info("Dropping old index table if exists...")
    indexer.db_handler.drop_index_table()

    # create the main index
    logger.info("Creating new index table...")
    indexer.db_handler.create_index_table()

    # Create the Manager and Queue
    manager = Manager()
    # Optional: limit size to prevent RAM overflow
    queue = manager.Queue(max_queue_size)

    # Start the DB Worker Process
    db_writer = Process(target=_db_worker, args=(queue, batch_rows, indexer.db_handler))
    db_writer.start()

    start_t = time.time()

    # TODO: work on Windows? (check mp_context-polars)
    # TODO: Timeout for _process_task/_db_worker?
    try:
        with ProcessPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(
                    _table_parsing_worker,
                    tables_path.joinpath(table_id),
                    load_opts,
                    indexer._clean_args,
                    indexer.xash_size,
                    indexer.max_cell_length,
                    queue,
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

    finally:
        end_ins_t = time.time()

        # Stop the DB worker
        queue.put(None)
        db_writer.join(30)
        if db_writer.is_alive():
            db_writer.terminate()
            db_writer.join()
        manager.shutdown()

    time_insertion = end_ins_t - start_t

    logger.info(f"Tables ingestion completed in {time_insertion:.2f} seconds.")
    logger.info(f"Correctly parsed {non_empty_tables} tables.")
    logger.info("Creating column indexes (this may take some time)...")

    # create indexes
    start_idx_t = time.time()
    indexer.db_handler.create_column_indexes()
    time_indexes_creation = time.time() - start_idx_t

    logger.info(f"Indexes created in {time_indexes_creation:.2f} seconds.")

    logger.debug("Closing DB...")
    indexer.db_handler.close()
    logger.info("Index creation completed.")

    time_total = time.time() - start_t
    return time_insertion, time_indexes_creation, time_total
