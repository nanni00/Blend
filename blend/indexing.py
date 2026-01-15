import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import cleandoc
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from time import time
from typing import Optional

import polars as pl
from tqdm import tqdm

from .blend import BLEND
from .db import DBHandler
from .utils import init_logger, parse_table


def _db_worker(queue: Optional[Queue], tmp_path: Optional[Path], db_handler: DBHandler):
    """Dedicated consumer process that handles all DB writes."""
    while True:
        if queue:
            data = queue.get()
            if data is None:  # Poison pill to stop the process
                break

            if not isinstance(data, pl.DataFrame):
                raise TypeError(f"Expected polars.DataFrame, found: {type(data)}")

            try:
                db_handler.save_data_to_duckdb([data])
            except Exception as e:
                print(f"DB Write Error: {e}")
        elif tmp_path:
            files = os.listdir(tmp_path)
            stop = False
            for filename in files:
                if filename == "STOP":
                    stop = True
                    break
                else:
                    file = tmp_path.joinpath(filename)
                    db_handler.save_data_to_duckdb(file.absolute())

            if stop:
                break


def _process_task(
    table_path: Path,
    load_opts: Optional[dict],
    clean_args: Optional[dict],
    xash_size: int,
    db_handler: DBHandler,
    queue: Optional[Queue],
    tmp_path: Optional[Path],
):
    table_id, df = parse_table(
        table_path,
        load_opts,
        clean_args,
        xash_size,
    )

    if isinstance(df, pl.DataFrame):
        if queue:
            queue.put(df)
        elif tmp_path:
            file = tmp_path.joinpath(f"{uuid.uuid4()}.parquet")
            df.write_parquet(file, compression_level=22)
            os.remove(file)

        return table_id, True
    return table_id, False


def index_tables(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    load_opts: Optional[dict] = None,
    tmp_path: Optional[Path] = None,
) -> tuple:
    """
    Index all the tables stored under the given tables path, considering
    it as a flat folder with only tables.

    :param indexer: A BLEND indexer instance.
    :param tables_path: The path to the folder containing the tables to index.
    :param logfile_path: The path to a logfile.
    :param max_workers: Maximum number of processes to instantiate.
    :param load_opts: A dictionary with Polars scan csv/parquet/... configuration options. See blend.utils.load_table.
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

    # Create the Manager and Queue
    manager = Manager()

    if not tmp_path:
        queue = manager.Queue(
            maxsize=100
        )  # Optional: limit size to prevent RAM overflow
    else:
        queue = None

    # Start the DB Worker Process
    db_writer = Process(target=_db_worker, args=(queue, None, indexer.db_handler))
    db_writer.start()

    start_t = time()

    # FIX: sometimes this doesn't work at all
    # even by changhing with different mp_context
    # (check on polars with multiproc setting),
    with ProcessPoolExecutor(max_workers) as executor:
        futures = {
            executor.submit(
                _process_task,
                tables_path.joinpath(table_id),
                load_opts,
                indexer._clean_args,
                indexer.xash_size,
                indexer.db_handler,  # ty: ignore
                queue,
                tmp_path,
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

    # Stop the DB worker
    if queue:
        queue.put(None)
    elif tmp_path:
        with open(tmp_path.joinpath("STOP"), "w") as file:
            file.write("STOP")
    db_writer.join()

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


def _index_tables_seq(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    load_opts: dict = {},
    batch_size: Optional[int] = None,
) -> tuple:
    """
    Index all the tables stored under the given tables path, considering
    it as a flat folder with only tables.

    :param indexer: A BLEND indexer instance.
    :param tables_path: The path to the folder containing the tables to index.
    :param logfile_path: The path to a logfile.
    :param max_workers: Maximum number of processes to instantiate. (not used - single process)
    :param load_opts: A dictionary with Polars scan csv/parquet/... configuration options.
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
            load_opts,
            indexer._clean_function_args,
            indexer.xash_size,
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
