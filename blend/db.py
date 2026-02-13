import random
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import duckdb
import pandas as pd
import polars as pl


class DBHandler(object):
    """Handles interactions with the DuckDB database for the BLEND index.

    Attributes:
        connection: The DuckDB connection object.
        cursor: The DuckDB cursor object.
        dbms: The database management system being used (default: "duckdb").
        use_pandas: Whether to return results as pandas DataFrames (default: True).
        db_path: The path to the DuckDB database file.
        index_table: The name of the table used for indexing.
        db_name: The name of the database.
        use_ml_optimizer: Whether to use the ML optimizer.
        frequency_dict: A dictionary mapping tokens to their frequencies.
    """

    def __init__(
        self,
        db_path: Path,
        index_table: Optional[str] = None,
        use_ml_optimizer: bool = False,
        freq_dict_path: str | None = None,
        use_pandas: bool = True,
    ) -> None:
        """Initializes the DBHandler.

        Args:
            db_path: The path to the DuckDB database file.
            index_table: The name of the table used for indexing. Defaults to "all_tables".
            use_ml_optimizer: Whether to use the ML optimizer. Defaults to False.
            freq_dict_path: Path to the frequency dictionary CSV file. Required if use_ml_optimizer is True.
            use_pandas: Whether to return results as pandas DataFrames. Defaults to True.

        Raises:
            FileNotFoundError: If the parent directory of db_path does not exist.
            AssertionError: If use_ml_optimizer is True but freq_dict_path is not provided.
        """
        self.connection = None
        self.cursor = None
        self.dbms = "duckdb"  # we'll use only duckdb
        self.use_pandas = use_pandas

        self.db_path = db_path
        if not self.db_path.parent.exists():
            raise FileNotFoundError(
                f"DB directory doesn't exist: {self.db_path.parent}"
            )

        self.index_table = index_table if isinstance(index_table, str) else "all_tables"
        self.db_name = self.db_path.stem.replace("-", "_")

        self.use_ml_optimizer = use_ml_optimizer

        # BLEND supports also the possibility to
        # optimize the general plan, but a frequency
        # dictionary is needed
        if self.use_ml_optimizer:
            assert freq_dict_path, (
                "Frequencies file must be provided to use ML optimizer"
            )

            df = pd.read_csv(freq_dict_path)
            self.frequency_dict = dict(zip(df["tokenized"], df["frequency"]))
        else:
            self.frequency_dict = {}

    def drop_index_table(self):
        """Drops the index table if it exists."""
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                DROP TABLE IF EXISTS {self.index_table} CASCADE;
                CHECKPOINT {self.db_name};
            """)

    def create_index_table(self):
        """Creates the index table."""
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                CREATE TABLE {self.index_table} (
                table_id             VARCHAR,
                column_id            UINTEGER,
                row_id               UINTEGER,
                quadrant             BOOLEAN,
                cell_value           VARCHAR,
                super_key            BYTEA,
                PRIMARY KEY (table_id, column_id, row_id)
            );""")

    def create_column_indexes(self):
        """Creates indexes on the cell_value column."""
        with duckdb.connect(self.db_path) as con:
            # con.sql(f"CREATE INDEX table_id_idx ON {self.index_table} (table_id);")
            con.sql(f"CREATE INDEX cell_value_idx ON {self.index_table} (cell_value);")

    def save_data_to_duckdb(self, data: pl.DataFrame | list[pl.DataFrame] | Path):
        """Saves data to the DuckDB database.

        Args:
            data: A polars DataFrame, a list of polars DataFrames, or a Path to a parquet file.
        """
        if isinstance(data, pl.DataFrame):
            data = [data]

        with duckdb.connect(self.db_path) as con:
            if isinstance(data, list):
                for df in data:
                    con.sql(f"INSERT INTO {self.index_table} SELECT * FROM df;")
            elif isinstance(data, Path):
                filename = data.absolute().as_posix()
                con.sql(
                    f"INSERT INTO {self.index_table} SELECT * FROM read_parquet('{filename}');"
                )

    def close(self) -> None:
        """Closes the database connection (placeholder)."""
        pass

    def clean_query(self, query: str) -> str:
        """Replaces the 'all_tables' index name with the actual index table name.

        Args:
            query: The SQL query string.

        Returns:
            The modified query string.
        """
        return query.replace("all_tables", f"{self.index_table}")

    def execute_and_fetchall(self, query: str) -> list[Union[tuple, list]]:
        """Executes a query and returns all results.

        Args:
            query: The SQL query string.

        Returns:
            A list of tuples or lists containing the query results.
        """
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(super_key)", "super_key")

        with duckdb.connect(self.db_path, read_only=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    def execute_and_fetchyield(self, query: str, params: Optional[tuple] = None):
        """Executes a query and yields results in batches.

        Args:
            query: The SQL query string.
            params: Optional parameters for the query.

        Yields:
            Rows from the query result.
        """
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(super_key)", "super_key")

        with duckdb.connect(self.db_path, read_only=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                while rows := cursor.fetchmany(size=1000):
                    for row in rows:
                        yield row

    def get_table_from_index(self, table_id: str) -> pd.DataFrame | pl.DataFrame:
        sql = f"""
        SELECT cell_value, column_id, row_id
        FROM all_tables
        WHERE table_id = {table_id}
        """

        results = self.execute_and_fetchall(sql)

        df = pd.DataFrame(
            results, columns=["cell_value", "column_id", "row_id"], dtype=str
        )
        df = df.drop_duplicates()
        df = df.pivot(index="row_id", columns="column_id", values="cell_value")
        df.index.name = None
        df.columns.name = None

        df = df if self.use_pandas else pl.from_pandas(df)

        return df

    def table_ids_to_sql(self, table_ids: list[int]) -> str:
        if len(table_ids) == 0:
            return "SELECT 0 AS table_id WHERE 1 = 0"

        if self.dbms == "postgres":
            return f"""
            SELECT * FROM (
                VALUES {" ,".join([f"('{table_id}')" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}(table_id)
            """

        return f"""
            SELECT table_id FROM (
            {" UNION ALL ".join([f"SELECT '{table_id}' AS table_id" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}
        """

    def get_token_frequencies(self, tokens: Iterable[str]) -> dict[str, int]:
        tokens = self.clean_value_collection(set(tokens))

        return {token: self.frequency_dict.get(token, 1) for token in tokens}

    def remove_table_from_index(self, table_id: str):
        sql = f"DELETE FROM all_tables WHERE table_id = '{table_id}'"

        self.execute_and_fetchall(sql)

    @staticmethod
    def clean_value_collection(values: Iterable[Any]) -> list[str]:
        return [
            str(v).replace("'", "''").strip() for v in values if str(v).lower() != "nan"
        ]

    @staticmethod
    def create_sql_list_str(values: Iterable[Any]) -> str:
        values = set(map(lambda x: str(x).replace("'", ""), values))
        return "'{}'".format("' , '".join(values))

    @staticmethod
    def create_sql_list_numeric(values: Iterable[Number]) -> str:
        return "{}".format(" , ".join(map(str, values)))

    @staticmethod
    def random_subquery_name() -> str:
        return f"subquery{random.random() * 1000000:.0f}"
