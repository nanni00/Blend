import random
from numbers import Number
from pathlib import Path
from typing import Iterable, Optional, Union, Any

import duckdb
import pandas as pd
import polars as pl


class DBHandler(object):
    def __init__(
        self,
        db_path: Path,
        index_table: Optional[str] = None,
        use_ml_optimizer: bool = False,
        freq_dict_path: str | None = None,
        use_pandas: bool = True,
    ) -> None:
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

    def drop_table(self):
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                DROP TABLE IF EXISTS {self.index_table} CASCADE;
                CHECKPOINT {self.db_name};
            """)

    def create_table(self):
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                CREATE TABLE {self.index_table} (
                cell_value           VARCHAR,
                table_id             VARCHAR,
                column_id            UINTEGER,
                row_id               UINTEGER,
                quadrant             BOOLEAN,
                super_key            BYTEA,
                PRIMARY KEY (table_id, column_id, row_id)
            );""")

    def create_column_indexes(self):
        with duckdb.connect(self.db_path) as con:
            con.sql(f"CREATE INDEX table_id_idx ON {self.index_table} (table_id);")
            con.sql(f"CREATE INDEX cell_value_idx ON {self.index_table} (cell_value);")

    def save_data_to_duckdb(self, data: dict | pl.DataFrame):
        with duckdb.connect(self.db_path) as con:
            con.sql(f"INSERT INTO {self.index_table} SELECT * FROM data;")

    def close(self) -> None:
        pass

    def clean_query(self, query: str) -> str:
        """Replaces the 'all_tables' index name"""
        return query.replace("all_tables", f"{self.index_table}")

    def execute_and_fetchall(self, query: str) -> list[Union[tuple, list]]:
        """Returns results"""
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(superkey)", "superkey")

        with duckdb.connect(self.db_path, read_only=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

    def execute_and_fetchyield(self, query: str, params: Optional[tuple] = None):
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(superkey)", "superkey")

        with duckdb.connect(self.db_path, read_only=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                while rows := cursor.fetchmany(size=1000):
                    for row in rows:
                        yield row

    def get_table_from_index(self, table_id: int) -> pd.DataFrame | pl.DataFrame:
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
        values = [str(x).replace("'", "") for x in values]
        return "'{}'".format("' , '".join(set(values)))

    @staticmethod
    def create_sql_list_numeric(values: Iterable[Number]) -> str:
        values = [str(x) for x in values]
        return "{}".format(" , ".join(values))

    @staticmethod
    def random_subquery_name() -> str:
        return f"subquery{random.random() * 1000000:.0f}"
