import random
from numbers import Number
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any

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

        self.index_table = index_table if isinstance(index_table, str) else "AllTables"
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
                CellValue           VARCHAR,
                TableId             VARCHAR,
                ColumnId            UINTEGER,
                RowId               UINTEGER,
                Quadrant            BOOLEAN,
                SuperKey            BYTEA,
                PRIMARY KEY (TableId, ColumnId, RowId)
            );""")

    def create_column_indexes(self):
        with duckdb.connect(self.db_path) as con:
            con.sql(f"CREATE INDEX TableId_idx ON {self.index_table} (TableId);")
            con.sql(f"CREATE INDEX CellValue_idx ON {self.index_table} (CellValue);")

    def save_data_to_duckdb(self, data: Dict | pl.DataFrame):
        # schema = [
        #     ("CellValue", pl.String),
        #     ("TableId", pl.String),
        #     ("ColumnId", pl.UInt32),
        #     ("RowId", pl.UInt32),
        #     ("Quadrant", pl.Boolean),
        #     ("SuperKey", pl.Binary),
        # ]
        #
        # # TODO: with pl.Series, maybe the lazy
        # # computation here is not correctly adopted
        # data = pl.LazyFrame(
        #     # [pl.Series(c, data[c], d) for c, d in schema],
        #     {c: data[c] for c, _ in schema},
        #     schema=schema,
        #     orient="col",
        # )

        with duckdb.connect(self.db_path) as con:
            con.sql(f"INSERT INTO {self.index_table} SELECT * FROM data;")

    def close(self) -> None:
        pass

    def clean_query(self, query: str) -> str:
        """Replaces the 'AllTables' index name"""
        return query.replace("AllTables", f"{self.index_table}")

    def execute_and_fetchall(self, query: str) -> List[Union[Tuple, List]]:
        """Returns results"""
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(superkey)", "superkey")

        try:
            with duckdb.connect(self.db_path, read_only=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
        except Exception as e:
            print(query)
            raise e

        return results

    def execute_and_fetchyield(self, query: str, params: Optional[tuple] = None):
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(superkey)", "superkey")

        try:
            with duckdb.connect(self.db_path, read_only=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    while rows := cursor.fetchmany(size=1000):
                        for row in rows:
                            yield row
        except Exception as e:
            print(f'Failure with query "{query}": {e}')
            raise e

    def get_table_from_index(self, table_id: int) -> pd.DataFrame | pl.DataFrame:
        sql = f"""
        SELECT CellValue, ColumnId, RowId
        FROM AllTables
        WHERE TableId = {table_id}
        """

        results = self.execute_and_fetchall(sql)

        df = pd.DataFrame(
            results, columns=["CellValue", "ColumnId", "RowId"], dtype=str
        )
        df = df.drop_duplicates()
        df = df.pivot(index="RowId", columns="ColumnId", values="CellValue")
        df.index.name = None
        df.columns.name = None

        df = df if self.use_pandas else pl.from_pandas(df)

        return df

    def table_ids_to_sql(self, table_ids: List[int]) -> str:
        if len(table_ids) == 0:
            return "SELECT 0 AS TableId WHERE 1 = 0"

        if self.dbms == "postgres":
            return f"""
            SELECT * FROM (
                VALUES {" ,".join([f"('{table_id}')" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}(TableId)
            """

        return f"""
            SELECT TableId FROM (
            {" UNION ALL ".join([f"SELECT '{table_id}' AS TableId" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}
        """

    def get_token_frequencies(self, tokens: Iterable[str]) -> dict[str, int]:
        tokens = self.clean_value_collection(set(tokens))

        return {token: self.frequency_dict.get(token, 1) for token in tokens}

    def remove_table_from_index(self, table_id: str):
        sql = f"DELETE FROM AllTables WHERE TableId = '{table_id}'"

        self.execute_and_fetchall(sql)

    @staticmethod
    def clean_value_collection(values: Iterable[Any]) -> List[str]:
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
