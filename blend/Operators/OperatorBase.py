from abc import ABC, abstractmethod
from typing import List

from ..DBHandler import DBHandler


class Operator(ABC):
    DB: DBHandler | None = None  # DBHandler()

    def __init__(self, k: int):
        self.k = k

    def run(self, additionals: str = "") -> List[int]:
        sql = self.create_sql_query(self.DB, additionals=additionals)
        result = self.DB.execute_and_fetchall(sql)
        return [r for r in result[: self.k]]

    def set_db(self, db: DBHandler):
        self.DB = db

    @abstractmethod
    def cost(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def ml_cost(self, db: DBHandler) -> float:
        raise NotImplementedError

    @abstractmethod
    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        raise NotImplementedError
