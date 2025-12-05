from typing import Iterable

from ...DBHandler import DBHandler
from .SeekerBase import Seeker


class SingleColumnOverlap(Seeker):
    def __init__(self, input_query_values: Iterable[str], k: int = 10) -> None:
        super().__init__(k)

        self.input = set(input_query_values)

        # The final output results list is sorted by
        # the COUNT(DISTINCT), applying a set semantic
        self.base_sql = """
        SELECT TableId, ColumnId, 
            COUNT(DISTINCT CellValue) AS DistinctTokensCount, 
            COUNT(CellValue) AS TokensCount FROM AllTables
        WHERE CellValue IN ($TOKENS$) $ADDITIONALS$
        GROUP BY TableId, ColumnId
        ORDER BY COUNT(DISTINCT CellValue) DESC
        LIMIT $TOPK$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        sql = self.base_sql.replace("$TOPK$", str(self.k))
        sql = sql.replace("$ADDITIONALS$", additionals)
        sql = sql.replace(
            "$TOKENS$", db.create_sql_list_str(db.clean_value_collection(self.input))
        )

        return sql

    def cost(self) -> int:
        return 4

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([list(self.input)], db)
