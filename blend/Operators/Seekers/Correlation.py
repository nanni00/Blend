from numbers import Number
import numpy as np

# FIX: move to pure polars syntax
import pandas as pd

from ...DBHandler import DBHandler
from .SeekerBase import Seeker


class Correlation(Seeker):
    def __init__(
        self,
        source_values: list[str],
        target_values: list[Number],
        k: int = 10,
        hash_size: int = 256,
    ) -> None:
        super().__init__(k)

        # NaNs are dropped and duplicates are removed
        # by grouping and aggregating by the average
        grouped = (
            pd.DataFrame({"source": source_values, "target": target_values})
            .dropna()
            .groupby("source")
            .mean()
        )

        # the input source save the index values, which,
        # after grouping, become the source values
        self.input_source = grouped.index.values

        # Here we keep the numerical values (i.e. those
        # on which we actually compute the correlation)
        self.input_target = grouped["target"].values

        # BLEND makes more flexible the hash size
        self.hash_size = hash_size

        # in the SQL query we do not return QCR score, we
        # are only interested in the order of the resutls
        self.base_sql = f"""
            SELECT 
                TableId, 
                catcol, 
                numcol,
                qcr
            FROM (
                SELECT TableId, catcol, numcol, 
                        (2 * SUM((CellValue IN ($TRUETOKENS$) = Quadrant)::INT) - COUNT(*)) AS score,
                        (2 * SUM((CellValue IN ($TRUETOKENS$) = Quadrant)::INT) - COUNT(*)) / COUNT(*) AS qcr,
                FROM (
                    SELECT
                        categorical.CellValue,
                        categorical.TableId,
                        categorical.ColumnId catcol,
                        numerical.ColumnId numcol,
                        SUM(numerical.Quadrant::INT) / COUNT(*) > 0.5 AS Quadrant,
                        COUNT(DISTINCT numerical.CellValue) AS num_unique,
                        MIN(numerical.CellValue) AS any_cellvalue
                    FROM (
                        SELECT * 
                        FROM AllTables 
                        WHERE 
                            RowId < {self.hash_size}
                        AND (
                                CellValue IN ($FALSETOKENS$)
                            OR 
                                CellValue IN ($TRUETOKENS$)
                            ) $ADDITIONALS$
                        ) categorical
                    JOIN (
                        SELECT * 
                        FROM AllTables 
                        WHERE 
                            RowId < {self.hash_size}
                        AND 
                            Quadrant IS NOT NULL $ADDITIONALS$
                        ) numerical
                    ON categorical.TableId = numerical.TableId 
                    AND categorical.RowId = numerical.RowId
                    GROUP BY categorical.TableId, categorical.ColumnId, numerical.ColumnId, categorical.CellValue
                ) grouped_cellvalues
                GROUP BY TableId, catcol, numcol
                HAVING 
                    COUNT(*) > 1 
                AND (
                        COUNT(DISTINCT any_cellvalue) > 1 
                    OR 
                        SUM(num_unique) > COUNT(*)
                    )
            ) scores
            WHERE catcol != numcol
            ORDER BY ABS(score) DESC
            LIMIT $TOPK$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # create an array of 0/1 values, where 1 if the relative target
        # value is above the average of the target numbers, 0 otherwise
        # the 0/1 value, is basically the quadrant indicator
        self.input_target = self.input_target.astype(float)
        target_average = np.mean(self.input_target)
        target_int = np.where(self.input_target >= target_average, 1, 0).astype(int)

        # clean the string values (remove 'nan', etc)
        self.input_source = db.clean_value_collection(self.input_source)

        # take the negative target values
        source_0 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 0]
        )

        # take the positive target values
        source_1 = db.create_sql_list_str(
            [key for key, qdr in zip(self.input_source, target_int) if qdr == 1]
        )

        sql = self.base_sql.replace("$TOPK$", str(self.k))
        sql = sql.replace("$ADDITIONALS$", additionals)
        sql = sql.replace("$FALSETOKENS$", source_0)
        sql = sql.replace("$TRUETOKENS$", source_1)

        return sql

    def cost(self) -> int:
        return 6

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([[token for token in self.input_source]], db)
