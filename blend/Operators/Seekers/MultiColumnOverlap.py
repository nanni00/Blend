from collections import defaultdict
from heapq import heappop, heappush
from typing import Any

import numpy as np
from tqdm import tqdm

# FIX: include polars
import pandas as pd

from ...DBHandler import DBHandler
from ...utils import calculate_xash
from .SeekerBase import Seeker


class MultiColumnOverlap(Seeker):
    def __init__(
        self,
        input_df: pd.DataFrame,
        k: int = 10,
        xash_size: int = 128,
        verbose: bool = False,
    ) -> None:
        super().__init__(k)
        self.input = input_df.copy().astype(str)
        self.xash_size = xash_size
        self.verbose = verbose

        # This is the base SQL query, which needs to be extended
        # to include the inner joins with all the user required
        # columns for the multi-column search
        self.base_sql = """
            SELECT firstcolumn.TableId, firstcolumn.RowId, firstcolumn.superkey, firstcolumn.CellValue,
                    firstcolumn.ColumnId $OTHER_SELECT_COLUMNS$
            FROM (
                SELECT TableId, RowId, CellValue, ColumnId, TO_BITSTRING(superkey) AS superkey
                FROM AllTables
                WHERE CellValue IN ($TOKENS$) $ADDITIONALS$
                ) AS firstcolumn $INNERJOINS$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        sql = self.base_sql.replace("$TOPK$", f"{self.k}")

        # The first column is treated outside the loop below
        firstcolumn_values = self.input[self.input.columns.values[0]]
        sql = sql.replace(
            "$TOKENS$",
            db.create_sql_list_str(db.clean_value_collection(firstcolumn_values)),
        )

        # For each column (except the first one) add an inner join to
        # the base SQL query, including a join on its specific values
        innerjoins = ""
        for column_index in range(1, len(self.input.columns.values)):
            column_values = self.input[self.input.columns.values[column_index]]
            column_name = db.random_subquery_name()

            value_collections = db.clean_value_collection(column_values)
            sql_list_str = db.create_sql_list_str(value_collections)

            innerjoins += f"""
                INNER JOIN
                    (
                        SELECT TableId, RowId, CellValue, ColumnId FROM AllTables
                        WHERE CellValue IN ({sql_list_str})
                            $ADDITIONALS$
                    ) AS col_{column_name}
                ON firstcolumn.TableId = col_{column_name}.TableID AND firstcolumn.RowId = col_{column_name}.RowId
            """

            # other_select_columns = f' , clm_{column_name}.CellValue, clm_{column_name}.ColumnId $OTHER_SELECT_COLUMNS$ '
            # sql = sql.replace('$OTHER_SELECT_COLUMNS$', other_select_columns)

        sql = sql.replace("$OTHER_SELECT_COLUMNS$", "")
        sql = sql.replace("$INNERJOINS$", innerjoins)
        sql = sql.replace("$ADDITIONALS$", additionals)

        candidates = db.execute_and_fetchall(sql)

        if self.verbose:
            print(f"#candidate rows = {len(candidates)}")

        # Run the MATE specific filters to prune irrelevant results
        results = self.run_filter(
            posting_lists=candidates,
            db=db,
            xash_size=self.xash_size,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"#filtered tables = {len(results)}")

        if len(results) == 0:
            return "SELECT * FROM AllTables WHERE 1 = 0;"

        return f"""
            SELECT TableId, JoinKeys, JoinabilityScore FROM (
            {
            " UNION ALL ".join(
                [
                    f"(SELECT '{table_id}' AS TableId, {
                        list(map(int, join_keys.split('_')))
                    } AS JoinKeys, {joinability_score} AS JoinabilityScore)"
                    for table_id, join_keys, joinability_score in results[: self.k]
                ]
            )
        }) AS ResultsSelection
        """

    def cost(self) -> int:
        return 10

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([list(col) for col in self.input.values.T], db)

    def run_filter(
        self,
        posting_lists: list,
        db: DBHandler,
        xash_size: int = 128,
        verbose: bool = False,
    ) -> list[tuple[int, str, float]]:
        # - Preprocessing
        PL_dictionary = defaultdict(list)
        PL_candidate_structure = {}
        for tablerow_superkey in tqdm(
            posting_lists, desc="Preprocessing posting lists: ", disable=not verbose
        ):
            table = tablerow_superkey[0]
            row = tablerow_superkey[1]
            superkey = tablerow_superkey[2]
            token = tablerow_superkey[3]
            colid = tablerow_superkey[4]
            tokens = [
                tablerow_superkey[x] for x in np.arange(5, len(tablerow_superkey), 2)
            ]
            cols = [
                tablerow_superkey[x] for x in np.arange(6, len(tablerow_superkey), 2)
            ]
            PL_dictionary[table].append((row, superkey, token, colid))
            PL_candidate_structure[(table, row)] = [tokens, cols]

        top_joinable_tables = []  # each item includes: Tableid, joinable_rows

        query_columns = self.input.columns.values

        # Calculate superkey for all input rows
        input_cpy = self.input.copy()
        input_cpy["SuperKey"] = input_cpy.apply(
            lambda row: self.hash_row_vals(row, xash_size), axis=1
        )

        # Get all rows grouped by first token of each row
        g = input_cpy.groupby([input_cpy.columns.values[0]])
        gd = defaultdict(list)
        for key, item in g:
            gd[str(key[0])] = g.get_group((key[0],)).values

        candidate_external_row_ids = []
        candidate_external_col_ids = []
        candidate_input_rows = []
        candidate_table_rows = []
        candidate_table_ids = []
        all_pls = 0
        total_approved = 0
        total_match = 0
        overlaps_dict = {}
        super_key_index = list(input_cpy.columns.values).index("SuperKey")
        checked_tables = 0
        max_table_check = 10000000

        for tableid in tqdm(
            sorted(PL_dictionary, key=lambda k: len(PL_dictionary[k]), reverse=True)[
                :max_table_check
            ],
            desc="Checking candidate tables: ",
            disable=not verbose,
        ):
            checked_tables += 1
            if checked_tables == max_table_check:
                # pruned = True
                break
            set_of_rowids = set()
            hitting_PLs = PL_dictionary[tableid]
            if len(top_joinable_tables) >= self.k and top_joinable_tables[0][0] >= len(
                hitting_PLs
            ):
                # pruned = True
                break
            already_checked_hits = 0

            for hit in tqdm(
                sorted(hitting_PLs),
                desc="Checking Hits: ",
                disable=not verbose,
                leave=False,
            ):
                if len(top_joinable_tables) >= self.k and (
                    (len(hitting_PLs) - already_checked_hits + len(set_of_rowids))
                    < top_joinable_tables[0][0]
                ):
                    break

                rowid = hit[0]
                superkey = int(hit[1], 2)
                token = hit[2]
                colid = hit[3]
                relevant_input_rows = gd[token]
                all_pls += len(relevant_input_rows)
                already_checked_hits += 1

                for input_row in relevant_input_rows:
                    if (input_row[super_key_index] | superkey) == superkey:
                        candidate_external_row_ids.append(rowid)
                        set_of_rowids.add(rowid)
                        candidate_external_col_ids.append(colid)
                        candidate_input_rows.append(input_row)
                        candidate_table_ids.append(tableid)
                        candidate_table_rows.append((tableid, rowid))

        if len(candidate_external_row_ids) == 0:
            if verbose:
                print("No candidate external row IDs found.")
        if len(candidate_external_row_ids) > 0:
            if verbose:
                print(f"#Candidate external row IDs: {len(candidate_external_row_ids)}")

            # We get a list of posting lists to evaluate as candidate matches, fetched
            # from the combination of the given table_id and row_id
            #
            # Below, we don't return all the results at once, but with a fetch-yield approach,
            # because sometimes there are very many many candidate PLs...
            joint_distinct_rows = tuple(map(int, set(candidate_external_row_ids)))
            joint_distinct_tableids = tuple(map(str, set(candidate_table_ids)))

            candidate_table_rows_as_tuple = [
                {"TableId": str(t[0]), "RowId": int(t[1])} for t in candidate_table_rows
            ]

            # NOTE: are joint_distinct_tableids and joint_distinct_row
            # really necessary for this step? Aren't they already included
            # thanks to the last WHERE-condition?
            query = """
            SELECT
                TableId, RowId,
                ColumnId, CellValue
            FROM (
                SELECT *
                FROM AllTables
                WHERE TableId IN ?
                AND RowId IN ?
                AND (TableId, RowId) IN (SELECT UNNEST(?, recursive := True))
            );
            """

            params = (
                joint_distinct_tableids,
                joint_distinct_rows,
                candidate_table_rows_as_tuple,
            )

            pls_to_evaluate = db.execute_and_fetchyield(query, params)

            # contains rowid that each rowid has dict that maps colids to tokenized
            table_row_dict = defaultdict(dict)

            if verbose:
                print("Evaluating remaining posting lists (fetch-yield)...")
            for table_id, row_id, col_id, cell_value in pls_to_evaluate:
                # here we are sure that (table_id, row_id) tuples are in candidate_table_rows,
                # since this condition is used in the above SQL query
                table_row_dict[(table_id, row_id)][col_id] = cell_value

            for i in tqdm(
                range(len(candidate_table_rows)),
                desc="Evaluating candidate table rows",
                total=len(candidate_table_rows),
                disable=not verbose,
            ):
                if candidate_table_rows[i] not in table_row_dict:
                    continue

                col_dict = table_row_dict[candidate_table_rows[i]]
                match, matched_columns = self.evaluate_rows(
                    candidate_input_rows[i], col_dict, query_columns
                )
                total_approved += 1
                if match:
                    total_match += 1
                    complete_matched_columns = "{}{}".format(
                        str(candidate_external_col_ids[i]), matched_columns
                    )
                    if candidate_table_ids[i] not in overlaps_dict:
                        overlaps_dict[candidate_table_ids[i]] = {}

                    if (
                        complete_matched_columns
                        in overlaps_dict[candidate_table_ids[i]]
                    ):
                        overlaps_dict[candidate_table_ids[i]][
                            complete_matched_columns
                        ] += 1
                    else:
                        overlaps_dict[candidate_table_ids[i]][
                            complete_matched_columns
                        ] = 1

            for tbl in set(candidate_table_ids):
                if tbl in overlaps_dict and len(overlaps_dict[tbl]) > 0:
                    join_keys = max(overlaps_dict[tbl], key=overlaps_dict[tbl].get)
                    joinability_score = overlaps_dict[tbl][join_keys]
                    if self.k <= len(top_joinable_tables):
                        if top_joinable_tables[0][0] < joinability_score:
                            _popped_table = heappop(top_joinable_tables)
                            heappush(
                                top_joinable_tables, [joinability_score, tbl, join_keys]
                            )
                    else:
                        heappush(
                            top_joinable_tables, [joinability_score, tbl, join_keys]
                        )

        # both original code of MATE and BLEND do not
        # return also the join keys, but I found more
        # useful to have them as part of the final results
        return [
            (tableid, join_keys, joinability_score)
            for joinability_score, tableid, join_keys in top_joinable_tables[::-1]
        ]

    def hash_row_vals(self, row: list[Any], xash_size: int = 128) -> int:
        hresult = 0
        for q in row:
            hvalue = calculate_xash(str(q), xash_size)
            hresult = hresult | hvalue
        return hresult

    def evaluate_rows(self, input_row, col_dict, query_columns):
        vals = list(col_dict.values())
        query_cols_arr = np.array(query_columns)
        query_degree = len(query_cols_arr)
        matching_column_order = ""
        for q in query_cols_arr[-(query_degree - 1) :]:
            q_index = list(query_columns).index(q)
            if input_row[q_index] not in vals:
                return False, ""
            else:
                for colid, val in col_dict.items():
                    if val == input_row[q_index]:
                        matching_column_order += "_{}".format(str(colid))
        return True, matching_column_order
