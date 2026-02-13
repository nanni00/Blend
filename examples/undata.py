import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from blend import BLEND
from blend.indexing import index_tables

data_path = Path(__file__).parent.parent.joinpath("examples", "example-data", "undata")

data_lake_path = data_path.joinpath("data-lake")
index_db_path = data_path.joinpath("undata.db")
logdir_path = data_path.joinpath("log")
queries_path = data_path.joinpath("queries")

data_path.exists()
tmp_path = data_path.joinpath("tmp")
tmp_path.mkdir(parents=True, exist_ok=True)

indexer = BLEND(
    index_db_path,
    clean_args={"lowercase": True, "filter_bad_tokens": True},
    max_cell_length=512,
)

load_opts = {"ignore_errors": True}
index_tables(indexer, data_lake_path, True, None, 8, load_opts, 100, tmp_path=tmp_path)
