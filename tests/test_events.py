from pathlib import Path

import polars as pl

from streamline.ingest.events import SCHEMA, build_parquet, load_events

RAW = """timestamp,visitorid,event,itemid,transactionid
1433221332117,257597,view,355908,
1433221332117,257597,view,355908,
1433221000000,1,addtocart,5,
1433222000000,1,transaction,5,4000
"""


def test_build_and_load_roundtrip(tmp_path: Path) -> None:
    csv = tmp_path / "events.csv"
    csv.write_text(RAW)
    out = tmp_path / "events.parquet"
    df = build_parquet(csv, out)

    assert df.schema == pl.Schema(SCHEMA)
    assert df.height == 3, "exact duplicate row should be dropped"
    assert df["ts_ms"].is_sorted()
    assert df.filter(pl.col("event") == "transaction")["transaction_id"].to_list() == [4000]
    assert load_events(out).equals(df)
