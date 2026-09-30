import pytest

from streamline.config import get_settings


@pytest.mark.integration
def test_live_stack_has_no_skew() -> None:
    """Redpanda -> Bytewax -> Redis vs the offline engine on one hour of real events."""
    if not get_settings().events_parquet.exists():
        pytest.skip("run `make data` first")
    from streamline.features.skew_check import run

    report = run("2015-09-04", hours=1)
    assert report.events > 0
    assert report.values_compared > 0
    assert report.mismatches == 0, report.examples
