"""Shared fixtures. Embeddings are switched off so tests are fast and deterministic."""

import datetime as dt
import os
import sys
from pathlib import Path

import pytest

os.environ["CVSCORE_EMBEDDINGS"] = "off"
os.environ["CVSCORE_CACHE"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

TODAY = dt.date(2026, 10, 28)


@pytest.fixture(scope="session")
def root() -> Path:
    """Project root."""
    return ROOT


@pytest.fixture(scope="session")
def today() -> dt.date:
    """Fixed reference date."""
    return TODAY


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory) -> Path:
    """Generate the synthetic PDFs into a temporary folder (never into the repo)."""
    import make_samples  # pylint: disable=import-outside-toplevel

    out = tmp_path_factory.mktemp("pdf")
    make_samples.cv_two_column(out / "cv_2col.pdf")
    make_samples.outlines_variant(out / "cv_2col.pdf", out / "cv_outlines.pdf")
    for name, text in make_samples.OFFERS.items():
        make_samples.simple_doc(out / name, make_samples.lines_from(text))
    make_samples.simple_doc(out / "list_sds.pdf", make_samples.lines_from(make_samples.LETTER_SDS))
    make_samples.simple_doc(out / "oferta_www.pdf", make_samples.WEB_OFFER)
    make_samples.simple_doc(out / "cv_tech.pdf", make_samples.lines_from(make_samples.CV_TECH, (
        "profil", "doświadczenie zawodowe", "wykształcenie", "umiejętności", "języki")))
    return out
