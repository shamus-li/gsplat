import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
EXAMPLES_DIR = REPO_ROOT / 'examples'
if EXAMPLES_DIR.as_posix() not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR.as_posix())

from examples.compute_dataset_alignment import compute_alignment  # noqa: E402
from examples.datasets.colmap import Parser  # noqa: E402
from examples.simple_trainer import (  # noqa: E402
    Config as TrainerConfig,
    _resolve_alignment_path,
)
from scripts.run_external_eval import _load_alignment_payload  # noqa: E402

DATA_ROOT = Path('../gs7/dataset/action-figure/iphone').resolve()
TRAIN_DIR = DATA_ROOT / 'train'
TEST_DIR = DATA_ROOT / 'test'
RESULT_DIR = Path('results/action-figure/iphone/combined').resolve()
ALIGN_ROOT = RESULT_DIR / 'alignments'
TRAIN_NORM_PATH = ALIGN_ROOT / 'train_normalization.npz'
LEGACY_ALIGN_PATH = ALIGN_ROOT / 'test_to_train.npz'

pytestmark = pytest.mark.skipif(
    not (TRAIN_DIR.exists() and TEST_DIR.exists()),
    reason='action-figure fixture is unavailable'
)


@lru_cache(maxsize=None)
def _parser_cache(path: str, normalize: bool, test_every: int) -> Parser:
    return Parser(
        data_dir=path,
        factor=1,
        normalize=normalize,
        test_every=test_every,
    )


def _make_parser(data_dir: Path) -> Parser:
    return _parser_cache(data_dir.as_posix(), True, 8)


def _alignment_file() -> Path:
    if LEGACY_ALIGN_PATH.exists():
        return LEGACY_ALIGN_PATH
    return TRAIN_NORM_PATH


def test_alignment_npz_matches_parser_transform():
    """Alignment artifact should encode train/test normalization relationship."""
    align_path = _alignment_file()
    if not align_path.exists():
        pytest.skip('alignment artifact is unavailable')
    train_parser = _make_parser(TRAIN_DIR)
    test_parser = _parser_cache(TEST_DIR.as_posix(), True, 1)
    payload = np.load(align_path)
    np.testing.assert_allclose(payload['base_transform'], train_parser.transform, atol=1e-5)
    if 'support_transform' in payload:
        np.testing.assert_allclose(payload['support_transform'], test_parser.transform, atol=1e-5)
    expected_align = train_parser.transform @ np.linalg.inv(test_parser.transform)
    np.testing.assert_allclose(payload['align_transform'], expected_align, atol=1e-5)


def test_auto_alignment_prefers_result_alignment():
    """Auto-resolution should pick the training alignment, not covisible metadata."""
    align_path = _alignment_file()
    if not align_path.exists():
        pytest.skip('alignment artifact is unavailable')
    cfg = TrainerConfig(
        disable_viewer=True,
        data_dir=str(TEST_DIR),
        data_factor=1,
        result_dir=str(RESULT_DIR),
    )

    resolved = _resolve_alignment_path(cfg)
    assert resolved == align_path
    assert cfg.dataset_transform_path == align_path.as_posix()

    covisible_align = DATA_ROOT / 'covisible' / 'test' / 'alignment.npz'
    assert resolved != covisible_align


def test_compute_alignment_matches_expected_alignment():
    """Helper should encode full train/test alignment."""
    train_parser = _parser_cache(TRAIN_DIR.as_posix(), True, 8)
    test_parser = _parser_cache(TEST_DIR.as_posix(), True, 1)

    align, base_transform, support_transform = compute_alignment(
        train_dir=TRAIN_DIR,
        subset_dir=TEST_DIR,
        train_test_every=8,
        eval_test_every=1,
    )

    expected_align = train_parser.transform @ np.linalg.inv(test_parser.transform)
    np.testing.assert_allclose(align, expected_align, atol=1e-5)
    np.testing.assert_allclose(base_transform, train_parser.transform, atol=1e-5)
    np.testing.assert_allclose(support_transform, test_parser.transform, atol=1e-5)


def test_alignment_loader_prefers_embedded_transform(tmp_path):
    """Loader should honor explicit align_transform even if base/support are reversed."""
    align = np.eye(4, dtype=np.float32)
    base = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)
    support = np.diag([3.0, 3.0, 3.0, 1.0]).astype(np.float32)
    payload = tmp_path / 'alignment.npz'
    np.savez(
        payload,
        align_transform=align,
        base_transform=base,
        support_transform=support,
    )

    loaded_align, loaded_base, loaded_support = _load_alignment_payload(payload)
    np.testing.assert_allclose(loaded_align, align)
    assert loaded_base is not None and loaded_support is not None
