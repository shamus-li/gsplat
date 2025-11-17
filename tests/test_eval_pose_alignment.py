#!/usr/bin/env python3
"""
Test that evaluation poses are correctly aligned to training poses.

This test detects cases where evaluation renders use incorrect poses that don't
match the ground truth test poses, which can happen when:
1. Alignment computation fails silently
2. Wrong alignment file is used (e.g., covisible alignment with different params)
3. Alignment is not applied correctly during evaluation
4. Pose optimization is enabled during evaluation (most common issue!)
"""
import sys
from pathlib import Path
from typing import List, Tuple

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
from examples.datasets.normalize import transform_cameras  # noqa: E402

DATA_ROOT = Path('../gs7/dataset').resolve()
RESULT_ROOT = Path('results').resolve()

# Scenes that were reported to have wrong poses
FAILING_SCENES = [
    ('action-figure', 'iphone'),
    ('chicken', 'iphone'),
    ('chicken', 'stereo'),
    ('espresso', 'iphone'),
    ('espresso', 'stereo'),
    ('optics', 'iphone'),
    ('optics', 'stereo'),
    ('salt-pepper', 'iphone'),
    ('salt-pepper', 'stereo'),
    ('shelf', 'iphone'),
    ('shelf', 'stereo'),
]


def get_all_scene_camera_pairs() -> List[Tuple[str, str]]:
    """Discover all scene/camera pairs in the dataset."""
    pairs = []
    if not DATA_ROOT.exists():
        return pairs

    for scene_dir in sorted(DATA_ROOT.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name in ['dynamic', 'static']:
            continue
        for cam_dir in scene_dir.iterdir():
            if not cam_dir.is_dir() or cam_dir.name not in ['iphone', 'stereo']:
                continue
            train_dir = cam_dir / 'train'
            test_dir = cam_dir / 'test'
            if train_dir.exists() and test_dir.exists():
                pairs.append((scene_dir.name, cam_dir.name))
    return pairs


def get_alignment_path(scene: str, camera: str) -> Path:
    """Get the alignment path that would be used during evaluation."""
    # Check both combined and filtered variants (dual training setup)
    for variant in ['combined', 'filtered', '']:
        if variant:
            result_dir = RESULT_ROOT / scene / camera / variant
        else:
            result_dir = RESULT_ROOT / scene / camera

        align_root = result_dir / 'alignments'
        legacy = align_root / 'test_to_train.npz'
        if legacy.exists():
            return legacy
        preferred = align_root / 'train_normalization.npz'
        if preferred.exists():
            return preferred

    # Check for covisible alignment as fallback
    covi_align = DATA_ROOT / scene / camera / 'covisible' / 'test' / 'alignment.npz'
    if covi_align.exists():
        return covi_align

    return None


def verify_alignment_correctness(
    scene: str,
    camera: str,
    train_test_every: int = 8,
    eval_test_every: int = 1,
) -> dict:
    """
    Verify that alignment file matches expected transformation.

    Returns dict with verification results.
    """
    train_dir = DATA_ROOT / scene / camera / 'train'
    test_dir = DATA_ROOT / scene / camera / 'test'

    if not (train_dir.exists() and test_dir.exists()):
        return {'skip': True, 'reason': 'missing directories'}

    align_path = get_alignment_path(scene, camera)
    if align_path is None:
        return {'skip': True, 'reason': 'no alignment file found'}

    # Load alignment file
    try:
        align_data = np.load(align_path)
    except Exception as e:
        return {
            'error': True,
            'message': f'Failed to load alignment from {align_path}: {e}',
            'align_path': align_path,
        }

    # Compute expected alignment
    try:
        train_parser = Parser(
            data_dir=str(train_dir),
            factor=1,
            normalize=True,
            test_every=train_test_every,
        )
        test_parser_norm = Parser(
            data_dir=str(test_dir),
            factor=1,
            normalize=True,
            test_every=eval_test_every,
        )
        test_parser_raw = Parser(
            data_dir=str(test_dir),
            factor=1,
            normalize=False,
            test_every=eval_test_every,
        )
    except Exception as e:
        return {
            'skip': True,
            'reason': f'Failed to create parsers: {e}',
        }

    align_transform = align_data.get('align_transform')
    base_transform = align_data.get('base_transform')
    support_transform = align_data.get('support_transform')

    if align_transform is not None:
        align_transform = np.asarray(align_transform, dtype=np.float32)
    if base_transform is not None:
        base_transform = np.asarray(base_transform, dtype=np.float32)
    if support_transform is not None:
        support_transform = np.asarray(support_transform, dtype=np.float32)

    expected_align = train_parser.transform @ np.linalg.inv(test_parser_norm.transform)

    align_matches = align_transform is not None and np.allclose(
        align_transform, expected_align, atol=1e-4
    )
    base_matches = base_transform is not None and np.allclose(
        base_transform, train_parser.transform, atol=1e-4
    )

    if support_transform is not None:
        support_matches = np.allclose(
            support_transform, test_parser_norm.transform, atol=1e-4
        )
    else:
        support_matches = False

    if align_transform is not None:
        aligned_test_poses = transform_cameras(
            align_transform, test_parser_norm.camtoworlds.copy()
        )
    elif base_transform is not None:
        aligned_test_poses = transform_cameras(
            base_transform, test_parser_raw.camtoworlds.copy()
        )
    else:
        return {
            'error': True,
            'message': f'Alignment file {align_path} lacks usable transforms',
            'align_path': align_path,
        }

    if align_transform is not None:
        align_max_diff = float(np.abs(align_transform - expected_align).max())
    elif base_transform is not None:
        align_max_diff = float(
            np.abs(base_transform - train_parser.transform).max()
        )
    else:
        align_max_diff = float('nan')

    # Mean up vector in training set
    train_ups = train_parser.camtoworlds[:, :3, 1]
    train_ups = train_ups / np.linalg.norm(train_ups, axis=1, keepdims=True)
    train_mean_up = train_ups.mean(axis=0)
    train_mean_up = train_mean_up / np.linalg.norm(train_mean_up)

    # Mean up vector in aligned test set
    test_ups = aligned_test_poses[:, :3, 1]
    test_ups = test_ups / np.linalg.norm(test_ups, axis=1, keepdims=True)
    test_mean_up = test_ups.mean(axis=0)
    test_mean_up = test_mean_up / np.linalg.norm(test_mean_up)

    # Cosine similarity (should be close to 1.0)
    up_cosine = np.dot(train_mean_up, test_mean_up)

    # Check camera center distances
    train_centers = train_parser.camtoworlds[:, :3, 3]
    test_centers = aligned_test_poses[:, :3, 3]

    # Compute scene scale (max distance from mean center)
    all_centers = np.concatenate([train_centers, test_centers], axis=0)
    scene_center = all_centers.mean(axis=0)
    scene_scale = np.max(np.linalg.norm(all_centers - scene_center, axis=1))

    # Check if test cameras are within reasonable distance of training cameras
    # (they should be, for same scene)
    test_mean_center = test_centers.mean(axis=0)
    train_mean_center = train_centers.mean(axis=0)
    center_offset = np.linalg.norm(test_mean_center - train_mean_center)
    center_offset_ratio = center_offset / scene_scale if scene_scale > 0 else 0

    return {
        'align_path': align_path,
        'is_covisible_alignment': 'covisible' in align_path.as_posix(),
        'align_matches_expected': align_matches,
        'align_max_diff': align_max_diff,
        'base_matches_expected': base_matches,
        'support_matches_expected': support_matches,
        'up_cosine': float(up_cosine),
        'center_offset_ratio': float(center_offset_ratio),
        'scene_scale': float(scene_scale),
        'num_train_cameras': len(train_parser.camtoworlds),
        'num_test_cameras': len(test_parser.camtoworlds),
    }


@pytest.mark.parametrize('scene,camera', FAILING_SCENES)
def test_failing_scenes_have_correct_alignment(scene: str, camera: str):
    """Test that previously failing scenes now have correct alignment."""
    train_dir = DATA_ROOT / scene / camera / 'train'
    test_dir = DATA_ROOT / scene / camera / 'test'

    if not (train_dir.exists() and test_dir.exists()):
        pytest.skip(f'Scene {scene}/{camera} data not available')

    result = verify_alignment_correctness(scene, camera)

    if result.get('skip'):
        pytest.skip(result.get('reason', 'unknown'))

    if result.get('error'):
        pytest.fail(result['message'])

    # Check alignment quality metrics
    assert result['align_matches_expected'], (
        f"Alignment file {result['align_path']} does not match expected transform. "
        f"Max diff: {result['align_max_diff']:.6f}"
    )

    assert result['up_cosine'] > 0.9, (
        f"Up vector alignment is poor (cosine={result['up_cosine']:.3f}). "
        f"This suggests poses are not correctly aligned. "
        f"Using: {result['align_path']}"
    )

    assert result['center_offset_ratio'] < 0.5, (
        f"Camera centers are too far apart (offset ratio={result['center_offset_ratio']:.3f}). "
        f"This suggests wrong coordinate frame. "
        f"Using: {result['align_path']}"
    )


def test_all_scenes_report():
    """Generate a report of alignment status for all scenes."""
    pairs = get_all_scene_camera_pairs()

    if not pairs:
        pytest.skip('No dataset available')

    results = []
    for scene, camera in pairs:
        result = verify_alignment_correctness(scene, camera)
        result['scene'] = scene
        result['camera'] = camera
        results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("ALIGNMENT VERIFICATION REPORT")
    print("="*80)

    issues = []
    for r in results:
        if r.get('skip'):
            continue
        if r.get('error'):
            issues.append(r)
            print(f"❌ {r['scene']}/{r['camera']}: ERROR - {r['message']}")
            continue

        status = "✓" if r['align_matches_expected'] and r['up_cosine'] > 0.9 else "✗"
        covi = " [COVISIBLE]" if r['is_covisible_alignment'] else ""

        print(f"{status} {r['scene']}/{r['camera']}{covi}")
        print(f"   Up cosine: {r['up_cosine']:.4f}, "
              f"Center offset: {r['center_offset_ratio']:.4f}, "
              f"Align matches: {r['align_matches_expected']}")

        if not r['align_matches_expected']:
            issues.append(r)
            print(f"   WARNING: Alignment mismatch! Max diff: {r['align_max_diff']:.6f}")
        if r['up_cosine'] < 0.9:
            if r not in issues:
                issues.append(r)
            print(f"   WARNING: Poor up vector alignment!")
        if r['is_covisible_alignment']:
            print(f"   INFO: Using covisible alignment (main alignment may be missing)")

    print("="*80)

    if issues:
        print(f"\n⚠️  Found {len(issues)} scenes with alignment issues:")
        for r in issues:
            print(f"  - {r['scene']}/{r['camera']}")
        print()


if __name__ == '__main__':
    # Run report when executed directly
    test_all_scenes_report()
