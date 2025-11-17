#!/usr/bin/env python3
"""Compare different alignment files for the same scene."""
import sys
from pathlib import Path
import numpy as np

def load_alignment(path: Path):
    """Load alignment transform from npz file."""
    data = np.load(path)
    base = data.get('base_transform')
    support = data.get('support_transform')
    if 'align_transform' in data:
        align = data['align_transform']
    elif base is not None and support is not None:
        # Alignments should map eval -> train; support represents the training
        # normalization and base represents the eval/test normalization.
        align = support @ np.linalg.inv(base)
    else:
        raise ValueError(f"Unknown alignment format in {path}")

    result = {'align_transform': align}
    if base is not None:
        result['base_transform'] = base
    if support is not None:
        result['support_transform'] = support

    return result

def compare_alignments(scene: str, camera: str):
    """Compare alignment files for a scene/camera pair."""
    print(f"\n{'='*80}")
    print(f"Comparing alignments for {scene}/{camera}")
    print(f"{'='*80}")

    # Paths to check
    main_align = Path(f"results/{scene}/{camera}/combined/alignments/test_to_train.npz")
    covi_results = Path(f"results/{scene}/{camera}/combined/covisible/test/alignment.npz")
    covi_dataset = Path(f"../gs7/dataset/{scene}/{camera}/covisible/test/alignment.npz")

    alignments = {}

    for name, path in [
        ('Main (results)', main_align),
        ('Covisible (results)', covi_results),
        ('Covisible (dataset)', covi_dataset),
    ]:
        if path.exists():
            try:
                alignments[name] = load_alignment(path)
                print(f"\n✓ {name}: {path}")
                print(f"  Modified: {path.stat().st_mtime}")
            except Exception as e:
                print(f"\n✗ {name}: {path}")
                print(f"  Error: {e}")
        else:
            print(f"\n- {name}: {path} (not found)")

    if len(alignments) < 2:
        print("\nNot enough alignment files to compare")
        return

    # Compare alignments
    print(f"\n{'-'*80}")
    print("Alignment comparison:")
    print(f"{'-'*80}")

    names = list(alignments.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            align1 = alignments[name1]['align_transform']
            align2 = alignments[name2]['align_transform']

            diff = np.abs(align1 - align2).max()
            matches = np.allclose(align1, align2, atol=1e-4)

            status = "✓ MATCH" if matches else "✗ DIFFER"
            print(f"\n{name1} vs {name2}: {status}")
            print(f"  Max diff: {diff:.6f}")

            if not matches:
                print(f"  Alignment 1:\n{align1}")
                print(f"  Alignment 2:\n{align2}")

if __name__ == '__main__':
    scenes = [
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

    for scene, camera in scenes:
        compare_alignments(scene, camera)
