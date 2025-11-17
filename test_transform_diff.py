#!/usr/bin/env python3
"""Test if gsplat and Difix3D Parsers produce different normalization transforms."""

import sys
import numpy as np
from pathlib import Path

# Test scene
scene = "action-figure"
iphone_train = f"/share/monakhova/shamus_data/multiplexed_pixels/dataset/{scene}/iphone/train"

print("=" * 80)
print("Testing Normalization Transform Difference")
print("=" * 80)
print()

# Test with gsplat Parser
print("1. Loading with gsplat Parser...")
sys.path.insert(0, '/home/wl757/multiplexed-pixels/gsplat/examples')
from datasets.colmap import Parser as GsplatParser
sys.path.pop(0)

gsplat_parser = GsplatParser(
    data_dir=iphone_train,
    factor=1,
    normalize=True,
    test_every=8,
)
gsplat_transform = gsplat_parser.transform.copy()
gsplat_camtoworlds = gsplat_parser.camtoworlds.copy()
gsplat_points = gsplat_parser.points.copy()

print(f"Loaded {len(gsplat_parser.image_names)} images")
print(f"First 5 image names: {gsplat_parser.image_names[:5]}")
print(f"Transform:\n{gsplat_transform}")
print(f"Camera 0 position: {gsplat_camtoworlds[0, :3, 3]}")
print(f"Scene scale: {gsplat_parser.scene_scale:.4f}")
print()

# Clear modules
for mod in list(sys.modules.keys()):
    if 'datasets' in mod or 'colmap' in mod or 'normalize' in mod:
        del sys.modules[mod]

# Test with Difix3D Parser
print("2. Loading with Difix3D Parser...")
difix_path = Path('/home/wl757/multiplexed-pixels/Difix3D/examples/gsplat')
sys.path.insert(0, str(difix_path))
from datasets.colmap import Parser as DifixParser  # type: ignore[import]
sys.path.pop(0)

difix_parser = DifixParser(
    data_dir=iphone_train,
    factor=1,
    normalize=True,
    test_every=8,
)
difix_transform = difix_parser.transform.copy()
difix_camtoworlds = difix_parser.camtoworlds.copy()
difix_points = difix_parser.points.copy()

print(f"Loaded {len(difix_parser.image_names)} images")
print(f"First 5 image names: {difix_parser.image_names[:5]}")
print(f"Transform:\n{difix_transform}")
print(f"Camera 0 position: {difix_camtoworlds[0, :3, 3]}")
print(f"Scene scale: {difix_parser.scene_scale:.4f}")
print()

# Compare
print("=" * 80)
print("3. Comparison:")
print("=" * 80)
print()

print(f"Image count matches: {len(gsplat_parser.image_names) == len(difix_parser.image_names)}")
print(f"Image names match: {gsplat_parser.image_names == difix_parser.image_names}")
if gsplat_parser.image_names != difix_parser.image_names:
    print("  First difference:")
    for i, (g, d) in enumerate(zip(gsplat_parser.image_names, difix_parser.image_names)):
        if g != d:
            print(f"    Index {i}: gsplat='{g}', difix='{d}'")
            break

print()
print("Transform difference:")
transform_diff = np.abs(gsplat_transform - difix_transform)
print(f"  Max difference: {np.max(transform_diff):.6f}")
print(f"  Mean difference: {np.mean(transform_diff):.6f}")
if np.max(transform_diff) > 1e-5:
    print("  ⚠ SIGNIFICANT DIFFERENCE!")
    print(f"  Difference matrix:\n{transform_diff}")

print()
print("Camera positions difference (first 3 cameras):")
for i in range(min(3, len(gsplat_camtoworlds))):
    pos_diff = np.linalg.norm(gsplat_camtoworlds[i, :3, 3] - difix_camtoworlds[i, :3, 3])
    print(f"  Camera {i}: {pos_diff:.6f}")

print()
print("Points difference:")
if len(gsplat_points) == len(difix_points):
    points_diff = np.linalg.norm(gsplat_points - difix_points, axis=1)
    print(f"  Mean distance: {np.mean(points_diff):.6f}")
    print(f"  Max distance: {np.max(points_diff):.6f}")
    print(f"  Median distance: {np.median(points_diff):.6f}")
else:
    print(f"  ⚠ Point count mismatch: gsplat={len(gsplat_points)}, difix={len(difix_points)}")

print()
print("Scene scale difference:")
scale_diff = abs(gsplat_parser.scene_scale - difix_parser.scene_scale)
print(f"  {scale_diff:.6f} ({scale_diff / gsplat_parser.scene_scale * 100:.2f}%)")

print()
print("=" * 80)
if np.max(transform_diff) > 1e-5 or gsplat_parser.image_names != difix_parser.image_names:
    print("RESULT: Parsers produce DIFFERENT results!")
    print("This WILL cause alignment mismatches during eval.")
else:
    print("RESULT: Parsers produce IDENTICAL results.")
    print("The alignment mismatch must be caused by something else.")
print("=" * 80)
