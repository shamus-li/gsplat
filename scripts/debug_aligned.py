import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,'examples')
from datasets.colmap import Parser, transform_cameras

TRAIN_DIR = Path('../gs7/dataset/action-figure/iphone/train').resolve()
TEST_DIR = Path('../gs7/dataset/action-figure/iphone/test').resolve()
train = Parser(str(TRAIN_DIR), factor=1, normalize=True, test_every=8)
test = Parser(str(TEST_DIR), factor=1, normalize=True, test_every=8)
align = np.load('results/action-figure/iphone/combined/alignments/test_to_train.npz')['align_transform']
cam_aligned = transform_cameras(align, test.camtoworlds.copy())
rel = np.linalg.inv(test.camtoworlds) @ cam_aligned
angles = []
for mat in rel:
    r = mat[:3,:3]
    angle = np.degrees(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))
    angles.append(angle)
print('mean', np.mean(angles))
