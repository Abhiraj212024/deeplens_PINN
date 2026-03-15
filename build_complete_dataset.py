import numpy as np
import glob
import os

def load_class(class_dir, label):
    raw_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
    # pre_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))

    # assert len(raw_files) == len(pre_files)

    X = []
    y = []

    for rf in raw_files:
        raw = np.load(rf)
        # (2, H, W)
        x = np.stack([raw[0]], axis=0)

        X.append(x)
        y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


X_no_train, y_no_train = load_class("dataset/train/no", label=0)
X_cdm_train, y_cdm_train = load_class("dataset/train/sphere", label=1)
X_ax_train , y_ax_train  = load_class("dataset/train/vort", label=2)

X_train = np.concatenate([X_no_train, X_cdm_train, X_ax_train], axis=0)
y_train = np.concatenate([y_no_train, y_cdm_train, y_ax_train], axis=0)
print(X_train.shape, y_train.shape)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

X_no_test, y_no_test = load_class("dataset/val/no", label=0)
X_cdm_test, y_cdm_test = load_class("dataset/val/sphere", label=1)
X_ax_test , y_ax_test  = load_class("dataset/val/vort", label=2)

X_test = np.concatenate([X_no_test, X_cdm_test, X_ax_test], axis=0)
y_test = np.concatenate([y_no_test, y_cdm_test, y_ax_test], axis=0)
print(X_test.shape, y_test.shape)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
