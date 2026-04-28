"""CNN sweep for the bond-Holstein model
(method of Broecker, Assaad and Trebst, ScaiPost Phys. Lect. Notes 11 (2019);
related to learning by confusion but with a balanced two-class training set
around each guessed critical value).

Inputs
------
A single CSV with shape (N_samples, L*L + 1).  The first L*L columns are one
electron-density snapshot, the last column is the sweep parameter (typically
the inverse temperature beta or the dimensionless inverse coupling 1/lambda).
The sweep values are expected to be on a uniform grid.

Method
------
For every guessed critical sweep value, the script picks `nskip` neighboring
sweep values around the guess (half on each side, omitting the guess itself),
relabels snapshots from the lower half as class 0 and from the upper half as
class 1, then trains a small CNN on the relabeled data.  A correct guess
yields the highest test accuracy.

Outputs
-------
Per-guess training curves and a `test_accuracies-<sID>.csv` file holding one
test accuracy per guessed critical value.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tensorflow  # noqa: F401  (initializes TF before keras imports)
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


def load_data(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    return np.array(rows, dtype=float)


def build_cnn(L: int) -> keras.Model:
    he = keras.initializers.HeNormal()
    return Sequential([
        keras.Input(shape=(L, L, 1)),
        Conv2D(8, (3, 3), padding="same", kernel_initializer=he),
        layers.Activation("relu"),
        MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        Conv2D(16, (3, 3), padding="same", kernel_initializer=he),
        layers.Activation("relu"),
        MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        Flatten(),
        Dense(64, kernel_initializer=he),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        Dense(2, activation="softmax"),
    ])


def run_sweep(
    *, input_csv: Path, output_dir: Path, sID: int, L: int,
    nskip: int, num_epochs: int, batch_size: int, lr: float, weight_decay: float,
):
    if nskip % 2 != 0:
        raise ValueError(f"nskip must be even (got {nskip}).")

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(input_csv)

    sweep = data[:, -1]
    unique_sweep = np.unique(sweep)
    NL = len(unique_sweep)
    print(f"Loaded {data.shape[0]} samples; {NL} distinct sweep values "
          f"in [{unique_sweep.min():.4f}, {unique_sweep.max():.4f}].")

    half = nskip // 2
    NG = NL - nskip
    trunc_sweep = unique_sweep[half:-half]
    print(f"nskip={nskip}, training {NG} guesses in "
          f"[{trunc_sweep.min():.4f}, {trunc_sweep.max():.4f}].")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=10, min_lr=1e-6, verbose=1),
    ]

    test_losses, test_accuracies = [], []
    Size = L * L

    for iL in range(NG):
        guess = trunc_sweep[iL]
        left = unique_sweep[iL : iL + half]
        right = unique_sweep[iL + half + 1 : iL + nskip + 1]

        mask_left = np.isin(sweep, left)
        mask_right = np.isin(sweep, right)
        X0 = data[mask_left, :Size]
        X1 = data[mask_right, :Size]
        n_min = min(len(X0), len(X1))
        X0, X1 = X0[:n_min], X1[:n_min]
        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n_min, dtype=int), np.ones(n_min, dtype=int)])

        perm = np.random.permutation(len(X))
        X = X[perm]
        y = keras.utils.to_categorical(y[perm], num_classes=2)

        N = len(X)
        n_train, n_val = int(0.8 * N), int(0.9 * N)
        X_train = X[:n_train].reshape(-1, L, L, 1)
        X_val = X[n_train:n_val].reshape(-1, L, L, 1)
        X_test = X[n_val:].reshape(-1, L, L, 1)
        Y_train = y[:n_train]
        Y_val = y[n_train:n_val]
        Y_test = y[n_val:]

        model = build_cnn(L)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
            metrics=["accuracy"],
        )
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                            epochs=num_epochs, batch_size=batch_size, verbose=1,
                            callbacks=callbacks)

        np.savetxt(output_dir / f"train_accs_iL{iL}-{sID}.csv",
                   np.array(history.history["accuracy"]), fmt="%.5f")
        np.savetxt(output_dir / f"val_accs_iL{iL}-{sID}.csv",
                   np.array(history.history["val_accuracy"]), fmt="%.5f")
        np.savetxt(output_dir / f"train_losses_iL{iL}-{sID}.csv",
                   np.array(history.history["loss"]), fmt="%.5f")
        np.savetxt(output_dir / f"val_losses_iL{iL}-{sID}.csv",
                   np.array(history.history["val_loss"]), fmt="%.5f")

        test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"  guess #{iL} (sweep={guess:.4f}): test acc = {test_accuracy:.5f}")

        np.savetxt(output_dir / f"test_accuracies-{sID}.csv",
                   np.array(test_accuracies), fmt="%.5f")
        np.savetxt(output_dir / f"test_losses-{sID}.csv",
                   np.array(test_losses), fmt="%.5f")
        np.savetxt(output_dir / f"sweep_values-{sID}.csv", unique_sweep, fmt="%.5f")
        np.savetxt(output_dir / f"trunc_sweep-{sID}.csv", trunc_sweep, fmt="%.5f")

    print(f"Wrote test accuracies to {output_dir}/test_accuracies-{sID}.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path, help="Snapshot CSV (last column is sweep param).")
    p.add_argument("output_dir", type=Path, help="Output directory.")
    p.add_argument("--sID", type=int, default=1)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--nskip", type=int, default=6,
                   help="Number of neighboring sweep values used for training (must be even).")
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run_sweep(
        input_csv=a.input_csv, output_dir=a.output_dir, sID=a.sID, L=a.L,
        nskip=a.nskip, num_epochs=a.num_epochs, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay,
    )


if __name__ == "__main__":
    main()
