"""Linear reservoir equalizer with identity activation and NLMS readout."""
import numpy as np


def _linear_rc_equalize(symbols, target, *, train_samples=None, taps=25,
                nodes=40, gamma=0.8, spectral_radius=0.9,
                seed=100, train_ratio=0.1, lms_step=0.05,
                lms_epochs=30):

    """Original RC with identity activation; readout training is unchanged."""
    if len(symbols) != len(target):
        raise ValueError("CAP output and teaching symbols must have identical lengths")
    count = len(target)

    half = taps // 2

    train_end = int(count * train_ratio) if train_samples is None else int(train_samples)

    if not 0 < train_end <= count:

        raise ValueError("Invalid RC training sample count")

    washout = half + 1

    if train_end <= washout + half:

        raise ValueError("Insufficient RC training symbols")

    symbols = symbols / np.sqrt(np.mean(np.abs(symbols[:train_end]) ** 2))

    scale = np.sqrt(np.mean(np.abs(target[:train_end]) ** 2))

    windows = np.lib.stride_tricks.sliding_window_view(

        np.pad(symbols, (half, half)), taps

    )

    # Fit input scaling using training data only.

    input_scale = max(np.max(np.abs(windows[:train_end - half])), 1e-12)

    inputs = np.column_stack((windows.real, windows.imag)) / input_scale

    rng = np.random.default_rng(seed)

    recurrent = rng.uniform(-1, 1, (nodes, nodes))

    recurrent *= spectral_radius / np.max(np.abs(np.linalg.eigvals(recurrent)))

    win = rng.uniform(-1, 1, (nodes, 2 * taps + 1))

    features = np.empty((count, nodes + 2 * taps + 1))

    state = np.zeros(nodes)

    for index, sample in enumerate(inputs):

        augmented = np.append(sample, 1.0)

        state = (1 - gamma) * state + gamma * (recurrent @ state + win @ augmented)

        features[index] = np.concatenate((state, augmented))

    desired = np.column_stack((target.real, target.imag)) / scale

    weights = np.zeros((features.shape[1], 2))

    # End before the training boundary because windows include future symbols.

    train_features = features[washout:train_end - half]

    train_desired = desired[washout:train_end - half]

    powers = np.einsum("ij,ij->i", train_features, train_features) + 1e-12

    for _ in range(lms_epochs):
        for feature, teacher, power in zip(train_features, train_desired, powers):
            error = teacher - feature @ weights
            weights += (lms_step / power) * np.outer(feature, error)
    prediction = features @ weights
    recovered = (prediction[:, 0] + 1j * prediction[:, 1]) * scale
    if not np.all(np.isfinite(recovered)):
        raise FloatingPointError("Nonfinite RC output")

    # Return full length: no deleted transient and no target shift.
    return recovered, train_end + half + 1, half
