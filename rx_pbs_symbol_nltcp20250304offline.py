"""Reproduce NN/NN20250320 Rx_PBS_symbol_NLTCP20250304offline.m."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lms_1downs_testnan import LMS_1DownS_Testnan
from lms_volterra_1downs_testnan import LMS_volterra_1DownS_Testnan
from cap_demodulation import CAPdemodulation
from nltcp_demodulation import NLTCP_demodulation

ROOT = Path(__file__).resolve().parent.parent
NN_ROOT = ROOT / "NN" / "NN\u6570\u636e20250320"
DATANO = 51200
UPSAMPLENO = 4
TRAIN_SAMPLES = 10000
VOLTERRA = ((33, 0.022, 17, 0.00008), (43, 0.013, 13, 0.00005))
SYMBOL_LMS = ((115, 0.0021), (115, 0.0011))


def biterr(recv_sym, ref_sym, bits_per_sym=None):
    """MATLAB biterr(X,Y) ratio for equal-length numeric symbol vectors.

    With no K argument, MATLAB chooses the width from BOTH arrays. NLTCP
    outputs can include 4/5, so a fixed PAM4 width does not reproduce MATLAB.
    """
    received = np.asarray(recv_sym).reshape(-1)
    reference = np.asarray(ref_sym).reshape(-1)
    if received.size == 0 or received.shape != reference.shape:
        raise ValueError("biterr requires nonempty, equal-length symbol vectors")
    for values in (received, reference):
        if np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("biterr inputs must be real and finite")
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("biterr inputs must be nonnegative integers")
    largest = int(max(received.max(), reference.max()))
    required_bits = max(1, largest.bit_length())
    if bits_per_sym is None:
        bits_per_sym = required_bits
    if int(bits_per_sym) != bits_per_sym or bits_per_sym < required_bits:
        raise ValueError("biterr bit width is too small or is not an integer")
    xor = np.bitwise_xor(received.astype(np.uint64), reference.astype(np.uint64))
    errors = sum(np.count_nonzero((xor >> bit) & 1) for bit in range(int(bits_per_sym)))
    return errors / (received.size * int(bits_per_sym))


def _safe_load_text(path):
    return np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)


def _decode(symbols):
    # The MATLAB decoder does not clip differential outputs to 0..3.
    decoded = []
    for component in (np.real(symbols), np.imag(symbols)):
        _, pam4 = NLTCP_demodulation(DATANO, component)
        decoded.append(pam4)
    return decoded


def _decoded_ber(decoded, references, taps=115):
    # MATLAB taps:end-taps includes the first element at Python index taps-1.
    interval = slice(taps - 1, -taps)
    return sum(biterr(received[interval], reference[interval])
               for received, reference in zip(decoded, references)) / 2


def run_one_case(n, return_stages=False):
    target = _safe_load_text(NN_ROOT / "v1.txt") + 1j * _safe_load_text(NN_ROOT / "v2.txt")
    avp1 = np.sqrt(np.mean(np.abs(target) ** 2))
    waveform = _safe_load_text(NN_ROOT / "txdata" / "RF0.105_data36QAM.txt")
    references = (
        _safe_load_text(NN_ROOT / "decimal_PAM4_1.txt"),
        _safe_load_text(NN_ROOT / "decimal_PAM4_2.txt"),
    )
    if len(target) != DATANO or any(len(ref) != DATANO for ref in references):
        raise ValueError("MATLAB reproduction requires exactly 51200 teaching symbols")
    outputs, channel_bers = [], []
    stages = {}
    for channel in (1, 2):
        received = _safe_load_text(NN_ROOT / "rxdata" / f"rx{channel}_{n}.txt")
        if len(received) != DATANO * UPSAMPLENO:
            raise ValueError("MATLAB reproduction requires 204800 waveform samples")
        # The MATLAB main script normalizes here, and the LMS function again.
        received = received / np.sqrt(np.mean(np.abs(received) ** 2))
        settings = VOLTERRA[channel - 1]
        coarse, train_y, train_e, wave_w = LMS_volterra_1DownS_Testnan(
            *settings, TRAIN_SAMPLES, received, waveform
        )
        symbols = CAPdemodulation(coarse, UPSAMPLENO, DATANO)
        taps, step = SYMBOL_LMS[channel - 1]
        recovered, _, symbol_e, symbol_w = LMS_1DownS_Testnan(
            taps, step, TRAIN_SAMPLES, symbols, target
        )
        outputs.append(recovered)
        decoded = _decode(recovered * avp1)
        channel_bers.append(_decoded_ber(decoded, references, taps))
        if return_stages:
            train_count = TRAIN_SAMPLES - max(settings[0], settings[2]) + 1
            stages[f"volterra{channel}"] = coarse
            stages[f"volterra{channel}_W"] = wave_w
            stages[f"volterra{channel}_y"] = train_y[:train_count]
            stages[f"volterra{channel}_E"] = train_e[:train_count]
            stages[f"cap{channel}"] = symbols
            stages[f"symbol{channel}"] = recovered
            stages[f"symbol{channel}_W"] = symbol_w
            stages[f"symbol{channel}_E"] = symbol_e[:TRAIN_SAMPLES - taps + 1]
            stages[f"decode{channel}_I"], stages[f"decode{channel}_Q"] = decoded
    denominator = sum(channel_bers)
    if denominator == 0:
        raise ZeroDivisionError("MATLAB PBS weights are undefined when both channel BERs are zero")
    # Preserve the reference's BER-derived weights, including its test-label use.
    combined = (
        channel_bers[1] / denominator * outputs[0] * avp1
        + channel_bers[0] / denominator * outputs[1] * avp1
    )
    decoded = _decode(combined)
    results = np.array([*channel_bers, _decoded_ber(decoded, references)])
    if return_stages:
        stages["combined"] = combined
        stages["decodePBS_I"], stages["decodePBS_Q"] = decoded
        return results, stages
    return results


def run_all_experiments(save_txt=True, plot=True):
    berall = np.empty((19, 3), dtype=float)
    for n in range(1, 20):
        print(f"Running case {n}/19 ...")
        berall[n - 1] = run_one_case(n)
    # Keep Python output separate from the original MATLAB berall.txt.
    if save_txt:
        np.savetxt(NN_ROOT / "berall1.txt", berall, fmt="%.16e")
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, 20), np.log10(berall))
        plt.legend(["CH1", "CH3", "PBS"])
        plt.xlabel("experiment index")
        plt.ylabel("log10(BER)")
        plt.title("BER evolution over experiments")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(NN_ROOT / "berall1.png", dpi=200)
        plt.show()
    return berall


if __name__ == "__main__":
    berall = run_all_experiments(save_txt=True, plot=True)
    print("==== Final berall matrix ====")
    print(berall)
