# lab1_library.py

import numpy as np

def quantize(a, n_bits):
    """
    Quantize a float value `a` into `n_bits`-wide signed integer range.
    """
    if np.isinf(a) or np.isnan(a):
        raise ValueError(f"Cannot quantize invalid value: {a}")

    max_q = (2 ** (n_bits - 1)) - 1
    min_q = - (2 ** (n_bits - 1))

    # Clamp to quantization range
    quantized = int(np.round(a))
    quantized = max(min_q, min(max_q, quantized))
    #print('a = ', a, ', n_bits = ', n_bits, ', max_q = ', max_q, ', min_q = ', min_q, ', q = ', quantized);
   
    return quantized


def dequantize(q_val):
    """
    Simple dequantization function.
    """
    return float(q_val)


def nadpcmc_encode(signal, n_bits, init_vector_size):
    # Basic DPCM-style encoding: store error between sample and previous sample
    encoded = []
    prev = 0
    for i, val in enumerate(signal):
        if i < init_vector_size:
            encoded.append(quantize(val, n_bits))
            prev = val
        else:
            err = val - prev
            q_err = quantize(err, n_bits)
            encoded.append(q_err)
            prev = prev + q_err
    return encoded


def nadpcmc_decode(encoded, n_bits, init_vector_size):
    # Decode encoded signal
    decoded = []
    prev = 0
    for i, val in enumerate(encoded):
        if i < init_vector_size:
            sample = dequantize(val)
        else:
            sample = prev + dequantize(val)
        decoded.append(sample)
        prev = sample
    return decoded


def evaluate(original, reconstructed, encoded, n_bits, init_vector_size):
    # Mean squared error
    mse = np.mean((np.array(original) - np.array(reconstructed))**2)

    # Total transmitted bits
    transmitted_bits = len(encoded) * n_bits

    # Compression ratio: assume original is 32-bit floats
    original_bits = len(original) * 32
    compression_ratio = original_bits / transmitted_bits if transmitted_bits else 0

    return mse, transmitted_bits, compression_ratio
