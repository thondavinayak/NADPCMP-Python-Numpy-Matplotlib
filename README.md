📂 lab1_nadpcmc_library.py — Function Summaries
1️⃣ def predict(phi, theta):
Inputs:

phi: numpy array, shape (h_depth,) (history vector)

theta: numpy array, shape (h_depth,) (adaptive predictor weights)

Outputs:

prediction: float (predicted sample value)

2️⃣ def calculate_error(sample, prediction):
Inputs:

sample: float (true current sample)

prediction: float (predicted value)

Outputs:

error: float (difference, i.e., prediction error)

3️⃣ def quantize(error, n_bits):
Inputs:

error: float (error to quantize)

n_bits: int (number of quantization bits)

Outputs:

q_error: int (quantized error, signed integer within n_bits range)

4️⃣ def dequantize(q_error, n_bits):
Inputs:

q_error: int (quantized error)

n_bits: int

Outputs:

error: float (dequantized error as float)

5️⃣ def update_theta(theta, learning_rate, q_error, phi):
Inputs:

theta: numpy array, shape (h_depth,) (current predictor weights)

learning_rate: float (learning rate for adaptive update)

q_error: float (quantized error for update)

phi: numpy array, shape (h_depth,)

Outputs:

new_theta: numpy array, shape (h_depth,) (updated predictor weights)

6️⃣ def reconstruct(prediction, q_error):
Inputs:

prediction: float (predicted value)

q_error: float (dequantized error)

Outputs:

reconstructed_sample: float (reconstructed sample)

7️⃣ def update_phi(phi, new_sample):
Inputs:

phi: numpy array, shape (h_depth,) (current history vector)

new_sample: float (new reconstructed sample)

Outputs:

new_phi: numpy array, shape (h_depth,) (updated history vector)

