import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import lab1_ndpcm_library2 as lab1_ndpcm_library

# Parameters
n_bits = 16
n = 100
h_depth = 3
amp = 10000

# Generate test data
x = np.linspace(0, 10 * pi, n)
f_original = np.sin(x)
f = (f_original+1) * amp

# Init TX and RX data
tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

tx_data.y_recreated[:h_depth] = f[:h_depth]
rx_data.y_recreated[:h_depth] = f[:h_depth]

# Main simulation loop
for k in range(1, n - 1):
    lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
    lab1_ndpcm_library.predict(tx_data, k)
    lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
    lab1_ndpcm_library.update_theta(tx_data, k, 1e-7)
    lab1_ndpcm_library.reconstruct(tx_data, k)

    rx_data.eq[k] = tx_data.eq[k]

    lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
    if k < tx_data.h_depth:
        rx_data.phi[k] = tx_data.phi[k]
        rx_data.theta[k] = tx_data.theta[k]
    lab1_ndpcm_library.predict(rx_data, k)
    lab1_ndpcm_library.update_theta_rx(rx_data, k, 1e-7)
    lab1_ndpcm_library.reconstruct(rx_data, k) 
    

# Compute error
e_all = np.abs(rx_data.y_recreated - f)
print("Cumulative error =", e_all.sum(), ", Average error =", np.average(e_all))
print("Bits transmitted =", n_bits * n)
print("Bitrate [kbps] =", n_bits * n / 1000)

tt = np.arange(n)

# Prepare for scatter plotting phi and theta
def plot_matrix_scatter(ax, matrix, title):
    k_vals, idx_vals, values = [], [], []
    for k in range(matrix.shape[0]):
        for idx in range(matrix.shape[1]):
            k_vals.append(k)
            idx_vals.append(idx)
            values.append(matrix[k, idx])
    sc = ax.scatter(k_vals, idx_vals, c=values, cmap='viridis', s=20)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Index")
    plt.colorbar(sc, ax=ax)

# Create one figure with subplots
fig, axs = plt.subplots(5, 2, figsize=(15, 18))
fig.suptitle(f"NADPCM Compression and Reconstruction (n_bits={n_bits})", fontsize=13)

# Row 1: Signal and predictions (TX)
axs[0, 0].plot(tt, f, label="Sensor Data", color='blue')
axs[0, 0].plot(tt, tx_data.y_hat, label="TX y_hat", linestyle='--', color='orange')
axs[0, 0].plot(tt, tx_data.y_recreated, label="TX y_recreated", color='green')
axs[0, 0].set_title("TX: Sensor, Prediction & Reconstructed Signal")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Row 1: RX predictions
axs[0, 1].plot(tt, rx_data.y_hat, label="RX y_hat", linestyle='--', color='orange')
axs[0, 1].plot(tt, rx_data.y_recreated, label="RX y_recreated", color='green')
axs[0, 1].set_title("RX: Prediction & Reconstructed Signal")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Row 2: Errors
axs[1, 0].scatter(tt, f - tx_data.y_recreated, s=10, label="TX Reconstruction Error")
axs[1, 0].scatter(tt, tx_data.e, s=10, label="TX Scatter e", color='red')
axs[1, 0].scatter(tt, tx_data.eq, s=10, label="TX Scatter eq", color='orange')
axs[1, 0].set_title("Error Signals")
axs[1, 0].legend()
axs[1, 0].grid(True)


axs[1, 1].scatter(tt, f - rx_data.y_recreated, s=10, label="RX Reconstruction Error")
#axs[1, 1].scatter(tt, rx_data.e, s=10, label="RX Scatter e", color='red')
axs[1, 1].scatter(tt, rx_data.eq, s=10, label="RX Scatter eq", color='orange')
axs[1, 1].set_title("Error Signals")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Row 3: Scatter phi
plot_matrix_scatter(axs[2, 0], tx_data.phi, "TX phi (history)")
plot_matrix_scatter(axs[2, 1], rx_data.phi, "RX phi (history)")

# Row 4: Scatter theta
plot_matrix_scatter(axs[3, 0], tx_data.theta, "TX theta (prediction weights)")
plot_matrix_scatter(axs[3, 1], rx_data.theta, "RX theta (prediction weights)")

# Row 5: Final error
axs[4, 0].plot(tt, e_all, label="|f - y_recreated|")
axs[4, 0].set_title("Absolute Reconstruction Error (RX)")
axs[4, 0].grid(True)
axs[4, 0].legend()


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

