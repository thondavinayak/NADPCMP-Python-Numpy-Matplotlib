from collections import namedtuple
import numpy as np
import lab1_library  # Assumes lab1_library.quantize exists

NAPCMC = namedtuple('NAPCMC', ['n', 'h_depth', 'n_bits',
                   'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated'])

def init(n, h_depth, n_bits):
    return NAPCMC(
        n=n,
        h_depth=h_depth,
        n_bits=n_bits,
        phi=np.zeros((n, h_depth)),
        theta=np.zeros((n, h_depth)),
        y_hat=np.zeros(n),
        e=np.zeros(n),
        eq=np.zeros(n),
        y_recreated=np.zeros(n)
    )

def prepare_params_for_prediction(data_block, k):
    h = data_block.h_depth
    phi_k = np.zeros(h)

    if k >= h:
        phi_k = data_block.y_recreated[k - h:k][::-1]
    else:
        phi_k[-k:] = data_block.y_recreated[:k][::-1]

    # Normalize phi to unit norm
    #norm = np.linalg.norm(phi_k) + 1e-8
    #phi_k = phi_k / norm

    data_block.phi[k] = phi_k
    data_block.theta[k] = data_block.theta[k - 1] if k > 0 else np.zeros(h)


def predict(data_block, k):
    if k == 0:
        data_block.y_hat[k] = 0.0
        return 0.0

    phi_k = data_block.phi[k]
    theta_k = data_block.theta[k - 1]
    
    y_hat = theta_k @ phi_k if np.isfinite(theta_k).all() and np.isfinite(phi_k).all() else 0.0
    
    #nonlinear_term = np.tanh(theta_k @ phi_k)
    #y_hat = nonlinear_term if np.isfinite(nonlinear_term) else 0.0
    
    data_block.y_hat[k] = y_hat


    return y_hat

def update_theta(data_block, k, learning_rate=1e-6):
    if k >= data_block.h_depth:
        phi_k = data_block.phi[k]
        e_k = data_block.e[k]
        if np.isfinite(e_k) and np.isfinite(phi_k).all():
            data_block.theta[k] = data_block.theta[k - 1] + learning_rate * e_k * phi_k
        else:
            data_block.theta[k] = data_block.theta[k - 1]



def update_theta_rx(data_block, k, learning_rate=1e-6):
    if k >= data_block.h_depth:
        phi_k = data_block.phi[k]
        eq_k = data_block.eq[k]

        if not np.isfinite(eq_k) or not np.all(np.isfinite(phi_k)):
            data_block.theta[k] = data_block.theta[k - 1]
            return

        theta_new = data_block.theta[k - 1] + learning_rate * eq_k * phi_k

        if np.all(np.isfinite(theta_new)):
            data_block.theta[k] = np.clip(theta_new, -1000, 1000)
        else:
            data_block.theta[k] = data_block.theta[k - 1]





def calculate_error(data_block, k, real_y):
    e_k = real_y - data_block.y_hat[k]
    eq_k = lab1_library.quantize(e_k, data_block.n_bits)
    data_block.e[k] = e_k
    data_block.eq[k] = eq_k
    return eq_k

def reconstruct(data_block, k):
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]
