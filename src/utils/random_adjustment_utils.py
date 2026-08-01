import numpy as np


def estimate_sigma_from_percentiles():
    # percentiles: list of (p in (0,1), observed radius r_p) taken from Fig. 3 in https://arxiv.org/pdf/2405.06593v2
    percentiles = [(0.1, 17.9), (0.25, 35.0), (0.5, 59.9), (0.75, 99.4), (0.9, 147.3), (0.99, 324.0)]

    # Estimate sigma from the observed percentiles
    ps = np.array([p for p, r in percentiles], dtype=float)
    rs = np.array([r for p, r in percentiles], dtype=float)

    eps = 1e-12
    ps = np.clip(ps, eps, 0.999999)  # avoid p=0 or p extremely close to 1

    a = np.sqrt(-2.0 * np.log(1.0 - ps))    # a_i = sqrt(-2 ln(1-p_i))
    sigma_per = rs / a                      # per-percentile sigma estimates
    sigma_ls = (a * rs).sum() / (a * a).sum()  # least-squares estimate

    print("sigma per-percentile:", sigma_per)
    print("sigma (LS):", sigma_ls)
    return sigma_ls


def generate_x_y_sample(sigma_value):
    thetas = np.random.uniform(0, 2*np.pi, size=1)
    rs_sample = np.random.rayleigh(scale=sigma_value, size=1)
    x_sample = rs_sample * np.cos(thetas)[0]
    y_sample = rs_sample * np.sin(thetas)[0]
    return x_sample, y_sample
