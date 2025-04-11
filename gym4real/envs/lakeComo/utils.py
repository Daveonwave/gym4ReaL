import numpy as np
import random

class Utils:

    @staticmethod
    def load_matrix(file_name, row, col):
        data = np.loadtxt(file_name)
        return data.reshape((row, col)).tolist()

    @staticmethod
    def load_vector(file_name, length):
        return np.loadtxt(file_name, max_rows=length).tolist()

    @staticmethod
    def load_array(file_name, length):
        return np.loadtxt(file_name, max_rows=length)

    @staticmethod
    def log_vector(vec, file_name):
        with open(file_name, 'w') as f:
            for val in vec:
                f.write(f"{val}\n")

    @staticmethod
    def log_vector_append(vec, file_name):
        with open(file_name, 'a') as f:
            for val in vec:
                f.write(f"{val}\n")

    @staticmethod
    def interp_lin(X, Y, x):
        return np.interp(x, X, Y, left=None, right=None)

    # Unit Conversions
    @staticmethod
    def gallon_to_cubic_feet(x):
        return x * 0.13368

    @staticmethod
    def inches_to_feet(x):
        return x * 0.08333

    @staticmethod
    def cubic_feet_to_cubic_meters(x):
        return x * 0.0283

    @staticmethod
    def feet_to_meters(x):
        return x * 0.3048

    @staticmethod
    def acre_to_squared_feet(x):
        return x * 43560

    @staticmethod
    def acre_feet_to_cubic_feet(x):
        return x * 43560

    @staticmethod
    def cubic_feet_to_acre_feet(x):
        return x / 43560

    # Vector Operations
    @staticmethod
    def compute_sum(g):
        return sum(g)

    @staticmethod
    def compute_max(g):
        return max(g)

    @staticmethod
    def compute_min(g):
        return min(g)

    @staticmethod
    def compute_mean(g):
        return np.mean(g)

    @staticmethod
    def compute_variance(g):
        return np.var(g)

    # Normalization
    @staticmethod
    def normalize_vector(X, m, M):
        return [(x - mi) / (Ma - mi) for x, mi, Ma in zip(X, m, M)]

    @staticmethod
    def denormalize_vector(X, m, M):
        return [x * (Ma - mi) + mi for x, mi, Ma in zip(X, m, M)]

    # Standardization
    @staticmethod
    def standardize_vector(X, m, s):
        return [(x - mi) / si for x, mi, si in zip(X, m, s)]

    @staticmethod
    def destandardize_vector(X, m, s):
        return [x * si + mi for x, mi, si in zip(X, m, s)]

    # Random number
    @staticmethod
    def generate_random_unif(lower_bound, upper_bound):
        return random.uniform(lower_bound, upper_bound)
