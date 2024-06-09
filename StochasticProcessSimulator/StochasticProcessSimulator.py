import numpy as np
from scipy.special import gamma, hyp2f1
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Function to plot 2D paths with a colormap
def plot_paths(paths, title, plot_variance=True, colmap_name = 'cool', envelope_col = 'yellow', background_col = 'white', envelope_width_multiple_of_sd = 5.2):
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap(colmap_name)
    colors = cmap(np.linspace(0, 1, paths.shape[0]))
    ax = plt.axes()
    # Setting the background color of the plot 
    # using set_facecolor() method
    ax.set_facecolor(background_col)
    for i in range(paths.shape[0]):
        plt.plot(paths[i], color=colors[i], alpha=0.8, linewidth=0.8)
    mean_path = np.mean(paths, axis=0)
    plt.plot(mean_path, color='black', linestyle='--', linewidth=2, label='Mean')
    plt.axhline(np.max(paths), color='orange', linestyle='--', linewidth=1, label='Max')
    plt.axhline(np.min(paths), color='green', linestyle='--', linewidth=1, label='Min')
    if plot_variance and paths.ndim == 2:
        variance = np.var(paths, axis=0)
        plt.fill_between(np.arange(len(mean_path)), mean_path - (envelope_width_multiple_of_sd/2)*np.sqrt(variance), mean_path + (envelope_width_multiple_of_sd/2)*np.sqrt(variance), color=envelope_col, alpha=0.35, label='Variance Envelope')
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
# Function to plot complex paths
def plot_complex_paths(paths, title, colmap_name = 'magma', background_col = 'white'):
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap(colmap_name)
    colors = cmap(np.linspace(0, 1, paths[0].shape[0]))
    ax = plt.axes()
    # Setting the background color of the plot 
    # using set_facecolor() method
    ax.set_facecolor(background_col)
    for i in range(paths[0].shape[0]):
        plt.plot(paths[0][i], paths[1][i], color=colors[i], alpha=0.8, linewidth=0.8)

    plt.title(title)
    plt.xlabel('Y_t')
    plt.ylabel('X_t')
    plt.show()
    
# Function to plot correlated 2D paths with different colormaps
def plot_correlated_paths(paths, title, background_col = 'white'):
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    # Setting the background color of the plot 
    # using set_facecolor() method
    ax.set_facecolor(background_col)
    for i in range(paths.shape[0]):
        plt.plot(paths[i, :, 0], color='cyan', alpha=0.8, linewidth=0.8)
        plt.plot(paths[i, :, 1], color='yellow', alpha=0.8, linewidth=0.8)
    mean_path1 = np.mean(paths[:, :, 0], axis=0)
    mean_path2 = np.mean(paths[:, :, 1], axis=0)
    plt.plot(mean_path1, color='black', linestyle='--', linewidth=2, label='Mean Path 1')
    plt.plot(mean_path2, color='blue', linestyle='--', linewidth=2, label='Mean Path 2')
    plt.axhline(np.max(paths[:, :, 0]), color='red', linestyle='--', linewidth=1, label='Max Path 1')
    plt.axhline(np.min(paths[:, :, 0]), color='green', linestyle='--', linewidth=1, label='Min Path 1')
    plt.axhline(np.max(paths[:, :, 1]), color='orange', linestyle='--', linewidth=1, label='Max Path 2')
    plt.axhline(np.min(paths[:, :, 1]), color='purple', linestyle='--', linewidth=1, label='Min Path 2')
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
# Function to plot 3D paths with a colormap
def plot_3d_paths(paths, title, colmap_name = 'magma'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap(colmap_name)
    colors = cmap(np.linspace(0, 1, paths.shape[0]))
    for i in range(paths.shape[0]):
        ax.plot(paths[i, :, 0], paths[i, :, 1], paths[i, :, 2], color=colors[i], alpha=0.8, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def plot_eigenvalues(eigenvalues, title):
    plt.figure(figsize=(12, 6))
    for i in range(eigenvalues.shape[0]):
        plt.plot(eigenvalues[i], alpha=0.8, linewidth=0.8)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Eigenvalue')
    plt.show()
    
def plot_bessel_paths(paths, title, plot_variance=True, colmap_name = 'cool', envelope_col = 'yellow', background_col = 'white', envelope_width_multiple_of_sd = 5):
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap(colmap_name)
    colors = cmap(np.linspace(0, 1, paths.shape[0]))
    ax = plt.axes()
    # Setting the background color of the plot 
    # using set_facecolor() method
    ax.set_facecolor(background_col)
    for i in range(paths.shape[0]):
        plt.plot(paths[i], color=colors[i], alpha=0.8, linewidth=0.8)
    mean_path = np.mean(paths, axis=0)
    upper_lim = np.mean(mean_path) + 5.2 * np.std(paths)
    lower_lim = np.mean(mean_path) - 5.2 * np.std(paths)
    plt.plot(mean_path, color='black', linestyle='--', linewidth=2, label='Mean')
    if plot_variance and paths.ndim == 2:
        variance = np.var(paths, axis=0)
        plt.fill_between(np.arange(len(mean_path)), mean_path - (envelope_width_multiple_of_sd/2)*np.sqrt(variance), mean_path + (envelope_width_multiple_of_sd/2)*np.sqrt(variance), color=envelope_col, alpha=0.35, label='Variance Envelope')
    plt.axhline(np.max(paths), color='orange', linestyle='--', linewidth=1, label='Max')
    plt.axhline(np.min(paths), color='green', linestyle='--', linewidth=1, label='Min')
    plt.ylim(lower_lim, upper_lim)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
def custom_a(t, X_t):
    return 0.05*X_t * X_t * np.sin(t) 
    
def custom_b(t, X_t):
    return 0.05*(-X_t*X_t*X_t) * np.cos(t)
        
class StochasticProcessSimulator:
    def __init__(self, process_type, num_paths=50, path_length=252, do_plot = False, plot_variance = True, colmap_name = 'cool', envelope_col = 'yellow', background_col = 'white', dt=1/252, time_exec = True, **kwargs):
        self.process_type = process_type
        self.num_paths = num_paths
        self.path_length = path_length
        self.do_plot = do_plot
        self.plot_variance = plot_variance
        self.colmap_name = colmap_name
        self.envelope_col = envelope_col
        self.background_col = background_col
        self.time_exec = time_exec
        self.dt = dt
        self.params = kwargs
        self.validate_params() 

    def validate_params(self):
        if self.process_type == 'OU':
            self.params.setdefault('kappa', 0.5)
            self.params.setdefault('theta', 1.3)
            self.params.setdefault('sigma', 0.55)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'CIR':
            self.params.setdefault('kappa', 0.55)
            self.params.setdefault('theta', 0.34)
            self.params.setdefault('sigma', 0.3)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'GBM':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
        elif self.process_type == '(Heston) GBMSA':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.4)
            self.params.setdefault('rho', 0.3)
            self.params.setdefault('v0', 0.04)
        elif self.process_type == 'VG':
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('theta', 0.1)
            self.params.setdefault('nu', 0.1)
            self.params.setdefault('S0', 1.0)
        elif self.process_type == 'VGSA':
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('theta', 0.4)
            self.params.setdefault('nu', 0.1)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('kappa', 0.5)
            self.params.setdefault('eta', 0.1)
            self.params.setdefault('lambda_', 0.1)
            self.params.setdefault('v0', 0.1)
        elif self.process_type == 'Merton':
            self.params.setdefault('mu', -0.3)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('lambda_', 0.1)
            self.params.setdefault('muJ', 0.7)
            self.params.setdefault('sigmaJ', 0.15)
        elif self.process_type == 'ATSM':
            self.params.setdefault('ar', 0.01)
            self.params.setdefault('br', np.array([0.02]))
            self.params.setdefault('B', np.array([[0.03]]))
            self.params.setdefault('Sigma', np.array([[0.1]]))
            self.params.setdefault('x0', np.array([0.015]))
        elif self.process_type == 'ATSM_SV':
            self.params.setdefault('ar', 0.01)
            self.params.setdefault('br', np.array([0.02]))
            self.params.setdefault('B', np.array([[0.03]]))
            self.params.setdefault('Sigma', np.array([[0.1]]))
            self.params.setdefault('x0', np.array([0.015]))
            self.params.setdefault('kappa', 0.74)
            self.params.setdefault('theta', 0.7)
            self.params.setdefault('lambda_', 0.9)
            self.params.setdefault('v0', 0.7)
        elif self.process_type == 'CEV':
            self.params.setdefault('mu', 0.3)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('beta', 0.5)
        elif self.process_type == 'BrownianBridge':
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('XT', 0.9)
        elif self.process_type == 'BrownianMeander':
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'BrownianExcursion':
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'CorrelatedBM':
            self.params.setdefault('rho', 0.9)
        elif self.process_type == 'dDimensionalBM':
            self.params.setdefault('d', 3)
        elif self.process_type == '3dGBM':
            self.params.setdefault('mu', np.array([0.3, 0.3, 0.3]))
            self.params.setdefault('sigma', np.eye(3) * 0.2)
            self.params.setdefault('S0', np.array([1.0, 1.0, 1.0]))
        elif self.process_type == 'CorrelatedGBM':
            self.params.setdefault('mu1', 0.3)
            self.params.setdefault('sigma1', 0.05)
            self.params.setdefault('S01', 1.0)
            self.params.setdefault('mu2', -0.3)
            self.params.setdefault('sigma2', 0.05)
            self.params.setdefault('S02', 1.0)
            self.params.setdefault('rho', -0.6)
        elif self.process_type == 'Vasicek':
            self.params.setdefault('a', 0.55)
            self.params.setdefault('b', -0.9)
            self.params.setdefault('sigma', 0.035)
            self.params.setdefault('r0', 0.15)
        elif self.process_type == 'ExponentialVasicek':
            self.params.setdefault('a', 0.1)
            self.params.setdefault('b', 0.05)
            self.params.setdefault('sigma', 0.01)
            self.params.setdefault('r0', 0.03)
            self.params.setdefault('rho', 0.1)
            self.params.setdefault('lambda', 0.1)
        elif self.process_type == 'GeneralBergomi':
            self.params.setdefault('r', 0.05)
            self.params.setdefault('q', 0.02)
            self.params.setdefault('xi', 0.1)
            self.params.setdefault('omega', 0.2)
        elif self.process_type == 'OneFactorBergomi':
            self.params.setdefault('omega', 0.2)
            self.params.setdefault('kappa', 0.3)
            self.params.setdefault('XT', 0.1)
            self.params.setdefault('X0', 0.05)
        elif self.process_type == 'RoughBergomi':
            self.params.setdefault('eta', 0.3)
            self.params.setdefault('nu', 0.5)
            self.params.setdefault('rho', 0.1)
            self.params.setdefault('xi0', 0.1)
        elif self.process_type == 'RoughVolatility':
            self.params.setdefault('B', lambda t: 0.1)
            self.params.setdefault('xi', 0.1)
            self.params.setdefault('g', lambda t: 0.35)
        elif self.process_type == 'DysonBM':
            self.params.setdefault('n', 1)
            self.params.setdefault('epsilon', 0.01)  # Small offset to avoid division by zero
        elif self.process_type == 'fBM':
            self.params.setdefault('H', 0.78)
            self.params.setdefault('sigma', 0.55)
        elif self.process_type == 'fIM':
            self.params.setdefault('H', 0.78)
            self.params.setdefault('sigma', 0.15)
        elif self.process_type == 'SABR':
            self.params.setdefault('alpha', 0.3)
            self.params.setdefault('beta', 0.5)
            self.params.setdefault('rho', -0.4)
            self.params.setdefault('F0', 0.4)
            self.params.setdefault('sigma0', 0.2)
        elif self.process_type == 'ShiftedSABR':
            self.params.setdefault('alpha', 0.3)
            self.params.setdefault('beta', 0.5)
            self.params.setdefault('rho', -0.4)
            self.params.setdefault('F0', 0.4)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('s', 0.47)
        elif self.process_type == '3dOU':
            self.params.setdefault('kappa', 0.9)
            self.params.setdefault('theta', np.array([1.0, 1.0, 1.0]))
            self.params.setdefault('sigma', np.eye(3))
            self.params.setdefault('X0', np.zeros(3))
        elif self.process_type == 'StickyBM':
            self.params.setdefault('mu', 0.05)
            self.params.setdefault('sticky_point', 0.18)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('sigma', 0.5)
        elif self.process_type == 'ReflectingBM':
            self.params.setdefault('lower_b', -1)
            self.params.setdefault('upper_b', 1)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('sigma', 1.0)
        elif self.process_type == 'BM':
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('mu', 0.0)
            self.params.setdefault('sigma', 1.0)
        elif self.process_type == 'DTDG':
            self.params.setdefault('ar', 0.01)
            self.params.setdefault('br', np.array([0.02]))
            self.params.setdefault('B', np.array([[0.03]]))
            self.params.setdefault('Sigma', np.array([[0.5]]))
            self.params.setdefault('x0', np.array([0.015]))
            self.params.setdefault('gamma', 0.5)
            self.params.setdefault('lambda_', 0.2)
            self.params.setdefault('v0', 0.1)
        elif self.process_type == 'CKLS':
            self.params.setdefault('alpha', 0.35)
            self.params.setdefault('beta', 0.55)
            self.params.setdefault('gamma', 0.5)
            self.params.setdefault('sigma', 0.05)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'HullWhite':
            self.params.setdefault('theta', lambda t: 0.05)
            self.params.setdefault('alpha', lambda t: 0.09)
            self.params.setdefault('sigma', lambda t: 0.01)
            self.params.setdefault('r0', 0.03)
        elif self.process_type == 'LotkaVolterra':
            self.params.setdefault('b', 0.1)
            self.params.setdefault('a', 0.35)
            self.params.setdefault('sigma', 0.45)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'TwoFactorHullWhite':
            self.params.setdefault('theta', lambda t: 0.05)
            self.params.setdefault('alpha', lambda t: 0.09)
            self.params.setdefault('sigma1', lambda t: 0.03)
            self.params.setdefault('sigma2', 0.09)
            self.params.setdefault('b', 0.03)
            self.params.setdefault('r0', 0.03)
            self.params.setdefault('u0', 0.3)
        elif self.process_type == 'BlackKarasinski':
            self.params.setdefault('theta', lambda t: 0.05)
            self.params.setdefault('phi', lambda t: 0.09)
            self.params.setdefault('sigma', lambda t: 0.03)
            self.params.setdefault('r0', 0.03)
        elif self.process_type == 'Bates':
            self.params.setdefault('gamma', 0.05)
            self.params.setdefault('q', 0.02)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('k', 0.3)
            self.params.setdefault('theta', 0.04)
            self.params.setdefault('sigma_v', 0.02)
            self.params.setdefault('lambda', 0.1)
            self.params.setdefault('muJ', 0.35)
            self.params.setdefault('sigmaJ', 0.1)
            self.params.setdefault('S0', 1.0)
        elif self.process_type == 'fSV':
            self.params.setdefault('sigma_star', 0.15)
            self.params.setdefault('gamma_h', 0.59)
            self.params.setdefault('sigma_h_star', 0.35)
            self.params.setdefault('y0', 0.5)
            self.params.setdefault('h0', 0.95)
            self.params.setdefault('H', 0.83)
            self.params.setdefault('sigma', 0.2)
        elif self.process_type == 'fOU':
            self.params.setdefault('nu', 0.8)
            self.params.setdefault('kappa', 8.1)
            self.params.setdefault('m', 0.05)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('H', 0.3)
            self.params.setdefault('sigma', 0.2)
        elif self.process_type == 'RfSV':
            self.params.setdefault('nu', 0.8)
            self.params.setdefault('H', 0.15)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('sigma0', 0.2)
        elif self.process_type == 'Bessel':
            self.params.setdefault('n', 2)
            self.params.setdefault('X0', 0.05)
        elif self.process_type == 'SquaredBessel':
            self.params.setdefault('delta', 2)
            self.params.setdefault('X0', 0.05)
        elif self.process_type == 'GARCH':
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.35)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('lambda0', 0.15)
        elif self.process_type == 'GARCHJump':
            self.params.setdefault('muJ', -0.015)
            self.params.setdefault('sigmaJ', 0.1)
            self.params.setdefault('lambda_', 0.1)
            self.params.setdefault('beta0', 0.9)
            self.params.setdefault('beta1', 0.5)
            self.params.setdefault('beta2', 0.35)
            self.params.setdefault('c', 0.25)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('h0', 0.1)
        elif self.process_type == 'VIX':
            self.params.setdefault('H', 0.35)
            self.params.setdefault('nu', 0.5)
            self.params.setdefault('rho', -0.7)
            self.params.setdefault('xi0', 0.1)
            self.params.setdefault('v0', 0.15)
        elif self.process_type == 'GFBM':
            self.params.setdefault('H', 0.9)
            self.params.setdefault('sigma', 2)
            self.params.setdefault('theta', 0.78)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'StochasticLogisticGrowth':
            self.params.setdefault('r', 3.0)
            self.params.setdefault('K', 1.5)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.1)
        elif self.process_type == 'StochasticExponentialDecay':
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 1.0)
        elif self.process_type == 'SinCosVectorNoiseIto':
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('A', 3.0)
            self.params.setdefault('B', 1.0)
        elif self.process_type == 'ExponentialOU':
            self.params.setdefault('theta', 0.7)
            self.params.setdefault('mu', 0.15)
            self.params.setdefault('sigma', 0.35)
            self.params.setdefault('X0', 0.1)
        elif self.process_type == 'SCP_mean_reverting':
            self.params.setdefault('kappa', 1.9)
            self.params.setdefault('mu', 0.15)
            self.params.setdefault('sigma', 0.35)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'SCP_modified_OU':
            self.params.setdefault('kappa', 3.3)
            self.params.setdefault('mu', 0.5)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'SCQuanto':
            self.params.setdefault('muS', 0.15)
            self.params.setdefault('sigmaS', 0.1)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('muR', 0.13)
            self.params.setdefault('sigmaR', 0.8)
            self.params.setdefault('R0', 1.0)
            self.params.setdefault('kappa', 0.75)
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('rho0', 0.25)
        elif self.process_type == 'SCP_tanh_Ito':
            self.params.setdefault('a', lambda t, X_t: 0.05*X_t * X_t * np.sin(t))
            self.params.setdefault('b', lambda t, X_t: 0.05*(-X_t*X_t*X_t) * np.cos(t))
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'SCP_arctan_Ito':
            self.params.setdefault('a', lambda t, X_t: 0.05*X_t * X_t * np.sin(t))
            self.params.setdefault('b', lambda t, X_t: 0.05*(-X_t*X_t*X_t) * np.cos(t))
            self.params.setdefault('X0', 0.1)
        elif self.process_type == 'Chen':
            self.params.setdefault('kappa', 0.5)
            self.params.setdefault('theta0', 0.1)
            self.params.setdefault('sigma0', 0.1)
            self.params.setdefault('r0', 0.03)
            self.params.setdefault('nu', 0.1)
            self.params.setdefault('zeta', 0.1)
            self.params.setdefault('alpha', 0.1)
            self.params.setdefault('beta', 0.1)
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('eta', 0.1)
        elif self.process_type == 'LongstaffSchwartz':
            self.params.setdefault('a_t', lambda t: 0.1)
            self.params.setdefault('c_t', lambda t: 0.1)
            self.params.setdefault('d_t', lambda t: 0.1)
            self.params.setdefault('f_t', lambda t: 0.1)
            self.params.setdefault('b', 0.1)
            self.params.setdefault('e', 0.1)
            self.params.setdefault('theta', 0.1)
            self.params.setdefault('mew', 0.1)
        elif self.process_type == 'BDT':
            self.params.setdefault('theta_t', lambda t: np.sin(t))
            self.params.setdefault('sigma_t', lambda t: 3*t*t+1)
            self.params.setdefault('sigma_t_prime', lambda t: 6*t+1)
            self.params.setdefault('r0', 0.05)
        elif self.process_type == 'HoLee':
            self.params.setdefault('theta_t', lambda t: np.sin(0.1*t))
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('r0', 0.03)
        elif self.process_type == 'CIR++':
            self.params.setdefault('a', 0.1)
            self.params.setdefault('b', 0.05)
            self.params.setdefault('sigma', 0.5)
            self.params.setdefault('phi_t', lambda t: 0.1*np.sin(t))
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'KWF':
            self.params.setdefault('theta_t', lambda t: 0.1*t*t)
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('r0', 0.03)
        elif self.process_type == 'fStochasticLogisticGrowth':
            self.params.setdefault('r', 1.15)
            self.params.setdefault('H', 0.3)
            self.params.setdefault('K', 1.0)
            self.params.setdefault('sigma', 0.5)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'fStochasticExponentialDecay':
            self.params.setdefault('sigma', 0.5)
            self.params.setdefault('H', 0.3)
            self.params.setdefault('X0', 1.0)
        elif self.process_type == 'tanh_fOU':
            self.params.setdefault('nu', 5.5)
            self.params.setdefault('kappa', 10.0)
            self.params.setdefault('m', 0.5)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('sigma', 0.15)
            self.params.setdefault('H', 0.85)
        elif self.process_type == 'ARIMA':
            self.params.setdefault('p', 2)
            self.params.setdefault('d', 1)
            self.params.setdefault('q', 3)
            self.params.setdefault('ar_params', np.array([5.0]))
            self.params.setdefault('ma_params', np.array([-3.0]))
        elif self.process_type == 'SinRFSV':
            self.params.setdefault('nu', 0.5)
            self.params.setdefault('H', 0.4)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('A', 1.0)
        elif self.process_type == 'TanhRFSV':
            self.params.setdefault('nu', 0.5)
            self.params.setdefault('H', 0.4)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('A', 1.0)
        elif self.process_type == 'ConicDiffusionMartingale':
            self.params.setdefault('b', lambda t: 0.5 * np.sin(t))
            self.params.setdefault('b_prime', lambda t: 0.5 * np.cos(t))
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.05)
        elif self.process_type == 'ConicUnifDiffusionMartingale':
            self.params.setdefault('b', lambda t: 0.5 * np.sin(t))
            self.params.setdefault('b_prime', lambda t: 0.5 * np.cos(t))
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('Z0', 0.05)
        elif self.process_type == 'ConicHalfUnifDiffusionMartingale':
            self.params.setdefault('b', lambda t: 0.1)
            self.params.setdefault('b_prime', lambda t: 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('Y0', 0.05)
        elif self.process_type == 'CircleBM':
            self.params.setdefault('c_x', 1)
            self.params.setdefault('c_y', 5)
        elif self.process_type == 'SinFourierDecompBB':
            self.params.setdefault('num_terms', 100)
        elif self.process_type == 'MixedFourierDecompBB':
            self.params.setdefault('num_terms', 100)
        elif self.process_type == 'Jacobi':
            self.params.setdefault('R_upper', 1.0)
            self.params.setdefault('R_lower', -1.0)
            self.params.setdefault('R', 0.0005)
            self.params.setdefault('k', 0.341)
            self.params.setdefault('rho0', 0.1)
            self.params.setdefault('sigma', 2.0)
        elif self.process_type == 'WrightFisherDiffusion':
            self.params.setdefault('kappa', 0.7)
            self.params.setdefault('sigma', 15.0)
            self.params.setdefault('X0', 0.95)
            self.params.setdefault('rho_', 0.7)
        elif self.process_type == 'WrightFisherSC':
            self.params.setdefault('kappa', 1.3)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('rho_', 0.1)
        elif self.process_type == 'PolynomialItoProcess':
            self.params.setdefault('a', (lambda x: 0.05 * x * x + np.sin(x) - 0.1))
            self.params.setdefault('b', (lambda x: 0.05 * (-x) + 0.15 * np.cos(x)))
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'WeibullDiffusion':
            self.params.setdefault('alpha', 0.4)
            self.params.setdefault('lambda_', 10)
            self.params.setdefault('k', 3)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'WeibullDiffusion2':
            self.params.setdefault('alpha', 2)
            self.params.setdefault('lambda_', 1)
            self.params.setdefault('k', 3)
            self.params.setdefault('X0', 0.015)
        elif self.process_type == 'kCorrelatedGBMs':
            self.params.setdefault('mu', np.array([0.1, 0.2, 0.3]))
            self.params.setdefault('sigma', np.array([0.2, 0.2, 0.2]))
            self.params.setdefault('S0', np.array([1, 1, 1]))
            self.params.setdefault('rho', np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]]))
        elif self.process_type == 'kCorrelatedBMs':
            self.params.setdefault('mu', np.array([0.1, 0.2, 0.3]))
            self.params.setdefault('sigma', np.array([0.2, 0.2, 0.2]))
            self.params.setdefault('X0', np.array([1, 1, 1]))
            self.params.setdefault('rho', np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]]))
        elif self.process_type == 'Poly_fOU':
            self.params.setdefault('nu', 0.09)
            self.params.setdefault('kappa', 0.07)
            self.params.setdefault('m', 0.15)
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('X0', 0.015)
            self.params.setdefault('H', 0.15)
            self.params.setdefault('coefficients', np.array([0.1, 0.15, -0.03, np.pi] + [0]*7))
        elif self.process_type == 'GaussTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('scale_', 0.1)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'fBM_WrightFisherDiffusion':
            self.params.setdefault('kappa', 0.5)
            self.params.setdefault('sigma_WFD', 2.0)
            self.params.setdefault('X0', 0.95)
            self.params.setdefault('rho_', 0.70)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 5.0)
        elif self.process_type == 'LaplaceTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('scale_', 0.1)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'CauchyTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.1)
            self.params.setdefault('sigma0', 0.1)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'triangularTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('l', -1)
            self.params.setdefault('m', 0)
            self.params.setdefault('r', 1.5)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 't_TanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'GumbelTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('scale_', 0.1)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'LogisticTanhPolyRFSV':
            self.params.setdefault('nu', 5.0)
            self.params.setdefault('H', 0.8)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('scale_', 0.1)
            self.params.setdefault('sigma0', 0.2)
            self.params.setdefault('coefficients', np.array([0, -0.01, 0.2, -0.1] + [0]*7))
        elif self.process_type == 'CIR2++':
            self.params.setdefault('kappa_x', 0.5)
            self.params.setdefault('theta_x', 0.7)
            self.params.setdefault('sigma_x', 0.2)
            self.params.setdefault('X0', 1.1)
            self.params.setdefault('kappa_y', 0.5)
            self.params.setdefault('theta_y', 0.4)
            self.params.setdefault('sigma_y', 0.3)
            self.params.setdefault('Y0', 1.015)
            self.params.setdefault('phi_t', (lambda t: 0.030 * t * np.sin(t/100)))
        elif self.process_type == 'SBM':
            self.params.setdefault('alpha', 2.0)
            self.params.setdefault('X0', 0.5) 
        elif self.process_type == 'ExpTC':
            self.params.setdefault('sigma', 2.0)
            self.params.setdefault('v', 3.0)
            self.params.setdefault('delta', 0.03)
            self.params.setdefault('X0', 0.05)
        elif self.process_type == 'srGBM':
            self.params.setdefault('mu', 0.5)
            self.params.setdefault('sigma', 0.5)
            self.params.setdefault('S0', 0.7)
            self.params.setdefault('eta', 0.4)
            self.params.setdefault('r', 2.0)
        elif self.process_type == 'MRSqrtDiff':
            self.params.setdefault('epsilon', 1.9)
            self.params.setdefault('theta', 0.7)
            self.params.setdefault('kappa', 0.8)
            self.params.setdefault('X0', 0.015)
        else:
            raise ValueError("Unsupported process type")
        
    def simulate(self):
        if self.process_type == 'OU':
            return self.simulate_OU()
        elif self.process_type == 'CIR':
            return self.simulate_CIR()
        elif self.process_type == 'GBM':
            return self.simulate_GBM()
        elif self.process_type == '(Heston) GBMSA':
            return self.simulate_GBMSA()
        elif self.process_type == 'VG':
            return self.simulate_VG()
        elif self.process_type == 'VGSA':
            return self.simulate_VGSA()
        elif self.process_type == 'Merton':
            return self.simulate_Merton()
        elif self.process_type == 'ATSM':
            return self.simulate_ATSM()
        elif self.process_type == 'ATSM_SV':
            return self.simulate_ATSM_SV()
        elif self.process_type == 'CEV':
            return self.simulate_CEV()
        elif self.process_type == 'BrownianBridge':
            return self.simulate_BrownianBridge()
        elif self.process_type == 'BrownianMeander':
            return self.simulate_BrownianMeander()
        elif self.process_type == 'BrownianExcursion':
            return self.simulate_BrownianExcursion()
        elif self.process_type == 'CorrelatedBM':
            return self.simulate_CorrelatedBM()
        elif self.process_type == 'dDimensionalBM':
            return self.simulate_dDimensionalBM()
        elif self.process_type == '3dGBM':
            return self.simulate_3dGBM()
        elif self.process_type == 'CorrelatedGBM':
            return self.simulate_CorrelatedGBM()
        elif self.process_type == 'Vasicek':
            return self.simulate_Vasicek()
        elif self.process_type == 'ExponentialVasicek':
            return self.simulate_ExponentialVasicek()
        elif self.process_type == 'GeneralBergomi':
            return self.simulate_GeneralBergomi()
        elif self.process_type == 'OneFactorBergomi':
            return self.simulate_OneFactorBergomi()
        elif self.process_type == 'RoughBergomi':
            return self.simulate_RoughBergomi()
        elif self.process_type == 'RoughVolatility':
            return self.simulate_RoughVolatility()
        elif self.process_type == 'DysonBM':
            return self.simulate_DysonBM()
        elif self.process_type == 'fBM':
            return self.simulate_fBM()
        elif self.process_type == 'fIM':
            return self.simulate_fIM()
        elif self.process_type == 'SABR':
            return self.simulate_SABR()
        elif self.process_type == 'ShiftedSABR':
            return self.simulate_ShiftedSABR()
        elif self.process_type == '3dOU':
            return self.simulate_3dOU()
        elif self.process_type == 'BM':
            return self.simulate_BM()
        elif self.process_type == 'StickyBM':
            return self.simulate_StickyBM()
        elif self.process_type == 'ReflectingBM':
            return self.simulate_ReflectingBM()
        elif self.process_type == 'DTDG':
            return self.simulate_DTDG()
        elif self.process_type == 'CKLS':
            return self.simulate_CKLS()
        elif self.process_type == 'HullWhite':
            return self.simulate_HullWhite()
        elif self.process_type == 'LotkaVolterra':
            return self.simulate_LotkaVolterra()
        elif self.process_type == 'TwoFactorHullWhite':
            return self.simulate_TwoFactorHullWhite()
        elif self.process_type == 'BlackKarasinski':
            return self.simulate_BlackKarasinski()
        elif self.process_type == 'Bates':
            return self.simulate_Bates()
        elif self.process_type == 'fSV':
            return self.simulate_fSV()
        elif self.process_type == 'fOU':
            return self.simulate_fOU()
        elif self.process_type == 'RfSV':
            return self.simulate_RfSV()
        elif self.process_type == 'Bessel':
            return self.simulate_Bessel()
        elif self.process_type == 'SquaredBessel':
            return self.simulate_SquaredBessel()
        elif self.process_type == 'GARCH':
            return self.simulate_GARCH()
        elif self.process_type == 'GARCHJump':
            return self.simulate_GARCHJump()
        elif self.process_type == 'VIX':
            return self.simulate_VIX()
        elif self.process_type == 'GFBM':
            return self.simulate_GFBM()
        elif self.process_type == 'StochasticLogisticGrowth':
            return self.simulate_StochasticLogisticGrowth()
        elif self.process_type == 'StochasticExponentialDecay':
            return self.simulate_StochasticExponentialDecay()
        elif self.process_type == 'SinCosVectorNoiseIto':
            return self.simulate_SinCosVectorNoiseIto()
        elif self.process_type == 'ExponentialOU':
            return self.simulate_ExponentialOU()
        elif self.process_type == 'SCP_tanh_Ito':
            return self.simulate_SCP_tanh_Ito()
        elif self.process_type == 'SCP_arctan_Ito':
            return self.simulate_SCP_arctan_Ito()
        elif self.process_type == 'SCP_mean_reverting':
            return self.simulate_SCP_mean_reverting()
        elif self.process_type == 'SCP_modified_OU':
            return self.simulate_SCP_modified_OU()
        elif self.process_type == 'SCQuanto':
            return self.simulate_SCQuanto()
        elif self.process_type == 'Chen': 
            return self.simulate_Chen()
        elif self.process_type == 'LongstaffSchwartz':
            return self.simulate_LongstaffSchwartz()
        elif self.process_type == 'BDT':
            return self.simulate_BDT()
        elif self.process_type == 'HoLee':
            return self.simulate_HoLee()
        elif self.process_type == 'CIR++':
            return self.simulate_CIRPlusPlus()
        elif self.process_type == 'KWF':
            return self.simulate_KWF()
        elif self.process_type == 'fStochasticLogisticGrowth':
            return self.simulate_fStochasticLogisticGrowth()
        elif self.process_type == 'fStochasticExponentialDecay':
            return self.simulate_fStochasticExponentialDecay()
        elif self.process_type == 'tanh_fOU':
            return self.simulate_tanh_fOU()
        elif self.process_type == 'ARIMA':
            return self.simulate_ARIMA()
        elif self.process_type == 'SinRFSV':
            return self.simulate_SinRFSV()
        elif self.process_type == 'TanhRFSV':
            return self.simulate_TanhRFSV()
        elif self.process_type == 'ConicDiffusionMartingale':
            return self.simulate_ConicDiffusionMartingale()
        elif self.process_type == 'ConicUnifDiffusionMartingale':
            return self.simulate_ConicUnifDiffusionMartingale()
        elif self.process_type == 'ConicHalfUnifDiffusionMartingale':
            return self.simulate_ConicHalfUnifDiffusionMartingale()
        elif self.process_type == 'CircleBM':
            return self.simulate_CircleBM()
        elif self.process_type == 'SinFourierDecompBB':
            return self.simulate_SinFourierDecompBB()
        elif self.process_type == 'MixedFourierDecompBB':
            return self.simulate_MixedFourierDecompBB()
        elif self.process_type == 'Jacobi':
            return self.simulate_Jacobi()
        elif self.process_type == 'WrightFisherDiffusion':
            return self.simulate_WrightFisherDiffusion()
        elif self.process_type == 'WrightFisherSC':
            return self.simulate_WrightFisherSC()
        elif self.process_type == 'PolynomialItoProcess':
            return self.simulate_PolynomialItoProcess()
        elif self.process_type == 'WeibullDiffusion':
            return self.simulate_WeibullDiffusion()
        elif self.process_type == 'WeibullDiffusion2':
            return self.simulate_WeibullDiffusion2()
        elif self.process_type == 'kCorrelatedGBMs':
            return self.simulate_kCorrelatedGBMs()
        elif self.process_type == 'kCorrelatedBMs':
            return self.simulate_kCorrelatedBMs()
        elif self.process_type == 'Poly_fOU':
            return self.simulate_Poly_fOU()
        elif self.process_type == 'GaussTanhPolyRFSV':
            return self.simulate_GaussTanhPolyRFSV()
        elif self.process_type == 'LaplaceTanhPolyRFSV':
            return self.simulate_LaplaceTanhPolyRFSV()
        elif self.process_type == 't_TanhPolyRFSV':
            return self.simulate_t_TanhPolyRFSV()
        elif self.process_type == 'CauchyTanhPolyRFSV':
            return self.simulate_CauchyTanhPolyRFSV()
        elif self.process_type == 'triangularTanhPolyRFSV':
            return self.simulate_triangularTanhPolyRFSV()
        elif self.process_type == 'GumbelTanhPolyRFSV': 
            return self.simulate_GumbelTanhPolyRFSV()
        elif self.process_type == 'LogisticTanhPolyRFSV': 
            return self.simulate_LogisticTanhPolyRFSV()
        elif self.process_type == 'fBM_WrightFisherDiffusion': 
            return self. simulat_fBM_WrightFisherDiffusion()
        elif self.process_type == 'CIR2++':
            return self.simulate_CIR2PlusPlus()
        elif self.process_type == 'SBM':
            return self.simulate_SBM()
        elif self.process_type == 'ExpTC':
            return self.simulate_ExpTC()
        elif self.process_type == 'srGBM':
            return self.simulate_srGBM()
        elif self.process_type == 'MRSqrtDiff':
            return self.simulate_MRSqrtDiff()

    def simulate_fOU_helper(self):
        start_time = time.time()
        H = self.params['H']
        sigma = self.params['sigma']
        kappa = self.params['kappa']
        m = self.params['m']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] - kappa * (paths[:, t-1] - m) * dt + fBM_paths[:, t]

        end_time = time.time()
        if self.time_exec == True: 
            print('Underlying fOU execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_fBM_helper(self):
        start_time = time.time()
        H = self.params['H']  # 0 < H < 1
        sigma = self.params['sigma']  # 0 < sigma
        dt = self.dt
        num_steps = self.path_length
        T = num_steps * dt
    
        # Generate time vector
        t = np.linspace(dt, T, num_steps)  # Start from dt to avoid division by zero
    
        # Define the kernel function K_H(t, s)
        def K_H(t, s, H):
            if t == s:
                return 0
            elif t > s:
                factor1 = (t - s) ** (H - 0.5)
                factor2 = (t / s) ** (H - 0.5)
                factor3 = hyp2f1(H - 0.5, 0.5 - H, H + 0.5, 1 - (t / s))
                return factor1 * factor2 * factor3 / gamma(H + 0.5)
            else:
                return 0
    
        # Compute the increments of a standard Brownian motion
        dB = np.sqrt(dt) * np.random.randn(self.num_paths, num_steps)
    
        # Compute the paths for fBM
        paths = np.zeros((self.num_paths, num_steps))
        for i in range(self.num_paths):
            for j in range(1, num_steps):
                integral = 0
                for k in range(j):
                    integral += K_H(t[j], t[k], H) * dB[i, k]
                paths[i, j] = integral
    
        paths *= sigma

        end_time = time.time()
        if self.time_exec == True: 
            print('Underlying fBM execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_ExpTC(self):
        start_time = time.time()
        sigma = self.params['sigma']
        v = self.params['v']
        delta = self.params['delta']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        
        x_paths = np.zeros((self.num_paths, num_steps))
        y_paths = np.zeros((self.num_paths, num_steps))
        tau_paths = np.zeros((self.num_paths, num_steps))
        x_paths[:, 0] = X0
        y_paths[:, 0] = (X0)**(1-delta/2)
        
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            x_paths[:, t] = x_paths[:, t-1] + delta * dt + 2 * np.sqrt(np.abs(x_paths[:, t-1])) * dW
            
            y_previous = y_paths[:, t-1]
            condition = (y_previous > 0) | ((1-delta)/(2-delta) >= 1)
            
            y_term = np.where(condition, np.power(np.abs(y_previous), (1-delta)/(2-delta)), y_previous)
            y_term[y_previous < 0] = 0  # To handle cases where y_previous is negative
            
            y_paths[:, t] = y_previous + v * y_previous * dt + sigma * y_term * dW
            
            tau_paths[:, t] = ((sigma**2)/(2*v*(2-delta)))*(1-np.exp(-(2*v*t/(2-delta))))

        if self.do_plot == True: 
            plot_paths(x_paths, f'{self.process_type} x_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(y_paths, f'{self.process_type} y_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(tau_paths, f'{self.process_type} tau_transformation_of_t Paths', colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
        
        return x_paths, y_paths, tau_paths
    
    def simulate_srGBM(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma']
        S0 = self.params['S0']
        eta = self.params['eta']
        r = self.params['r']  # Resetting rate
        dt = self.dt
        num_steps = self.path_length
        
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            Zt = np.random.rand(self.num_paths) < (r * dt)
            paths[:, t] = np.where(Zt, S0, paths[:, t-1] + paths[:, t-1] * (mu * dt + sigma * dW))

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_SBM(self):
        start_time = time.time()
        alpha = self.params['alpha']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + np.sqrt(alpha) * (t ** ((alpha - 1) / 2)) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_MRSqrtDiff(self):
        start_time = time.time()
        epsilon = self.params['epsilon']
        theta = self.params['theta']
        kappa = self.params['kappa']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + epsilon * np.sqrt(np.abs(paths[:, t-1])) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_CIR2PlusPlus(self):
        start_time = time.time()
        kappa_x = self.params['kappa_x']
        theta_x = self.params['theta_x']
        sigma_x = self.params['sigma_x']
        X0 = self.params['X0']
        kappa_y = self.params['kappa_y']
        theta_y = self.params['theta_y']
        sigma_y = self.params['sigma_y']
        Y0 = self.params['Y0']
        phi_t = self.params['phi_t']
        dt = self.dt
        num_steps = self.path_length
        
        x_paths = np.zeros((self.num_paths, num_steps))
        y_paths = np.zeros((self.num_paths, num_steps))
        r_paths = np.zeros((self.num_paths, num_steps))
        x_paths[:, 0] = X0
        y_paths[:, 0] = Y0
        r_paths[:, 0] = X0 + Y0 
        
        for t in range(1, num_steps):
            dW_x = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW_y = np.sqrt(dt) * np.random.randn(self.num_paths)
            x_paths[:, t] = x_paths[:, t-1] + kappa_x * (theta_x - x_paths[:, t-1]) * dt + sigma_x * np.sqrt(np.abs(x_paths[:, t-1])) * dW_x
            y_paths[:, t] = y_paths[:, t-1] + kappa_y * (theta_y - y_paths[:, t-1]) * dt + sigma_y * np.sqrt(np.abs(y_paths[:, t-1])) * dW_y
            r_paths[:, t] = x_paths[:, t] + y_paths[:, t] + phi_t(t * dt)

        if self.do_plot == True:
            plot_paths(x_paths, f'{self.process_type} x_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(y_paths, f'{self.process_type} y_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(r_paths, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            phi_t = lambda t: 0.030 * t * np.sin(num_steps/100)
            phi = np.array([phi_t(ti) for ti in range(num_steps)])
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_steps), phi, label='$\phi(t)$')
            plt.title('CIR2++ $\phi(t)$ as a function of t')
            plt.xlabel('Time Steps')
            plt.ylabel('$\phi(t)$')
            plt.legend()
            plt.grid(True)
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return x_paths, y_paths, r_paths, phi_t
    
    def simulate_LogisticTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        scale_ = self.params['scale_']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.logistic(loc=0.0, scale=t*scale_, size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
                
    def simulate_GumbelTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        scale_ = self.params['scale_']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.gumbel(loc=0.0, scale=t*scale_, size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_triangularTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        l = self.params['l']
        m = self.params['m']
        r = self.params['r']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.triangular(l, m, r, self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_CauchyTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.standard_cauchy(size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_t_TanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.standard_t(self.num_paths-1, size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_LaplaceTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        scale_ = self.params['scale_']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.laplace(loc=0.0, scale=t*scale_, size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulat_fBM_WrightFisherDiffusion(self):
        start_time = time.time()
        kappa = self.params['kappa']
        sigma_WFD = self.params['sigma_WFD']
        X0 = self.params['X0']
        rho_ = self.params['rho_']
        H = self.params['H']
        sigma = self.params['sigma']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()  
        for t in range(1, num_steps):
            paths[:, t] = ((paths[:, t-1] + kappa * (rho_ - paths[:, t-1]) * dt + sigma * np.sqrt(
                np.maximum(0, (1 - (paths[:, t-1] * paths[:, t-1])))) * (fBM_paths[:, t] - fBM_paths[:, t-1]))/2 + 1/2)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_kCorrelatedGBMs(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma']
        S0 = self.params['S0']
        rho = self.params['rho']
        dt = self.dt
        num_steps = self.path_length
        num_assets = len(S0)
    
        # Cholesky decomposition
        L = np.linalg.cholesky(rho)
    
        paths = np.zeros((self.num_paths, num_steps, num_assets))
        paths[:, 0, :] = S0
    
        for t in range(1, num_steps):
            dW = np.dot(L, np.random.randn(self.num_paths, num_assets).T).T * np.sqrt(dt)
            paths[:, t, :] = paths[:, t-1, :] * np.exp((mu - 0.5 * np.diag(sigma)**2) * dt + np.dot(dW, sigma.T))

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_kCorrelatedBMs(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        rho = self.params['rho']
        dt = self.dt
        num_steps = self.path_length
        num_assets = len(X0)
    
        # Cholesky decomposition
        L = np.linalg.cholesky(rho)
    
        paths = np.zeros((self.num_paths, num_steps, num_assets))
        paths[:, 0, :] = X0
    
        for t in range(1, num_steps):
            dW = np.dot(L, np.random.randn(self.num_paths, num_assets).T).T * np.sqrt(dt)
            paths[:, t, :] = paths[:, t-1, :] + mu * dt + np.dot(dW, sigma.T)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_Poly_fOU(self):
        start_time = time.time()
        nu = self.params['nu']
        kappa = self.params['kappa']
        m = self.params['m']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        H = self.params['H']
        coefficients = self.params['coefficients']
        num_steps = self.path_length
        dt = self.dt
        fOU_paths = self.simulate_fOU_helper()
    
        transformed_paths = np.zeros_like(fOU_paths)
        for i in range(len(coefficients)):
            transformed_paths += coefficients[i] * (fOU_paths ** i)

        paths = transformed_paths

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
    
    def simulate_GaussTanhPolyRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        scale_ = self.params['scale_']
        sigma0 = self.params['sigma0']
        coefficients = self.params['coefficients']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.zeros_like(paths[:, t-1])
            # random.seed(t), change seed each t ? remove tanh? have user set scale? 
            normal_multiplier = np.random.normal(loc=0.0, scale=t*scale_, size=self.num_paths)
            for i in range(len(coefficients)):
                increment += coefficients[i] * ((fBM_paths[:, t] - fBM_paths[:, t-1]) ** i)
                paths[:, t] = paths[:, t-1] + normal_multiplier * np.tanh(increment / nu)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, num_steps * dt, num_steps)
            polynomial = np.zeros_like(t)
            for i, coef in enumerate(coefficients):
                polynomial += coef * (t ** i)
            plt.plot(t, polynomial, label='Polynomial Transformation')
            plt.title(f'Polynomial Transformation for {self.process_type}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_Jacobi(self):
        start_time = time.time()
        R_ = self.params['R_lower']
        R_plus = self.params['R_upper']
        R = self.params['R']
        k = self.params['k']
        rho0 = self.params['rho0']
        sigma = self.params['sigma']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = rho0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + k * (R - paths[:, t-1]) * dt + sigma * np.sqrt((paths[:, t-1] - R_) * (R_plus - paths[:, t-1])) * dW
            paths[:, t] = np.clip(paths[:, t], R_, R_plus)  # Reflecting boundaries
        
        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_WrightFisherDiffusion(self):
        start_time = time.time()
        kappa = self.params['kappa']
        sigma = self.params['sigma']
        rho_ = self.params['rho_'] #-1 <= rho_ <= 1
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = ((paths[:, t-1] + kappa * (rho_ - paths[:, t-1]) * dt + sigma * np.sqrt(
                np.maximum(0, (1 - (paths[:, t-1] * paths[:, t-1])))) * dW)/2 + 1/2)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_WrightFisherSC(self):
        start_time = time.time()
        kappa = self.params['kappa']
        sigma = self.params['sigma']
        rho_ = self.params['rho_'] #-1 <= rho_ <= 1
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (rho_ - paths[:, t-1]) * dt + sigma * np.sqrt(
                np.maximum(0, (1 - (paths[:, t-1] * paths[:, t-1])))) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col) 

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_PolynomialItoProcess(self):
        start_time = time.time()
        a = self.params['a']
        b = self.params['b']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + b(paths[:, t-1]) * dt + np.sqrt(np.maximum(a(paths[:, t-1]), 0)) * dW
            
        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col) 

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_WeibullDiffusion(self):
        start_time = time.time()
        alpha = self.params['alpha']
        lambda_ = self.params['lambda_']
        k = self.params['k']
        mu_W = lambda_ * gamma(1 + 1/k)
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_t = paths[:, t-1]

            b1 = 2 * alpha / ((k / lambda_) * (X_t / lambda_)**(k - 1) * np.exp(-(X_t / lambda_)**k))
            b2 = lambda_ * gamma(1 + 1/k, (X_t / lambda_)**k) - mu_W * np.exp(-(X_t / lambda_)**k)
            b = np.sqrt(np.maximum(b1 * b2, 0))

            a = -alpha * (X_t - mu_W)

            paths[:, t] = X_t + a * dt + b * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col) 

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_WeibullDiffusion2(self):
        start_time = time.time()
        alpha = self.params['alpha']
        lambda_ = self.params['lambda_']
        k = self.params['k']
        sigma_W = lambda_ * np.sqrt(np.maximum(gamma(1 + 2/k) - lambda_ * lambda_ * (gamma(1 + 1/k))**2, 0))
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_t = paths[:, t-1]

            b = np.sqrt(2 * alpha) * sigma_W

            # Prevent overflow in power operation
            safe_exp = np.maximum(X_t / lambda_, 1e-8)
            a = (alpha * sigma_W**2 * k / X_t) * (((k - 1) / k) - safe_exp**k)

            paths[:, t] = X_t + a * dt + b * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_ConicDiffusionMartingale(self):
        start_time = time.time()
        b = self.params['b']
        b_prime = self.params['b_prime']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            b_t = b(t*dt)
            b_prime_t = b_prime(t*dt)
            paths[:, t] = paths[:, t-1] + b_t * dt + b_prime_t * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col) 

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ConicUnifDiffusionMartingale(self):
        start_time = time.time()
        b = self.params['b']
        b_prime = self.params['b_prime']
        sigma = self.params['sigma']
        Z0 = self.params['Z0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = Z0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            b_t = b(t*dt)
            b_prime_t = b_prime(t*dt)
            paths[:, t] = paths[:, t-1] + b_t * dt + b_prime_t * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ConicHalfUnifDiffusionMartingale(self):
        start_time = time.time()
        b = self.params['b']
        b_prime = self.params['b_prime']
        sigma = self.params['sigma']
        Y0 = self.params['Y0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = Y0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            b_t = b(t*dt)
            b_prime_t = b_prime(t*dt)
            paths[:, t] = paths[:, t-1] + b_t * dt + b_prime_t * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_CircleBM(self):
        start_time = time.time()
        dt = self.dt
        num_steps = self.path_length
        num_paths = self.num_paths
        c_y = self.params['c_y']
        c_x = self.params['c_x']
        paths_X = np.zeros((num_paths, num_steps))
        paths_Y = np.zeros((num_paths, num_steps))
        
        for t in range(1, num_steps):
            dB = np.sqrt(dt) * np.random.randn(num_paths)
            B_t = np.cumsum(dB)  # B(t) is the cumulative sum of dB
            
            # Update Y(t) and X(t) using the provided SDEs
            paths_X[:, t] = paths_X[:, t-1] - c_x * np.sin(B_t) * dB - 0.5 * np.cos(B_t) * dt
            paths_Y[:, t] = paths_Y[:, t-1] + c_y * np.cos(B_t) * dB - 0.5 * np.sin(B_t) * dt

        if self.do_plot == True: 
            # Stack paths_Y and paths_X along the first axis
            stacked_paths = paths_Y, paths_X # np.stack((paths_Y, paths_X), axis=1)
            plot_complex_paths(stacked_paths, 'CircleBM Process', colmap_name = self.colmap_name, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths_Y, paths_X

    def simulate_SinFourierDecompBB(self):
        start_time = time.time()
        T = self.path_length
        num_terms = self.params['num_terms']
        dt = self.dt
        num_steps = self.path_length
        num_paths = self.num_paths
        paths = np.zeros((num_paths, num_steps))
        
        for path in range(num_paths):
            for t in range(1, num_steps):
                dW = np.sqrt(dt) * np.random.randn()
                for n in range(1, num_terms + 1):
                    coef = (np.sqrt(2) * np.random.normal(0, 1)) / (np.pi * n)
                    paths[path, t] += coef * np.sin(n * np.pi * t * dt / T) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_MixedFourierDecompBB(self):
        start_time = time.time()
        T = self.path_length
        num_terms = self.params['num_terms']
        dt = self.dt
        num_steps = self.path_length
        num_paths = self.num_paths
        paths = np.zeros((num_paths, num_steps))
        
        for path in range(num_paths):
            for t in range(1, num_steps):
                dW = np.sqrt(dt) * np.random.randn()
                for n in range(1, num_terms + 1):
                    coef_sin = np.random.normal(0, 1)
                    coef_cos = np.random.normal(0, 1)
                    paths[path, t] += coef_sin * np.sin(n * np.pi * t * dt / T) * dW
                    paths[path, t] += coef_cos * np.cos(n * np.pi * t * dt / T) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_fStochasticLogisticGrowth(self):
        start_time = time.time()
        r = self.params['r']
        K = self.params['K']
        sigma = self.params['sigma']
        H = self.params['H']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] + r * paths[:, t-1] * (1 - paths[:, t-1] / K) * dt + sigma * paths[:, t-1] * (fBM_paths[:, t] - fBM_paths[:, t-1])

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_fStochasticExponentialDecay(self):
        start_time = time.time()
        sigma = self.params['sigma']
        X0 = self.params['X0']
        H = self.params['H']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] * np.exp(-dt + sigma * (fBM_paths[:, t] - fBM_paths[:, t-1]))

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_tanh_fOU(self):
        start_time = time.time()
        X0 = self.params['X0']
        H = self.params['H']
        nu = self.params['nu']
        kappa = self.params['kappa']
        m = self.params['m']
        sigma = self.params['sigma']
        fOU_paths = self.simulate_fOU_helper()

        if self.do_plot == True: 
            plot_paths(np.tanh(fOU_paths), f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return np.tanh(fOU_paths)

    def simulate_ARIMA(self):
        start_time = time.time()
        p = self.params['p']
        d = self.params['d']
        q = self.params['q']
        ar_params = self.params['ar_params']
        ma_params = self.params['ma_params']
        num_steps = self.path_length
    
        # Generate white noise
        white_noise = np.random.normal(0, 1, (self.num_paths, num_steps))
    
        # Placeholder for the ARIMA process
        paths = np.zeros((self.num_paths, num_steps))
    
        for i in range(self.num_paths):
            arima_process = sm.tsa.ArmaProcess(ar_params, ma_params)
            paths[i] = arima_process.generate_sample(num_steps)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_SinRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        A = self.params['A']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.sin(A * (fBM_paths[:, t] - fBM_paths[:, t-1]))
            paths[:, t] = paths[:, t-1] + nu * increment

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_TanhRFSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        A = self.params['A']
        dt = self.dt
        num_steps = self.path_length
    
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            increment = np.tanh(A * (fBM_paths[:, t] - fBM_paths[:, t-1]))
            paths[:, t] = paths[:, t-1] + nu * increment

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_Chen(self):
        start_time = time.time()
        kappa = self.params['kappa']
        theta0 = self.params['theta0']
        sigma0 = self.params['sigma0']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        rt = np.zeros((self.num_paths, num_steps))
        theta_t = np.zeros((self.num_paths, num_steps))
        sigma_t = np.zeros((self.num_paths, num_steps))
        rt[:, 0] = r0
        theta_t[:, 0] = theta0
        sigma_t[:, 0] = sigma0
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW3 = np.sqrt(dt) * np.random.randn(self.num_paths)
            rt[:, t] = rt[:, t-1] + kappa * (theta_t[:, t-1] - rt[:, t-1]) * dt + np.sqrt(rt[:, t-1]) * sigma_t[:, t-1] * dW1
            theta_t[:, t] = theta_t[:, t-1] + self.params.get('nu', 0.1) * (self.params.get('zeta', 0.1) - theta_t[:, t-1]) * dt + self.params.get('alpha', 0.1) * np.sqrt(theta_t[:, t-1]) * dW2
            sigma_t[:, t] = sigma_t[:, t-1] + self.params.get('mu', 0.1) * (self.params.get('beta', 0.1) - sigma_t[:, t-1]) * dt + self.params.get('eta', 0.1) * np.sqrt(sigma_t[:, t-1]) * dW3

        if self.do_plot == True: 
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(theta_t, f'{self.process_type} theta_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(sigma_t, f'{self.process_type} sigma_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return rt, theta_t, sigma_t
    
    def simulate_LongstaffSchwartz(self):
        start_time = time.time()
        a_t = self.params['a_t']
        c_t = self.params['c_t']
        d_t = self.params['d_t']
        f_t = self.params['f_t']
        b = self.params['b']
        e = self.params['e']
        theta = self.params['theta']
        mew = self.params['mew']
        dt = self.dt
        num_steps = self.path_length
        Xt = np.zeros((self.num_paths, num_steps))
        Yt = np.zeros((self.num_paths, num_steps))
        rt = np.zeros((self.num_paths, num_steps))
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW3 = np.sqrt(dt) * np.random.randn(self.num_paths)
            Xt[:, t] = Xt[:, t-1] + (a_t(t*dt) - b * Xt[:, t-1]) * dt + np.sqrt(Xt[:, t-1]) * c_t(t*dt) * dW1
            Yt[:, t] = Yt[:, t-1] + (d_t(t*dt) - e * Yt[:, t-1]) * dt + np.sqrt(Yt[:, t-1]) * f_t(t*dt) * dW2
            rt[:, t] = mew * Xt[:, t] + theta * Yt[:, t] + self.params.get('sigma_t', 0.1) * np.sqrt(Yt[:, t]) * dW3

        if self.do_plot == True: 
            plot_paths(Xt, f'{self.process_type} X_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(Yt, f'{self.process_type} Y_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return Xt, Yt, rt
    
    def simulate_BDT(self):
        start_time = time.time()
        theta_t = self.params['theta_t']
        sigma_t = self.params['sigma_t']
        sigma_t_prime = self.params['sigma_t_prime']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        rt = np.zeros((self.num_paths, num_steps))
        rt[:, 0] = r0
        for t in range(1, num_steps):
            current_time = t * dt
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            rt_prev = np.maximum(rt[:, t-1], 1e-6)  # Ensure no zero or negative rates
            log_rt_prev = np.log(rt_prev)
            drift_term = theta_t(current_time) + (sigma_t_prime(current_time) * log_rt_prev / sigma_t(current_time))
            rt[:, t] = rt_prev * np.exp(drift_term * dt + sigma_t(current_time) * dW)

        if self.do_plot == True: 
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return rt
    
    def simulate_HoLee(self):
        start_time = time.time()
        theta_t = self.params['theta_t']
        sigma = self.params['sigma']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        rt = np.zeros((self.num_paths, num_steps))
        rt[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            rt[:, t] = rt[:, t-1] + theta_t(t*dt) * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return rt
    
    def simulate_CIRPlusPlus(self):
        start_time = time.time()
        a = self.params['a']
        b = self.params['b']
        sigma = self.params['sigma']
        phi_t = self.params['phi_t']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        xt = np.zeros((self.num_paths, num_steps))
        rt = np.zeros((self.num_paths, num_steps))
        xt[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            xt[:, t] = xt[:, t-1] + a * (b - xt[:, t-1]) * dt + sigma * np.sqrt(np.abs(xt[:, t-1])) * dW
            rt[:, t] = xt[:, t] + phi_t(t*dt)

        if self.do_plot == True: 
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return rt
    
    def simulate_KWF(self):
        start_time = time.time()
        theta_t = self.params['theta_t']
        sigma = self.params['sigma']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        rt = np.zeros((self.num_paths, num_steps))
        rt[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            rt[:, t] = rt[:, t-1] + theta_t(t*dt) * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(rt, f'{self.process_type} r_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return rt

    def simulate_SCP_tanh_Ito(self):
        start_time = time.time()
        # Implementation of SCP_tanh process simulation
        dt = self.dt
        steps = self.path_length
        X_paths = np.zeros((self.num_paths, steps))
        rho_paths = np.zeros((self.num_paths, steps))
        X0 = self.params['X0']
        rho0 = np.tanh(X0)
        a = self.params['a']
        b = self.params['b']
        X_paths[:, 0] = X0
        rho_paths[:, 0] = rho0
    
        for t in range(1, steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            a_t = a((t-1) * dt, X_paths[:, t-1])
            b_t = b((t-1) * dt, X_paths[:, t-1])
            X_paths[:, t] = X_paths[:, t-1] + a_t * dt + b_t * dW
            a_t_tilda = a(t * dt, np.arctan(rho_paths[:, t-1]))
            b_t_tilda = b(t * dt, np.arctan(rho_paths[:, t-1]))
            rho_paths[:, t] = rho_paths[:, t-1] + (1 - rho_paths[:, t-1]**2) * ((a_t_tilda - rho_paths[:, t-1] * b_t_tilda**2) * dt + b_t_tilda * dW)

        if self.do_plot == True: 
            plot_paths(X_paths, f'{self.process_type} X_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rho_paths, f'{self.process_type} rho_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return X_paths, rho_paths
     
    def simulate_SCP_arctan_Ito(self):
        start_time = time.time()
        # Implementation of SCP_arctan process simulation
        dt = self.dt
        steps = self.path_length
        X_paths = np.zeros((self.num_paths, steps))
        rho_paths = np.zeros((self.num_paths, steps))
        X0 = self.params['X0']
        rho0 = (2/np.pi) * np.arctan((np.pi/2) * X0)
        a = self.params['a']
        b = self.params['b']
        X_paths[:, 0] = X0
        rho_paths[:, 0] = rho0
    
        for t in range(1, steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_t = X_paths[:, t-1]
            a_t = a((t-1) * dt, X_t)
            b_t = b((t-1) * dt, X_t)
            X_paths[:, t] = X_t + a_t * dt + b_t * dW
    
            rho_t = rho_paths[:, t-1]
            a_t_tilda = a((t-1) * dt, np.arctan(rho_t))
            b_t_tilda = b((t-1) * dt, np.arctan(rho_t))
            rho_paths[:, t] = rho_t + ((a_t_tilda / (1 + np.tan(rho_t * np.pi / 2)**2)) - 
                                       (np.pi * b_t_tilda**2 * np.tan(rho_t * np.pi / 2) / 
                                       (2 * (1 + np.tan(rho_t * np.pi / 2)**2)**2))) * dt + \
                                      (b_t_tilda / (1 + np.tan(rho_t * np.pi / 2)**2)) * dW

        if self.do_plot == True: 
            plot_paths(X_paths, f'{self.process_type} X_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rho_paths, f'{self.process_type} rho_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return X_paths, rho_paths

    def simulate_SCP_mean_reverting(self):
        start_time = time.time()
        kappa = self.params['kappa']
        mu = self.params['mu']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        
        # Simulating the mean-reverting process
        X_paths = np.zeros((self.num_paths, num_steps))
        X_paths[:, 0] = X0
        rho_paths = np.zeros((self.num_paths, num_steps))
        rho_paths[:, 0] = np.tanh(X0)
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_paths[:, t] = X_paths[:, t-1] + (kappa * (mu - np.tanh(X_paths[:, t-1]))/(1-((np.tanh(X_paths[:, t-1])))*(np.tanh(X_paths[:, t-1])))) * dt + (sigma / np.sqrt(1 - np.tanh(X_paths[:, t-1])**2)) * dW
            # Calculating the correlation paths
            rho_paths[:, t] = rho_paths[:, t-1] + ((kappa*(mu-rho_paths[:, t-1]))-sigma*sigma*rho_paths[:, t-1]) * dt + sigma*np.sqrt(np.abs(1-(rho_paths[:, t-1])*(rho_paths[:, t-1])))*dW

        if self.do_plot == True: 
            plot_paths(X_paths, f'{self.process_type} X_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rho_paths, f'{self.process_type} rho_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return X_paths, rho_paths

    def simulate_SCP_modified_OU(self):
        start_time = time.time()
        kappa = self.params['kappa']
        mu = self.params['mu']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        
        # Simulating the modified OU process
        X_paths = np.zeros((self.num_paths, num_steps))
        X_paths[:, 0] = X0
        rho_paths = np.zeros((self.num_paths, num_steps))
        rho_paths[:, 0] = np.tanh(X0)
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_paths[:, t] = X_paths[:, t-1] + kappa * (mu - np.tanh(X_paths[:, t-1])) * dt + sigma * dW
            # Calculating the correlation paths
            rho_paths[:, t] = rho_paths[:, t-1] + (((kappa*(mu-rho_paths[:, t-1]))-sigma*sigma*rho_paths[:, t-1]) * dt + sigma*np.sqrt(np.abs(1-(rho_paths[:, t-1])*(rho_paths[:, t-1])))*dW)*(1-(rho_paths[:, t-1])*(rho_paths[:, t-1]))

        if self.do_plot == True: 
            plot_paths(X_paths, f'{self.process_type} X_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rho_paths, f'{self.process_type} rho_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return X_paths, rho_paths

    def simulate_SCQuanto(self):
        start_time = time.time()
        muS = self.params['muS']
        sigmaS = self.params['sigmaS']
        S0 = self.params['S0']
        muR = self.params['muR']
        sigmaR = self.params['sigmaR']
        R0 = self.params['R0']
        kappa = self.params['kappa']
        mu = self.params['mu']
        sigma = self.params['sigma']
        rho0 = self.params['rho0']
        dt = self.dt
        num_steps = self.path_length

        S_paths = np.zeros((self.num_paths, num_steps))
        R_paths = np.zeros((self.num_paths, num_steps))
        rho_paths = np.zeros((self.num_paths, num_steps))
        S_paths[:, 0] = S0
        R_paths[:, 0] = R0
        rho_paths[:, 0] = rho0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW_S = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW_R = rho_paths[:, t-1] * dW_S + np.sqrt(np.abs(1 - rho_paths[:, t-1]**2)) * np.sqrt(dt) * np.random.randn(self.num_paths)
            S_paths[:, t] = S_paths[:, t-1] * np.exp((muS - 0.5 * sigmaS**2) * dt + sigmaS * dW_S)
            R_paths[:, t] = R_paths[:, t-1] * np.exp((muR - 0.5 * sigmaR**2) * dt + sigmaR * dW_R)
            rho_paths[:, t] = rho_paths[:, t-1] + kappa * (mu - rho_paths[:, t-1]) * (1 - rho_paths[:, t-1]**2) * dt + sigma * (1 - rho_paths[:, t-1]**2) * dW

        if self.do_plot == True: 
            plot_paths(S_paths, f'{self.process_type} S_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(R_paths, f'{self.process_type} R_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(rho_paths, f'{self.process_type} rho_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return S_paths, R_paths, rho_paths
        
    def simulate_GFBM(self):
        start_time = time.time()
        H = self.params['H']
        sigma = self.params['sigma']
        theta = self.params['theta']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()

        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] + theta * (paths[:, t-1] - X0) * dt + fBM_paths[:, t] * sigma

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_StochasticLogisticGrowth(self):
        start_time = time.time()
        r = self.params['r']
        K = self.params['K']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + r * paths[:, t-1] * (1 - paths[:, t-1] / K) * dt + sigma * paths[:, t-1] * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_StochasticExponentialDecay(self):
        start_time = time.time()
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp(-dt + sigma * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_SinCosVectorNoiseIto(self):
        start_time = time.time()
        X0 = self.params['X0']
        sigma = self.params['sigma']
        A = self.params['A']
        B = self.params['B']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0

        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + A * np.sin(paths[:, t-1]) * dW1 + B * np.cos(paths[:, t-1]) * dW2

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ExponentialOU(self):
        start_time = time.time()
        theta = self.params['theta']
        mu = self.params['mu']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = np.exp(X0)

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp((theta * (mu - np.log(paths[:, t-1]))) * dt + sigma * dW)
            
        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_OU(self):
        start_time = time.time()
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma'] # 0 < sigma
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_DTDG(self):
        start_time = time.time()
        ar = self.params['ar']
        br = self.params['br']
        B = self.params['B']
        Sigma = self.params['Sigma'] # 0 < Sigma
        x0 = self.params['x0']
        gamma = self.params['gamma']
        lambd = self.params['lambda_']
        v0 = self.params['v0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, len(np.array([x0]))))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0, :] = x0
        vol_paths[:, 0] = v0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, len(x0))
            dW_vol = np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + gamma * (lambd - vol_paths[:, t-1]) * dt + np.sqrt(vol_paths[:, t-1]) * dW_vol
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            paths[:, t, :] = paths[:, t-1, :] + (ar + B @ paths[:, t-1, :].T).T * dt + (Sigma * np.sqrt(vol_paths[:, t]).reshape(-1, 1)) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} Vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths

    def simulate_GARCHJump(self):
        start_time = time.time()
        beta0 = self.params['beta0']
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        c = self.params['c']
        lambda_ = self.params['lambda_']
        muJ = self.params['muJ']
        sigmaJ = self.params['sigmaJ']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        h = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = self.params['S0']  # Initial asset price
        h[:, 0] = self.params['h0']  # Initial volatility
    
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            jumps = np.random.poisson(lambda_ * dt, size=self.num_paths)
            jump_sizes = np.random.normal(muJ, sigmaJ, size=(self.num_paths, jumps.max()))
            J = np.array([jump_sizes[i, :jumps[i]].sum() for i in range(self.num_paths)])
            var_J = np.var(J)
            h[:, t] = beta0 * dt + h[:, t-1] * (beta1 - 1) * dt + beta2 * h[:, t-1] * (
                (J - np.mean(J)) / np.sqrt(var_J) if var_J > 0 else 0)**2 * dt
            h[:, t] = np.maximum(h[:, t], 0)  # Ensure h is non-negative
    
            paths[:, t] = paths[:, t-1] * np.exp(-0.5 * h[:, t] * dt + np.sqrt(h[:, t]) * dW + J)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_VIX(self):
        start_time = time.time()
        H = self.params['H']
        nu = self.params['nu']
        rho = self.params['rho']
        xi0 = self.params['xi0']
        dt = self.dt
        num_steps = self.path_length

        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = xi0

        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dZ = rho * dW + np.sqrt(1 - rho**2) * np.random.randn(self.num_paths) * np.sqrt(dt)
            paths[:, t] = paths[:, t-1] + nu * dZ

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_CKLS(self):
        start_time = time.time()
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + (alpha + beta * paths[:, t-1]) * dt + sigma * (paths[:, t-1]**gamma) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths
        
    def simulate_fOU(self):
        start_time = time.time()
        H = self.params['H']
        sigma = self.params['sigma']
        kappa = self.params['kappa']
        m = self.params['m']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        fBM_paths = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] - kappa * (paths[:, t-1] - m) * dt + fBM_paths[:, t]

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_fSV(self):
        start_time = time.time()
        sigma_star = self.params['sigma_star']
        gamma_h = self.params['gamma_h']
        sigma_h_star = self.params['sigma_h_star']
        y0 = self.params['y0']
        h0 = self.params['h0']
        H = self.params['H']
        dt = self.dt
        num_steps = self.path_length
        y_paths = np.zeros((self.num_paths, num_steps))
        h_paths = np.zeros((self.num_paths, num_steps))
        y_paths[:, 0] = y0
        h_paths[:, 0] = h0
        Wt = np.random.randn(self.num_paths, num_steps) * np.sqrt(dt)
        BHt = self.simulate_fBM_helper()
    
        for t in range(1, num_steps):
            y_paths[:, t] = y_paths[:, t-1] + sigma_star * np.exp(h_paths[:, t-1] / 2) * Wt[:, t]
            h_paths[:, t] = h_paths[:, t-1] + gamma_h * dt + BHt[:, t]

        if self.do_plot == True: 
            plot_paths(y_paths, f'{self.process_type} y_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(h_paths, f'{self.process_type} h_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return y_paths, h_paths
    
    def simulate_RfSV(self):
        start_time = time.time()
        nu = self.params['nu']
        H = self.params['H']
        sigma = self.params['sigma']
        sigma0 = self.params['sigma0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = sigma0
        BHt = self.simulate_fBM_helper()
        plt.clf()
        plt.close('all')
        for t in range(1, num_steps):
            paths[:, t] = paths[:, t-1] + nu * (BHt[:, t] - BHt[:, t-1])

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_Bessel(self):
        start_time = time.time()
        n = self.params['n']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = int(self.path_length)
        
        # Initialize the paths
        paths = np.zeros((self.num_paths, num_steps + 1))
        paths[:, 0] = X0
        
        for t in range(1, num_steps + 1):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            X_prev = paths[:, t - 1]
            drift_term = ((n - 1) / 2) * (dt / np.maximum(X_prev, 1e-1))
            diffusion_term = dW
            paths[:, t] = X_prev + diffusion_term + drift_term

        if self.do_plot == True: 
            plot_bessel_paths(paths, 'Bessel Process', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_SquaredBessel(self):
        start_time = time.time()
        delta = self.params['delta']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + delta * dt + 2 * np.sqrt(np.maximum(paths[:, t-1], 0.0001)) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_GARCH(self):
        start_time = time.time()
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        lambda0 = self.params['lambda0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = lambda0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + sigma * paths[:, t-1] * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_HullWhite(self):
        start_time = time.time()
        theta = self.params['theta']
        alpha = self.params['alpha']
        sigma = self.params['sigma']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + (theta(t*dt) - alpha(t*dt) * paths[:, t-1]) * dt + sigma(t*dt) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_LotkaVolterra(self):
        start_time = time.time()
        b = self.params['b']
        a = self.params['a']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + paths[:, t-1] * (b - a * paths[:, t-1]) * dt + sigma * paths[:, t-1] * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_TwoFactorHullWhite(self):
        start_time = time.time()
        theta = self.params['theta']
        alpha = self.params['alpha']
        sigma1 = self.params['sigma1']
        sigma2 = self.params['sigma2']
        b = self.params['b']
        r0 = self.params['r0']
        u0 = self.params['u0']
        dt = self.dt
        num_steps = self.path_length
        r_paths = np.zeros((self.num_paths, num_steps))
        u_paths = np.zeros((self.num_paths, num_steps))
        r_paths[:, 0] = r0
        u_paths[:, 0] = u0
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = np.sqrt(dt) * np.random.randn(self.num_paths)
            u_paths[:, t] = u_paths[:, t-1] - b * u_paths[:, t-1] * dt + sigma2 * dW2
            r_paths[:, t] = r_paths[:, t-1] + (theta(t*dt) + u_paths[:, t] - alpha(t*dt) * r_paths[:, t-1]) * dt + sigma1(t*dt) * dW1

        if self.do_plot == True: 
            plot_paths(u_paths, f'{self.process_type} u Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(r_paths, f'{self.process_type} r Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return r_paths, u_paths

    def simulate_ExponentialVasicek(self):
        start_time = time.time()
        a = self.params['a']
        b = self.params['b']
        sigma = self.params['sigma']
        r0 = self.params['r0']
        rho = self.params['rho']
        lambda_ = self.params['lambda']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dN = np.random.poisson(lambda_ * dt, self.num_paths)
            paths[:, t] = paths[:, t-1] + a * (b - paths[:, t-1]) * dt + sigma * dW + rho * dN

        if self.do_plot == True: 
            plot_paths(np.exp(paths), f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return np.exp(paths)

    def simulate_GeneralBergomi(self):
        start_time = time.time()
        r = self.params['r']
        q = self.params['q']
        xi = self.params['xi']
        omega = self.params['omega']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = 1
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * (1 + (r - q) * dt + xi * np.sqrt(omega * t) * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_OneFactorBergomi(self):
        start_time = time.time()
        omega = self.params['omega']
        kappa = self.params['kappa']
        XT = self.params['XT']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp((omega - kappa * paths[:, t-1]) * dt + XT * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_RoughBergomi(self):
        start_time = time.time()
        eta = self.params['eta']
        nu = self.params['nu']
        rho = self.params['rho']
        xi0 = self.params['xi0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = xi0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + eta * (nu * paths[:, t-1]**2 - paths[:, t-1]**3) * dt + rho * np.sqrt(paths[:, t-1]) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_RoughVolatility(self):
        start_time = time.time()
        B = self.params['B']
        xi = self.params['xi']
        g = self.params['g']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = 1
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * (1 + B(t*dt) * dt + xi * np.sqrt(g(t*dt) * paths[:, t-1]) * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_BlackKarasinski(self):
        start_time = time.time()
        theta = self.params['theta']
        phi = self.params['phi']
        sigma = self.params['sigma']
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        r_paths = np.zeros((self.num_paths, num_steps))
        r_paths[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            r_paths[:, t] = r_paths[:, t-1] * np.exp((theta(t*dt) - phi(t*dt) * np.log(r_paths[:, t-1])) * dt + sigma(t*dt) * dW)

        if self.do_plot == True: 
            plot_paths(r_paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return r_paths

    def simulate_Bates(self):
        start_time = time.time()
        gamma = self.params['gamma']
        q = self.params['q']
        sigma = self.params['sigma']
        mu = self.params['mu']
        k = self.params['k']
        theta = self.params['theta']
        sigma_v = self.params['sigma_v']
        lambda_ = self.params['lambda']
        muJ = self.params['muJ']
        sigmaJ = self.params['sigmaJ']
        S0 = self.params['S0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        vol_paths[:, 0] = theta
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW_v = np.sqrt(dt) * np.random.randn(self.num_paths)
            jumps = np.random.poisson(lambda_ * dt, size=self.num_paths)
            jump_sizes = np.random.normal(muJ, sigmaJ, size=(self.num_paths, jumps.max()))
            J = np.array([jump_sizes[i, :jumps[i]].sum() for i in range(self.num_paths)])
            vol_paths[:, t] = vol_paths[:, t-1] + k * (theta - vol_paths[:, t-1]) * dt + sigma_v * np.sqrt(vol_paths[:, t-1]) * dW_v
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * vol_paths[:, t]) * dt + np.sqrt(vol_paths[:, t]) * dW + J)
        s_paths=paths

        if self.do_plot == True: 
            plot_paths(s_paths, f'{self.process_type} S_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} sigma_t Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return s_paths, vol_paths
        
    def simulate_BM(self):
        start_time = time.time()
        X0 = self.params['X0']
        mu = self.params['mu']
        sigma = self.params['sigma']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + sigma*dW + mu

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_3dOU(self):
        start_time = time.time()
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma'] # 0 < sigma
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, 3))
        paths[:, 0, :] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, 3)
            paths[:, t, :] = paths[:, t-1, :] + kappa * (theta - paths[:, t-1, :]) * dt + np.dot(dW, sigma.T)

        if self.do_plot == True: 
            plot_3d_paths(paths, '3d OU', colmap_name = self.colmap_name)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_CIR(self):
        start_time = time.time()
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma'] # 0 < sigma
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + sigma * np.sqrt(paths[:, t-1]) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_GBM(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma'] # 0 < sigma
        S0 = self.params['S0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_GBMSA(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma'] # 0 < sigma
        S0 = self.params['S0']
        kappa = self.params['kappa']
        theta = self.params['theta']
        rho = self.params['rho']
        v0 = self.params['v0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        vol_paths[:, 0] = v0
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + kappa * (theta - vol_paths[:, t-1]) * dt + sigma * np.sqrt(vol_paths[:, t-1]) * dW2
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)  # Ensure variance is non-negative
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * vol_paths[:, t]) * dt + np.sqrt(vol_paths[:, t]) * dW1)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths

    def simulate_VG(self):
        start_time = time.time()
        sigma = self.params['sigma'] # 0 < sigma
        theta = self.params['theta']
        nu = self.params['nu']
        S0 = self.params['S0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dG = np.random.gamma(dt / nu, nu, size=self.num_paths)
            dW = np.sqrt(dG) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp(theta * dG + sigma * dW)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_VGSA(self):
        start_time = time.time()
        sigma = self.params['sigma'] # 0 < sigma
        theta = self.params['theta']
        nu = self.params['nu']
        S0 = self.params['S0']
        kappa = self.params['kappa']
        eta = self.params['eta']
        lambd = self.params['lambda_']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        vol_paths[:, 0] = self.params['v0']
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + kappa * (eta - vol_paths[:, t-1]) * dt + lambd * np.sqrt(vol_paths[:, t-1]) * dW
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            dG = np.random.gamma(dt / nu, nu, size=self.num_paths)
            dW2 = np.sqrt(dG) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp(theta * dG + sigma * dW2)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths

    def simulate_Merton(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma'] # 0 < sigma
        S0 = self.params['S0']
        lambd = self.params['lambda_']
        muJ = self.params['muJ']
        sigmaJ = self.params['sigmaJ'] # 0 < sigmaJ
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            jumps = np.random.poisson(lambd * dt, size=self.num_paths)
            jump_sizes = np.random.normal(muJ, sigmaJ, size=(self.num_paths, jumps.max()))
            J = np.array([jump_sizes[i, :jumps[i]].sum() for i in range(self.num_paths)])
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + J)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ATSM(self):
        start_time = time.time()
        ar = self.params['ar']
        br = self.params['br']
        B = self.params['B']
        Sigma = self.params['Sigma'] # 0 < Sigma
        x0 = self.params['x0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, len(x0)))
        paths[:, 0, :] = x0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, len(x0))
            paths[:, t, :] = paths[:, t-1, :] + (ar + B @ paths[:, t-1, :].T).T * dt + Sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ATSM_SV(self):
        start_time = time.time()
        ar = self.params['ar']
        br = self.params['br']
        B = self.params['B']
        Sigma = self.params['Sigma'] # 0 < Sigma
        x0 = self.params['x0']
        kappa = self.params['kappa']
        theta = self.params['theta']
        lambd = self.params['lambda_']
        v0 = self.params['v0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, len(x0)))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0, :] = x0
        vol_paths[:, 0] = v0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, len(x0))
            dW_vol = np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + kappa * (theta - vol_paths[:, t-1]) * dt + lambd * np.sqrt(vol_paths[:, t-1]) * dW_vol
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            paths[:, t, :] = paths[:, t-1, :] + (ar + B @ paths[:, t-1, :].T).T * dt + (Sigma * np.sqrt(vol_paths[:, t]).reshape(-1, 1)) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths

    def simulate_CEV(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma'] # 0 < sigma
        S0 = self.params['S0']
        beta = self.params['beta']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + mu * paths[:, t-1] * dt + sigma * paths[:, t-1]**beta * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_BrownianBridge(self):
        start_time = time.time()
        X0 = self.params['X0']
        XT = self.params['XT']
        dt = self.dt
        num_steps = self.path_length
        T = num_steps * dt
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            t_dt = t * dt
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + ((XT - paths[:, t-1]) / (T - t_dt)) * dt + dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_BrownianMeander(self):
        start_time = time.time()
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = np.maximum(paths[:, t-1] + dW, 0)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_BrownianExcursion(self):
        start_time = time.time()
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        T = num_steps * dt
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            t_dt = t * dt
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = np.maximum(paths[:, t-1] + ((0 - paths[:, t-1]) / (T - t_dt)) * dt + dW, 0)

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_StickyBM(self):
        start_time = time.time()
        mu = self.params['mu']
        sigma = self.params['sigma']
        sticky_point = self.params.get('sticky_point', 0.0)
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + sigma * dW
            paths[:, t] += mu * (np.abs(paths[:, t-1] - sticky_point) < 1e-5) * dt

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_ReflectingBM(self):
        start_time = time.time()
        lower_b = self.params.get('lower_b', -np.inf)
        upper_b = self.params.get('upper_b', np.inf)
        X0 = self.params['X0']
        sigma = self.params['sigma']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + sigma * dW
            paths[:, t] = np.maximum(paths[:, t], lower_b)  # Reflect at lower boundary
            paths[:, t] = np.minimum(paths[:, t], upper_b)  # Reflect at upper boundary

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_CorrelatedBM(self):
        start_time = time.time()
        rho = self.params['rho'] # -1 < rho < 1
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, 2))
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t, 0] = paths[:, t-1, 0] + dW1
            paths[:, t, 1] = paths[:, t-1, 1] + dW2

        if self.do_plot == True: 
            plot_correlated_paths(paths, f'{self.process_type} Process', background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_dDimensionalBM(self):
        start_time = time.time()
        d = self.params['d']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, d))
        cov_matrix = np.eye(d)
        cholesky_decomp = np.linalg.cholesky(cov_matrix)
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, d)
            dW = np.dot(dW, cholesky_decomp)
            paths[:, t, :] = paths[:, t-1, :] + dW

        if self.do_plot == True: 
            plot_3d_paths(paths, '3D Brownian Motion', colmap_name = self.colmap_name)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_3dGBM(self):
        start_time = time.time()
        mu = self.params['mu'] # Drift vector
        sigma = self.params['sigma'] # Volatility matrix
        S0 = self.params['S0'] # Initial value vector
        dt = self.dt
        num_steps = self.path_length
        d = len(S0)
        
        paths = np.zeros((self.num_paths, num_steps, d))
        paths[:, 0, :] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, d)
            paths[:, t, :] = paths[:, t-1, :] * np.exp((mu - 0.5 * np.diag(sigma)**2) * dt + np.dot(dW, sigma.T))

        if self.do_plot == True: 
            plot_3d_paths(paths, '3d GBM', colmap_name = self.colmap_name)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_CorrelatedGBM(self):
        start_time = time.time()
        mu1 = self.params['mu1']
        sigma1 = self.params['sigma1'] # 0 < sigma1
        S01 = self.params['S01']
        mu2 = self.params['mu2']
        sigma2 = self.params['sigma2'] # 0 < sigma2
        S02 = self.params['S02']
        rho = self.params['rho'] # -1 < rho < 1
        dt = self.dt
        num_steps = self.path_length
        paths1 = np.zeros((self.num_paths, num_steps))
        paths2 = np.zeros((self.num_paths, num_steps))
        paths1[:, 0] = S01
        paths2[:, 0] = S02
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            paths1[:, t] = paths1[:, t-1] * np.exp((mu1 - 0.5 * sigma1**2) * dt + sigma1 * dW1)
            paths2[:, t] = paths2[:, t-1] * np.exp((mu2 - 0.5 * sigma2**2) * dt + sigma2 * dW2)

        if self.do_plot == True: 
            plot_correlated_paths(np.stack((paths1, paths2), axis=-1), f'{self.process_type} Process', background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return np.stack((paths1, paths2), axis=-1)

    def simulate_Vasicek(self):
        start_time = time.time()
        a = self.params['a']
        b = self.params['b']
        sigma = self.params['sigma'] # 0 < sigma
        r0 = self.params['r0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = r0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + a * (b - paths[:, t-1]) * dt + sigma * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_DysonBM(self):
        start_time = time.time()
        n = self.params['n']
        epsilon = self.params['epsilon'] 
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, n))
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, n)
            for i in range(n):
                paths[:, t, i] = paths[:, t-1, i] + dW[:, i]
                for j in range(n):
                    if i != j:
                        paths[:, t, i] += dt / (paths[:, t-1, i] - paths[:, t-1, j] + epsilon)
                        
        if self.do_plot == True: 
            num_paths, path_length, n = paths.shape
            eigenvalues = np.zeros((num_paths, path_length, n))
            for i in range(num_paths):
                for t in range(path_length):
                    matrix = np.outer(paths[i, t, :], paths[i, t, :])
                    eigvals = np.linalg.eigvalsh(matrix)
                    eigenvalues[i, t, :] = np.sort(eigvals)
            plot_paths(paths[:, :, 0], f'{self.process_type} Process', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)  # Plot first component of DysonBM paths
            plot_eigenvalues(eigenvalues[:, :, 0], f'{self.process_type} Process Eigenvalues')  # Plot first eigenvalue of DysonBM paths

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_fBM(self):
        start_time = time.time()
        H = self.params['H']  # 0 < H < 1
        sigma = self.params['sigma']  # 0 < sigma
        dt = self.dt
        num_steps = self.path_length
        T = num_steps * dt
    
        # Generate time vector
        t = np.linspace(dt, T, num_steps)  # Start from dt to avoid division by zero
    
        # Define the kernel function K_H(t, s)
        def K_H(t, s, H):
            if t == s:
                return 0
            elif t > s:
                factor1 = (t - s) ** (H - 0.5)
                factor2 = (t / s) ** (H - 0.5)
                factor3 = hyp2f1(H - 0.5, 0.5 - H, H + 0.5, 1 - (t / s))
                return factor1 * factor2 * factor3 / gamma(H + 0.5)
            else:
                return 0
    
        # Compute the increments of a standard Brownian motion
        dB = np.sqrt(dt) * np.random.randn(self.num_paths, num_steps)
    
        # Compute the paths for fBM
        paths = np.zeros((self.num_paths, num_steps))
        for i in range(self.num_paths):
            for j in range(1, num_steps):
                integral = 0
                for k in range(j):
                    integral += K_H(t[j], t[k], H) * dB[i, k]
                paths[i, j] = integral
    
        paths *= sigma

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths


    def simulate_fIM(self):
        start_time = time.time()
        H = self.params['H']  # 0 < H < 1
        sigma = self.params['sigma']  # 0 < sigma
        dt = self.dt
        num_steps = self.path_length

        # Initialize paths
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = 1e-4  # Small initial value to avoid issues with zero

        for i in range(self.num_paths):
            for t in range(1, num_steps):
                dW = np.sqrt(dt) * np.random.randn()  # Brownian increment
                increment = sigma * np.abs(paths[i, t-1])**(1 - 1/(2*H)) * dW
                paths[i, t] = paths[i, t-1] + increment

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths

    def simulate_SABR(self):
        start_time = time.time()
        alpha = self.params['alpha'] # 0 <= alpha 
        beta = self.params['beta']   # 0 <= beta <= 1
        rho = self.params['rho'] # -1 < rho < 1
        F0 = self.params['F0'] # 0 < F0
        sigma0 = self.params['sigma0'] # 0 < sigma0
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = F0
        vol_paths[:, 0] = sigma0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dZ = rho * dW + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + alpha * vol_paths[:, t-1] * dZ
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            paths[:, t] = paths[:, t-1] + vol_paths[:, t] * (paths[:, t-1]**beta) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths

    def simulate_ShiftedSABR(self):
        start_time = time.time()
        alpha = self.params['alpha'] # 0 <= alpha 
        beta = self.params['beta']  # 0 <= beta <= 1
        rho = self.params['rho'] # -1 < rho < 1
        F0 = self.params['F0']  # 0 < F0
        sigma0 = self.params['sigma0'] # 0 < sigma0
        s = self.params['s'] # F0 <= s
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        vol_paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = F0
        vol_paths[:, 0] = sigma0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            dZ = rho * dW + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            vol_paths[:, t] = vol_paths[:, t-1] + alpha * vol_paths[:, t-1] * dZ
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 0)
            paths[:, t] = paths[:, t-1] + vol_paths[:, t] * ((paths[:, t-1] + s)**beta) * dW

        if self.do_plot == True: 
            plot_paths(paths, f'{self.process_type} Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)
            plot_paths(vol_paths, f'{self.process_type} vol Paths', plot_variance=self.plot_variance, colmap_name=self.colmap_name, envelope_col = self.envelope_col, background_col=self.background_col)

        end_time = time.time()
        if self.time_exec == True: 
            print('Execution time: ', end_time - start_time)
            
        return paths, vol_paths
