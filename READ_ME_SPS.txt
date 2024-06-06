StochasticProcessSimulator requires the following imports: 
- import numpy as np
- from scipy.special import gamma, hyp2f1
- import statsmodels.api as sm
- import matplotlib.pyplot as plt
- import time
- import pandas as pd
- from mpl_toolkits.mplot3d import Axes3D

The function StochasticProcessSimulator() has the following arguments: 
- process_type = [str] without a default, 
- do_plot = [bool] with False as default, 
- plot_variance = [bool] with True as default, 
- colmap_name = [str] with 'cool' as default, 
- time_exec = [bool] with True as default, 
- envelope_col = [str] with 'yellow' as default, 
- background_col = [str] with 'white' as default,
- num_paths = [int] with 50 as default, 
- path_length = [int] with 252 as default (a year, so 252 steps per path),
- dt = [float] with 1/252 as default (a day, so 1/252 step_length since want paths of length 1 year)

For colmap_name try one of the following options: 
- 'parula', 'turbo', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'sky', 'abyss', 'jet', 'lines', 'colorcube', 'prism', 'flag' 

For envelope_col and background_col try the following options: 
- "black", "dimgray", "dimgray", "gray", "grey", "darkgray", "darkgrey", "silver", "lightgray", "lightgrey", 
    "gainsboro", "whitesmoke", "white", "snow", "rosybrown", "lightcoral", "indianred", "brown", "firebrick", "maroon", 
    "darkred", "red", "mistyrose", "salmon", "tomato", "darksalmon", "coral", "orangered", "lightsalmon", "sienna", 
    "seashell", "chocolate", "saddlebrown", "sandybrown", "peachpuff", "peru", "linen", "bisque", "darkorange", 
    "burlywood", "antiquewhite", "tan", "navajowhite", "blanchedalmond", "papayawhip", "moccasin", "orange", "wheat", 
    "oldlace", "floralwhite", "darkgoldenrod", "goldenrod", "cornsilk", "gold", "lemonchiffon", "khaki", "palegoldenrod", 
    "darkkhaki", "ivory", "beige", "lightyellow", "lightgoldenrodyellow", "olive", "yellow", "olivedrab", "yellowgreen", 
    "darkolivegreen", "greenyellow", "chartreuse", "lawngreen", "lightgreen", "forestgreen", "limegreen", "darkgreen", 
    "green", "lime", "seagreen", "mediumseagreen", "springgreen", "mediumspringgreen", "mediumaquamarine", "aquamarine", 
    "turquoise", "lightseagreen", "mediumturquoise", "azure", "lightcyan", "paleturquoise", "darkslategray", 
    "darkslategrey", "teal", "darkcyan", "aqua", "cyan", "darkturquoise", "cadetblue", "powderblue", "lightblue", 
    "deepskyblue", "skyblue", "lightskyblue", "steelblue", "aliceblue", "dodgerblue", "lightsteelblue", "cornflowerblue", 
    "royalblue", "ghostwhite", "lavender", "midnightblue", "navy", "darkblue", "mediumblue", "blue", "slateblue", 
    "darkslateblue", "mediumslateblue", "mediumpurple", "rebeccapurple", "blueviolet", "indigo", "darkorchid", 
    "darkviolet", "mediumorchid", "thistle", "plum", "violet", "purple", "darkmagenta", "fuchsia", "magenta", "orchid", 
    "mediumvioletred", "deeppink", "hotpink", "palevioletred", "lavenderblush", "crimson", "pink", "lightpink"

List of parameters of the models and their default values: 
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
