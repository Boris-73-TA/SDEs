import numpy as np

class StochasticProcessSimulator:
    def __init__(self, process_type, num_paths=1000, path_length=100, dt=0.01, **kwargs):
        self.process_type = process_type
        self.num_paths = num_paths
        self.path_length = path_length
        self.dt = dt
        self.params = kwargs
        self.validate_params()
        
    def validate_params(self):
        # Add default parameters and validation for each process
        if self.process_type == 'OU':
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.0)
        elif self.process_type == 'CIR':
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('X0', 0.0)
        elif self.process_type == 'GBM':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
        elif self.process_type == 'GBMSA':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('rho', 0.0)
            self.params.setdefault('v0', 0.1)
        elif self.process_type == 'VG':
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('nu', 0.2)
            self.params.setdefault('S0', 1.0)
        elif self.process_type == 'VGSA':
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('nu', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('eta', 0.0)
            self.params.setdefault('lambda_', 0.2)
        elif self.process_type == 'Merton':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('lambda_', 0.1)
            self.params.setdefault('muJ', 0.0)
            self.params.setdefault('sigmaJ', 0.1)
        elif self.process_type == 'ATSM':
            self.params.setdefault('ar', 0.01)
            self.params.setdefault('br', np.array([0.02]))
            self.params.setdefault('B', np.array([[0.03]]))
            self.params.setdefault('Sigma', np.array([[0.1]]))
            self.params.setdefault('x0', np.array([0.0]))
        elif self.process_type == 'ATSM_SV':
            self.params.setdefault('ar', 0.01)
            self.params.setdefault('br', np.array([0.02]))
            self.params.setdefault('B', np.array([[0.03]]))
            self.params.setdefault('Sigma', np.array([[0.1]]))
            self.params.setdefault('x0', np.array([0.0]))
            self.params.setdefault('kappa', 1.0)
            self.params.setdefault('theta', 0.0)
            self.params.setdefault('lambda_', 0.2)
            self.params.setdefault('v0', 0.1)
        elif self.process_type == 'CEV':
            self.params.setdefault('mu', 0.1)
            self.params.setdefault('sigma', 0.2)
            self.params.setdefault('S0', 1.0)
            self.params.setdefault('beta', 0.5)
        elif self.process_type == 'BrownianBridge':
            self.params.setdefault('X0', 0.0)
            self.params.setdefault('XT', 0.0)
        elif self.process_type == 'BrownianMeander':
            self.params.setdefault('X0', 0.0)
        elif self.process_type == 'BrownianExcursion':
            self.params.setdefault('X0', 0.0)
        elif self.process_type == 'CorrelatedBM':
            self.params.setdefault('rho', 0.5)
        elif self.process_type == 'dDimensionalBM':
            self.params.setdefault('d', 2)
        elif self.process_type == 'CorrelatedGBM':
            self.params.setdefault('mu1', 0.1)
            self.params.setdefault('sigma1', 0.2)
            self.params.setdefault('S01', 1.0)
            self.params.setdefault('mu2', 0.1)
            self.params.setdefault('sigma2', 0.2)
            self.params.setdefault('S02', 1.0)
            self.params.setdefault('rho', 0.5)
        else:
            raise ValueError("Unsupported process type")
        
    def simulate(self):
        if self.process_type == 'OU':
            return self.simulate_OU()
        elif self.process_type == 'CIR':
            return self.simulate_CIR()
        elif self.process_type == 'GBM':
            return self.simulate_GBM()
        elif self.process_type == 'GBMSA':
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
        elif self.process_type == 'CorrelatedGBM':
            return self.simulate_CorrelatedGBM()

    def simulate_OU(self):
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + sigma * dW
        return paths

    def simulate_CIR(self):
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) * dt + sigma * np.sqrt(paths[:, t-1]) * dW
        return paths

    def simulate_GBM(self):
        mu = self.params['mu']
        sigma = self.params['sigma']
        S0 = self.params['S0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        return paths

    def simulate_GBMSA(self):
        mu = self.params['mu']
        sigma = self.params['sigma']
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
        return paths

    def simulate_VG(self):
        sigma = self.params['sigma']
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
        return paths

    def simulate_VGSA(self):
        sigma = self.params['sigma']
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
        return paths

    def simulate_Merton(self):
        mu = self.params['mu']
        sigma = self.params['sigma']
        S0 = self.params['S0']
        lambd = self.params['lambda_']
        muJ = self.params['muJ']
        sigmaJ = self.params['sigmaJ']
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
        return paths

    def simulate_ATSM(self):
        ar = self.params['ar']
        br = self.params['br']
        B = self.params['B']
        Sigma = self.params['Sigma']
        x0 = self.params['x0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, len(x0)))
        paths[:, 0, :] = x0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths, len(x0))
            paths[:, t, :] = paths[:, t-1, :] + (ar + B @ paths[:, t-1, :].T).T * dt + Sigma * dW
        return paths

    def simulate_ATSM_SV(self):
        ar = self.params['ar']
        br = self.params['br']
        B = self.params['B']
        Sigma = self.params['Sigma']
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
        return paths

    def simulate_CEV(self):
        mu = self.params['mu']
        sigma = self.params['sigma']
        S0 = self.params['S0']
        beta = self.params['beta']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = S0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = paths[:, t-1] + mu * paths[:, t-1] * dt + sigma * paths[:, t-1]**beta * dW
        return paths

    def simulate_BrownianBridge(self):
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
        return paths

    def simulate_BrownianMeander(self):
        X0 = self.params['X0']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps))
        paths[:, 0] = X0
        for t in range(1, num_steps):
            dW = np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t] = np.maximum(paths[:, t-1] + dW, 0)
        return paths

    def simulate_BrownianExcursion(self):
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
        return paths

    def simulate_CorrelatedBM(self):
        rho = self.params['rho']
        dt = self.dt
        num_steps = self.path_length
        paths = np.zeros((self.num_paths, num_steps, 2))
        for t in range(1, num_steps):
            dW1 = np.sqrt(dt) * np.random.randn(self.num_paths)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(self.num_paths)
            paths[:, t, 0] = paths[:, t-1, 0] + dW1
            paths[:, t, 1] = paths[:, t-1, 1] + dW2
        return paths

    def simulate_dDimensionalBM(self):
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
        return paths

    def simulate_CorrelatedGBM(self):
        mu1 = self.params['mu1']
        sigma1 = self.params['sigma1']
        S01 = self.params['S01']
        mu2 = self.params['mu2']
        sigma2 = self.params['sigma2']
        S02 = self.params['S02']
        rho = self.params['rho']
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
        return np.stack((paths1, paths2), axis=-1)
