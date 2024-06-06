# SDEs
- [Git Homepage](https://github.com/Boris-73-TA/SDEs/)
- [Pip Repository](https://pypi.org/project/StochasticProcessSimulator/)
  
- Simulating SDEs using Euler discretization.

<img width="538" alt="Screenshot 2024-05-15 at 08 49 20" src="https://github.com/Boris-73-TA/SDEs/assets/129144076/0916880c-03a7-4fca-9841-5ceec3836094">

- Euler scheme does not require derivatives, but Milstein and Runge-Kutta discretization schemes do...

- List of processes implemented:
  - 'BM', 'GBM', 'OU', 'ExponentialOU', '(Heston) GBMSA', 'srGBM', 'SBM', 'BrownianBridge', 'BrownianMeander',
     'BrownianExcursion', 'DysonBM', 'StickyBM', 'ReflectingBM', 'CorrelatedBM', 
     'CorrelatedGBM', 'CircleBM', 'dDimensionalBM', '3dGBM', '3dOU', 
     'StochasticLogisticGrowth', 'StochasticExponentialDecay', 'SinCosVectorNoiseIto',
     'PolynomialItoProcess', 'WrightFisherDiffusion', 'WeibullDiffusion', 'WeibullDiffusion2', 
     'ExpTC', 'MRSqrtDiff',

     'VG', 'VGSA', 'Merton', 'ATSM', 'ATSM_SV', 'CEV', 'CIR', 'Vasicek', 'ExponentialVasicek', 'SABR', 'ShiftedSABR',  
     'DTDG', 'CKLS', 'HullWhite', 'LotkaVolterra', 'TwoFactorHullWhite', 'BlackKarasinski', 'Chen', 
     'LongstaffSchwartz', 'BDT', 'HoLee', 'CIR++', 'CIR2++', 'KWF', 

     'Bates','GeneralBergomi', 'OneFactorBergomi', 'RoughBergomi', 'RoughVolatility',
     'RfSV', 'SinRFSV', 'TanhRFSV', 'ARIMA', 'GARCH', 'GARCHJump', 'VIX', 'GaussTanhPolyRFSV', 'LaplaceTanhPolyRFSV', 
      't_TanhPolyRFSV', 'CauchyTanhPolyRFSV', 'triangularTanhPolyRFSV', 'GumbelTanhPolyRFSV', 'LogisticTanhPolyRFSV',

     'SCP_mean_reverting', 'SCP_modified_OU', 'SCP_tanh_Ito', 'SCP_arctan_Ito', 'SCQuanto', 'WrightFisherSC', 'Jacobi',

     'fBM', 'GFBM', 'fOU', 'fIM', 'tanh_fOU', 'Poly_fOU', 'fStochasticLogisticGrowth', 
     'fStochasticExponentialDecay', 'fBM_WrightFisherDiffusion', 'fSV', 'Bessel', 'SquaredBessel', 
     'ConicDiffusionMartingale', 'ConicUnifDiffusionMartingale', 
     'ConicHalfUnifDiffusionMartingale', 'SinFourierDecompBB', 'MixedFourierDecompBB',
     'kCorrelatedGBMs', 'kCorrelatedBMs'
    
- StochasticProcessSimulator.py script can be used to obtain paths from the desired SDE/process.
- The main function of interest is StochasticProcessSimulator() which produces the paths and includes the options of plotting and recording execution time(s).
- READ_ME_SPS.txt describes all of the function arguments and their default settings.
- Simulating_SDEs_Euler_all_plots.ipynb shows an example of how this function can be used to produce plots of all the types of SDEs.
- Simulating_SDEs_Euler_simple_examples.ipynb (versions 1-4 of notebook) shows some simple examples of how this function can be used to produce stylized plots...
- NB: In StochasticProcessSimulator(), 'kCorrelatedGBMs' and 'kCorrelatedBMs' are implemented without the option of producing plots...

- 3dBMCube.ipynb and 3dBMSphere.ipynb are two examples demonstrating how StochasticProcessSimulator() can be integrated with more specific plotting requirements to produce plots of 3d BM paths constrained to a cube and sphere, respectively. 

## Installation

StochasticProcessSimulator is available on [pypi](https://pypi.python.org/pypi) and can be
installed as follows

```
pip install StochasticProcessSimulator
```

## Dependencies

StochasticProcessSimulator relies on

- ``numpy``  for random number generation
- ``scipy`` and ``statsmodels`` for support for a number of one-dimensional distributions.
- ``matplotlib`` for creating visualisations

## Quick-Start

StochasticProcessSimulator allows you to simulate and plot paths from different stochastic processes in simple way.

For instance, the following code

```python
from StochasticProcessSimulator import StochasticProcessSimulator as SPS
simulator = SPS(
            process_type='GBM', do_plot=True)
paths = simulator.simulate()

```

produces the following output:

<p align="center">
<img src="https://raw.githubusercontent.com/"   style="display:block;float:none;margin-left:auto;margin-right:auto;width:80%">
</p>


- Future work:
  - add handling of parameter inputs that break validity conditions... 
  - add Milstein and Runge-Kutta discretization schemes...
  - add more SDEs/processes... 
- This is a very rough implementation, and there are lots of improvements to be made. Any suggestions and improvements are welcome and appreciated. 
