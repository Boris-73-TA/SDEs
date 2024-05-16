# SDEs
- Simulating SDEs using Euler's discretization.

<img width="538" alt="Screenshot 2024-05-15 at 08 49 20" src="https://github.com/Boris-73-TA/SDEs/assets/129144076/0916880c-03a7-4fca-9841-5ceec3836094">

- List of processes implemented:
  - 'OU', 'CIR', 'GBM', 'GBMSA', 'VG', 'VGSA', 'Merton',
    'ATSM', 'ATSM_SV', 'CEV', 'BrownianBridge', 'BrownianMeander',
    'BrownianExcursion', 'CorrelatedBM', 'CorrelatedGBM', 'dDimensionalBM'
- StochasticProcessSimulator.py script can be used to obtain paths from desired SDE.
- Simulating_SDEs_Euler.ipynb shows an example of how this can be used to produce plots of all the types of SDEs. 
- Future work:
  - add Milstein and Runge-Kutta discretization schemes
  - add more SDEs such as Vasicek, Hull-White, Chan–Karolyi Longstaff–Sanders, and Ito diffusions
- This is a very rough implementation, and there are lots of improvements to be made. Any suggestions are welcome. 
