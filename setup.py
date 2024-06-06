from setuptools import setup, find_packages

setup(
    name='StochasticProcessSimulator',
    version='0.1.0',
    author='Boris Ter-Avanesov',
    author_email='bt2522@columbia.edu',
    description='A package for simulating and plotting stochastic processes using discretization of SDEs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Boris-73-TA/SDEs',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pandas>=1.5.2",
        "numpy>=1.23.5",
        "scipy>=1.9.3",
        "matplotlib>=3.6.2",
        "statsmodels>=0.13.5"
    ],
)
