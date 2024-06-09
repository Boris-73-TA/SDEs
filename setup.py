from setuptools import setup, find_packages

setup(
    name='StochasticProcessSimulator',
    version='0.1.2', 
    author='Boris Ter-Avanesov',
    author_email='bt2522@columbia.edu',
    description='A package for simulating and plotting stochastic processes using discretization of SDEs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Boris-73-TA/SDEs',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'statsmodels',
        'matplotlib',
        'pandas',
    ],
)
