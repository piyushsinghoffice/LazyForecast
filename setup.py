from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = "\n" + f.read()

VERSION = '0.0.1'

# Short description
DESCRIPTION = "LazyForecast is a Python library for performing univariate time series analysis using a lazy forecasting approach. This approach is designed to provide quick and simple forecasting models without requiring extensive configuration or parameter tuning."

# Required packages
INSTALL_PACKAGES = ["ipython", "matplotlib", "numpy", "pandas", "pmdarima", "scikit-learn", "tensorflow", "tqdm"]

TAGS = ["arima", "timeseries", "forecasting", "pyramid", "pmdarima", 'pyramid-arima', "scikit-learn", "statsmodels", "tensorflow", "tensor", "machine", "learning"]

setup(
    name="lazyforecast",
    version=VERSION,
    author="Piyush Singh",
    author_email="piyush.singh.office@gmail.com",
    url="https://github.com/piyushsinghoffice/LazyForecast.git",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires= INSTALL_PACKAGES,
    keywords= TAGS,
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.9",
)
