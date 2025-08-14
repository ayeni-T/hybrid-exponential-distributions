Hybrid Exponential Distributions Documentation
==============================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

A comprehensive Python package for fitting hybrid and modified probability distributions involving exponential distributions.

Overview
--------

The **Hybrid Exponential Distributions** package provides implementations of four novel probability distributions:

* **Exponential-Gamma Distribution (EGD)** - Base distribution using product methodology
* **Exponential-Exponential Distribution (EED)** - Uses ED-X methodology  
* **Rayleigh-Exponential-Gamma Distribution (REGD)** - T-X family with Rayleigh base
* **Exponential-Gamma-Rayleigh Distribution (EGRD)** - T-X family with EG base

Key Features
------------

✅ **Novel Distributions**: First Python implementation of these hybrid distributions

✅ **Robust Estimation**: Maximum likelihood estimation with numerical optimization

✅ **Model Comparison**: AIC/BIC-based selection with model averaging

✅ **Reliability Analysis**: Survival functions, hazard rates, MTTF calculations

✅ **Statistical Testing**: Kolmogorov-Smirnov goodness-of-fit tests

✅ **Rich Visualization**: Publication-ready plots and charts

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install hybrid-exponential-distributions

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from hybrid_exp_package import HybridDistributionFactory
   
   # Generate sample data
   data = np.random.gamma(shape=2.0, scale=1.5, size=1000)
   
   # Compare all distributions
   factory = HybridDistributionFactory()
   results = factory.compare_distributions(data)
   
   # Get best fitting distribution
   best_dist = list(results.values())[0]['distribution']
   
   # Plot results
   best_dist.plot(data)

Installation Guide
==================

Requirements
------------

* Python 3.8 or higher
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* Matplotlib >= 3.5.0
* Pandas >= 1.3.0

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install hybrid-exponential-distributions

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/ayeni-T/hybrid-exponential-distributions.git
   cd hybrid-exponential-distributions
   pip install -e .

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hybrid_exp_package
   print(hybrid_exp_package.__version__)
   
   # List available distributions
   from hybrid_exp_package import HybridDistributionFactory
   factory = HybridDistributionFactory()
   print(factory.list_distributions())

API Reference
=============

.. automodule:: hybrid_exp_package
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
------------

Base Distribution
~~~~~~~~~~~~~~~~~

.. autoclass:: hybrid_exp_package.BaseHybridDistribution
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Factory Class
~~~~~~~~~~~~~

.. autoclass:: hybrid_exp_package.HybridDistributionFactory
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Distribution Classes
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hybrid_exp_package.ExponentialGammaDistribution
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: hybrid_exp_package.ExponentialExponentialDistribution
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: hybrid_exp_package.RayleighExponentialGammaDistribution
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: hybrid_exp_package.ExponentialGammaRayleighDistribution
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Examples
========

This section provides comprehensive examples of using the package.

Basic Distribution Fitting
---------------------------

.. code-block:: python

   import numpy as np
   from hybrid_exp_package import ExponentialGammaDistribution
   
   # Generate sample data
   np.random.seed(42)
   data = np.random.gamma(shape=2.0, scale=1.5, size=1000)
   
   # Fit Exponential-Gamma distribution
   eg_dist = ExponentialGammaDistribution()
   fit_results = eg_dist.fit(data)
   
   print("Fitted parameters:", fit_results['params'])
   print("AIC:", fit_results['aic'])
   
   # Goodness-of-fit test
   gof_results = eg_dist.goodness_of_fit(data)
   print("p-value:", gof_results['p_value'])

Model Comparison
----------------

.. code-block:: python

   from hybrid_exp_package import HybridDistributionFactory
   
   # Compare all distributions
   factory = HybridDistributionFactory()
   results = factory.compare_distributions(data)
   
   # Display comparison
   for name, result in results.items():
       print(f"{name}: AIC = {result['aic']:.2f}")
   
   # Calculate model weights
   weights = factory.calculate_aic_weights(results)
   print("Model weights:", weights)

Reliability Analysis
--------------------

.. code-block:: python

   from hybrid_exp_package import ExponentialExponentialDistribution
   
   # Simulate failure times
   failure_times = np.random.exponential(scale=1000, size=200)
   
   # Fit distribution
   ee_dist = ExponentialExponentialDistribution()
   fit_results = ee_dist.fit(failure_times)
   
   # Calculate reliability metrics
   mission_times = [500, 1000, 2000, 5000]
   for t in mission_times:
       reliability = ee_dist.survival([t], **fit_results['params'])[0]
       print(f"Time {t}: Reliability = {reliability:.4f}")

Mathematical Theory
===================

This section covers the mathematical foundations of the hybrid distributions.

Exponential-Gamma Distribution
------------------------------

The Exponential-Gamma Distribution (EGD) combines exponential and gamma characteristics using the product methodology.

**Probability Density Function:**

.. math::

   f(x) = \frac{\lambda^{\alpha+1} x^{\alpha-1} e^{-\lambda x}}{\Gamma(\alpha)}

where :math:`x > 0`, :math:`\alpha > 0` (shape parameter), and :math:`\lambda > 0` (rate parameter).

**Cumulative Distribution Function:**

.. math::

   F(x) = \frac{\gamma(\alpha, \lambda x)}{\Gamma(\alpha)}

where :math:`\gamma(\alpha, z)` is the lower incomplete gamma function.

Exponential-Exponential Distribution
------------------------------------

The Exponential-Exponential Distribution (EED) uses the ED-X methodology.

**Probability Density Function:**

.. math::

   g(x) = \lambda^2 e^{-\lambda^2 x}

**Cumulative Distribution Function:**

.. math::

   F(x) = 1 - e^{-\lambda^2 x}

**Moments:**

The r-th moment is given by:

.. math::

   \mu_r' = \frac{\Gamma(r+1)}{\lambda^{2r}}

T-X Family Distributions
------------------------

The Rayleigh-Exponential-Gamma Distribution (REGD) and Exponential-Gamma-Rayleigh Distribution (EGRD) are based on the Transformer-X (T-X) methodology.

For a baseline distribution with PDF :math:`f(x)` and CDF :math:`F(x)`, the T-X family is defined by:

.. math::

   g(x) = \frac{f(x)}{F(x)} \left(-\log(1-F(x))\right)^{\alpha-1} \exp\left(-\lambda\left(-\log(1-F(x))\right)^{\beta}/2\right)

This methodology provides great flexibility in creating new distributions with desirable properties.

Contributing
============

We welcome contributions to this project!

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/ayeni-T/hybrid-exponential-distributions.git
      cd hybrid-exponential-distributions

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      # On Windows:
      venv\Scripts\activate

4. Install in development mode:

   .. code-block:: bash

      pip install -e .

Code Style
-----------

* Follow PEP 8
* Use type hints where appropriate
* Write comprehensive docstrings
* Add tests for new features

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest tests/ --cov=hybrid_exp_package

Submitting Changes
------------------

1. Create a feature branch
2. Make your changes and add tests
3. Ensure all tests pass
4. Submit a pull request with a clear description

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.

Citation
========

If you use this package in your research, please cite:

.. code-block:: bibtex

   @software{hybrid_exponential_distributions,
     title = {Hybrid Exponential Distributions: A Python Package},
     author = {Ayeni, Taiwo Michael},
     year = {2025},
     url = {https://github.com/ayeni-T/hybrid-exponential-distributions}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`