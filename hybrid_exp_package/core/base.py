"""
Base class for all hybrid distributions
"""

import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt


class BaseHybridDistribution(ABC):
    """
    Abstract base class for all hybrid distributions.
    Provides common interface and functionality.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.fitted = False
        
    @abstractmethod
    def pdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Probability density function"""
        pass
    
    @abstractmethod
    def cdf(self, x: np.ndarray, **params) -> np.ndarray:
        """Cumulative distribution function"""
        pass
    
    @abstractmethod
    def _negative_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """Negative log-likelihood for MLE optimization"""
        pass
    
    @abstractmethod
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter estimates for optimization"""
        pass
    
    def survival(self, x: np.ndarray, **params) -> np.ndarray:
        """Survival function S(x) = 1 - F(x)"""
        return 1 - self.cdf(x, **params)
    
    def hazard(self, x: np.ndarray, **params) -> np.ndarray:
        """Hazard function h(x) = f(x) / S(x)"""
        survival_vals = self.survival(x, **params)
        survival_vals = np.where(survival_vals < 1e-15, 1e-15, survival_vals)
        return self.pdf(x, **params) / survival_vals
    
    def moment(self, r: int, **params) -> float:
        """Calculate r-th moment (numerical integration)"""
        from scipy.integrate import quad
        
        def integrand(x):
            return (x ** r) * self.pdf(x, **params)
        
        try:
            result, _ = quad(integrand, 0, np.inf, limit=100)
            return result
        except:
            result, _ = quad(integrand, 0, 50, limit=100)
            return result
    
    def fit(self, data: np.ndarray, method: str = 'mle', **kwargs) -> Dict[str, Any]:
        """Fit distribution parameters to data"""
        if method.lower() == 'mle':
            return self._fit_mle(data, **kwargs)
        else:
            raise ValueError(f"Method '{method}' not supported")
    
    def _fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Maximum Likelihood Estimation"""
        data = np.asarray(data)
        
        if np.any(data <= 0):
            raise ValueError("All data values must be positive")
        
        initial_params = self._get_initial_params(data)
        bounds = [(0.001, None)] * len(initial_params)
        
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            self.params = dict(zip(self._param_names, result.x))
            self.fitted = True
            
            log_likelihood = -result.fun
            n_params = len(initial_params)
            n_obs = len(data)
            
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n_obs) - 2 * log_likelihood
            
            return {
                'params': self.params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'convergence': result,
                'n_params': n_params,
                'n_obs': n_obs
            }
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")
    
    def goodness_of_fit(self, data: np.ndarray, test: str = 'ks') -> Dict[str, float]:
        """Perform goodness-of-fit tests"""
        if not self.fitted:
            raise ValueError("Distribution must be fitted first")
            
        data = np.asarray(data)
        
        if test.lower() == 'ks':
            sorted_data = np.sort(data)
            n = len(data)
            
            theoretical_cdf = self.cdf(sorted_data, **self.params)
            empirical_cdf = np.arange(1, n + 1) / n
            
            ks_statistic = np.max(np.abs(theoretical_cdf - empirical_cdf))
            
            alpha = 0.05
            critical_value = 1.36 / np.sqrt(n)
            
            p_value = 2 * np.exp(-2 * n * ks_statistic**2)
            p_value = min(p_value, 1.0)
            
            return {
                'test': 'Kolmogorov-Smirnov',
                'statistic': ks_statistic,
                'critical_value': critical_value,
                'p_value': p_value,
                'reject_null': ks_statistic > critical_value
            }
        else:
            raise ValueError(f"Test '{test}' not implemented")
    
    def plot(self, data: Optional[np.ndarray] = None, x_range: Optional[Tuple[float, float]] = None):
        """Plot PDF, CDF, and data histogram if provided"""
        if not self.fitted and data is None:
            raise ValueError("Either fit the distribution or provide data")
        
        if x_range is None and data is not None:
            x_range = (0, np.max(data) * 1.2)
        elif x_range is None:
            x_range = (0, 10)
        
        x = np.linspace(x_range[0], x_range[1], 1000)
        x = x[x > 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{self.name} Distribution', fontsize=16)
        
        # PDF plot
        if self.fitted:
            pdf_vals = self.pdf(x, **self.params)
            axes[0, 0].plot(x, pdf_vals, 'r-', lw=2, label='Fitted PDF')
        
        if data is not None:
            axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, 
                           color='skyblue', label='Data', edgecolor='black')
        
        axes[0, 0].set_title('Probability Density Function')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('f(x)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CDF plot
        if self.fitted:
            cdf_vals = self.cdf(x, **self.params)
            axes[0, 1].plot(x, cdf_vals, 'b-', lw=2, label='Fitted CDF')
        
        if data is not None:
            sorted_data = np.sort(data)
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[0, 1].plot(sorted_data, y_vals, 'ro', alpha=0.6, 
                           markersize=4, label='Empirical CDF')
        
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('F(x)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Survival function
        if self.fitted:
            survival_vals = self.survival(x, **self.params)
            axes[1, 0].plot(x, survival_vals, 'g-', lw=2, label='Survival')
            axes[1, 0].set_title('Survival Function')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('S(x)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Hazard function
        if self.fitted:
            hazard_vals = self.hazard(x, **self.params)
            hazard_vals = np.where(hazard_vals > 100, 100, hazard_vals)
            axes[1, 1].plot(x, hazard_vals, 'm-', lw=2, label='Hazard')
            axes[1, 1].set_title('Hazard Rate Function')
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('h(x)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()