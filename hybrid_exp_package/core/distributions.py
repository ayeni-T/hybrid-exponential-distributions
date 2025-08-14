"""
Individual hybrid distribution implementations
Based on the four research papers
"""

import numpy as np
from scipy.special import gamma, gammainc
from .base import BaseHybridDistribution


class ExponentialGammaDistribution(BaseHybridDistribution):
    """
    Exponential-Gamma Distribution (EGD)
    From Paper 4: Base distribution using product method
    PDF: f(x) = (λ^(α+1) * x^(α-1) * e^(-λx)) / Γ(α)
    """
    
    def __init__(self):
        super().__init__("Exponential-Gamma")
        self._param_names = ['alpha', 'lambda_param']
    
    def pdf(self, x: np.ndarray, alpha: float, lambda_param: float) -> np.ndarray:
        """PDF of Exponential-Gamma distribution"""
        x = np.asarray(x)
        x = np.where(x <= 0, np.finfo(float).eps, x)  # Handle x <= 0
        
        return (lambda_param**(alpha + 1) * x**(alpha - 1) * 
                np.exp(-lambda_param * x)) / gamma(alpha)
    
    def cdf(self, x: np.ndarray, alpha: float, lambda_param: float) -> np.ndarray:
        """CDF using incomplete gamma function"""
        x = np.asarray(x)
        x = np.where(x <= 0, 0, x)
        
        return gammainc(alpha, lambda_param * x)
    
    def moment(self, r: int, alpha: float, lambda_param: float) -> float:
        """Analytical r-th moment"""
        try:
            return (lambda_param**alpha * gamma(alpha + r)) / \
                   (gamma(alpha) * (2*lambda_param)**(alpha + r))
        except:
            # Fallback to numerical integration
            return super().moment(r, alpha=alpha, lambda_param=lambda_param)
    
    def _negative_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        alpha, lambda_param = params
        if alpha <= 0 or lambda_param <= 0:
            return np.inf
        
        try:
            pdf_vals = self.pdf(data, alpha, lambda_param)
            if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
                return np.inf
            return -np.sum(np.log(pdf_vals))
        except:
            return np.inf
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Method of moments initial estimates"""
        mean_data = np.mean(data)
        var_data = np.var(data)
        
        # Prevent division by zero
        if var_data < 1e-10:
            var_data = 1e-10
        
        # Method of moments estimators
        lambda_init = mean_data / var_data
        alpha_init = mean_data * lambda_init
        
        return np.array([max(alpha_init, 0.1), max(lambda_init, 0.1)])


class ExponentialExponentialDistribution(BaseHybridDistribution):
    """
    Exponential-Exponential Distribution (EED)
    From Paper 1: Uses ED-X methodology
    PDF: g(x) = λ²e^(-λ²x)
    """
    
    def __init__(self):
        super().__init__("Exponential-Exponential")
        self._param_names = ['lambda_param']
    
    def pdf(self, x: np.ndarray, lambda_param: float) -> np.ndarray:
        """PDF: λ²e^(-λ²x)"""
        x = np.asarray(x)
        x = np.where(x <= 0, 0, x)
        
        return lambda_param**2 * np.exp(-lambda_param**2 * x)
    
    def cdf(self, x: np.ndarray, lambda_param: float) -> np.ndarray:
        """CDF: 1 - e^(-λ²x)"""
        x = np.asarray(x)
        x = np.where(x <= 0, 0, x)
        
        return 1 - np.exp(-lambda_param**2 * x)
    
    def moment(self, r: int, lambda_param: float) -> float:
        """Analytical r-th moment: Γ(r+1) / λ^(2r)"""
        return gamma(r + 1) / (lambda_param**(2*r))
    
    def _negative_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        lambda_param = params[0]
        if lambda_param <= 0:
            return np.inf
        
        try:
            # Log-likelihood: n*log(λ²) - λ²*Σx_i
            n = len(data)
            return -(n * np.log(lambda_param**2) - lambda_param**2 * np.sum(data))
        except:
            return np.inf
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """MLE estimate: λ = sqrt(n / (2*Σx_i))"""
        sum_data = np.sum(data)
        if sum_data <= 0:
            sum_data = len(data)  # Fallback
        
        lambda_init = np.sqrt(len(data) / (2 * sum_data))
        return np.array([max(lambda_init, 0.1)])


class RayleighExponentialGammaDistribution(BaseHybridDistribution):
    """
    Rayleigh-Exponential-Gamma Distribution (REGD)
    From Paper 2: T-X family with Rayleigh base distribution
    """
    
    def __init__(self):
        super().__init__("Rayleigh-Exponential-Gamma")
        self._param_names = ['theta', 'alpha', 'sigma', 'lambda_param']
    
    def pdf(self, x: np.ndarray, theta: float, alpha: float, 
            sigma: float, lambda_param: float) -> np.ndarray:
        """Simplified PDF approximation for computational stability"""
        x = np.asarray(x)
        x = np.where(x <= 0, np.finfo(float).eps, x)
        
        # Simplified form based on T-X methodology
        term1 = (lambda_param**(alpha+1) * x**(alpha-1) * 
                (2**alpha)) / gamma(alpha)
        term2 = np.exp(-(4*lambda_param*sigma**2 - theta**2) / (2*sigma**2))
        
        # Add small constant to avoid numerical issues
        return term1 * term2 + 1e-10
    
    def cdf(self, x: np.ndarray, theta: float, alpha: float, 
            sigma: float, lambda_param: float) -> np.ndarray:
        """CDF using numerical integration"""
        x = np.asarray(x)
        if np.isscalar(x):
            x = np.array([x])
        
        cdf_vals = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= 0:
                cdf_vals[i] = 0
            else:
                # Numerical integration approximation
                from scipy.integrate import quad
                try:
                    integrand = lambda t: self.pdf(np.array([t]), theta, alpha, 
                                                 sigma, lambda_param)[0]
                    result, _ = quad(integrand, 0, xi, limit=50)
                    cdf_vals[i] = min(result, 1.0)
                except:
                    # Fallback approximation
                    cdf_vals[i] = 1 - np.exp(-xi / (theta + 1))
        
        return cdf_vals
    
    def _negative_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        theta, alpha, sigma, lambda_param = params
        if any(p <= 0 for p in params):
            return np.inf
        
        try:
            pdf_vals = self.pdf(data, theta, alpha, sigma, lambda_param)
            if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
                return np.inf
            return -np.sum(np.log(pdf_vals))
        except:
            return np.inf
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Initial parameter estimates"""
        mean_data = np.mean(data)
        std_data = np.std(data)
        
        return np.array([1.0, 2.0, max(std_data, 0.1), 1.0/max(mean_data, 0.1)])


class ExponentialGammaRayleighDistribution(BaseHybridDistribution):
    """
    Exponential-Gamma-Rayleigh Distribution (EGRD)
    From Paper 3: T-X family with Exponential-Gamma base
    """
    
    def __init__(self):
        super().__init__("Exponential-Gamma-Rayleigh")
        self._param_names = ['alpha', 'theta', 'lambda_param']
    
    def pdf(self, x: np.ndarray, alpha: float, theta: float, 
            lambda_param: float) -> np.ndarray:
        """PDF based on T-X methodology"""
        x = np.asarray(x)
        x = np.where(x <= 0, np.finfo(float).eps, x)
        
        return (theta**alpha * 2 * lambda_param * x**(2*alpha - 1)) / \
               gamma(alpha) * np.exp(-2 * theta * x)
    
    def cdf(self, x: np.ndarray, alpha: float, theta: float, 
            lambda_param: float) -> np.ndarray:
        """CDF using gamma function approximation"""
        x = np.asarray(x)
        x = np.where(x <= 0, 0, x)
        
        # Approximation using incomplete gamma function
        return gammainc(alpha, 2 * theta * x)
    
    def moment(self, r: int, alpha: float, theta: float, 
               lambda_param: float) -> float:
        """r-th moment approximation"""
        return (2 * theta**alpha * gamma(2*alpha + r)) / \
               ((2*theta)**(2*alpha + r) * gamma(alpha))
    
    def _negative_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        alpha, theta, lambda_param = params
        if alpha <= 0 or theta <= 0 or lambda_param <= 0:
            return np.inf
        
        try:
            pdf_vals = self.pdf(data, alpha, theta, lambda_param)
            if np.any(pdf_vals <= 0) or np.any(~np.isfinite(pdf_vals)):
                return np.inf
            return -np.sum(np.log(pdf_vals))
        except:
            return np.inf
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Initial parameter estimates"""
        mean_data = np.mean(data)
        return np.array([2.0, 1.0/max(mean_data, 0.1), 1.0]) 
