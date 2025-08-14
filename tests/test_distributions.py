"""
Test cases for hybrid distributions
"""

import pytest
import numpy as np
from hybrid_exp_package import (
    ExponentialGammaDistribution,
    ExponentialExponentialDistribution,
    HybridDistributionFactory
)


class TestExponentialGammaDistribution:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dist = ExponentialGammaDistribution()
        np.random.seed(42)
        # Generate test data from gamma distribution (approximation)
        self.data = np.random.gamma(shape=2.0, scale=1.5, size=100)
    
    def test_pdf_positive(self):
        """Test that PDF values are positive"""
        x = np.array([0.1, 1.0, 2.0, 5.0])
        pdf_vals = self.dist.pdf(x, alpha=2.0, lambda_param=1.0)
        
        assert np.all(pdf_vals >= 0)
        assert np.all(np.isfinite(pdf_vals))
    
    def test_cdf_properties(self):
        """Test CDF properties"""
        x = np.array([0.1, 1.0, 2.0, 5.0])
        cdf_vals = self.dist.cdf(x, alpha=2.0, lambda_param=1.0)
        
        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf_vals) >= 0)
        # CDF values should be between 0 and 1
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1)
    
    def test_parameter_estimation(self):
        """Test parameter estimation"""
        fit_results = self.dist.fit(self.data)
        
        # Check that parameters are positive
        assert fit_results['params']['alpha'] > 0
        assert fit_results['params']['lambda_param'] > 0
        
        # Check that fit statistics are reasonable
        assert np.isfinite(fit_results['log_likelihood'])
        assert np.isfinite(fit_results['aic'])
        assert np.isfinite(fit_results['bic'])
    
    def test_moments(self):
        """Test moment calculations"""
        alpha, lambda_param = 2.0, 1.0
        
        # First moment should be positive
        moment1 = self.dist.moment(1, alpha=alpha, lambda_param=lambda_param)
        assert moment1 > 0
        
        # Second moment should be larger than first
        moment2 = self.dist.moment(2, alpha=alpha, lambda_param=lambda_param)
        assert moment2 > moment1
    
    def test_survival_and_hazard(self):
        """Test survival and hazard functions"""
        alpha, lambda_param = 2.0, 1.0
        x = np.array([0.1, 1.0, 2.0])
        
        survival_vals = self.dist.survival(x, alpha=alpha, lambda_param=lambda_param)
        hazard_vals = self.dist.hazard(x, alpha=alpha, lambda_param=lambda_param)
        
        # Survival should be decreasing
        assert np.all(np.diff(survival_vals) <= 0)
        # Hazard should be positive
        assert np.all(hazard_vals >= 0)


class TestExponentialExponentialDistribution:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dist = ExponentialExponentialDistribution()
        np.random.seed(42)
        # Generate test data from exponential distribution
        self.data = np.random.exponential(scale=2.0, size=100)
    
    def test_pdf_properties(self):
        """Test PDF properties"""
        x = np.array([0.1, 1.0, 2.0, 5.0])
        pdf_vals = self.dist.pdf(x, lambda_param=1.0)
        
        assert np.all(pdf_vals >= 0)
        assert np.all(np.isfinite(pdf_vals))
    
    def test_cdf_at_zero(self):
        """Test CDF at zero"""
        cdf_val = self.dist.cdf(np.array([0]), lambda_param=1.0)[0]
        assert cdf_val == 0
    
    def test_parameter_estimation(self):
        """Test parameter estimation"""
        fit_results = self.dist.fit(self.data)
        
        assert fit_results['params']['lambda_param'] > 0
        assert np.isfinite(fit_results['log_likelihood'])
        
    def test_analytical_moments(self):
        """Test analytical moment calculation"""
        lambda_param = 1.5
        
        # Test first few moments
        for r in range(1, 4):
            moment_val = self.dist.moment(r, lambda_param=lambda_param)
            assert moment_val > 0
            assert np.isfinite(moment_val)


class TestHybridDistributionFactory:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.factory = HybridDistributionFactory()
        np.random.seed(42)
        self.data = np.random.gamma(shape=2.0, scale=1.0, size=50)  # Smaller sample for speed
    
    def test_list_distributions(self):
        """Test listing available distributions"""
        dist_list = self.factory.list_distributions()
        
        assert isinstance(dist_list, list)
        assert len(dist_list) > 0
        assert 'exponential_gamma' in dist_list
        assert 'exponential_exponential' in dist_list
    
    def test_create_distribution(self):
        """Test creating distribution instances"""
        dist = self.factory.create_distribution('exponential_gamma')
        assert isinstance(dist, ExponentialGammaDistribution)
        
        with pytest.raises(ValueError):
            self.factory.create_distribution('nonexistent_distribution')
    
    def test_compare_distributions(self):
        """Test comparing distributions"""
        # Use a subset of distributions for faster testing
        distributions_to_test = ['exponential_gamma', 'exponential_exponential']
        
        results = self.factory.compare_distributions(
            self.data, 
            distributions=distributions_to_test
        )
        
        assert isinstance(results, dict)
        assert len(results) <= len(distributions_to_test)
        
        # Check that results are sorted by AIC
        aic_values = [r['aic'] for r in results.values()]
        assert aic_values == sorted(aic_values)
    
    def test_aic_weights(self):
        """Test AIC weight calculation"""
        distributions_to_test = ['exponential_gamma', 'exponential_exponential']
        results = self.factory.compare_distributions(
            self.data,
            distributions=distributions_to_test
        )
        
        weights = self.factory.calculate_aic_weights(results)
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        # All weights should be positive
        assert all(w > 0 for w in weights.values())


def test_package_imports():
    """Test that all main components can be imported"""
    from hybrid_exp_package import (
        BaseHybridDistribution,
        ExponentialGammaDistribution,
        ExponentialExponentialDistribution,
        HybridDistributionFactory
    )
    
    # Test that classes can be instantiated
    eg_dist = ExponentialGammaDistribution()
    ee_dist = ExponentialExponentialDistribution()
    factory = HybridDistributionFactory()
    
    assert eg_dist.name == "Exponential-Gamma"
    assert ee_dist.name == "Exponential-Exponential"


def test_data_validation():
    """Test that distributions handle invalid data properly"""
    dist = ExponentialGammaDistribution()
    
    # Test negative data
    with pytest.raises(ValueError):
        bad_data = np.array([-1, 0, 1, 2])
        dist.fit(bad_data)
    
    # Test empty data
    with pytest.raises(Exception):
        empty_data = np.array([])
        dist.fit(empty_data) 
