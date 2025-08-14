"""
Quick test script to verify package functionality
"""

import numpy as np
from hybrid_exp_package import HybridDistributionFactory

def test_basic_functionality():
    """Test that the package works with real data"""
    print("Testing Hybrid Exponential Distributions Package")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    data = np.random.gamma(shape=2.0, scale=1.5, size=500)
    
    print(f"Test data: {len(data)} observations")
    print(f"Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    # Test factory
    factory = HybridDistributionFactory()
    available_dists = factory.list_distributions()
    print(f"Available distributions: {available_dists}")
    
    # Test fitting a single distribution
    print("\nTesting single distribution fitting...")
    eg_dist = factory.create_distribution('exponential_gamma')
    fit_results = eg_dist.fit(data)
    
    print("✅ Single distribution fitting works!")
    print(f"Parameters: {fit_results['params']}")
    print(f"AIC: {fit_results['aic']:.2f}")
    
    # Test goodness of fit
    gof_results = eg_dist.goodness_of_fit(data)
    print(f"✅ Goodness-of-fit test works! p-value: {gof_results['p_value']:.4f}")
    
    print("\n🎉 Package is working correctly!")

if __name__ == "__main__":
    test_basic_functionality() 
