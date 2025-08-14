"""
Basic usage examples for the hybrid exponential distributions package
"""

import numpy as np
import matplotlib.pyplot as plt
from hybrid_exp_package import HybridDistributionFactory, ExponentialGammaDistribution


def basic_fitting_example():
    """Basic example of fitting a distribution to data"""
    print("=== Basic Distribution Fitting Example ===")
    
    # Generate sample data
    np.random.seed(42)
    true_data = np.random.gamma(shape=2.5, scale=1.2, size=1000)
    
    print(f"Generated {len(true_data)} data points")
    print(f"Sample statistics: mean={np.mean(true_data):.3f}, std={np.std(true_data):.3f}")
    
    # Create and fit a single distribution
    eg_dist = ExponentialGammaDistribution()
    
    print("\nFitting Exponential-Gamma Distribution...")
    fit_results = eg_dist.fit(true_data)
    
    print(f"Fitted parameters:")
    for param, value in fit_results['params'].items():
        print(f"  {param}: {value:.4f}")
    
    print(f"AIC: {fit_results['aic']:.3f}")
    print(f"BIC: {fit_results['bic']:.3f}")
    
    # Calculate theoretical moments
    alpha_est = fit_results['params']['alpha']
    lambda_est = fit_results['params']['lambda_param']
    
    print(f"\nMoments comparison:")
    print(f"  Sample mean: {np.mean(true_data):.4f}")
    try:
        theoretical_mean = eg_dist.moment(1, alpha=alpha_est, lambda_param=lambda_est)
        print(f"  Theoretical mean: {theoretical_mean:.4f}")
    except:
        print(f"  Theoretical mean: Could not calculate")
    
    # Goodness-of-fit test
    try:
        gof_results = eg_dist.goodness_of_fit(true_data)
        print(f"\nGoodness-of-fit test ({gof_results['test']}):")
        print(f"  Statistic: {gof_results['statistic']:.4f}")
        print(f"  P-value: {gof_results['p_value']:.4f}")
        print(f"  Reject null hypothesis: {gof_results['reject_null']}")
    except Exception as e:
        print(f"  Goodness-of-fit test failed: {e}")
    
    return true_data, eg_dist


def model_comparison_example():
    """Example of comparing multiple distributions"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Generate mixed data (more complex)
    np.random.seed(123)
    data1 = np.random.exponential(scale=1.0, size=300)
    data2 = np.random.gamma(shape=2, scale=1.5, size=700)
    mixed_data = np.concatenate([data1, data2])
    np.random.shuffle(mixed_data)
    
    print(f"Mixed dataset: {len(mixed_data)} observations")
    print(f"Data range: [{np.min(mixed_data):.3f}, {np.max(mixed_data):.3f}]")
    
    # Compare distributions (use subset for faster execution)
    factory = HybridDistributionFactory()
    distributions_to_test = ['exponential_gamma', 'exponential_exponential']
    print(f"\nComparing {len(distributions_to_test)} distributions...")
    
    try:
        results = factory.compare_distributions(mixed_data, distributions=distributions_to_test)
        
        print("\nComparison Results (sorted by AIC):")
        print("-" * 60)
        print(f"{'Distribution':<30} {'AIC':<10} {'BIC':<10}")
        print("-" * 60)
        
        for dist_name, result in results.items():
            print(f"{dist_name:<30} {result['aic']:<10.2f} {result['bic']:<10.2f}")
        
        # Get best distribution
        best_dist_name = list(results.keys())[0]
        print(f"\nBest fitting distribution: {best_dist_name}")
        
        # Calculate AIC weights
        weights = factory.calculate_aic_weights(results)
        print("\nAIC Weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
            
        return mixed_data, results
        
    except Exception as e:
        print(f"Model comparison failed: {e}")
        return mixed_data, {}


def reliability_analysis_example():
    """Example of reliability analysis"""
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Simulate component failure times
    np.random.seed(456)
    failure_times = np.random.exponential(scale=1000, size=200)  # hours
    
    # Fit distribution
    from hybrid_exp_package import ExponentialExponentialDistribution
    ee_dist = ExponentialExponentialDistribution()
    
    try:
        fit_results = ee_dist.fit(failure_times)
        
        lambda_est = fit_results['params']['lambda_param']
        print(f"Fitted parameter λ = {lambda_est:.6f}")
        
        # Calculate reliability at different times
        mission_times = [500, 1000, 2000, 5000]  # hours
        
        print("\nReliability Analysis:")
        print("-" * 50)
        print(f"{'Time (hrs)':<12} {'Reliability':<12} {'Hazard Rate':<12}")
        print("-" * 50)
        
        for t in mission_times:
            reliability = ee_dist.survival(np.array([t]), lambda_param=lambda_est)[0]
            hazard = ee_dist.hazard(np.array([t]), lambda_param=lambda_est)[0]
            
            print(f"{t:<12} {reliability:<12.4f} {hazard:<12.6f}")
        
        # Mean time to failure
        try:
            mttf = ee_dist.moment(1, lambda_param=lambda_est)
            print(f"\nMean Time to Failure: {mttf:.1f} hours")
        except:
            print(f"\nMean Time to Failure: Could not calculate")
        
        return failure_times, ee_dist
        
    except Exception as e:
        print(f"Reliability analysis failed: {e}")
        return failure_times, None


def demonstrate_all_distributions():
    """Demonstrate all four distributions"""
    print("\n" + "=" * 60)
    print("ALL DISTRIBUTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Test all distributions
    factory = HybridDistributionFactory()
    available_dists = factory.list_distributions()
    
    print(f"Available distributions: {available_dists}")
    
    # Generate test data
    np.random.seed(789)
    test_data = np.random.gamma(shape=2, scale=1, size=100)
    
    print(f"\nTesting each distribution with {len(test_data)} data points...")
    print("-" * 70)
    print(f"{'Distribution':<30} {'Status':<10} {'AIC':<10} {'Parameters'}")
    print("-" * 70)
    
    for dist_name in available_dists:
        try:
            dist = factory.create_distribution(dist_name)
            fit_results = dist.fit(test_data)
            
            aic = fit_results['aic']
            n_params = len(fit_results['params'])
            status = "✅ Success"
            
        except Exception as e:
            aic = "Failed"
            n_params = "N/A"
            status = "❌ Failed"
        
        print(f"{dist_name:<30} {status:<10} {str(aic):<10} {n_params}")


if __name__ == "__main__":
    print("Hybrid Exponential Distributions - Usage Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        print("Running examples...")
        
        data1, dist1 = basic_fitting_example()
        data2, results2 = model_comparison_example() 
        data3, dist3 = reliability_analysis_example()
        demonstrate_all_distributions()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("• The package successfully fits hybrid exponential distributions")
        print("• Model comparison helps select the best distribution")
        print("• Reliability analysis provides practical engineering insights")
        print("• All distributions are accessible through a unified interface")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc() 
