"""
Factory class for creating and managing hybrid distributions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from .distributions import (
    ExponentialGammaDistribution,
    ExponentialExponentialDistribution,
    RayleighExponentialGammaDistribution,
    ExponentialGammaRayleighDistribution
)


class HybridDistributionFactory:
    """
    Factory class to create and manage different hybrid distributions
    """
    
    _distributions = {
        'exponential_gamma': ExponentialGammaDistribution,
        'exponential_exponential': ExponentialExponentialDistribution,
        'rayleigh_exponential_gamma': RayleighExponentialGammaDistribution,
        'exponential_gamma_rayleigh': ExponentialGammaRayleighDistribution
    }
    
    @classmethod
    def create_distribution(cls, dist_type: str):
        """Create a distribution instance"""
        if dist_type not in cls._distributions:
            available = list(cls._distributions.keys())
            raise ValueError(f"Unknown distribution type '{dist_type}'. "
                           f"Available: {available}")
        
        return cls._distributions[dist_type]()
    
    @classmethod
    def list_distributions(cls) -> List[str]:
        """List available distribution types"""
        return list(cls._distributions.keys())
    
    @classmethod
    def compare_distributions(cls, data: np.ndarray, 
                            distributions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple distributions on the same data using AIC/BIC
        
        Parameters:
        -----------
        data : array-like
            Sample data
        distributions : list, optional
            List of distribution types to compare (default: all)
            
        Returns:
        --------
        dict : Comparison results sorted by AIC
        """
        if distributions is None:
            distributions = cls.list_distributions()
        
        results = {}
        
        for dist_type in distributions:
            try:
                print(f"Fitting {dist_type}...")
                dist = cls.create_distribution(dist_type)
                fit_result = dist.fit(data)
                
                results[dist_type] = {
                    'params': fit_result['params'],
                    'log_likelihood': fit_result['log_likelihood'],
                    'aic': fit_result['aic'],
                    'bic': fit_result['bic'],
                    'n_params': fit_result['n_params'],
                    'distribution': dist
                }
            except Exception as e:
                print(f"Failed to fit {dist_type}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No distributions could be fitted successfully")
        
        # Sort by AIC (lower is better)
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['aic']))
        
        return sorted_results
    
    @classmethod
    def get_comparison_summary(cls, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary DataFrame from comparison results
        
        Parameters:
        -----------
        comparison_results : dict
            Results from compare_distributions
            
        Returns:
        --------
        pd.DataFrame : Summary table
        """
        data = []
        for dist_name, results in comparison_results.items():
            data.append({
                'Distribution': dist_name.replace('_', ' ').title(),
                'Log-Likelihood': results['log_likelihood'],
                'AIC': results['aic'],
                'BIC': results['bic'],
                'Parameters': len(results['params']),
                'Best_AIC': dist_name == list(comparison_results.keys())[0]
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('AIC').reset_index(drop=True)
    
    @classmethod
    def calculate_aic_weights(cls, comparison_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate AIC weights for model averaging
        
        Parameters:
        -----------
        comparison_results : dict
            Results from compare_distributions
            
        Returns:
        --------
        dict : AIC weights for each distribution
        """
        aic_values = [results['aic'] for results in comparison_results.values()]
        min_aic = min(aic_values)
        
        weights = {}
        for dist_name, results in comparison_results.items():
            delta_aic = results['aic'] - min_aic
            weights[dist_name] = np.exp(-0.5 * delta_aic)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for dist_name in weights:
            weights[dist_name] /= total_weight
        
        return weights
    
    @classmethod
    def fit_best_distribution(cls, data: np.ndarray, 
                            distributions: Optional[List[str]] = None) -> tuple:
        """
        Fit all distributions and return the best one
        
        Parameters:
        -----------
        data : array-like
            Sample data
        distributions : list, optional
            List of distribution types to compare
            
        Returns:
        --------
        tuple : (best_distribution_instance, fit_results, comparison_results)
        """
        comparison_results = cls.compare_distributions(data, distributions)
        
        # Get best distribution
        best_dist_name = list(comparison_results.keys())[0]
        best_dist = comparison_results[best_dist_name]['distribution']
        best_fit_results = comparison_results[best_dist_name]
        
        return best_dist, best_fit_results, comparison_results 
