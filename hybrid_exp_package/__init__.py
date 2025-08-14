"""
Hybrid Exponential Distributions Package
======================================

A comprehensive Python package for fitting hybrid and modified probability 
distributions involving exponential distributions.
"""

__version__ = "1.0.0"
__author__ = "Ayeni Taiwo Michael"
__license__ = "MIT"

from .core.base import BaseHybridDistribution
from .core.distributions import (
    ExponentialGammaDistribution,
    ExponentialExponentialDistribution,
    RayleighExponentialGammaDistribution,
    ExponentialGammaRayleighDistribution
)
from .core.factory import HybridDistributionFactory

__all__ = [
    'BaseHybridDistribution',
    'ExponentialGammaDistribution', 
    'ExponentialExponentialDistribution',
    'RayleighExponentialGammaDistribution',
    'ExponentialGammaRayleighDistribution',
    'HybridDistributionFactory'
] 
