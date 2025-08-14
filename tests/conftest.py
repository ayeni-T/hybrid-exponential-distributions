"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)


@pytest.fixture
def small_sample_data():
    """Small sample dataset for quick tests"""
    np.random.seed(42)
    return np.random.gamma(shape=2.0, scale=1.0, size=50)


@pytest.fixture  
def large_sample_data():
    """Larger sample dataset for thorough tests"""
    np.random.seed(42)
    return np.random.gamma(shape=2.0, scale=1.0, size=500)


@pytest.fixture
def exponential_data():
    """Sample data from exponential distribution"""
    np.random.seed(42)
    return np.random.exponential(scale=1.5, size=100) 
