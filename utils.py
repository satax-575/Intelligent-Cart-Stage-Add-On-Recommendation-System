"""
Utility functions for the recommendation system
Includes logging, timing, caching abstractions, and helper functions
"""

import time
import logging
import hashlib
import json
import pickle
from functools import wraps
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


# Setup logging
def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(LOGS_DIR / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Timing decorator
def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {elapsed_ms:.2f}ms")
        return result
    return wrapper


# Cache abstraction (Redis-like interface)
class CacheManager:
    """
    Simple in-memory cache manager with TTL support
    In production, replace with Redis/Memcached
    """
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if key in self.expiry and datetime.now() > self.expiry[key]:
                # Expired
                del self.cache[key]
                del self.expiry[key]
                return None
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache with optional TTL"""
        self.cache[key] = value
        if ttl_seconds:
            self.expiry[key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.expiry.clear()
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instance
cache_manager = CacheManager()


def cached(ttl_seconds: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{cache_manager.generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl_seconds)
            return result
        return wrapper
    return decorator


# Data validation utilities
def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame"):
    """Validate that DataFrame has required columns"""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} missing required columns: {missing_cols}")


def validate_positive(value: float, name: str):
    """Validate that value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


# Feature engineering helpers
def cyclical_encoding(values: np.ndarray, max_value: int) -> tuple:
    """Encode cyclical features (hour, day) as sin/cos"""
    sin_encoded = np.sin(2 * np.pi * values / max_value)
    cos_encoded = np.cos(2 * np.pi * values / max_value)
    return sin_encoded, cos_encoded


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Safe division with fill value for division by zero"""
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def clip_outliers(series: pd.Series, lower_percentile: float = 1, upper_percentile: float = 99) -> pd.Series:
    """Clip outliers based on percentiles"""
    lower = series.quantile(lower_percentile / 100)
    upper = series.quantile(upper_percentile / 100)
    return series.clip(lower, upper)


# Model persistence
def save_model(model: Any, filepath: str):
    """Save model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Metric computation helpers
def compute_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute Precision@K
    
    Precision@K = (# relevant items in top-K) / K
    
    FIXED: Use descending sort to get HIGHEST scores
    """
    if len(y_true) < k:
        k = len(y_true)
    
    # Get indices of top-k HIGHEST scores (descending order)
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevant_in_top_k = np.sum(y_true[top_k_indices])
    return relevant_in_top_k / k


def compute_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute Recall@K
    
    Recall@K = (# relevant items in top-K) / (total # relevant items)
    
    FIXED: Use descending sort to get HIGHEST scores
    """
    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0
    
    if len(y_true) < k:
        k = len(y_true)
    
    # Get indices of top-k HIGHEST scores (descending order)
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevant_in_top_k = np.sum(y_true[top_k_indices])
    return relevant_in_top_k / total_relevant


def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain)
    
    NDCG@K = DCG@K / IDCG@K
    
    FIXED: 
    1. Use descending sort to get HIGHEST scores
    2. Return 0.0 when IDCG = 0 (no relevant items)
    3. Handle edge cases properly
    """
    if len(y_true) < k:
        k = len(y_true)
    
    # Get indices of top-k HIGHEST scores (descending order)
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        # Gain is 2^relevance - 1, discounted by log2(position + 1)
        dcg += (2 ** y_true[idx] - 1) / np.log2(i + 2)
    
    # IDCG: Ideal DCG (if items were perfectly ranked)
    ideal_indices = np.argsort(y_true)[::-1][:k]
    idcg = 0.0
    for i, idx in enumerate(ideal_indices):
        idcg += (2 ** y_true[idx] - 1) / np.log2(i + 2)
    
    # FIXED: Return 0.0 when no relevant items (IDCG = 0)
    if idcg == 0.0 or np.isclose(idcg, 0.0):
        return 0.0
    
    return dcg / idcg


# Business metrics
def compute_aov_lift(baseline_aov: float, treatment_aov: float) -> float:
    """Compute AOV lift percentage"""
    if baseline_aov == 0:
        return 0.0
    return (treatment_aov - baseline_aov) / baseline_aov


def compute_attach_rate(n_accepted: int, n_shown: int) -> float:
    """Compute attach rate (acceptance rate)"""
    if n_shown == 0:
        return 0.0
    return n_accepted / n_shown


def compute_c2o_ratio(n_orders: int, n_carts: int) -> float:
    """Compute cart-to-order ratio"""
    if n_carts == 0:
        return 0.0
    return n_orders / n_carts


# Data generation utilities (for testing/demo)
def generate_synthetic_interactions(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic interaction data for testing"""
    np.random.seed(seed)
    
    data = {
        'session_id': [f'sess_{i}' for i in range(n_samples)],
        'user_id': np.random.randint(1, 1000, n_samples),
        'item_id': np.random.randint(1, 500, n_samples),
        'restaurant_id': np.random.randint(1, 100, n_samples),
        'recommended_flag': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'clicked': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'added_to_cart': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'ordered': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    }
    
    return pd.DataFrame(data)


# Session management
class SessionContext:
    """Context manager for tracking session-level operations"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = None
        self.operations = []
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.time() - self.start_time) * 1000
        logger = logging.getLogger(__name__)
        logger.info(f"Session {self.session_id} completed in {elapsed:.2f}ms")
    
    def log_operation(self, operation: str, duration_ms: float):
        """Log individual operation within session"""
        self.operations.append({
            'operation': operation,
            'duration_ms': duration_ms
        })


# Batch processing utilities
def batch_process(items: List[Any], batch_size: int, process_fn: Callable) -> List[Any]:
    """Process items in batches"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    return results


# Statistical utilities
def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 1000, 
                                  confidence: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return lower, upper


# Initialize logger for this module
logger = setup_logger(__name__, "utils.log")
