"""
Configuration file for Zomato Add-On Recommendation System
Centralizes all hyperparameters, paths, and system settings
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
TRAIN_TEST_SPLIT_DATE = "2024-02-15"  # Temporal split
VALIDATION_DAYS = 7
TEST_DAYS = 7

# Candidate generation config
N_CANDIDATES_COOCCURRENCE = 30
N_CANDIDATES_POPULAR = 20
N_CANDIDATES_PERSONALIZED = 30
TOTAL_CANDIDATES_TARGET = 80
MIN_COOCCURRENCE_COUNT = 3
PMI_THRESHOLD = 0.1

# Ranking model config
RANKING_MODEL_TYPE = "lightgbm"  # Options: lightgbm, xgboost, wide_deep
LIGHTGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "num_threads": 4,
    "device": "gpu",  # Use GPU
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}
N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50

# Feature engineering config
USER_FEATURES = [
    "recency_days", "order_frequency_30d", "lifetime_orders",
    "avg_order_value", "veg_preference_score", "price_sensitivity_score",
    "historical_addon_attach_rate", "avg_cart_size", "beverage_affinity",
    "dessert_affinity", "starter_affinity"
]

RESTAURANT_FEATURES = [
    "price_range", "rating", "delivery_rating", "avg_prep_time",
    "order_volume_30d", "chain_flag"
]

CART_FEATURES = [
    "cart_total", "cart_item_count", "avg_item_price", "price_std",
    "veg_ratio", "has_beverage", "has_dessert", "has_starter",
    "dominant_category_starter", "dominant_category_main",
    "dominant_category_beverage", "dominant_category_dessert"
]

ITEM_FEATURES = [
    "price", "veg_flag", "margin_proxy", "popularity_score",
    "historical_attach_rate", "category_starter", "category_main",
    "category_beverage", "category_dessert", "category_other"
]

CONTEXT_FEATURES = [
    "hour_sin", "hour_cos", "is_weekend", "is_breakfast",
    "is_lunch", "is_dinner", "is_late_night"
]

INTERACTION_FEATURES = [
    "price_distance_from_cart_avg", "price_ratio_to_cart_avg",
    "complementarity_score", "category_missing_in_cart",
    "same_restaurant_flag", "user_item_affinity"
]

# Inference config
TOP_N_RECOMMENDATIONS = 10
INFERENCE_TIMEOUT_MS = 300
BATCH_SIZE_INFERENCE = 100
CACHE_TTL_SECONDS = 3600

# Cold start config
COLD_START_POPULAR_ITEMS = 20
COLD_START_SEGMENT_PRIOR = True
COLD_START_CUISINE_FALLBACK = True

# Diversity & reranking config
DIVERSITY_LAMBDA = 0.3  # Weight for diversity vs relevance
MAX_ITEMS_PER_CATEGORY = 4
MIN_CATEGORIES_IN_RECOMMENDATIONS = 2
MMR_LAMBDA = 0.7  # Relevance vs diversity tradeoff

# Evaluation config
EVALUATION_K_VALUES = [5, 8, 10]
SEGMENT_ANALYSIS = ["budget", "regular", "premium"]
TIME_SLOT_ANALYSIS = ["breakfast", "lunch", "dinner", "late_night"]

# Business metrics config
TARGET_AOV_LIFT = 0.15  # 15% lift target
TARGET_ATTACH_RATE = 0.25  # 25% attach rate target
TARGET_C2O_IMPROVEMENT = 0.05  # 5% improvement

# Negative sampling config
NEGATIVE_SAMPLING_RATIO = 4  # 4 negatives per positive
NEGATIVE_SAMPLING_STRATEGY = "mixed"  # Options: random, popular, hard

# Model retraining config
RETRAINING_FREQUENCY_DAYS = 7
INCREMENTAL_TRAINING = True
MIN_SAMPLES_FOR_RETRAINING = 10000

# Feature store config (abstraction for production)
FEATURE_REFRESH_STRATEGY = {
    "static_features": "daily",  # Restaurant, item features
    "user_features": "hourly",   # User history, preferences
    "session_features": "realtime"  # Cart state, context
}

# Logging config
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API config
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
API_TIMEOUT = 30

# Experiment tracking
ENABLE_AB_TESTING = True
EXPERIMENT_LOGGING_SAMPLE_RATE = 0.1  # Log 10% of requests

# Model versioning
MODEL_VERSION = "v1.0.0"
MODEL_REGISTRY_PATH = MODELS_DIR / "registry"
