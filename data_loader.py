"""
Data loading and preprocessing module
Handles loading data from various sources and initial preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from tqdm import tqdm

from config import (
    DATA_DIR, TRAIN_TEST_SPLIT_DATE, VALIDATION_DAYS, TEST_DAYS,
    NEGATIVE_SAMPLING_RATIO, NEGATIVE_SAMPLING_STRATEGY
)
from utils import setup_logger, timing_decorator, validate_dataframe

logger = setup_logger(__name__, "data_loader.log")


class DataLoader:
    """
    Handles loading and preprocessing of all data tables
    In production, this would connect to data warehouse/feature store
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.users_df = None
        self.restaurants_df = None
        self.items_df = None
        self.carts_df = None
        self.interactions_df = None
    
    @timing_decorator
    def load_users(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load users table"""
        if filepath:
            self.users_df = pd.read_csv(filepath)
        else:
            # Try to load from generated data first
            try:
                self.users_df = pd.read_csv(self.data_dir / 'users.csv')
                logger.info("Loaded users from data/users.csv")
            except FileNotFoundError:
                logger.warning("data/users.csv not found, generating synthetic data")
                self.users_df = self._generate_synthetic_users()
        
        logger.info(f"Loaded {len(self.users_df)} users")
        return self.users_df
    
    @timing_decorator
    def load_restaurants(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load restaurants table"""
        if filepath:
            self.restaurants_df = pd.read_csv(filepath)
        else:
            try:
                self.restaurants_df = pd.read_csv(self.data_dir / 'restaurants.csv')
                logger.info("Loaded restaurants from data/restaurants.csv")
            except FileNotFoundError:
                logger.warning("data/restaurants.csv not found, generating synthetic data")
                self.restaurants_df = self._generate_synthetic_restaurants()
        
        logger.info(f"Loaded {len(self.restaurants_df)} restaurants")
        return self.restaurants_df
    
    @timing_decorator
    def load_items(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load items table"""
        if filepath:
            self.items_df = pd.read_csv(filepath)
        else:
            try:
                self.items_df = pd.read_csv(self.data_dir / 'items.csv')
                logger.info("Loaded items from data/items.csv")
            except FileNotFoundError:
                logger.warning("data/items.csv not found, generating synthetic data")
                self.items_df = self._generate_synthetic_items()
        
        logger.info(f"Loaded {len(self.items_df)} items")
        return self.items_df
    
    @timing_decorator
    def load_carts(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load cart sessions table"""
        if filepath:
            self.carts_df = pd.read_csv(filepath)
        else:
            self.carts_df = self._generate_synthetic_carts()
        
        logger.info(f"Loaded {len(self.carts_df)} cart sessions")
        return self.carts_df
    
    @timing_decorator
    def load_interactions(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load interaction logs"""
        if filepath:
            self.interactions_df = pd.read_csv(filepath)
            self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        else:
            try:
                # Try to load from order_items.csv (realistic data)
                order_items_df = pd.read_csv(self.data_dir / 'order_items.csv')
                orders_df = pd.read_csv(self.data_dir / 'orders.csv')
                
                # Validate required columns exist
                required_order_items_cols = ['order_id', 'item_id', 'added_timestamp', 'is_recommended', 'added_after_recommendation']
                required_orders_cols = ['order_id', 'user_id', 'restaurant_id']
                
                missing_order_items = [col for col in required_order_items_cols if col not in order_items_df.columns]
                missing_orders = [col for col in required_orders_cols if col not in orders_df.columns]
                
                if missing_order_items or missing_orders:
                    logger.error(f"Missing columns - order_items: {missing_order_items}, orders: {missing_orders}")
                    raise ValueError("Required columns missing in data files")
                
                # Convert order_items to interactions format
                self.interactions_df = order_items_df.merge(
                    orders_df[['order_id', 'user_id', 'restaurant_id']], 
                    on='order_id',
                    how='left'
                )
                
                # Validate merge was successful
                if 'restaurant_id' not in self.interactions_df.columns:
                    logger.error("CRITICAL: restaurant_id not found after merge")
                    raise ValueError("Merge failed: restaurant_id column missing")
                
                # Check for null values in critical columns
                null_counts = {
                    'user_id': self.interactions_df['user_id'].isna().sum(),
                    'restaurant_id': self.interactions_df['restaurant_id'].isna().sum(),
                    'item_id': self.interactions_df['item_id'].isna().sum()
                }
                
                if any(null_counts.values()):
                    logger.warning(f"Found null values after merge: {null_counts}")
                    logger.warning("Dropping rows with null values in critical columns")
                    self.interactions_df = self.interactions_df.dropna(subset=['user_id', 'restaurant_id', 'item_id'])
                
                # Rename columns to match expected format
                self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['added_timestamp'])
                self.interactions_df['added_to_cart'] = 1  # All items in order_items were added
                self.interactions_df['recommended_flag'] = self.interactions_df['is_recommended']
                self.interactions_df['clicked'] = self.interactions_df['added_after_recommendation']
                self.interactions_df['ordered'] = 1  # Assume all were ordered
                
                # Create session_id from order_id
                self.interactions_df['session_id'] = 'sess_' + self.interactions_df['order_id'].astype(str)
                
                # Final validation
                required_output_cols = ['session_id', 'user_id', 'item_id', 'restaurant_id', 'timestamp', 'added_to_cart']
                missing_output = [col for col in required_output_cols if col not in self.interactions_df.columns]
                if missing_output:
                    logger.error(f"CRITICAL: Missing required output columns: {missing_output}")
                    raise ValueError(f"Output validation failed: {missing_output}")
                
                logger.info("[OK] Loaded interactions from data/order_items.csv")
                logger.info(f"[OK] Validated columns: {required_output_cols}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load from CSV files: {str(e)}, generating synthetic data")
                self.interactions_df = self._generate_synthetic_interactions()
        
        logger.info(f"Loaded {len(self.interactions_df)} interactions")
        logger.info(f"Columns in interactions_df: {list(self.interactions_df.columns)}")
        return self.interactions_df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all tables"""
        return {
            'users': self.load_users(),
            'restaurants': self.load_restaurants(),
            'items': self.load_items(),
            'carts': self.load_carts(),
            'interactions': self.load_interactions()
        }
    
    # Synthetic data generation methods (for demo/testing)
    def _generate_synthetic_users(self, n_users: int = 5000) -> pd.DataFrame:
        """Generate synthetic user data"""
        np.random.seed(42)
        
        cuisines = ['North Indian', 'South Indian', 'Chinese', 'Italian', 'Continental']
        segments = ['budget', 'regular', 'premium']
        
        data = {
            'user_id': range(1, n_users + 1),
            'recency_days': np.random.exponential(10, n_users).astype(int),
            'order_frequency_30d': np.random.poisson(5, n_users),
            'lifetime_orders': np.random.poisson(20, n_users),
            'avg_order_value': np.random.normal(400, 150, n_users).clip(100, 2000),
            'veg_preference_score': np.random.beta(2, 2, n_users),
            'price_sensitivity_score': np.random.beta(2, 2, n_users),
            'segment': np.random.choice(segments, n_users, p=[0.3, 0.5, 0.2]),
            'historical_addon_attach_rate': np.random.beta(2, 5, n_users),
            'avg_cart_size': np.random.poisson(3, n_users) + 1,
            'beverage_affinity': np.random.beta(2, 3, n_users),
            'dessert_affinity': np.random.beta(2, 4, n_users),
            'starter_affinity': np.random.beta(3, 3, n_users),
        }
        
        # Add preferred cuisines as comma-separated string
        data['preferred_cuisines'] = [
            ','.join(np.random.choice(cuisines, size=np.random.randint(1, 4), replace=False))
            for _ in range(n_users)
        ]
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_restaurants(self, n_restaurants: int = 500) -> pd.DataFrame:
        """Generate synthetic restaurant data"""
        np.random.seed(43)
        
        cuisines = ['North Indian', 'South Indian', 'Chinese', 'Italian', 'Continental', 
                   'Fast Food', 'Desserts', 'Beverages']
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune']
        
        data = {
            'restaurant_id': range(1, n_restaurants + 1),
            'cuisine_type': np.random.choice(cuisines, n_restaurants),
            'city': np.random.choice(cities, n_restaurants),
            'price_range': np.random.choice([1, 2, 3, 4], n_restaurants, p=[0.2, 0.4, 0.3, 0.1]),
            'rating': np.random.normal(3.8, 0.5, n_restaurants).clip(2.0, 5.0),
            'delivery_rating': np.random.normal(3.5, 0.6, n_restaurants).clip(2.0, 5.0),
            'avg_prep_time': np.random.normal(30, 10, n_restaurants).clip(15, 60),
            'order_volume_30d': np.random.poisson(500, n_restaurants),
            'chain_flag': np.random.choice([0, 1], n_restaurants, p=[0.7, 0.3]),
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_items(self, n_items: int = 5000) -> pd.DataFrame:
        """Generate synthetic item data"""
        np.random.seed(44)
        
        categories = ['starter', 'main', 'beverage', 'dessert', 'side']
        
        # Distribute items across restaurants
        restaurant_ids = np.random.choice(range(1, 501), n_items)
        
        data = {
            'item_id': range(1, n_items + 1),
            'restaurant_id': restaurant_ids,
            'category': np.random.choice(categories, n_items, p=[0.2, 0.4, 0.2, 0.15, 0.05]),
            'veg_flag': np.random.choice([0, 1], n_items, p=[0.4, 0.6]),
            'price': np.random.lognormal(4.5, 0.8, n_items).clip(50, 1000),
            'margin_proxy': np.random.beta(3, 2, n_items),
            'popularity_score': np.random.beta(2, 5, n_items),
            'historical_attach_rate': np.random.beta(1, 10, n_items),
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_carts(self, n_carts: int = 20000) -> pd.DataFrame:
        """Generate synthetic cart sessions"""
        np.random.seed(45)
        
        data = {
            'session_id': [f'sess_{i}' for i in range(1, n_carts + 1)],
            'user_id': np.random.randint(1, 5001, n_carts),
            'restaurant_id': np.random.randint(1, 501, n_carts),
            'cart_item_count': np.random.poisson(3, n_carts) + 1,
            'cart_total': np.random.lognormal(5.5, 0.6, n_carts).clip(100, 3000),
            'veg_ratio': np.random.beta(3, 3, n_carts),
            'dominant_category': np.random.choice(['starter', 'main', 'beverage', 'dessert'], n_carts),
        }
        
        # Generate cart items as comma-separated item IDs
        data['cart_items'] = [
            ','.join(map(str, np.random.choice(range(1, 5001), size=count, replace=False)))
            for count in data['cart_item_count']
        ]
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_interactions(self, n_interactions: int = 100000) -> pd.DataFrame:
        """Generate synthetic interaction logs"""
        np.random.seed(46)
        
        # Generate timestamps over 60 days
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(minutes=np.random.randint(0, 60*24*60)) 
                     for _ in range(n_interactions)]
        
        data = {
            'session_id': [f'sess_{np.random.randint(1, 20001)}' for _ in range(n_interactions)],
            'user_id': np.random.randint(1, 5001, n_interactions),
            'item_id': np.random.randint(1, 5001, n_interactions),
            'restaurant_id': np.random.randint(1, 501, n_interactions),
            'recommended_flag': np.random.choice([0, 1], n_interactions, p=[0.6, 0.4]),
            'clicked': np.random.choice([0, 1], n_interactions, p=[0.85, 0.15]),
            'added_to_cart': np.random.choice([0, 1], n_interactions, p=[0.88, 0.12]),
            'ordered': np.random.choice([0, 1], n_interactions, p=[0.92, 0.08]),
            'timestamp': timestamps,
        }
        
        df = pd.DataFrame(data)
        
        # Make interactions more realistic: if added_to_cart=1, then clicked=1
        df.loc[df['added_to_cart'] == 1, 'clicked'] = 1
        df.loc[df['ordered'] == 1, 'added_to_cart'] = 1
        
        logger.info("[OK] Generated synthetic interactions with restaurant_id column")
        
        return df.sort_values('timestamp').reset_index(drop=True)


class DataSplitter:
    """
    Handles temporal train/validation/test splitting
    Critical for time-series nature of recommendation data
    """
    
    def __init__(self, split_date: str = TRAIN_TEST_SPLIT_DATE, 
                 validation_days: int = VALIDATION_DAYS,
                 test_days: int = TEST_DAYS):
        self.split_date = pd.to_datetime(split_date)
        self.validation_days = validation_days
        self.test_days = test_days
    
    @timing_decorator
    def split_temporal(self, df: pd.DataFrame, 
                      timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform temporal split: train / validation / test
        Ensures no data leakage from future to past
        """
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Calculate split dates
        test_start = self.split_date
        validation_start = test_start - timedelta(days=self.test_days)
        train_end = validation_start
        
        # Split data
        train_df = df[df[timestamp_col] < train_end].copy()
        val_df = df[(df[timestamp_col] >= validation_start) & 
                    (df[timestamp_col] < test_start)].copy()
        test_df = df[df[timestamp_col] >= test_start].copy()
        
        logger.info(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info(f"Train period: up to {train_end}")
        logger.info(f"Validation period: {validation_start} to {test_start}")
        logger.info(f"Test period: from {test_start}")
        
        return train_df, val_df, test_df


class NegativeSampler:
    """
    Handles negative sampling for training ranking models
    Critical for learning from implicit feedback
    """
    
    def __init__(self, strategy: str = NEGATIVE_SAMPLING_STRATEGY,
                 ratio: int = NEGATIVE_SAMPLING_RATIO):
        self.strategy = strategy
        self.ratio = ratio
        self.item_popularity = None
    
    def fit(self, interactions_df: pd.DataFrame):
        """Compute item popularity for popularity-based sampling"""
        self.item_popularity = (
            interactions_df.groupby('item_id')
            .size()
            .reset_index(name='count')
        )
        self.item_popularity['prob'] = (
            self.item_popularity['count'] / self.item_popularity['count'].sum()
        )
    
    @timing_decorator
    def sample_negatives(self, interactions_df: pd.DataFrame,
                        items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample negative examples for each positive interaction
        
        Strategies:
        - random: Uniform random sampling
        - popular: Sample based on item popularity
        - mixed: 50% random + 50% popular
        """
        positives = interactions_df[interactions_df['added_to_cart'] == 1].copy()
        
        negatives_list = []
        
        print(f"\n[PROCESSING] Sampling negatives for {len(positives)} positive interactions...")
        for _, row in tqdm(positives.iterrows(), total=len(positives), desc="Negative Sampling"):
            session_id = row['session_id']
            user_id = row['user_id']
            restaurant_id = row['restaurant_id']
            
            # Get candidate items from same restaurant
            restaurant_items = items_df[items_df['restaurant_id'] == restaurant_id]['item_id'].values
            
            if len(restaurant_items) == 0:
                continue
            
            # Sample negatives
            n_negatives = min(self.ratio, len(restaurant_items) - 1)
            
            if self.strategy == 'random':
                negative_items = np.random.choice(restaurant_items, size=n_negatives, replace=False)
            elif self.strategy == 'popular' and self.item_popularity is not None:
                # Sample based on popularity
                item_probs = self.item_popularity[
                    self.item_popularity['item_id'].isin(restaurant_items)
                ]
                if len(item_probs) > 0:
                    negative_items = np.random.choice(
                        item_probs['item_id'].values,
                        size=n_negatives,
                        replace=False,
                        p=item_probs['prob'].values / item_probs['prob'].sum()
                    )
                else:
                    negative_items = np.random.choice(restaurant_items, size=n_negatives, replace=False)
            else:  # mixed
                n_random = n_negatives // 2
                n_popular = n_negatives - n_random
                negative_items = np.concatenate([
                    np.random.choice(restaurant_items, size=n_random, replace=False),
                    np.random.choice(restaurant_items, size=n_popular, replace=False)
                ])
            
            # Create negative samples
            for item_id in negative_items:
                negatives_list.append({
                    'session_id': session_id,
                    'user_id': user_id,
                    'item_id': item_id,
                    'restaurant_id': restaurant_id,
                    'added_to_cart': 0,
                    'timestamp': row['timestamp']
                })
        
        negatives_df = pd.DataFrame(negatives_list)
        
        # Combine positives and negatives
        combined_df = pd.concat([
            positives[['session_id', 'user_id', 'item_id', 'restaurant_id', 'added_to_cart', 'timestamp']],
            negatives_df
        ], ignore_index=True)
        
        # VALIDATION: Ensure each session has both positives and negatives
        self._validate_candidate_set(combined_df)
        
        logger.info(f"Generated {len(negatives_df)} negative samples for {len(positives)} positives")
        print(f"[OK] Generated {len(negatives_df):,} negatives → Total: {len(combined_df):,} samples")
        
        return combined_df
    
    def _validate_candidate_set(self, combined_df: pd.DataFrame):
        """Validate that each session has proper positive and negative samples"""
        logger.info("Validating candidate set...")
        
        session_stats = combined_df.groupby('session_id')['added_to_cart'].agg(['sum', 'count'])
        session_stats.columns = ['n_positives', 'total']
        session_stats['n_negatives'] = session_stats['total'] - session_stats['n_positives']
        
        # Check for sessions with no positives
        no_positives = session_stats[session_stats['n_positives'] == 0]
        if len(no_positives) > 0:
            logger.error(f"Found {len(no_positives)} sessions with NO positives")
            logger.error(f"Sample sessions: {no_positives.head().index.tolist()}")
            raise ValueError(f"Invalid candidate set: {len(no_positives)} sessions have no positives")
        
        # Check for sessions with no negatives
        no_negatives = session_stats[session_stats['n_negatives'] == 0]
        if len(no_negatives) > 0:
            logger.error(f"Found {len(no_negatives)} sessions with NO negatives")
            logger.error(f"Sample sessions: {no_negatives.head().index.tolist()}")
            raise ValueError(f"Invalid candidate set: {len(no_negatives)} sessions have no negatives")
        
        # Check for sessions with too few negatives (< 5)
        few_negatives = session_stats[session_stats['n_negatives'] < 5]
        if len(few_negatives) > 0:
            logger.warning(f"Found {len(few_negatives)} sessions with < 5 negatives")
            logger.warning(f"This may affect ranking quality")
        
        # Log statistics
        logger.info(f"Candidate set validation passed:")
        logger.info(f"  - Total sessions: {len(session_stats)}")
        logger.info(f"  - Avg positives per session: {session_stats['n_positives'].mean():.2f}")
        logger.info(f"  - Avg negatives per session: {session_stats['n_negatives'].mean():.2f}")
        logger.info(f"  - Min negatives per session: {session_stats['n_negatives'].min()}")
        logger.info(f"  - Max negatives per session: {session_stats['n_negatives'].max()}")
        
        print(f"[OK] Validation passed: {len(session_stats):,} sessions with proper pos/neg samples")
