"""
Advanced Feature Engineering V2.0
Complete redesign with focus on:
1. No data leakage
2. High-signal features
3. Interaction features
4. Sequential features
5. Context features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for recommendation system
    
    Feature Categories:
    1. User features (behavior, preferences)
    2. Item features (properties, popularity)
    3. Cart context features (current cart state)
    4. Interaction features (user-item affinity)
    5. Sequential features (item transitions)
    6. Context features (time, location)
    """
    
    def __init__(self):
        self.user_stats = None
        self.item_stats = None
        self.item_cooccurrence = None
        self.category_transitions = None
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: pd.DataFrame):
        """Precompute statistics for feature engineering"""
        logger.info("Fitting feature engineer...")
        
        # Only use training data (past interactions)
        train_interactions = interactions_df[interactions_df['added_to_cart'] == 1].copy()
        
        # 1. User statistics
        self._compute_user_stats(train_interactions, users_df)
        
        # 2. Item statistics
        self._compute_item_stats(train_interactions, items_df)
        
        # 3. Item co-occurrence
        self._compute_item_cooccurrence(train_interactions)
        
        # 4. Category transitions
        self._compute_category_transitions(train_interactions, items_df)
        
        logger.info("Feature engineer fitted")
        
    def _compute_user_stats(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame):
        """Compute user-level statistics"""
        user_stats = interactions_df.groupby('user_id').agg({
            'item_id': 'count',  # order frequency
            'timestamp': ['min', 'max']  # first and last order
        })
        user_stats.columns = ['order_count', 'first_order', 'last_order']
        
        # Merge with user demographics
        user_stats = user_stats.merge(
            users_df[['user_id', 'user_segment', 'veg_preference', 'preferred_cuisine']],
            on='user_id',
            how='left'
        )
        
        self.user_stats = user_stats
        logger.info(f"Computed stats for {len(user_stats)} users")
        
    def _compute_item_stats(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame):
        """Compute item-level statistics"""
        item_stats = interactions_df.groupby('item_id').agg({
            'user_id': 'nunique',  # unique users
            'session_id': 'count'  # total orders
        })
        item_stats.columns = ['unique_users', 'order_count']
        
        # Compute popularity score
        item_stats['popularity_score'] = (
            item_stats['order_count'] / item_stats['order_count'].sum()
        )
        
        # Merge with item properties
        item_stats = item_stats.merge(
            items_df[['item_id', 'price', 'category', 'veg_flag']],
            on='item_id',
            how='left'
        )
        
        self.item_stats = item_stats
        logger.info(f"Computed stats for {len(item_stats)} items")
        
    def _compute_item_cooccurrence(self, interactions_df: pd.DataFrame):
        """Compute item co-occurrence matrix"""
        # Group by session to get items ordered together
        session_items = interactions_df.groupby('session_id')['item_id'].apply(list)
        
        cooccurrence = {}
        for items in session_items:
            for i, item1 in enumerate(items):
                if item1 not in cooccurrence:
                    cooccurrence[item1] = Counter()
                for item2 in items[i+1:]:
                    cooccurrence[item1][item2] += 1
                    if item2 not in cooccurrence:
                        cooccurrence[item2] = Counter()
                    cooccurrence[item2][item1] += 1
        
        self.item_cooccurrence = cooccurrence
        logger.info(f"Computed co-occurrence for {len(cooccurrence)} items")
        
    def _compute_category_transitions(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame):
        """Compute category transition probabilities"""
        # Merge to get categories
        df = interactions_df.merge(
            items_df[['item_id', 'category']],
            on='item_id',
            how='left'
        )
        
        # Group by session and get category sequences
        session_categories = df.groupby('session_id')['category'].apply(list)
        
        transitions = Counter()
        for categories in session_categories:
            for i in range(len(categories) - 1):
                transition = (categories[i], categories[i+1])
                transitions[transition] += 1
        
        # Convert to probabilities
        total_transitions = sum(transitions.values())
        transition_probs = {
            k: v / total_transitions
            for k, v in transitions.items()
        }
        
        self.category_transitions = transition_probs
        logger.info(f"Computed {len(transition_probs)} category transitions")
        
    def engineer_features(self,
                         candidates_df: pd.DataFrame,
                         users_df: pd.DataFrame,
                         items_df: pd.DataFrame,
                         carts_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer all features for candidate items
        
        CRITICAL: NO DATA LEAKAGE
        - Only use past information
        - No target-based features
        - No future information
        """
        logger.info(f"Engineering features for {len(candidates_df):,} candidates...")
        
        df = candidates_df.copy()
        
        # 1. User features
        df = self._add_user_features(df, users_df)
        
        # 2. Item features
        df = self._add_item_features(df, items_df)
        
        # 3. Cart context features
        if carts_df is not None:
            df = self._add_cart_features(df, carts_df, items_df)
        
        # 4. Interaction features
        df = self._add_interaction_features(df)
        
        # 5. Context features
        df = self._add_context_features(df)
        
        # 6. Sequential features
        df = self._add_sequential_features(df, items_df)
        
        logger.info(f"Feature engineering complete: {df.shape}")
        
        return df
    
    def _add_user_features(self, df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        """Add user-level features"""
        # Merge user demographics
        df = df.merge(
            users_df[['user_id', 'user_segment', 'veg_preference', 'order_frequency', 'preferred_cuisine']],
            on='user_id',
            how='left'
        )
        
        # Encode categorical features
        df['segment_budget'] = (df['user_segment'] == 'budget').astype(int)
        df['segment_regular'] = (df['user_segment'] == 'regular').astype(int)
        df['segment_premium'] = (df['user_segment'] == 'premium').astype(int)
        
        df['veg_preference_veg'] = (df['veg_preference'] == 'veg').astype(int)
        df['veg_preference_nonveg'] = (df['veg_preference'] == 'non-veg').astype(int)
        df['veg_preference_both'] = (df['veg_preference'] == 'both').astype(int)
        
        df['order_freq_low'] = (df['order_frequency'] == 'low').astype(int)
        df['order_freq_medium'] = (df['order_frequency'] == 'medium').astype(int)
        df['order_freq_high'] = (df['order_frequency'] == 'high').astype(int)
        
        # Merge user stats if available
        if self.user_stats is not None:
            df = df.merge(
                self.user_stats[['order_count']],
                left_on='user_id',
                right_index=True,
                how='left'
            )
            df['user_order_count'] = df['order_count'].fillna(0)
            df.drop('order_count', axis=1, inplace=True)
        
        return df
    
    def _add_item_features(self, df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Add item-level features"""
        # Merge item properties
        df = df.merge(
            items_df[['item_id', 'price', 'category', 'veg_flag', 'popularity_score']],
            on='item_id',
            how='left'
        )
        
        # Price features
        df['log_price'] = np.log1p(df['price'])
        df['price_squared'] = df['price'] ** 2
        
        # Category one-hot encoding
        categories = ['main', 'drink', 'dessert', 'snack', 'starter', 'beverage']
        for cat in categories:
            df[f'category_{cat}'] = (df['category'] == cat).astype(int)
        
        # Veg flag
        df['is_veg'] = df['veg_flag'].astype(int)
        
        # Merge item stats if available
        if self.item_stats is not None:
            df = df.merge(
                self.item_stats[['unique_users', 'order_count', 'popularity_score']],
                left_on='item_id',
                right_index=True,
                how='left',
                suffixes=('', '_stat')
            )
            df['item_unique_users'] = df['unique_users'].fillna(0)
            df['item_order_count'] = df['order_count'].fillna(0)
            df['item_popularity'] = df['popularity_score_stat'].fillna(0)
            df.drop(['unique_users', 'order_count', 'popularity_score_stat'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def _add_cart_features(self, df: pd.DataFrame, carts_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Add cart context features"""
        # Merge cart items
        cart_info = carts_df.groupby('session_id')['item_id'].apply(list).to_dict()
        
        df['cart_items'] = df['session_id'].map(cart_info)
        df['cart_size'] = df['cart_items'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Get cart item details
        def get_cart_stats(cart_items):
            if not isinstance(cart_items, list) or len(cart_items) == 0:
                return {
                    'cart_total_price': 0,
                    'cart_avg_price': 0,
                    'cart_has_main': 0,
                    'cart_has_drink': 0,
                    'cart_has_dessert': 0,
                    'cart_veg_ratio': 0.5
                }
            
            cart_item_details = items_df[items_df['item_id'].isin(cart_items)]
            
            return {
                'cart_total_price': cart_item_details['price'].sum(),
                'cart_avg_price': cart_item_details['price'].mean(),
                'cart_has_main': int('main' in cart_item_details['category'].values),
                'cart_has_drink': int('drink' in cart_item_details['category'].values),
                'cart_has_dessert': int('dessert' in cart_item_details['category'].values),
                'cart_veg_ratio': cart_item_details['veg_flag'].mean()
            }
        
        cart_stats = df['cart_items'].apply(get_cart_stats)
        cart_stats_df = pd.DataFrame(cart_stats.tolist())
        df = pd.concat([df, cart_stats_df], axis=1)
        
        df.drop('cart_items', axis=1, inplace=True)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user-item interaction features"""
        # User-item affinity (based on preferences)
        df['user_item_veg_match'] = (
            (df['veg_preference_veg'] == 1) & (df['is_veg'] == 1)
        ).astype(int)
        
        df['user_item_price_match'] = 0
        df.loc[df['segment_budget'] == 1, 'user_item_price_match'] = (df['price'] < 150).astype(int)
        df.loc[df['segment_premium'] == 1, 'user_item_price_match'] = (df['price'] > 200).astype(int)
        
        # Item co-occurrence with cart items
        if self.item_cooccurrence is not None:
            def compute_cooccurrence_score(row):
                item_id = row['item_id']
                cart_items = row.get('cart_items_list', [])
                
                if not isinstance(cart_items, list) or len(cart_items) == 0:
                    return 0.0
                
                if item_id not in self.item_cooccurrence:
                    return 0.0
                
                score = sum(
                    self.item_cooccurrence[item_id].get(cart_item, 0)
                    for cart_item in cart_items
                )
                return score / len(cart_items)
            
            # Note: This requires cart_items_list which we'll add if available
            # For now, skip this feature
            pass
        
        return df
    
    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add context features (time, location)"""
        # Time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Time of day
            df['is_breakfast'] = ((df['hour'] >= 6) & (df['hour'] < 10)).astype(int)
            df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] < 15)).astype(int)
            df['is_dinner'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
            df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        return df
    
    def _add_sequential_features(self, df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Add sequential/transition features"""
        # Category transition probability
        if self.category_transitions is not None:
            # This requires knowing the last item in cart
            # For now, we'll add a simplified version
            pass
        
        # Price progression (is this item more expensive than cart average?)
        if 'cart_avg_price' in df.columns:
            df['price_vs_cart_avg'] = df['price'] - df['cart_avg_price']
            df['price_ratio_to_cart'] = df['price'] / (df['cart_avg_price'] + 1)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude IDs and target)"""
        exclude_cols = [
            'order_id', 'session_id', 'user_id', 'item_id', 'restaurant_id',
            'added_to_cart', 'timestamp', 'added_timestamp',
            'user_segment', 'veg_preference', 'order_frequency', 'preferred_cuisine',
            'category', 'dominant_category', 'city', 'cuisine_type',
            # Leaky features
            'is_recommended', 'added_after_recommendation', 'recommended_flag',
            'clicked', 'ordered', 'position_in_cart'
        ]
        
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                dtype = df[col].dtype
                if dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'int8', 'uint8']:
                    feature_cols.append(col)
                else:
                    logger.warning(f"Excluding non-numeric column: {col} (dtype: {dtype})")
        
        logger.info(f"Selected {len(feature_cols)} feature columns")
        
        return feature_cols
