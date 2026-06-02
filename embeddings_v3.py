"""
Embedding Module V3.0
Learn latent representations for items and users

Components:
1. Item embeddings (Word2Vec-style on order sequences)
2. User embeddings (aggregated from item history)
3. Similarity features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ItemEmbeddingTrainer:
    """
    Train item embeddings using Word2Vec on order sequences
    Treats orders as sentences and items as words
    """
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 window: int = 5,
                 min_count: int = 2,
                 workers: int = 4,
                 epochs: int = 10):
        """
        Args:
            embedding_dim: Dimension of embedding vectors
            window: Context window size
            min_count: Minimum item frequency
            workers: Number of parallel workers
            epochs: Training epochs
        """
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.item_embeddings = None
        
    def fit(self, interactions_df: pd.DataFrame):
        """
        Train embeddings on order sequences
        
        Args:
            interactions_df: DataFrame with order_id/session_id and item_id
        """
        logger.info("Training item embeddings...")
        
        # Group items by order to create sequences
        group_col = 'order_id' if 'order_id' in interactions_df.columns else 'session_id'
        
        # Sort by timestamp within each order
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.sort_values(['timestamp'])
        
        # Create sequences (list of item lists)
        sequences = interactions_df.groupby(group_col)['item_id'].apply(list).tolist()
        
        logger.info(f"Training on {len(sequences)} order sequences...")
        logger.info(f"Embedding dim: {self.embedding_dim}, Window: {self.window}")
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=1,  # Skip-gram (better for rare items)
            negative=5,  # Negative sampling
            seed=42
        )
        
        # Extract embeddings as dictionary
        self.item_embeddings = {
            item: self.model.wv[item]
            for item in self.model.wv.index_to_key
        }
        
        logger.info(f"Trained embeddings for {len(self.item_embeddings)} items")
        
        return self
    
    def get_embedding(self, item_id: str) -> np.ndarray:
        """Get embedding for a single item"""
        if item_id in self.item_embeddings:
            return self.item_embeddings[item_id]
        else:
            # Return zero vector for unknown items
            return np.zeros(self.embedding_dim)
    
    def get_embeddings(self, item_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple items"""
        return np.array([self.get_embedding(item_id) for item_id in item_ids])
    
    def get_similar_items(self, item_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most similar items using cosine similarity"""
        if item_id not in self.item_embeddings:
            return []
        
        try:
            similar = self.model.wv.most_similar(item_id, topn=top_n)
            return similar
        except:
            return []
    
    def save(self, filepath: str):
        """Save embeddings to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.item_embeddings,
                'embedding_dim': self.embedding_dim,
                'model': self.model
            }, f)
        logger.info(f"Embeddings saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ItemEmbeddingTrainer':
        """Load embeddings from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls(embedding_dim=data['embedding_dim'])
        trainer.item_embeddings = data['embeddings']
        trainer.model = data['model']
        
        logger.info(f"Embeddings loaded from {filepath}")
        return trainer


class UserEmbeddingGenerator:
    """
    Generate user embeddings by aggregating item embeddings
    from user's order history
    """
    
    def __init__(self, item_embeddings: Dict[str, np.ndarray]):
        """
        Args:
            item_embeddings: Dictionary mapping item_id to embedding vector
        """
        self.item_embeddings = item_embeddings
        self.embedding_dim = len(next(iter(item_embeddings.values())))
        self.user_embeddings = {}
        
    def fit(self, interactions_df: pd.DataFrame, recency_weight: bool = True):
        """
        Generate user embeddings from interaction history
        
        Args:
            interactions_df: DataFrame with user_id, item_id, timestamp
            recency_weight: Whether to weight recent items more
        """
        logger.info("Generating user embeddings...")
        
        # Sort by timestamp if available
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.sort_values('timestamp')
        
        # Group by user
        for user_id, group in interactions_df.groupby('user_id'):
            item_ids = group['item_id'].tolist()
            
            # Get item embeddings
            embeddings = []
            weights = []
            
            for i, item_id in enumerate(item_ids):
                if item_id in self.item_embeddings:
                    embeddings.append(self.item_embeddings[item_id])
                    
                    # Recency weight: more recent items get higher weight
                    if recency_weight:
                        weight = np.exp(-0.1 * (len(item_ids) - i - 1))
                    else:
                        weight = 1.0
                    weights.append(weight)
            
            if len(embeddings) > 0:
                # Weighted average
                embeddings = np.array(embeddings)
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                user_embedding = np.average(embeddings, axis=0, weights=weights)
                self.user_embeddings[user_id] = user_embedding
        
        logger.info(f"Generated embeddings for {len(self.user_embeddings)} users")
        
        return self
    
    def get_embedding(self, user_id: str) -> np.ndarray:
        """Get embedding for a single user"""
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id]
        else:
            # Return zero vector for unknown users
            return np.zeros(self.embedding_dim)
    
    def get_embeddings(self, user_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple users"""
        return np.array([self.get_embedding(user_id) for user_id in user_ids])


class EmbeddingFeatureEngineer:
    """
    Generate embedding-based features for recommendation
    """
    
    def __init__(self, 
                 item_embeddings: Dict[str, np.ndarray],
                 user_embeddings: Dict[str, np.ndarray]):
        """
        Args:
            item_embeddings: Item embedding dictionary
            user_embeddings: User embedding dictionary
        """
        self.item_embeddings = item_embeddings
        self.user_embeddings = user_embeddings
        self.embedding_dim = len(next(iter(item_embeddings.values())))
        
    def add_embedding_features(self, 
                               df: pd.DataFrame,
                               cart_items_col: str = None) -> pd.DataFrame:
        """
        Add embedding-based features to dataframe
        
        Features:
        1. user_item_similarity: user_embedding · item_embedding
        2. last_item_similarity: last_item_embedding · candidate_embedding
        3. cart_similarity: avg(cart_embeddings) · candidate_embedding
        4. embedding_norm: ||item_embedding||
        
        Args:
            df: DataFrame with user_id, item_id
            cart_items_col: Column containing list of cart items (optional)
        """
        logger.info("Adding embedding features...")
        
        # 1. User-item similarity
        user_embeddings = np.array([
            self.user_embeddings.get(uid, np.zeros(self.embedding_dim))
            for uid in df['user_id']
        ])
        
        item_embeddings = np.array([
            self.item_embeddings.get(iid, np.zeros(self.embedding_dim))
            for iid in df['item_id']
        ])
        
        # Dot product (cosine similarity if normalized)
        df['user_item_similarity'] = np.sum(user_embeddings * item_embeddings, axis=1)
        
        # 2. Embedding norm (magnitude)
        df['item_embedding_norm'] = np.linalg.norm(item_embeddings, axis=1)
        df['user_embedding_norm'] = np.linalg.norm(user_embeddings, axis=1)
        
        # 3. Cart similarity (if cart items available)
        if cart_items_col and cart_items_col in df.columns:
            cart_similarities = []
            
            for idx, row in df.iterrows():
                cart_items = row[cart_items_col]
                candidate_embedding = item_embeddings[idx]
                
                if isinstance(cart_items, list) and len(cart_items) > 0:
                    # Get cart embeddings
                    cart_embs = np.array([
                        self.item_embeddings.get(item, np.zeros(self.embedding_dim))
                        for item in cart_items
                    ])
                    
                    # Average cart embedding
                    avg_cart_emb = np.mean(cart_embs, axis=0)
                    
                    # Similarity
                    similarity = np.dot(avg_cart_emb, candidate_embedding)
                    cart_similarities.append(similarity)
                else:
                    cart_similarities.append(0.0)
            
            df['cart_item_similarity'] = cart_similarities
        
        # 4. Last item similarity (if we can infer last item)
        # This would require additional logic to track last item per session
        # For now, we'll skip this feature
        
        logger.info(f"Added {3 if cart_items_col else 2} embedding features")
        
        return df
    
    def get_embedding_cluster(self, item_id: str, n_clusters: int = 10) -> List[str]:
        """
        Get items in the same embedding cluster
        Useful for hard negative sampling
        """
        if item_id not in self.item_embeddings:
            return []
        
        item_emb = self.item_embeddings[item_id]
        
        # Compute similarities to all items
        similarities = []
        for other_id, other_emb in self.item_embeddings.items():
            if other_id != item_id:
                sim = np.dot(item_emb, other_emb)
                similarities.append((other_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [item_id for item_id, _ in similarities[:n_clusters]]


def train_embeddings_pipeline(interactions_df: pd.DataFrame,
                              embedding_dim: int = 64,
                              save_path: str = None) -> Tuple[ItemEmbeddingTrainer, UserEmbeddingGenerator]:
    """
    Complete pipeline to train item and user embeddings
    
    Args:
        interactions_df: DataFrame with order_id, user_id, item_id, timestamp
        embedding_dim: Embedding dimension
        save_path: Path to save embeddings (optional)
    
    Returns:
        Tuple of (item_trainer, user_generator)
    """
    logger.info("Starting embedding training pipeline...")
    
    # 1. Train item embeddings
    item_trainer = ItemEmbeddingTrainer(embedding_dim=embedding_dim)
    item_trainer.fit(interactions_df)
    
    # 2. Generate user embeddings
    user_generator = UserEmbeddingGenerator(item_trainer.item_embeddings)
    user_generator.fit(interactions_df, recency_weight=True)
    
    # 3. Save if path provided
    if save_path:
        item_trainer.save(save_path)
    
    logger.info("Embedding training pipeline complete")
    
    return item_trainer, user_generator
