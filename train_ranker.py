"""
Ranking Model Training Module - Stage 2 of Two-Stage Architecture
Trains LightGBM ranker to predict probability of add-to-cart
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import (
    LIGHTGBM_PARAMS, N_ESTIMATORS, EARLY_STOPPING_ROUNDS,
    MODELS_DIR, MODEL_VERSION
)
from utils import setup_logger, timing_decorator, save_model, load_model

logger = setup_logger(__name__, "train_ranker.log")


class RankingModel:
    """
    LightGBM-based ranking model for add-on recommendations
    Uses lambdarank objective for learning-to-rank
    """
    
    def __init__(self, params: Dict = None):
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {}
    
    @timing_decorator
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: np.ndarray,
             groups_train: np.ndarray,
             X_val: pd.DataFrame = None,
             y_val: np.ndarray = None,
             groups_val: np.ndarray = None) -> 'RankingModel':
        """
        Train the ranking model
        
        Args:
            X_train: Training features
            y_train: Training labels (added_to_cart)
            groups_train: Group sizes for ranking (sessions)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            groups_val: Validation group sizes (optional)
        """
        logger.info("Starting ranking model training...")
        logger.info(f"Training samples: {len(X_train)}, Groups: {len(groups_train)}")
        print(f"\n[START] Training LightGBM Ranker on GPU...")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Training groups: {len(groups_train):,}")
        print(f"   Features: {len(X_train.columns)}")
        
        self.feature_names = list(X_train.columns)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            group=groups_train,
            feature_name=self.feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None and groups_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                feature_name=self.feature_names,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
            logger.info(f"Validation samples: {len(X_val)}, Groups: {len(groups_val)}")
            print(f"   Validation samples: {len(X_val):,}")
            print(f"   Validation groups: {len(groups_val):,}")
        
        # Training callbacks with progress bar
        callbacks = [
            lgb.log_evaluation(period=10),
            lgb.record_evaluation(self.training_history)
        ]
        
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS))
        
        # Train model
        print(f"\n   Training {N_ESTIMATORS} trees with GPU acceleration...")
        with tqdm(total=N_ESTIMATORS, desc="Training Progress", unit="tree") as pbar:
            def callback(env):
                pbar.update(1)
            
            callbacks.append(callback)
            
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=N_ESTIMATORS,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")
        print(f"\n[OK] Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best NDCG@10: {self.model.best_score.get('valid', {}).get('ndcg@10', 'N/A')}")
        
        return self
    
    @timing_decorator
    def train_with_cv(self,
                     X: pd.DataFrame,
                     y: np.ndarray,
                     groups: np.ndarray,
                     n_folds: int = 3) -> Dict:
        """
        Train with cross-validation using GroupKFold
        Ensures sessions are not split across folds
        """
        logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        # Create group labels for GroupKFold
        group_labels = np.repeat(np.arange(len(groups)), groups)
        
        gkf = GroupKFold(n_splits=n_folds)
        
        cv_scores = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, group_labels)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training Fold {fold + 1}/{n_folds}")
            logger.info(f"{'='*50}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y[val_idx]
            
            # Compute groups for this fold
            train_group_labels = group_labels[train_idx]
            val_group_labels = group_labels[val_idx]
            
            groups_train_fold = np.bincount(train_group_labels)
            groups_val_fold = np.bincount(val_group_labels)
            
            # Train fold model
            fold_model = RankingModel(self.params)
            fold_model.train(
                X_train_fold, y_train_fold, groups_train_fold,
                X_val_fold, y_val_fold, groups_val_fold
            )
            
            fold_models.append(fold_model)
            
            # Record validation score
            if 'valid' in fold_model.training_history:
                best_score = fold_model.training_history['valid']['ndcg@10'][-1]
                cv_scores.append(best_score)
                logger.info(f"Fold {fold + 1} best NDCG@10: {best_score:.4f}")
        
        # Use the best fold model
        best_fold = np.argmax(cv_scores)
        self.model = fold_models[best_fold].model
        self.feature_names = fold_models[best_fold].feature_names
        self.feature_importance = fold_models[best_fold].feature_importance
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'best_fold': best_fold
        }
        
        logger.info(f"\nCross-validation completed:")
        logger.info(f"Mean NDCG@10: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        logger.info(f"Using model from fold {best_fold + 1}")
        
        return cv_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict add-to-cart probability for candidates
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches training
        X_ordered = X[self.feature_names]
        
        predictions = self.model.predict(X_ordered)
        
        return predictions
    
    def predict_top_n(self, 
                     X: pd.DataFrame,
                     groups: np.ndarray,
                     n: int = 10) -> List[np.ndarray]:
        """
        Predict and return top N items per group (session)
        """
        predictions = self.predict(X)
        
        top_n_per_group = []
        start_idx = 0
        
        for group_size in groups:
            end_idx = start_idx + group_size
            group_preds = predictions[start_idx:end_idx]
            
            # Get top N indices
            top_n_indices = np.argsort(group_preds)[-n:][::-1]
            top_n_per_group.append(top_n_indices)
            
            start_idx = end_idx
        
        return top_n_per_group
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot NDCG
        for dataset in self.training_history:
            if 'ndcg@10' in self.training_history[dataset]:
                ndcg_scores = self.training_history[dataset]['ndcg@10']
                axes[0].plot(ndcg_scores, label=dataset)
        
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('NDCG@10')
        axes[0].set_title('NDCG@10 During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot NDCG@5 if available
        for dataset in self.training_history:
            if 'ndcg@5' in self.training_history[dataset]:
                ndcg_scores = self.training_history[dataset]['ndcg@5']
                axes[1].plot(ndcg_scores, label=dataset)
        
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('NDCG@5')
        axes[1].set_title('NDCG@5 During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, filepath: str = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODELS_DIR / f"ranker_{MODEL_VERSION}.pkl"
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.params,
            'training_history': self.training_history
        }
        
        save_model(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RankingModel':
        """Load model from disk"""
        model_data = load_model(filepath)
        
        ranker = cls(params=model_data['params'])
        ranker.model = model_data['model']
        ranker.feature_names = model_data['feature_names']
        ranker.feature_importance = model_data['feature_importance']
        ranker.training_history = model_data.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")
        
        return ranker
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "n_features": len(self.feature_names),
            "n_trees": self.model.num_trees(),
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "top_10_features": self.feature_importance.head(10)['feature'].tolist() if self.feature_importance is not None else []
        }


def train_production_model(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          feature_cols: List[str],
                          save_path: str = None) -> RankingModel:
    """
    Train production-ready ranking model
    Convenience function for end-to-end training
    """
    logger.info("Training production ranking model...")
    
    # Double-check: filter out any non-numeric columns from feature_cols
    numeric_feature_cols = []
    for col in feature_cols:
        if col in train_df.columns:
            dtype = train_df[col].dtype
            if dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'int8', 'float16', 'uint8', 'uint16', 'uint32', 'uint64']:
                numeric_feature_cols.append(col)
            else:
                logger.warning(f"Skipping non-numeric feature column: {col} (dtype: {dtype})")
    
    logger.info(f"Using {len(numeric_feature_cols)} numeric features for training")
    
    # Prepare training data
    X_train = train_df[numeric_feature_cols]
    y_train = train_df['added_to_cart'].values
    groups_train = train_df.groupby('session_id').size().values
    
    # Prepare validation data
    X_val = val_df[numeric_feature_cols]
    y_val = val_df['added_to_cart'].values
    groups_val = val_df.groupby('session_id').size().values
    
    # Train model
    ranker = RankingModel()
    ranker.train(X_train, y_train, groups_train, X_val, y_val, groups_val)
    
    # Plot diagnostics
    ranker.plot_feature_importance(
        top_n=20,
        save_path=MODELS_DIR / "feature_importance.png"
    )
    ranker.plot_training_history(
        save_path=MODELS_DIR / "training_history.png"
    )
    
    # Save model
    if save_path:
        ranker.save(save_path)
    else:
        ranker.save()
    
    logger.info("Production model training completed")
    
    return ranker
