"""
V2 + V3 Embeddings Integration
Quick win: Add embedding features to V2 pipeline

Expected improvements:
- AUC: +5-8%
- NDCG@10: +5-10%
- Minimal code changes
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import MODELS_DIR, DATA_DIR
from utils import setup_logger
from data_loader import DataLoader, DataSplitter
from fast_negative_sampler import UltraFastNegativeSampler  # OPTIMIZED!
from feature_engineering_v2 import AdvancedFeatureEngineer
from train_ranker import RankingModel
from evaluation_v2 import AdvancedEvaluator

# V3 modules
from embeddings_v3 import train_embeddings_pipeline, EmbeddingFeatureEngineer

logger = setup_logger(__name__, "main_v2_v3.log")


def main():
    """
    V2 pipeline enhanced with V3 embeddings
    """
    print("\n" + "="*80)
    print("[V2+V3] ZOMATO RECOMMENDATION SYSTEM - V2 WITH V3 EMBEDDINGS")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n[STEP 1] LOADING DATA")
    print("-"*80)
    
    data_loader = DataLoader()
    data = data_loader.load_all()
    
    users_df = data['users']
    restaurants_df = data['restaurants']
    items_df = data['items']
    carts_df = data['carts']
    interactions_df = data['interactions']
    
    print(f"✓ Loaded {len(interactions_df):,} interactions")
    
    # ========================================================================
    # STEP 2: TEMPORAL SPLIT
    # ========================================================================
    print("\n[STEP 2] TEMPORAL SPLIT")
    print("-"*80)
    
    splitter = DataSplitter()
    train_interactions, val_interactions, test_interactions = splitter.split_temporal(
        interactions_df
    )
    
    print(f"✓ Train: {len(train_interactions):,}")
    print(f"✓ Val: {len(val_interactions):,}")
    print(f"✓ Test: {len(test_interactions):,}")
    
    # ========================================================================
    # STEP 3: TRAIN V3 EMBEDDINGS (NEW!)
    # ========================================================================
    print("\n[STEP 3] TRAINING V3 EMBEDDINGS")
    print("-"*80)
    
    print("  Training item and user embeddings...")
    item_trainer, user_generator = train_embeddings_pipeline(
        train_interactions,
        embedding_dim=64,
        save_path=MODELS_DIR / "embeddings_v3.pkl"
    )
    
    print(f"✓ Item embeddings: {len(item_trainer.item_embeddings)} items")
    print(f"✓ User embeddings: {len(user_generator.user_embeddings)} users")
    
    # ========================================================================
    # STEP 4: ULTRA-FAST NEGATIVE SAMPLING (OPTIMIZED!)
    # ========================================================================
    print("\n[STEP 4] ULTRA-FAST NEGATIVE SAMPLING (40 negatives)")
    print("-"*80)
    
    sampler = UltraFastNegativeSampler(n_negatives=40)
    sampler.fit(train_interactions, items_df)
    
    train_samples = sampler.sample_negatives(train_interactions, items_df)
    val_samples = sampler.sample_negatives(val_interactions, items_df)
    test_samples = sampler.sample_negatives(test_interactions, items_df)
    
    print(f"✓ Train samples: {len(train_samples):,}")
    
    # ========================================================================
    # STEP 5: FEATURE ENGINEERING (V2 + V3)
    # ========================================================================
    print("\n[STEP 5] FEATURE ENGINEERING (V2 + V3 EMBEDDINGS)")
    print("-"*80)
    
    # V2 feature engineering
    print("  V2 features...")
    v2_engineer = AdvancedFeatureEngineer()
    v2_engineer.fit(train_interactions, items_df, users_df)
    
    train_df = v2_engineer.engineer_features(
        train_samples, users_df, items_df, None  # Don't need carts_df
    )
    val_df = v2_engineer.engineer_features(
        val_samples, users_df, items_df, None
    )
    test_df = v2_engineer.engineer_features(
        test_samples, users_df, items_df, None
    )
    
    print(f"  ✓ V2 features: {train_df.shape[1]} columns")
    
    # V3 embedding features (NEW!)
    print("  V3 embedding features...")
    v3_engineer = EmbeddingFeatureEngineer(
        item_trainer.item_embeddings,
        user_generator.user_embeddings
    )
    
    train_df = v3_engineer.add_embedding_features(train_df)
    val_df = v3_engineer.add_embedding_features(val_df)
    test_df = v3_engineer.add_embedding_features(test_df)
    
    print(f"  ✓ V3 features added: {train_df.shape[1]} total columns")
    
    # Get feature columns
    feature_cols = v2_engineer.get_feature_names(train_df)
    
    print(f"✓ Total features: {len(feature_cols)}")
    
    # ========================================================================
    # STEP 6: TRAIN RANKING MODEL
    # ========================================================================
    print("\n[STEP 6] TRAINING LIGHTGBM RANKER (WITH V3 EMBEDDINGS)")
    print("-"*80)
    
    X_train = train_df[feature_cols]
    y_train = train_df['added_to_cart'].values
    group_col = 'order_id' if 'order_id' in train_df.columns else 'session_id'
    groups_train = train_df.groupby(group_col).size().values
    
    X_val = val_df[feature_cols]
    y_val = val_df['added_to_cart'].values
    groups_val = val_df.groupby(group_col).size().values
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {len(feature_cols)}")
    
    ranker = RankingModel()
    ranker.train(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val
    )
    
    # Save model
    model_path = MODELS_DIR / "ranker_v2_v3.pkl"
    ranker.save(model_path)
    
    print(f"✓ Model saved to {model_path}")
    
    # ========================================================================
    # STEP 7: EVALUATION
    # ========================================================================
    print("\n[STEP 7] EVALUATION")
    print("-"*80)
    
    evaluator = AdvancedEvaluator(k_values=[5, 10, 20])
    results = evaluator.evaluate(
        ranker, test_df, feature_cols,
        users_df, items_df
    )
    
    # ========================================================================
    # STEP 8: COMPARISON WITH V2
    # ========================================================================
    print("\n[STEP 8] COMPARISON")
    print("-"*80)
    
    ranking_metrics = results['ranking_metrics']
    business_metrics = results['business_metrics']
    
    print("\n  V2 Baseline (Expected):")
    print("    AUC: 0.70-0.75")
    print("    NDCG@10: 0.50-0.65")
    print("    Recall@10: 0.50-0.65")
    
    print("\n  V2+V3 Embeddings (Actual):")
    print(f"    AUC: {ranking_metrics['auc']:.4f}")
    print(f"    NDCG@10: {ranking_metrics['ndcg@10']:.4f}")
    print(f"    Recall@10: {ranking_metrics['recall@10']:.4f}")
    
    # Calculate improvement
    baseline_auc = 0.725  # midpoint of 0.70-0.75
    improvement_auc = (ranking_metrics['auc'] - baseline_auc) / baseline_auc * 100
    
    print(f"\n  Improvement:")
    print(f"    AUC: {improvement_auc:+.1f}%")
    
    # ========================================================================
    # STEP 9: SAVE RESULTS
    # ========================================================================
    print("\n[STEP 9] SAVING RESULTS")
    print("-"*80)
    
    report_path = MODELS_DIR / "evaluation_report_v2_v3.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("V2 + V3 EMBEDDINGS EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("RANKING METRICS:\n")
        f.write("-"*80 + "\n")
        for metric, value in ranking_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nBUSINESS METRICS:\n")
        f.write("-"*80 + "\n")
        for metric, value in business_metrics.items():
            if isinstance(value, float):
                if value < 1:
                    f.write(f"{metric}: {value:.2%}\n")
                else:
                    f.write(f"{metric}: {value:.2f}\n")
    
    print(f"✓ Report saved to {report_path}")
    
    # Plot feature importance
    ranker.plot_feature_importance(
        top_n=20,
        save_path=MODELS_DIR / "feature_importance_v2_v3.png"
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("[SUCCESS] V2 + V3 EMBEDDINGS TRAINING COMPLETE!")
    print("="*80)
    
    print("\n📊 KEY METRICS:")
    print(f"  AUC: {ranking_metrics['auc']:.4f}")
    print(f"  NDCG@10: {ranking_metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {ranking_metrics['recall@10']:.4f}")
    
    print("\n💼 BUSINESS IMPACT:")
    print(f"  Attach Rate: {business_metrics['attach_rate']:.2%}")
    print(f"  AOV Lift: {business_metrics['aov_lift']:.2%}")
    
    print("\n🎯 V3 FEATURES ADDED:")
    print("  ✓ Item embeddings (64-dim)")
    print("  ✓ User embeddings (aggregated)")
    print("  ✓ User-item similarity")
    print("  ✓ Embedding norms")
    
    print("\n📁 FILES SAVED:")
    print(f"  Embeddings: {MODELS_DIR / 'embeddings_v3.pkl'}")
    print(f"  Model: {model_path}")
    print(f"  Report: {report_path}")
    
    print("\n" + "="*80 + "\n")
    
    return ranker, evaluator, results


if __name__ == "__main__":
    ranker, evaluator, results = main()
