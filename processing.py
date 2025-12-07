#!/usr/bin/env python3
"""
Post-process Retinal Features: Sample-level Averaging and Validation
======================================================================

This script performs:
1. Sample-level averaging: Multiple images per SampleID → 1 row per sample
2. Validation: Compare PVBM vs Custom implementations on random images
3. Merge with labels for significance testing
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


def average_by_sample(df: pd.DataFrame, id_col: str = 'SampleID') -> pd.DataFrame:
    """
    Average features across multiple images per sample.
    
    Args:
        df: DataFrame with image-level features
        id_col: Column name for sample identifier
        
    Returns:
        DataFrame with one row per sample (averaged features)
    """
    feature_cols = [col for col in df.columns if col not in [id_col, 'image_id']]
    
    # Group by SampleID and take mean of numeric columns
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['image_id'] = 'count'  # Count number of images per sample
    
    df_avg = df.groupby(id_col).agg(agg_dict).reset_index()
    df_avg = df_avg.rename(columns={'image_id': 'n_images'})
    
    return df_avg


def validate_features(pvbm_df: pd.DataFrame, custom_df: pd.DataFrame, 
                      n_samples: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Compare PVBM and Custom features on random sample of images.
    
    Args:
        pvbm_df: DataFrame with PVBM features
        custom_df: DataFrame with Custom features
        n_samples: Number of random images to validate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with validation results
    """
    np.random.seed(random_seed)
    
    # Find common images
    common_images = list(set(pvbm_df['image_id']) & set(custom_df['image_id']))
    
    if len(common_images) == 0:
        print("No common images found!")
        return pd.DataFrame()
    
    # Sample random images
    sample_images = np.random.choice(common_images, size=min(n_samples, len(common_images)), replace=False)
    
    print(f"\n{'='*80}")
    print(f"VALIDATION: Comparing PVBM vs Custom on {len(sample_images)} random images")
    print(f"{'='*80}")
    
    # Get feature columns (exclude ID columns)
    feature_cols = [col for col in pvbm_df.columns if col not in ['SampleID', 'image_id']]
    
    validation_results = []
    
    for img_id in sample_images:
        print(f"\n{'='*60}")
        print(f"Image: {img_id}")
        print(f"{'='*60}")
        
        pvbm_row = pvbm_df[pvbm_df['image_id'] == img_id].iloc[0]
        custom_row = custom_df[custom_df['image_id'] == img_id].iloc[0]
        
        for col in feature_cols:
            pvbm_val = pvbm_row[col]
            custom_val = custom_row[col]
            
            if pd.isna(pvbm_val) and pd.isna(custom_val):
                match_status = "Both NaN"
                diff = np.nan
            elif pd.isna(pvbm_val) or pd.isna(custom_val):
                match_status = "One NaN"
                diff = np.nan
            else:
                diff = abs(pvbm_val - custom_val)
                rel_diff = diff / (abs(pvbm_val) + 1e-10) * 100
                if rel_diff < 5:
                    match_status = "Match (<5%)"
                elif rel_diff < 20:
                    match_status = "Close (5-20%)"
                else:
                    match_status = f"Diff ({rel_diff:.1f}%)"
            
            validation_results.append({
                'image_id': img_id,
                'feature': col,
                'pvbm_value': pvbm_val,
                'custom_value': custom_val,
                'abs_diff': diff,
                'status': match_status
            })
            
            # Print only divergent features
            if match_status not in ["Both NaN", "Match (<5%)"]:
                print(f"  {col}: PVBM={pvbm_val:.4f}, Custom={custom_val:.4f} [{match_status}]")
    
    return pd.DataFrame(validation_results)


def compute_feature_correlations(pvbm_df: pd.DataFrame, custom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation between PVBM and Custom for each feature.
    """
    # Merge on image_id
    merged = pvbm_df.merge(custom_df, on='image_id', suffixes=('_pvbm', '_custom'))
    
    feature_cols = [col for col in pvbm_df.columns if col not in ['SampleID', 'image_id']]
    
    results = []
    for col in feature_cols:
        pvbm_col = f"{col}_pvbm"
        custom_col = f"{col}_custom"
        
        if pvbm_col in merged.columns and custom_col in merged.columns:
            valid = merged[[pvbm_col, custom_col]].dropna()
            
            if len(valid) >= 10:
                r, p = pearsonr(valid[pvbm_col], valid[custom_col])
                results.append({
                    'feature': col,
                    'pearson_r': r,
                    'p_value': p,
                    'n_valid': len(valid),
                    'pvbm_mean': valid[pvbm_col].mean(),
                    'custom_mean': valid[custom_col].mean(),
                    'mean_diff': valid[custom_col].mean() - valid[pvbm_col].mean()
                })
            else:
                results.append({
                    'feature': col,
                    'pearson_r': np.nan,
                    'p_value': np.nan,
                    'n_valid': len(valid),
                    'pvbm_mean': np.nan,
                    'custom_mean': np.nan,
                    'mean_diff': np.nan
                })
    
    return pd.DataFrame(results)


def merge_with_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame,
                       features_id_col: str = 'SampleID',
                       labels_id_col: str = 'Project_Dummy_ID') -> pd.DataFrame:
    """
    Merge features with labels (Case/Control).
    """
    # Rename ID column for merge
    labels_temp = labels_df.rename(columns={labels_id_col: features_id_col})
    
    # Merge
    merged = features_df.merge(labels_temp, on=features_id_col, how='left')
    
    return merged

