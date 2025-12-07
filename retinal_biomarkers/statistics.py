"""
Statistical Analysis Module for Retinal Features
=================================================

Performs statistical significance testing between case and control groups.
Includes proper multiple testing correction (FDR) and effect size calculation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
from statsmodels.stats.multitest import multipletests


def analyze_significance(features_df: pd.DataFrame,
                         group_col: str = 'Case/Control',
                         control_label: str = 'Control',
                         case_labels: Optional[list] = None,
                         alpha: float = 0.05,
                         fdr_method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Analyze statistical significance of features between case and control groups.
    
    Args:
        features_df: DataFrame with features and group labels
        group_col: Column name containing group labels
        control_label: Label for control group
        case_labels: List of labels to group as 'case' (default: all non-Control)
        alpha: Significance level for FDR correction
        fdr_method: Method for FDR correction ('fdr_bh' = Benjamini-Hochberg)
        
    Returns:
        DataFrame with test results including p-values, FDR-adjusted p-values, and effect sizes
    """
    # Identify feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['SampleID', 'image_id', 'n_images', group_col, 
                                  'Project_Dummy_ID', 'Study_Dummy_ID', 'Case/Control_y']]
    
    # Filter numeric columns only
    feature_cols = [col for col in feature_cols 
                   if pd.api.types.is_numeric_dtype(features_df[col])]
    
    # Group data
    if case_labels is None:
        # Default: all non-Control are Case
        df_control = features_df[features_df[group_col] == control_label]
        df_case = features_df[(features_df[group_col].notna()) & 
                              (features_df[group_col] != control_label)]
    else:
        df_control = features_df[features_df[group_col] == control_label]
        df_case = features_df[features_df[group_col].isin(case_labels)]
    
    print(f"Control group: {len(df_control)} samples")
    print(f"Case group: {len(df_case)} samples")
    print(f"Testing {len(feature_cols)} features...")
    
    results = []
    p_values = []
    
    for feature in feature_cols:
        control_vals = df_control[feature].dropna()
        case_vals = df_case[feature].dropna()
        
        if len(control_vals) < 3 or len(case_vals) < 3:
            # Not enough data
            results.append({
                'feature': feature,
                'control_n': len(control_vals),
                'case_n': len(case_vals),
                'control_mean': np.nan,
                'case_mean': np.nan,
                'control_std': np.nan,
                'case_std': np.nan,
                'mean_diff': np.nan,
                'p_value_mw': np.nan,
                'p_value_ttest': np.nan,
                'cohens_d': np.nan
            })
            p_values.append(np.nan)
            continue
        
        # Compute statistics
        control_mean = control_vals.mean()
        control_std = control_vals.std()
        case_mean = case_vals.mean()
        case_std = case_vals.std()
        mean_diff = case_mean - control_mean
        
        # Mann-Whitney U test (non-parametric)
        try:
            stat_mw, p_mw = stats.mannwhitneyu(control_vals, case_vals, alternative='two-sided')
        except:
            p_mw = np.nan
        
        # Independent t-test (parametric)
        try:
            stat_t, p_t = stats.ttest_ind(control_vals, case_vals)
        except:
            p_t = np.nan
        
        # Cohen's d (effect size)
        try:
            pooled_std = np.sqrt(((len(control_vals) - 1) * control_std**2 + 
                                  (len(case_vals) - 1) * case_std**2) / 
                                 (len(control_vals) + len(case_vals) - 2))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
        except:
            cohens_d = np.nan
        
        results.append({
            'feature': feature,
            'control_n': len(control_vals),
            'case_n': len(case_vals),
            'control_mean': control_mean,
            'case_mean': case_mean,
            'control_std': control_std,
            'case_std': case_std,
            'mean_diff': mean_diff,
            'p_value_mw': p_mw,
            'p_value_ttest': p_t,
            'cohens_d': cohens_d
        })
        p_values.append(p_mw)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # FDR correction (use Mann-Whitney p-values)
    valid_p = np.array([p if not np.isnan(p) else 1.0 for p in p_values])
    
    try:
        reject, p_adjusted, _, _ = multipletests(valid_p, alpha=alpha, method=fdr_method)
        results_df['p_adj'] = p_adjusted
        results_df['significant_fdr'] = reject
    except:
        results_df['p_adj'] = np.nan
        results_df['significant_fdr'] = False
    
    # Sort by adjusted p-value
    results_df = results_df.sort_values('p_adj')
    
    return results_df


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1: Array of values for group 1
        group2: Array of values for group 2
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Effect size
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    d_abs = abs(d)
    
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def format_p_value(p: float) -> str:
    """
    Format p-value for display.
    
    Args:
        p: P-value
        
    Returns:
        Formatted string
    """
    if np.isnan(p):
        return "N/A"
    elif p < 0.001:
        return f"{p:.4f}***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def print_significance_summary(results_df: pd.DataFrame, alpha: float = 0.05):
    """
    Print a formatted summary of significance results.
    
    Args:
        results_df: DataFrame from analyze_significance()
        alpha: Significance threshold
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY")
    print("="*80)
    
    total_features = len(results_df)
    significant = results_df[results_df['p_adj'] < alpha]
    
    print(f"\nTotal features tested: {total_features}")
    print(f"Significant features (FDR-adjusted p < {alpha}): {len(significant)}")
    
    if len(significant) > 0:
        cohens_d_header = "Cohen's d"
        print(f"\n{'Feature':<40} {'p_adj':<12} {cohens_d_header:<10} {'Effect'}")
        print("-"*80)
        
        for _, row in significant.iterrows():
            effect = interpret_effect_size(row['cohens_d'])
            sig_marker = "***" if row['p_adj'] < 0.001 else "**" if row['p_adj'] < 0.01 else "*"
            print(f"{row['feature']:<40} {row['p_adj']:<12.4f}{sig_marker} "
                  f"{row['cohens_d']:>+8.3f}  {effect}")
    
    print("="*80)
