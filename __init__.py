"""
Custom Retinal Biomarkers Package

A comprehensive toolkit for extracting retinal vascular biomarkers from fundus images.
Provides transparent, validated implementations of geometric, fractal, and clinical features.
"""

__version__ = "1.0.0"
__author__ = "AIM Lab"

from .extractor import RetinalFeatureExtractor
from .geometry import GeometricalAnalysis
from .fractal import FractalAnalysis
from .cre import CREAnalysis
from .processing import average_by_sample, validate_features, compute_feature_correlations, merge_with_labels
from .statistics import analyze_significance

__all__ = [
    'RetinalFeatureExtractor',
    'GeometricalAnalysis',
    'FractalAnalysis',
    'CREAnalysis',
    'average_by_sample',
    'validate_features',
    'compute_feature_correlations',
    'merge_with_labels',
    'analyze_significance',
]
