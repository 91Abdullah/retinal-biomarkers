"""
Fractal Analysis Module for Retinal Vessels
"""

import numpy as np
from typing import Tuple
from scipy import stats
import skimage.transform as sktransform


class FractalAnalysis:
    """
    Custom implementation of multifractal dimension computation.
    Replaces PVBM's MultifractalVBMs class.
    """
    
    def __init__(self, n_dim: int = 10, n_rotations: int = 5, optimize: bool = True,
                 min_proba: float = 0.01, max_proba: float = 0.98):
        """
        Initialize fractal analyzer.
        
        Args:
            n_dim: Maximum dimension for singularity length
            n_rotations: Number of rotations for optimization
            optimize: Whether to optimize over rotations
            min_proba: Minimum box occupancy probability
            max_proba: Maximum box occupancy probability
        """
        self.n_dim = n_dim
        self.n_rotations = n_rotations
        self.optimize = optimize
        self.box_occupancy_proba = (min_proba, max_proba)
    
    def _probability_per_pixel(self, segmentation: np.ndarray, box_size: int, 
                                occupancy: Tuple[float, float]) -> np.ndarray:
        """
        Calculate box probabilities using box-counting method.
        Matches PVBM's implementation exactly.
        
        Args:
            segmentation: Binary image
            box_size: Size of boxes (k)
            occupancy: (min, max) probability thresholds
            
        Returns:
            Array of normalized probabilities for each box
        """
        h, w = segmentation.shape
        k = box_size
        
        # Calculate box sums using the same approach as PVBM
        # Sum along vertical axis first, then horizontal
        n_boxes_y = (h + k - 1) // k
        n_boxes_x = (w + k - 1) // k
        
        if n_boxes_y == 0 or n_boxes_x == 0:
            return np.array([])
        
        # Create matrix of box pixel counts
        M = np.zeros((n_boxes_y, n_boxes_x))
        
        for i in range(n_boxes_y):
            for j in range(n_boxes_x):
                y_start = i * k
                y_end = min((i + 1) * k, h)
                x_start = j * k
                x_end = min((j + 1) * k, w)
                M[i, j] = np.sum(segmentation[y_start:y_end, x_start:x_end])
        
        # Calculate probability per box (occupancy ratio)
        p = M / (k ** 2)
        
        # Filter by occupancy thresholds
        condition = (p >= occupancy[0]) & (p <= occupancy[1])
        
        # Get sum of pixel counts for valid boxes
        denom = np.sum(M[condition])
        if denom == 0:
            return np.array([])
        
        # Normalize by total pixel count in valid boxes (like PVBM)
        P = M[condition] / denom
        
        return P
    
    def _get_multifractal_dimension(self, segmentation: np.ndarray, q: float) -> Tuple[float, float, float, float]:
        """
        Calculate D_q fractal dimension for a given q.
        
        Args:
            segmentation: Binary image (must have min=0, max=1)
            q: q parameter for multifractal
            
        Returns:
            Tuple of (D_q, R^2, f_q, alpha_q)
        """
        # Validate input
        if len(segmentation.shape) != 2:
            return np.nan, np.nan, np.nan, np.nan
        if segmentation.max() != 1 or segmentation.min() != 0:
            # Try to fix by binarizing
            segmentation = (segmentation > 0.5).astype(float)
            if segmentation.max() != 1:
                return np.nan, np.nan, np.nan, np.nan
        
        # Minimal dimension
        p = min(segmentation.shape)
        
        # Linear sampling of box sizes
        epsilons = np.linspace(5, int(0.6 * p), 20).astype(np.int64)
        
        counts = []
        fq_numerators = []
        alpha_q_numerators = []
        used_epsilons = []
        
        for size in epsilons:
            P = self._probability_per_pixel(segmentation, size, self.box_occupancy_proba)
            
            if P.size == 0 or np.sum(P) == 0:
                P = self._probability_per_pixel(segmentation, size, (0.001, 1.0))
            
            if P.size == 0 or np.sum(P) == 0:
                continue
            
            p_positive = P[P > 0]
            if p_positive.size == 0:
                continue
            
            # Calculate I (partition function)
            I = np.sum(p_positive ** q)
            if I == 0:
                continue
            
            # Mu values (normalized probabilities)
            mu = (p_positive ** q) / I
            
            # f(q) numerator
            fq_numerator = np.sum(mu * np.log(mu + 1e-10))
            
            # alpha(q) numerator
            alpha_q_numerator = np.sum(mu * np.log(p_positive + 1e-10))
            
            # D_q calculation
            if q == 1:
                dq = -np.sum(p_positive * np.log(p_positive + 1e-10))
            else:
                dq = np.log(I + 1e-10) / (1 - q)
            
            counts.append(dq)
            fq_numerators.append(fq_numerator)
            alpha_q_numerators.append(alpha_q_numerator)
            used_epsilons.append(size)
        
        # Need at least 2 points for regression
        if len(counts) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        used_epsilons = np.array(used_epsilons)
        
        # Linear regression for D_q
        # D_q is the slope of dq values vs log(1/epsilon) = -log(epsilon)
        # For box-counting: N(epsilon) ~ epsilon^(-D), so log(N) ~ -D * log(epsilon)
        dq_slope, _, r_value, _, _ = stats.linregress(-np.log(used_epsilons), np.array(counts))
        dq_r2 = r_value ** 2
        
        # Linear regression for f(q) - uses positive log(epsilon) like PVBM
        fq_slope, _, _, _, _ = stats.linregress(np.log(used_epsilons), np.array(fq_numerators))
        
        # Linear regression for alpha(q) - uses positive log(epsilon) like PVBM
        alpha_slope, _, _, _, _ = stats.linregress(np.log(used_epsilons), np.array(alpha_q_numerators))
        
        return dq_slope, dq_r2, fq_slope, alpha_slope
    
    def compute_multifractals(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Compute multifractal biomarkers (D0, D1, D2, singularity length).
        
        Args:
            segmentation: Binary image (N x N with values 0.0 and 1.0)
            
        Returns:
            Array of [D0, D1, D2, singularity_length]
        """
        rotations = self.n_rotations if self.optimize else 1
        dim_list = [0, 1, 2, -self.n_dim, self.n_dim]
        
        angles = np.linspace(0, 360, rotations)
        
        all_results = []
        
        for angle in angles:
            # Rotate image
            if angle != 0:
                rotated = sktransform.rotate(segmentation, angle, resize=True, cval=0, mode='constant')
                # Re-binarize after rotation
                rotated = (rotated > 0.5).astype(np.float64)
            else:
                rotated = segmentation.astype(np.float64)
            
            # Calculate D_q for each dimension
            rotation_results = []
            for q in dim_list:
                dq_result = self._get_multifractal_dimension(rotated, q)
                rotation_results.append(dq_result)
            
            all_results.append(rotation_results)
        
        all_results = np.array(all_results)  # Shape: (rotations, len(dim_list), 4)
        
        if self.optimize and rotations > 1:
            # Choose best rotation based on D0, D1, D2
            first_three_dqs = all_results[:, :3, 0]  # D_q values for q=0,1,2
            
            # Find rotation with least NaN values and best R^2
            best_idx = 0
            best_score = -np.inf
            
            for i in range(rotations):
                valid = np.sum(np.isfinite(first_three_dqs[i]))
                r2_sum = np.nansum(all_results[i, :3, 1])  # Sum of R^2 values
                score = valid * 10 + r2_sum
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            best_results = all_results[best_idx]
        else:
            best_results = all_results[0]
        
        # Extract D0, D1, D2
        D0 = best_results[0, 0] if np.isfinite(best_results[0, 0]) else np.nan
        D1 = best_results[1, 0] if np.isfinite(best_results[1, 0]) else np.nan
        D2 = best_results[2, 0] if np.isfinite(best_results[2, 0]) else np.nan
        
        # Singularity length from alpha values at q=-n_dim and q=+n_dim
        alpha_neg = best_results[3, 3]  # alpha at q=-n_dim
        alpha_pos = best_results[4, 3]  # alpha at q=+n_dim
        singularity_length = abs(alpha_neg - alpha_pos) if np.isfinite(alpha_neg) and np.isfinite(alpha_pos) else np.nan
        
        return np.array([D0, D1, D2, singularity_length])


# =============================================================================
# Custom CRE (Central Retinal Equivalent) Analysis
# =============================================================================

