"""
Central Retinal Equivalent (CRE) Analysis Module
"""

import numpy as np
from typing import Dict, List
from skimage.morphology import skeletonize
from scipy import ndimage


class CREAnalysis:
    """
    Custom implementation of CRAE/CRVE computation.
    Replaces PVBM's CREVBMs class.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def crae_hubbard(w1: float, w2: float) -> float:
        """Hubbard formula for CRAE."""
        val = 0.87 * w1**2 + 1.01 * w2**2 - 0.22 * w1 * w2 - 10.76
        return np.sqrt(max(0, val))
    
    @staticmethod
    def crve_hubbard(w1: float, w2: float) -> float:
        """Hubbard formula for CRVE."""
        val = 0.72 * w1**2 + 0.91 * w2**2 + 450.05
        return np.sqrt(max(0, val))
    
    @staticmethod
    def crae_knudtson(w1: float, w2: float) -> float:
        """Knudtson formula for CRAE."""
        return 0.88 * np.sqrt(w1**2 + w2**2)
    
    @staticmethod
    def crve_knudtson(w1: float, w2: float) -> float:
        """Knudtson formula for CRVE."""
        return 0.95 * np.sqrt(w1**2 + w2**2)
    
    def _central_equivalent_recursive(self, widths: List[float], formula) -> float:
        """
        Recursively compute central retinal equivalent.
        
        Args:
            widths: List of vessel widths
            formula: Formula function (crae/crve_hubbard or knudtson)
            
        Returns:
            Central equivalent value
        """
        # Filter invalid values
        widths = np.array(widths, dtype=float)
        widths = widths[np.isfinite(widths) & (widths > 0)]
        widths = np.sort(widths)
        
        if len(widths) == 0:
            return np.nan
        if len(widths) == 1:
            return widths[0]
        
        # Handle odd number of vessels
        pivot = None
        if len(widths) % 2 != 0:
            idx_pivot = len(widths) // 2
            pivot = widths[idx_pivot]
        
        # Pair smallest with largest
        new_widths = []
        for i in range(len(widths) // 2):
            try:
                val = formula(widths[i], widths[-(i+1)])
                if np.isfinite(val) and val > 0:
                    new_widths.append(val)
            except:
                continue
        
        if pivot is not None and np.isfinite(pivot) and pivot > 0:
            new_widths.append(pivot)
        
        if len(new_widths) == 0:
            return pivot if pivot is not None else np.nan
        
        return self._central_equivalent_recursive(new_widths, formula)
    
    def _measure_vessel_widths_in_zone(self, segmentation: np.ndarray, skeleton: np.ndarray,
                                        center_x: int, center_y: int, 
                                        inner_radius: float, outer_radius: float) -> List[float]:
        """
        Measure vessel widths in an annular zone around the optic disc.
        
        The zone is typically 0.5-1.0 disc diameters from disc edge (Zone B).
        
        Args:
            segmentation: Binary vessel segmentation
            skeleton: Vessel skeleton
            center_x, center_y: Optic disc center
            inner_radius: Inner radius of measurement zone
            outer_radius: Outer radius of measurement zone
            
        Returns:
            List of vessel width measurements
        """
        h, w = segmentation.shape
        
        # Create zone mask
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        zone_mask = (dist >= inner_radius) & (dist <= outer_radius)
        
        # Get skeleton points in zone
        skeleton_in_zone = skeleton * zone_mask
        
        # Label connected components in skeleton
        from scipy import ndimage
        labeled, n_components = ndimage.label(skeleton_in_zone)
        
        widths = []
        
        for i in range(1, n_components + 1):
            component_mask = labeled == i
            
            # Get skeleton points for this component
            skel_points = np.where(component_mask)
            if len(skel_points[0]) < 3:
                continue
            
            # Measure width at multiple points along the skeleton
            component_widths = []
            
            for idx in range(0, len(skel_points[0]), max(1, len(skel_points[0]) // 5)):
                py, px = skel_points[0][idx], skel_points[1][idx]
                
                # Measure width perpendicular to skeleton direction
                width = self._measure_width_at_point(segmentation, py, px)
                if width > 0:
                    component_widths.append(width)
            
            if component_widths:
                # Use median width for this vessel
                widths.append(np.median(component_widths))
        
        return widths
    
    def _measure_width_at_point(self, segmentation: np.ndarray, y: int, x: int, max_width: int = 50) -> float:
        """
        Measure vessel width at a given skeleton point.
        
        Args:
            segmentation: Binary vessel segmentation
            y, x: Point coordinates
            max_width: Maximum search distance
            
        Returns:
            Vessel width at this point
        """
        h, w = segmentation.shape
        
        # Check multiple directions to find perpendicular to vessel
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        widths = []
        
        for dy, dx in directions:
            # Count pixels in positive direction
            count_pos = 0
            for step in range(1, max_width):
                ny, nx = y + step * dy, x + step * dx
                if 0 <= ny < h and 0 <= nx < w:
                    if segmentation[ny, nx] > 0:
                        count_pos += 1
                    else:
                        break
                else:
                    break
            
            # Count pixels in negative direction
            count_neg = 0
            for step in range(1, max_width):
                ny, nx = y - step * dy, x - step * dx
                if 0 <= ny < h and 0 <= nx < w:
                    if segmentation[ny, nx] > 0:
                        count_neg += 1
                    else:
                        break
                else:
                    break
            
            total_width = count_pos + count_neg + 1  # +1 for center point
            widths.append(total_width)
        
        # Return minimum (perpendicular direction should give smallest width)
        return min(widths) if widths else 0
    
    def compute_central_retinal_equivalents(self, segmentation: np.ndarray, skeleton: np.ndarray,
                                             center_x: int, center_y: int, radius: int,
                                             is_artery: bool = True) -> Dict[str, float]:
        """
        Compute CRAE or CRVE.
        
        Args:
            segmentation: Binary vessel segmentation
            skeleton: Vessel skeleton
            center_x, center_y: Optic disc center
            radius: Optic disc radius
            is_artery: True for CRAE, False for CRVE
            
        Returns:
            Dictionary with knudtson and hubbard values
        """
        # Measurement zone: 0.5 to 1.0 disc diameters from disc edge
        # This corresponds to Zone B in the ARIC study protocol
        inner_radius = radius + 0.5 * radius  # 0.5 disc diameters from edge
        outer_radius = radius + 1.5 * radius  # 1.0 disc diameters from edge
        
        # Measure vessel widths in zone
        widths = self._measure_vessel_widths_in_zone(
            segmentation, skeleton, center_x, center_y, inner_radius, outer_radius
        )
        
        if len(widths) == 0:
            prefix = 'crae' if is_artery else 'crve'
            return {f'{prefix}_knudtson': np.nan, f'{prefix}_hubbard': np.nan}
        
        # Use largest 6 vessels
        widths = sorted(widths, reverse=True)[:6]
        
        if is_artery:
            crae_k = self._central_equivalent_recursive(widths.copy(), self.crae_knudtson)
            crae_h = self._central_equivalent_recursive(widths.copy(), self.crae_hubbard)
            return {'crae_knudtson': crae_k, 'crae_hubbard': crae_h}
        else:
            crve_k = self._central_equivalent_recursive(widths.copy(), self.crve_knudtson)
            crve_h = self._central_equivalent_recursive(widths.copy(), self.crve_hubbard)
            return {'crve_knudtson': crve_k, 'crve_hubbard': crve_h}


# =============================================================================
# Main Feature Extractor Class
# =============================================================================

