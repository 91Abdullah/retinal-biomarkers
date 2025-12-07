"""
Main Feature Extractor for Retinal Biomarkers
"""

import os
import sys
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from typing import Dict, Tuple

# Add PVBM to path (only for DiscSegmenter)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PVBM'))
from PVBM.DiscSegmenter import DiscSegmenter

# Import our analysis classes
from .geometry import GeometricalAnalysis
from .fractal import FractalAnalysis
from .cre import CREAnalysis

class RetinalFeatureExtractor:

    """

    Extract comprehensive retinal features from vessel segmentations.

    Uses custom implementations except for disc detection (PVBM DiscSegmenter).

    """

    

    def __init__(self, use_disc_segmentation: bool = True, prob_threshold: float = 0.5):

        """

        Initialize feature extractor.

        

        Args:

            use_disc_segmentation: Whether to use PVBM DiscSegmenter

            prob_threshold: Threshold to binarize probability maps

        """

        self.prob_threshold = prob_threshold

        self.use_disc_segmentation = use_disc_segmentation

        

        # Initialize custom analyzers

        self.geom_analyzer = CustomGeometricalAnalysis()

        self.fractal_analyzer = CustomFractalAnalysis(n_dim=10, n_rotations=5, optimize=True)

        self.cre_analyzer = CustomCREAnalysis()

        

        # Initialize PVBM disc segmenter (only PVBM component used)

        if use_disc_segmentation:

            try:

                self.disc_segmenter = DiscSegmenter()

            except Exception as e:

                print(f"Warning: Could not initialize DiscSegmenter: {e}")

                self.disc_segmenter = None

        else:

            self.disc_segmenter = None

    

    def load_segmentation(self, seg_path: str) -> np.ndarray:

        """Load and binarize segmentation image."""

        if not os.path.exists(seg_path):

            return None

        

        img = np.array(Image.open(seg_path).convert('L'))

        

        if img.max() > 1:

            img = img / 255.0

        

        binary = (img > self.prob_threshold).astype(np.uint8)

        return binary

    

    def detect_disc_from_vessels(self, artery_seg: np.ndarray, vein_seg: np.ndarray) -> Tuple[int, int, int]:

        """Estimate disc center from vessel convergence (fallback method)."""

        combined = np.maximum(artery_seg, vein_seg)

        h, w = combined.shape

        

        center_x = w // 2

        center_y = h // 2

        estimated_radius = min(w, h) // 6

        

        return center_x, center_y, estimated_radius

    

    def segment_disc(self, image_path: str) -> Tuple[int, int, int]:

        """Segment optic disc using PVBM DiscSegmenter."""

        if self.disc_segmenter is None:

            return None, None, None

        

        try:

            disc_seg = self.disc_segmenter.segment(image_path)

            disc_mask = np.array(disc_seg)

            

            if disc_mask.max() > 0:

                coords = np.where(disc_mask > 0)

                if len(coords[0]) > 0:

                    center_y = int(np.mean(coords[0]))

                    center_x = int(np.mean(coords[1]))

                    disc_area = np.sum(disc_mask > 0)

                    radius = int(np.sqrt(disc_area / np.pi))

                    return center_x, center_y, radius

        except Exception as e:

            pass

        

        return None, None, None

    

    def extract_geometrical_features(self, binary_seg: np.ndarray) -> Dict[str, float]:

        """Extract geometrical features using custom implementation."""

        if binary_seg is None or binary_seg.sum() == 0:

            return {

                'area': np.nan, 'perimeter': np.nan, 'tortuosity': np.nan,

                'length': np.nan, 'endpoints': np.nan, 'intersections': np.nan,

                'branching_angle_mean': np.nan, 'branching_angle_std': np.nan,

                'branching_angle_median': np.nan, 'vessel_density': np.nan,

            }

        

        try:

            skeleton = skeletonize(binary_seg).astype(np.uint8)

            

            features = {}

            features['area'] = self.geom_analyzer.compute_area(binary_seg)

            

            try:

                perim, _ = self.geom_analyzer.compute_perimeter(binary_seg)

                features['perimeter'] = float(perim)

            except:

                features['perimeter'] = np.nan

            

            try:

                median_tor, length, _, _, _ = self.geom_analyzer.compute_tortuosity_length(skeleton)

                features['tortuosity'] = float(median_tor) if not np.isnan(median_tor) else np.nan

                features['length'] = float(length) if not np.isnan(length) else np.nan

            except:

                features['tortuosity'] = np.nan

                features['length'] = np.nan

            

            try:

                endpoints, intersections, _, _ = self.geom_analyzer.compute_particular_points(skeleton)

                features['endpoints'] = float(endpoints)

                features['intersections'] = float(intersections)

            except:

                features['endpoints'] = np.nan

                features['intersections'] = np.nan

            

            try:

                mean_ba, std_ba, median_ba, _, _ = self.geom_analyzer.compute_branching_angles(skeleton)

                features['branching_angle_mean'] = float(mean_ba) if not np.isnan(mean_ba) else np.nan

                features['branching_angle_std'] = float(std_ba) if not np.isnan(std_ba) else np.nan

                features['branching_angle_median'] = float(median_ba) if not np.isnan(median_ba) else np.nan

            except:

                features['branching_angle_mean'] = np.nan

                features['branching_angle_std'] = np.nan

                features['branching_angle_median'] = np.nan

            

            total_pixels = binary_seg.shape[0] * binary_seg.shape[1]

            features['vessel_density'] = float(features['area'] / total_pixels) if total_pixels > 0 else np.nan

            

        except Exception as e:

            return {

                'area': np.nan, 'perimeter': np.nan, 'tortuosity': np.nan,

                'length': np.nan, 'endpoints': np.nan, 'intersections': np.nan,

                'branching_angle_mean': np.nan, 'branching_angle_std': np.nan,

                'branching_angle_median': np.nan, 'vessel_density': np.nan,

            }

        

        return features

    

    def extract_fractal_features(self, binary_seg: np.ndarray) -> Dict[str, float]:

        """Extract multifractal features using custom implementation."""

        if binary_seg is None or binary_seg.sum() == 0:

            return {'D0': np.nan, 'D1': np.nan, 'D2': np.nan}

        

        try:

            vessel_ratio = binary_seg.sum() / (binary_seg.shape[0] * binary_seg.shape[1])

            if vessel_ratio < 0.001:

                return {'D0': np.nan, 'D1': np.nan, 'D2': np.nan}

            

            target_size = 512

            resized = cv2.resize(binary_seg.astype(np.uint8), (target_size, target_size),

                                interpolation=cv2.INTER_NEAREST)

            resized = (resized > 0).astype(np.float64)

            

            if resized.sum() < 100:

                return {'D0': np.nan, 'D1': np.nan, 'D2': np.nan}

            

            mf_features = self.fractal_analyzer.compute_multifractals(resized)

            

            if mf_features is None or len(mf_features) < 3:

                return {'D0': np.nan, 'D1': np.nan, 'D2': np.nan}

            

            return {

                'D0': float(mf_features[0]) if np.isfinite(mf_features[0]) else np.nan,

                'D1': float(mf_features[1]) if np.isfinite(mf_features[1]) else np.nan,

                'D2': float(mf_features[2]) if np.isfinite(mf_features[2]) else np.nan,

            }

        except:

            return {'D0': np.nan, 'D1': np.nan, 'D2': np.nan}

    

    def extract_cre_features(self, binary_seg: np.ndarray, center_x: int, center_y: int,

                             radius: int, is_artery: bool = True) -> Dict[str, float]:

        """Extract CRE features using custom implementation."""

        prefix = 'crae' if is_artery else 'crve'

        

        if binary_seg is None or binary_seg.sum() == 0:

            return {f'{prefix}_knudtson': np.nan, f'{prefix}_hubbard': np.nan}

        

        if center_x is None or center_y is None or radius is None or radius <= 0:

            return {f'{prefix}_knudtson': np.nan, f'{prefix}_hubbard': np.nan}

        

        try:

            skeleton = skeletonize(binary_seg).astype(np.uint8)

            result = self.cre_analyzer.compute_central_retinal_equivalents(

                binary_seg.astype(float), skeleton.astype(float),

                int(center_x), int(center_y), int(radius), is_artery

            )

            return result

        except:

            return {f'{prefix}_knudtson': np.nan, f'{prefix}_hubbard': np.nan}

    

    def extract_all_features(self, artery_path: str, vein_path: str,

                             original_image_path: str = None) -> Dict[str, float]:

        """Extract all 33 features from artery and vein segmentations."""

        features = {}

        

        # Load segmentations

        artery_seg = self.load_segmentation(artery_path) if artery_path else None

        vein_seg = self.load_segmentation(vein_path) if vein_path else None

        

        # Get disc information

        center_x, center_y, radius = None, None, None

        

        if original_image_path and self.disc_segmenter:

            center_x, center_y, radius = self.segment_disc(original_image_path)

        

        if center_x is None and (artery_seg is not None or vein_seg is not None):

            artery_for_disc = artery_seg if artery_seg is not None else np.zeros((100, 100))

            vein_for_disc = vein_seg if vein_seg is not None else np.zeros((100, 100))

            center_x, center_y, radius = self.detect_disc_from_vessels(artery_for_disc, vein_for_disc)

        

        features['disc_center_x'] = float(center_x) if center_x is not None else np.nan

        features['disc_center_y'] = float(center_y) if center_y is not None else np.nan

        features['disc_radius'] = float(radius) if radius is not None else np.nan

        

        # Artery geometrical features

        artery_geom = self.extract_geometrical_features(artery_seg)

        for key, value in artery_geom.items():

            features[f'artery_{key}'] = value

        

        # Artery CRE features

        artery_cre = self.extract_cre_features(artery_seg, center_x, center_y, radius, is_artery=True)

        features.update(artery_cre)

        

        # Artery fractal features

        artery_fractal = self.extract_fractal_features(artery_seg)

        for key, value in artery_fractal.items():

            features[f'artery_{key}'] = value

        

        # Vein geometrical features

        vein_geom = self.extract_geometrical_features(vein_seg)

        for key, value in vein_geom.items():

            features[f'vein_{key}'] = value

        

        # Vein CRE features

        vein_cre = self.extract_cre_features(vein_seg, center_x, center_y, radius, is_artery=False)

        features.update(vein_cre)

        

        # Vein fractal features

        vein_fractal = self.extract_fractal_features(vein_seg)

        for key, value in vein_fractal.items():

            features[f'vein_{key}'] = value

        

        # Derived features: AVR

        crae_k = features.get('crae_knudtson', np.nan)

        crve_k = features.get('crve_knudtson', np.nan)

        crae_h = features.get('crae_hubbard', np.nan)

        crve_h = features.get('crve_hubbard', np.nan)

        

        if not np.isnan(crae_k) and not np.isnan(crve_k) and crve_k > 0:

            features['avr_knudtson'] = crae_k / crve_k

        else:

            features['avr_knudtson'] = np.nan

        

        if not np.isnan(crae_h) and not np.isnan(crve_h) and crve_h > 0:

            features['avr_hubbard'] = crae_h / crve_h

        else:

            features['avr_hubbard'] = np.nan

        

        return features





def extract_sample_id(image_name: str) -> str:

    """Extract SampleID from image name (first part before underscore)."""

    base_name = os.path.splitext(image_name)[0]

    parts = base_name.split('_')

    return parts[0] if parts else base_name


