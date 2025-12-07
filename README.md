# Retinal Biomarkers

A Python package for extracting retinal vascular biomarkers from fundus images.

## Features

- **Geometric Analysis**: Area, perimeter, tortuosity, vessel length, branching angles
- **Fractal Analysis**: Multifractal dimensions (D₀, D₁, D₂)
- **Clinical Metrics**: CRAE/CRVE with Knudtson & Hubbard formulas
- **Statistical Analysis**: FDR-corrected significance testing with effect sizes

## Installation

```bash
pip install retinal-biomarkers
```

## Quick Start

```python
from retinal_biomarkers import RetinalFeatureExtractor

# Initialize extractor
extractor = RetinalFeatureExtractor()

# Extract features from vessel segmentations
features = extractor.extract_all_features(
    artery_path='path/to/artery_segmentation.png',
    vein_path='path/to/vein_segmentation.png',
    original_image_path='path/to/fundus_image.jpg'
)
```

## Features Extracted

The package extracts 35 retinal biomarkers:

- **Disc Features (3)**: center coordinates, radius
- **Artery Features (15)**: geometric, fractal, and clinical metrics
- **Vein Features (15)**: geometric, fractal, and clinical metrics  
- **Derived Features (2)**: artery-vein ratios

## Author

Abdullah Basit (abdullah.basit@hotmail.com)

## License

MIT
