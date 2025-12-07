# Changelog

## [1.0.0] - 2025-12-07

### Added
- Initial release of retinal_biomarkers package
- GeometricalAnalysis class for vessel geometry metrics
- FractalAnalysis class for multifractal dimensions
- CREAnalysis class for CRAE/CRVE calculation
- RetinalFeatureExtractor main interface
- Statistical analysis with FDR correction
- Sample-level averaging utilities
- Validation framework against PVBM

### Features
- Extracts 35 retinal biomarkers from vessel segmentations
- Enhanced branching angle calculation with recursive 20-pixel walk
- Multi-rotation fractal optimization (r=0.94-0.96 validation)
- Zone-based CRE calculation with Knudtson & Hubbard formulas
- FDR correction (Benjamini-Hochberg) for multiple testing
- Cohen's d effect size calculation
