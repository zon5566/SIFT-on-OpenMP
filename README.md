# SIFT-on-OpenMP

SIFT is a feature detection algorithm that detects interest points in multiple scales of the image and represent each key point with a relative orientation histogram which make this feature descriptor scale-invariant and also orientation-invariant. Most of the parts in this pipeline can be parallelized, which is also our main goal of the optimization.

In this project, we implemented a parallel version of the Scale-invariant Feature Transform (SIFT) [1] on OpenMP. The environment is Stempede2 from xsede.org.

## Algorithms

According to the original paper, the SIFT algorithm can be divided into several steps as follows:

1. Scale-Space Extrema Detection
  Build the image pyramid containing octaves with 
2. Accurate Keypoint Localization
3. Orientation Assignment
4. Keypoint Descriptor

## OpenMP Parallelization

First, we analyzed the serial running time and percentage in the whole algorithm on three different sizes of images.

| Stage | Small 700x542 | Medium 1850x1280 | Large 6000x4000 |
| --- | -----------: | -----------: | -----------: |
| initSigmaList | 5.0e-5 (0.0055%) | 5.7e-5 (0.0011%) | 5.2e-5 (0.0001%)
| initGaussian | 4.8e-4 (0.053%) | 6.6e-4 (0.012%) | 0.0013 (0.0026%) |
| **createPyramid** | **0.608 (67.2%)** | **3.37 (62.8%)** | **33.8 (67.8%)** |
| **createDoG** | **0.025 (2.7%)** | **0.082 (1.5%)** | **0.64 (1.29%)** |
| **getExtrema** | **0.16 (17.7%)** | **1.1 (19.7%)** | **10.4 (20.9%)** |
| rejectOutlier | 0.0058 (0.64%) | 0.050 (0.93%) | 0.38 (0.77%) | 
