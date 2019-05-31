# SIFT-on-OpenMP

SIFT is a feature detection algorithm that detects interest points in multiple scales of the image and represent each key point with a relative orientation histogram which make this feature descriptor scale-invariant and also orientation-invariant. Most of the parts in this pipeline can be parallelized, which is also our main goal of the optimization.

In this project, we implemented a parallel version of the [Scale-invariant Feature Transform](http://new.csd.uwo.ca/Courses/CS9840a/PossibleStudentPapers/iccv99.pdf) (SIFT) on OpenMP. The environment is Stempede2 from xsede.org.

For more detailed works and analysis, please visit our [writing report](https://drive.google.com/file/d/1M_80JQCrcZF60oKN9WuyB4ZGpqj5yuvs/view?usp=sharing).

## Execution

To run the program, run ```make``` to compile (using icpc) and generate the executable file ```sift```, which accepts two arguments:

```./sift <input image path> <#threads>```

## OpenMP Parallelization

First, we analyzed the serial running time (in second) and percentage in the whole algorithm on three different sizes of images.

| Stage | Small 700x542 | Medium 1850x1280 | Large 6000x4000 |
| --- | -----------: | -----------: | -----------: |
| initSigmaList | 5.0e-5 (0.0055%) | 5.7e-5 (0.0011%) | 5.2e-5 (0.0001%)
| initGaussian | 4.8e-4 (0.053%) | 6.6e-4 (0.012%) | 0.0013 (0.0026%) |
| **createPyramid** | **0.608 (67.2%)** | **3.37 (62.8%)** | **33.8 (67.8%)** |
| **createDoG** | **0.025 (2.72%)** | **0.082 (1.5%)** | **0.64 (1.29%)** |
| **getExtrema** | **0.16 (17.7%)** | **1.1 (19.7%)** | **10.4 (20.9%)** |
| rejectOutlier | 0.0058 (0.64%) | 0.050 (0.93%) | 0.38 (0.77%) | 
| **getOrientation** | **0.064 (7.11%)** | **0.73 (13.7%)** | **4.41 (8.85%)** | 
| **getDescriptor** | **0.042 (4.61%)** | **0.072 (1.35%)** | **0.19 (0.39%)** | 

The bold rows are the ones we decided to parallelize, and we could see for all sizes of the image, they all have a total > 99% of the whole running time. Therefore, parallelizing them is clearly easy to justify. We list the order of these 5 stages below:

* createPyramid(): Build the Scale-space octaves which each image within each of it is convolved with given specific sigma.

* getExtrema(): From Difference of Gaussain images, extract points that comply to the maximum/minimum of 3x3x3 cube rule.

* getOrientation(): For each keypoint location, calculate main orientation & magnitude (could be more than one) and bundle this information to build a new keypoint + orientation vector.

* creatDoG(): For each Scale-space octaves, subtract all adjacent levels of image to form a Difference of Gaussain pyramid.

* getDescriptor(): For each keypoint, correct features to 0 degree of orientation, and compute a 16*16 local region as 4x4 - 8 bin orientation histogram to a 128-dimension vector as its descriptor.

## Result
Time cost versus number of threads and the speedup

<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/time_700x542.png" width="400"><img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/time_1850x1280.png" width="400">
<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/time_6000x4000.png" width="400"><img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/speedup.png" width="400">

keypoints result on the image (700 x 542)

|<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/corgi.png" width="400">|<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/corgi_opencv.png" width="400">|
| :---: | :---: |
| Original image | OpenCV result |
|<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/corgi_ours.png" width="400">|<img src="https://github.com/zon5566/SIFT-on-OpenMP/blob/master/image/corgi_readable.png" width="400">|
| Ours | Ours (readable) |

