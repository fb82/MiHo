<h1 align="center"> 🧹🪣 MOP+MiHo+NCC 🖼️👀 <br> <br> Image Matching Filtering and Refinement <br> by Planes and Beyond </h1>

<p align="center">
  <a href="https://sites.google.com/view/fbellavia/home?authuser=0">Fabio Bellavia</a>
  ·
  <a href="https://ericzzj1989.github.io">Zhenjun Zhao</a>
  ·
  <a href="https://3dom.fbk.eu/people/profile/luca-morelli">Luca Morelli</a>
  ·
  <a href="https://3dom.fbk.eu/people/profile/remondino">Fabio Remondino</a>
</p>

## What is it?
MOP+MiHo+NCC is a sequence of non-deep modules to filter and refine image matches:
1. Multiple Overlapping Planes (MOP) removes outlier matches while jointly clusters inlier matches into planes by an iterative RANSAC-based strategy;
2. Middle Homography (MiHo) improves MOP planar homography by minimizing the relative patch distortion in the plane reprojection;
3. Normalized Cross Correlation (NCC) refines keypoint position on the patches after the planar transformation.

## Is it good?
MOP+MiHo generally improves the matches when used as RANSAC pre-processing, not degrading the final results in any case. MOP+MiHo+NCC introduces additional improvements in the case of corner-like matches, predominant for instance in Key.Net and SuperPoint, while in case of blob-like matches, predominant for instance in SIFT, seems to degrade the results.

**Comparative evaluation is available [here](https://github.com/fb82/MiHo/tree/main/data/results/benchmark%20results/short)**. Unlike other benchmarks, it is assumed that *no camera instrinsics are available* (the most general and maybe realistic case), so in the tables $\text{AUC}^{F}_\measuredangle$ means that pose is estimated by retrieving the essential matrix by adding camera intrinsics *after* fundamental matrix estimation, while $\text{AUC}^{E}_\measuredangle$ indicates that the pose is estimated normalizing keypoint coordinates through intrinsics *before* directly computing the essential matrix by the 5-point method using `cv2.findEssentialMat`. AUC considering metric translation error instead of the the angular one is reported as $\text{AUC}^{F}_\square$ and $\text{AUC}^{E}_\square$ in the case of the [IMC PhotoTourism dataset](https://www.kaggle.com/competitions/image-matching-challenge-2022/data).



## 

## Where can I find more details?
The early idea of the approach can be found in the paper [Progressive keypoint localization and
refinement in image matching (FAPER 2023)](https://drive.google.com/file/d/1sxQOpbBTvvqnpfR98HGHQafquww6JuaK/edit), *further details will be available soon in an upcoming paper*.
