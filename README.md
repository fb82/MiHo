<h1 align="center"> 🧹🪣 MOP+MiHo+NCC 🖼️👀 <br> <br> Image Matching Filtering and Refinement <br> by Planes and Beyond </h1>

<p align="center">
  <a href="https://sites.google.com/view/fbellavia/home?authuser=0">Fabio Bellavia*</a>
  ·
  <a href="https://ericzzj1989.github.io">Zhenjun Zhao*</a>
  ·
  <a href="https://3dom.fbk.eu/people/profile/luca-morelli">Luca Morelli</a>
  ·
  <a href="https://3dom.fbk.eu/people/profile/remondino">Fabio Remondino</a>
</p>

<p align="center">
    <img src="https://github.com/fb82/MiHo/blob/main/data/other/lions.png" alt="example" width=80%>
    <br>
    <em>Planar clusters assigned by MOP+MiHo, outlier matches are marked <br> with black diamonds, NCC keypoint shifts are not shown.</em>
</p>

## What is it?
MOP+MiHo+NCC is a modular, non-deep method designed to filter and refine image matches. This approach enhances the quality of image matching by utilizing multiple techniques:
1. Multiple Overlapping Planes (MOP) removes outlier matches while jointly clustering inlier matches into planes by an iterative RANSAC-based strategy;
2. Middle Homography (MiHo) improves planar homography from MOP by minimizing relative patch distortion in the plane reprojection;
3. Normalized Cross Correlation (NCC) refines keypoint positions on patches after planar transformation.

## Is it good?
MOP+MiHo consistently improves match quality when used a pre-processing step for RANSAC, without negatively affecting the final results in any case. MOP+MiHo+NCC introduces additional improvements in the case of corner-like matches, which are common in methods such as [Key.Net](https://github.com/axelBarroso/Key.Net) or [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork). However, for blob-like matches, which are more typical of methods like SIFT, MOP+MiHo+NCC unfortunately tends to degrade performance.

**Comparative evaluation is available [here](https://github.com/fb82/MiHo/tree/main/data/results/benchmark%20results/short)**.

Unlike other benchmarks, it assumes *no camera intrinsics are available*, which reflects a more general and realistic scenario. In the tables, $\text{AUC}^{F}_ \measuredangle$ represents the pose estimated by retrieving the essential matrix by adding camera intrinsics *after* computing the fundamental matrix, while $\text{AUC}^{E}_ \measuredangle$ indicates that the pose is estimated by normalizing keypoint coordinates through intrinsics *before* directly computing the essential matrix by the 5-point method using `cv2.findEssentialMat`. AUCs considering metric instead of angular translation errors are reported as $\text{AUC}^{F}_ \square$ and $\text{AUC}^{E}_ \square$ in the case of [IMC PhotoTourism](https://www.kaggle.com/competitions/image-matching-challenge-2022/data), for which about 13K random image pairs have been chosen.

As better RANSAC, [MAGSAC](https://github.com/danini/magsac) is employed with the thresholds of 1 and 0.75 px, indicated respectively by $\text{MAGSAC}_ \uparrow$ and $\text{MAGSAC}_ \downarrow$. Ablation studies with further RANSAC implementations ([DegenSAC](https://github.com/ducha-aiki/pydegensac) and [PoseLib](https://github.com/PoseLib/PoseLib)) and thresholds are available [here](https://github.com/fb82/MiHo/tree/main/data/results/RANSAC%20ablation%20results/short). Unlike defaults, for all RANSAC implementations, the maximum number of iterations was incremented to 2000 for better performances.

## How to use?
Check the comments and run `miho_ncc_demo.py`. Notice that in the code MOP is denoted as `miho_unduplex` while MOP+MiHo as `miho_duplex`; in case you get a Qt window displaying related errors, try to uncomment `matplotlib.use('tkagg')` in `src/miho.py`. Decreasing the `max_iters` params for `ransac_middle_args` in `get_avg_hom`, e.g. to 1500, 1000, or 500, can improve the speed at the expense of the accuracy; this is not recommended for SIFT, whose matches are more contaminated by outliers than other matching pipelines, also because in the experiments SIFT pipeline NNR threshold is 0.95 (quite high) for an effective evaluation of the match filtering. 

To run the full benchmark, use `run_bench.py` and generate result tables with `save_bench.py`. If you want to limit the methods or datasets being compared, you can comment out specific lines in the `pipes`, `pipe_heads`, `pipe_ransacs` and `benchmark_data` sections of the code. For RANSAC ablation, use the corresponding scripts `run_ransac_ablation.py` and `save_ransac_ablation.py`. Additional scripts like `intrinsiscs_bench.py` and `intrinsics_other_bench.py` provide more detailed reports on camera intrinsics, while `corr_bench.py` generates the correlation between different measurement errors. All detailed results are available in the `data/results` directory. 

## Where can I find more details?
The initial idea of this approach can be found in the paper [Progressive keypoint localization and
refinement in image matching](https://drive.google.com/file/d/1sxQOpbBTvvqnpfR98HGHQafquww6JuaK/edit) presented at FAPER2023. *More details will be available in an upcoming paper*.
