@echo off
REM Defeault example
REM python ./miho_buffered_rot_patch_pytorch.py data/im1.png data/im2.png data/matches.mat


REM LightGlue
REM python ./thirdparty/deep-image-matching/main.py --dir ./img_test/sacre_coeur --pipeline superpoint+lightglue --force
REM python ./thirdparty/deep-image-matching/scripts/export_from_database.py ./img_test/sacre_coeur/results_superpoint+lightglue_matching_lowres_quality_high/database.db x1y1x2y2
REM python ./miho_buffered_rot_patch_pytorch.py ./img_test/sacre_coeur/images/sacre_coeur_A.jpg ./img_test/sacre_coeur/images/sacre_coeur_B.jpg ./img_test/sacre_coeur/results_superpoint+lightglue_matching_lowres_quality_high/matches/sacre_coeur_A_sacre_coeur_B.txt


REM SIFT
REM python ./thirdparty/deep-image-matching/main.py --dir ./img_test/sacre_coeur --pipeline sift+kornia_matcher --force
REM python ./thirdparty/deep-image-matching/scripts/export_from_database.py ./img_test/sacre_coeur/results_sift+kornia_matcher_matching_lowres_quality_high/database.db x1y1x2y2
python ./miho_buffered_rot_patch_pytorch.py ./img_test/sacre_coeur/images/sacre_coeur_A.jpg ./img_test/sacre_coeur/images/sacre_coeur_B.jpg ./img_test/sacre_coeur/results_sift+kornia_matcher_matching_lowres_quality_high/matches/sacre_coeur_A_sacre_coeur_B.txt


