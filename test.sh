# TODO
# Aggiungere modifica Fabio
# Eliminate ransac verification
# Loop on a dataset

# CONFIG
pipeline="superpoint+lightglue"
working_dir="./img_test/cyprus"
img0=img0.jpg
img1=img1.jpg

# MAIN
python ./thirdparty/deep-image-matching/main.py --dir "${working_dir}" --pipeline "${pipeline}" --strategy bruteforce --force --skip_reconstruction
python ./thirdparty/deep-image-matching/scripts/export_from_database.py "${working_dir}/results_${pipeline}_bruteforce_quality_high/database.db" x1y1x2y2
python ./miho_buffered_rot_patch_pytorch.py "${working_dir}/images/${img0}" "${working_dir}/images/${img1}" "${working_dir}/results_${pipeline}_bruteforce_quality_high/matches/${img0%%.*}_${img1%%.*}.txt"