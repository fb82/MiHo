import scipy.io as sio
import numpy as np
import cv2
import os
import os.path as osp
import warnings
import argparse
import subprocess
from pathlib import Path
import shutil

from PIL import Image
from miho_buffered_rot_patch_pytorch import *

import sqlite3
import yaml
from importlib import import_module
from pprint import pprint
from deep_image_matching import logger, timer
from deep_image_matching.config import Config
from deep_image_matching.image_matching import ImageMatching
from deep_image_matching.io.h5_to_db import export_to_colmap


def resize_image(img, max_side_length):
    height, width = img.shape[:2]
    if max(height, width) > max_side_length:
        scale_factor = max_side_length / max(height, width)
        img_resized = cv2.resize(
            img, (int(width * scale_factor), int(height * scale_factor))
        )
        return img_resized
    else:
        return img


def GeneratePlot(
    img0_path: Path,
    img1_path: Path,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
):
    # Load images
    img0 = cv2.imread(str(img0_path))
    img1 = cv2.imread(str(img1_path))
    # Convert keypoints to integers
    kpts0_int = np.round(kpts0).astype(int)
    kpts1_int = np.round(kpts1).astype(int)
    # Create a new image to draw matches
    img_matches = np.zeros(
        (max(img0.shape[0], img1.shape[0]), img0.shape[1] + img1.shape[1], 3),
        dtype=np.uint8,
    )
    img_matches[: img0.shape[0], : img0.shape[1]] = img0
    img_matches[: img1.shape[0], img0.shape[1] :] = img1
    # Show keypoints
    for kpt in kpts0_int:
        kpt = tuple(kpt)
        cv2.circle(img_matches, kpt, 3, (0, 0, 255), -1)
    for kpt in kpts1_int:
        kpt = tuple(kpt + np.array([img0.shape[1], 0]))
        cv2.circle(img_matches, kpt, 3, (0, 0, 255), -1)
    # Draw lines and circles for matches
    for match in matches:
        pt1 = tuple(kpts0_int[match[0]])
        pt2 = tuple(np.array(kpts1_int[match[1]]) + np.array([img0.shape[1], 0]))
        # Draw a line connecting the keypoints
        cv2.line(img_matches, pt1, pt2, (0, 255, 0), 1)
        # Draw circles around keypoints
        cv2.circle(img_matches, pt1, 3, (255, 0, 0), -1)
        cv2.circle(img_matches, pt2, 3, (255, 0, 0), -1)
    img_matches_resized = resize_image(img_matches, 1000)
    # Show the image with matches
    cv2.imshow(f"Verified matches   {img0_path.name} - {img1_path.name}", img_matches_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def dbReturnMatches(database_path, min_num_matches):
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = {}
        raw_matches = {}
        matches = {}

        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name

        # Raw matches
        cursor.execute(
            "SELECT pair_id, data FROM matches WHERE rows>=?;",
            (min_num_matches,),
        )

        for row in cursor:
            pair_id = row[0]
            _matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1]
            image_name2 = images[image_id2]
            raw_matches["{} {}".format(image_name1, image_name2)] = _matches

        # Geometric verified matches
        cursor.execute(
            "SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;",
            (min_num_matches,),
        )

        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1]
            image_name2 = images[image_id2]
            matches["{} {}".format(image_name1, image_name2)] = inlier_matches

        cursor.close()
        connection.close()

        return images, raw_matches, matches

    else:
        print("Database does not exist")
        quit()


def dbReturnKeypoints(database_path):    
    if os.path.exists(database_path):
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()

        images = {}
        keypoints = {}
        cursor.execute("SELECT image_id, camera_id, name FROM images;")
        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name

        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        for row in cursor:
            image_id = row[0]
            # kpts = np.fromstring(row[3], np.float32).reshape(-1, 6)
            # kpts = np.fromstring(row[3], np.float32).reshape(-1, 4)
            kpts = np.fromstring(row[3], np.float32).reshape(-1, 2)
            keypoints[images[image_id]] = kpts

    return images, keypoints

def relative_pose_error(R_gt, t_gt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    # t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    # R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def error_auc(errors, thr):
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))

    last_index = np.searchsorted(errors, thr)
    y = recall[:last_index] + [recall[last_index-1]]
    x = errors[:last_index] + [thr]
    return np.trapz(y, x) / thr    


if __name__ == '__main__':
    
    # Input params
    apply_miho = False
    angular_thresholds = [5, 10, 20]
    pixel_thr = 0.5
    pipeline = "superpoint+lightglue"
    path_to_GT = Path(r"/home/threedom/Desktop/SCANNET_DATA/scannet.mat")
    working_dir = Path(r"/home/threedom/Desktop/SCANNET_DATA/DATA/")


    # Load data
    data = sio.loadmat(
                    path_to_GT,
                    simplify_cells=True,
                    matlab_compatible=True,
                    chars_as_strings=True
                    )
    im1 = ["".join(data['im1'][i]).strip()
           for i in range(data['im1'].shape[0])]
    im2 = ["".join(data['im2'][i]).strip()
           for i in range(data['im2'].shape[0])]
    K1 = data['K1']
    K2 = data['K2']
    R_gt = data['R']
    t_gt = data['T']


    # Store all the image pairs per scene
    pairs_per_scene = {}
    for pair_idx in range(len(im1)):
        scene1, _, image1 = im1[pair_idx].split("/", 2)
        scene2, _, image2 = im2[pair_idx].split("/", 2)
        if scene1 not in list(pairs_per_scene.keys()):
            pairs_per_scene[scene1] = []
        assert(scene1== scene2)
        pairs_per_scene[scene1].append((image1, image2, pair_idx))
    

    # Setup DIM
    data["results"] = {}
    data["results"]['R_errs'] = []
    data["results"]['t_errs'] = []
    data["results"]['inliers'] = []
    
    #for scene in list(pairs_per_scene.keys()):
    for scene in ["scene0707_00", "scene0708_00"]:
    #for scene in ["scene0707_00"]:
        with open(working_dir / scene / "pairs.txt", 'w') as pair_file:
            for pair in pairs_per_scene[scene]:
                pair_file.write(f'{pair[0]} {pair[1]}\n')
            
        # Run DIM as library to extract and match features
        cli_params = {
            "dir": f"{working_dir / scene}",
            "images": working_dir / scene / 'color',
            "pipeline": f"{pipeline}",
            "strategy": "custom_pairs",
            "pair_file": f"{working_dir / scene}/pairs.txt",
            "tiling": "none",
            "skip_reconstruction": True,
            "force": True,
            "camera_options": "../config/cameras.yaml",
            "openmvg": None,
            "verbose": True,
        }

        config = Config(cli_params)
        config.general["min_inliers_per_pair"] = 8
        config.general["gv_threshold"] = 1000 # 1000 pixel error threshold to disable DIM ransac, I can add an option to completly disable this step
        config.extractor["max_keypoints"] = 8000
        config.save()

        imgs_dir = config.general["image_dir"]
        output_dir = config.general["output_dir"]
        matching_strategy = config.general["matching_strategy"]
        extractor = config.extractor["name"]
        matcher = config.matcher["name"]

        img_matching = ImageMatching(
            imgs_dir=imgs_dir,
            output_dir=output_dir,
            matching_strategy=matching_strategy,
            local_features=extractor,
            matching_method=matcher,
            pair_file=config.general["pair_file"],
            retrieval_option=config.general["retrieval"],
            overlap=config.general["overlap"],
            existing_colmap_model=config.general["db_path"],
            custom_config=config.as_dict(),
        )


        pair_path = img_matching.generate_pairs()
        feature_path = img_matching.extract_features()
        match_path = img_matching.match_pairs(feature_path)

        camera_options = {
           'general' : {
            "camera_model" : "pinhole", # ["simple-pinhole", "pinhole", "simple-radial", "opencv"]
            "single_camera" : True,
           },
           'cam0' : {
                "camera_model" : "pinhole",
                "images" : "DSC_6468.JPG,DSC_6468.JPG",
           },
           'cam1' : {
                "camera_model" : "pinhole",
                "images" : "",
           },
        }

        # You can also read matches directly from h5 file, the advantage is that in a second moment the database.db can be imported to COLMAP
        # to visualize matches
        database_path = output_dir / "database.db"
        export_to_colmap(
            img_dir=imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            database_path=database_path,
            camera_options=camera_options,
        )

        images, keypoints = dbReturnKeypoints(database_path)
        images, raw_matches, matches = dbReturnMatches(database_path, min_num_matches=1)

        for pair in pairs_per_scene[scene]:
            img0, img1, pair_idx = pair[0], pair[1], pair[2]
            pair01 = f"{Path(img0).name} {Path(img1).name}"
            pair10 = f"{Path(img1).name} {Path(img0).name}"

            im1 = Image.open(working_dir / scene / "images" / img0)            
            im2 = Image.open(working_dir / scene / "images" / img1)

            m01 = np.empty((0,4))
                
            if pair01 in list(matches.keys()):
                m01 = matches[pair01]
                k0 = keypoints[f"{Path(img0).name}"][m01[:,0],:]
                k1 = keypoints[f"{Path(img1).name}"][m01[:,1],:]
                #GeneratePlot(working_dir / scene / "images" / Path(img0), working_dir / scene / "images" / Path(img1), keypoints[f"{Path(img0).name}"], keypoints[f"{Path(img1).name}"], m01)
                # To visualize matching just import database.db in COLMAP GUI
                m01 = np.hstack((k0,k1))

            if pair10 in list(matches.keys()):
                m10 = matches[pair10]
                k1 = keypoints[f"{Path(img1).name}"][m10[:,0],:]
                k0 = keypoints[f"{Path(img0).name}"][m10[:,1],:]
                m01 = np.hstack((k0,k1))
                
            if m01.shape[0] != 0:
                if apply_miho:
                    params = miho.all_params()
                    params['get_avg_hom']['rot_check'] = True
                    mihoo = miho(params)
                    mihoo.planar_clustering(m01[:, :2], m01[:, 2:])
                    mihoo.attach_images(im1, im2)
                    w = 15  
                    pt1_, pt2_, Hs_ = refinement_init(mihoo.im1, mihoo.im2, mihoo.Hidx, mihoo.Hs, mihoo.pt1, mihoo.pt2, mihoo, w=w, img_patches=False)        
                    pt1__, pt2__, Hs__, val, T = refinement_norm_corr(mihoo.im1, mihoo.im2, Hs_, pt1_, pt2_, w=w, ref_image=['both'], subpix=True, img_patches=False)
                    kp1, kp2 = pt1__.numpy(), pt2__.numpy()
                else:
                    kp1, kp2 = m01[:, :2], m01[:, 2:]

                if (kp1.shape[0] > 8):
                    Rt = estimate_pose(kp1, kp2, K1[pair_idx], K2[pair_idx], pixel_thr)
                else:
                    Rt = None

            else:
                Rt = None

            if Rt is None:
                data["results"]['R_errs'].append(np.inf)
                data["results"]['t_errs'].append(np.inf)
                data["results"]['inliers'].append(
                    np.array([]).astype('bool'))
            else:
                R, t, inliers = Rt
                t_err, R_err = relative_pose_error(
                    R_gt[pair_idx], t_gt[pair_idx], R, t, ignore_gt_t_thr=0.0)
                data["results"]['R_errs'].append(R_err)
                data["results"]['t_errs'].append(t_err)
                data["results"]['inliers'].append(inliers)


    aux = np.stack(
        ([data["results"]['R_errs'], data["results"]['t_errs']]), axis=1)
    max_Rt_err = np.max(aux, axis=1)

    tmp = np.concatenate((aux, np.expand_dims(
        np.max(aux, axis=1), axis=1)), axis=1)

    for a in angular_thresholds:
        #auc_R = error_auc(np.squeeze(data["results"]['R_errs']), a)
        #auc_t = error_auc(np.squeeze(data["results"]['t_errs']), a)
        #auc_max_Rt = error_auc(np.squeeze(max_Rt_err), a)
        #data["results"]['pose_error_auc_@' +
        #             str(a)] = np.asarray([auc_R, auc_t, auc_max_Rt])

        print("scene", scene, 'pose_error_acc_@' + str(a), np.sum(tmp < a, axis=0)/np.shape(tmp)[0])



    #sio.savemat(osp.join('table_res', pipeline +'_'+ "scannet" +
    #            '_pose_error_'+str(pixel_thr)+'.mat'), data, do_compression=True)