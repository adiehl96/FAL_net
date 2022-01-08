import numpy as np
import cv2
import os.path as path
from test_utils.eval_kitti_utils import (
    generate_depth_map,
    read_text_lines,
    read_file_data,
    get_focal_length_baseline,
    compute_errors,
)


def main(
    pred_disparities,
    gt_path,
    min_depth=1e-3,
    max_depth=80,
    eigen_crop=False,
    garg_crop=True,
):

    num_samples = 697

    text_file_path = path.join(
        path.dirname(path.abspath(__file__)), "../splits/KITTI/eigen_test_classic.txt"
    )

    test_files = read_text_lines(text_file_path)
    gt_files, gt_calib, im_sizes, _im_files, cams = read_file_data(test_files, gt_path)

    gt_depths = []
    pred_depths = []
    for t_id in range(num_samples):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        depth = generate_depth_map(
            gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True
        )
        gt_depths.append(depth.astype(np.float32))

        disp_pred = cv2.resize(
            pred_disparities[t_id],
            (im_sizes[t_id][1], im_sizes[t_id][0]),
            interpolation=cv2.INTER_LINEAR,
        )
        disp_pred = disp_pred * disp_pred.shape[1]

        # need to convert from disparity to depth
        focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
        depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        if garg_crop or eigen_crop:
            gt_height, gt_width = gt_depth.shape

            # crop used by Garg ECCV16
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            if garg_crop:
                crop = np.array(
                    [
                        0.40810811 * gt_height,
                        0.99189189 * gt_height,
                        0.03594771 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)
            # crop we found by trial and error to reproduce Eigen NIPS14 results
            elif eigen_crop:
                crop = np.array(
                    [
                        0.3324324 * gt_height,
                        0.91351351 * gt_height,
                        0.0359477 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(
            gt_depth[mask], pred_depth[mask]
        )

    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            "abs_rel", "sq_rel", "rms", "log_rms", "d1_all", "a1", "a2", "a3"
        )
    )
    print(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            abs_rel.mean(),
            sq_rel.mean(),
            rms.mean(),
            log_rms.mean(),
            d1_all.mean(),
            a1.mean(),
            a2.mean(),
            a3.mean(),
        )
    )
