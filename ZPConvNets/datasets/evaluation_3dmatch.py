import os
import glob
from os.path import join
import numpy as np
from sklearn.neighbors import KDTree
from multiprocessing import Pool
import vgtk.pc as pctk

def read_key_point(path):
    with open(path, 'r') as fin:
        point_ids_str = fin.readlines()
    point_ids = [int(i) for i in point_ids_str if i.strip()]
    return np.array(point_ids)

def read_feature(path, descriptor_name='ours'):
    def read_npz(path):
        return np.load(path)['data']
    def read_np(path):
        return np.load(path)

    if descriptor_name == 'ours' or 'lmvd':
        return read_np(path)
    elif descriptor_name == '3DSmooth':
        return read_npz(path)
    else:
        raise ValueError('No such descriptor')

def read_gt_log(path):
    fragment_pairs = []
    gt_transforms = []

    with open(path, 'r') as fin:
        lines = fin.readlines()
    for i in range(len(lines)//5):
        line = lines[i*5]
        data = line.split()
        fragment_pairs.append([int(data[0]), int(data[1])])
        gt_transform = []
        for j in range(4):
            line = lines[i*5+j+1]
            gt_transform.append(list(map(float, line.split())))
        gt_transforms.append(gt_transform)

    return np.array(fragment_pairs), np.array(gt_transforms)

def hom_transform(points, T, translation=True):
    if translation:
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = points @ T.T
        points = points[:, :3]
    else:
        points = points[:, :3] @ T[:3, :3].T
    return points


def evaluate_fragment_pair(src_frag_id, tgt_frag_id,
                           src_pc_path, tgt_pc_path, 
                           src_kp_path, tgt_kp_path,
                           src_feat_path, tgt_feat_path, 
                           gt_transform, tau1=0.1, descriptor='ours'):
    print("Evaluating frag %d and frag %d"%(src_frag_id, tgt_frag_id))
        
    src_point_cloud = pctk.load_ply(src_pc_path)
    tgt_point_cloud = pctk.load_ply(tgt_pc_path)
    src_key_point_ids = read_key_point(src_kp_path)
    tgt_key_point_ids = read_key_point(tgt_kp_path)

    src_feats = read_feature(src_feat_path, descriptor_name=descriptor)
    tgt_feats = read_feature(tgt_feat_path, descriptor_name=descriptor)
    assert src_feats.ndim == 2

    result_log = [src_frag_id, tgt_frag_id]

    src_key_point_locs = src_point_cloud[src_key_point_ids]
    tgt_key_point_locs = tgt_point_cloud[tgt_key_point_ids]

    src_KDT = KDTree(src_feats)
    tgt_KDT = KDTree(tgt_feats)

    _, src_tgt_nn_ids = tgt_KDT.query(src_feats, k=1)
    _, tgt_src_nn_ids = src_KDT.query(tgt_feats, k=1)
    # only care the closest one
    src_tgt_nn_ids = src_tgt_nn_ids.squeeze()
    tgt_src_nn_ids = tgt_src_nn_ids.squeeze()

    # currently use tgt->src->tgt
    mutual_closest_ids = (np.arange(src_tgt_nn_ids.shape[0]) == src_tgt_nn_ids[tgt_src_nn_ids])
    src_match_point_locs = src_key_point_locs[tgt_src_nn_ids[mutual_closest_ids]]
    tgt_match_point_locs = tgt_key_point_locs[mutual_closest_ids]
    # n_matching = tgt_match_point_locs.shape[0]

    # tgt_match_point_locs = np.hstack([tgt_match_point_locs, np.ones([n_matching, 1])])
    # tgt_match_point_locs = tgt_match_point_locs @ gt_transform.T
    # tgt_match_point_locs = tgt_match_point_locs[:, :3]

    # src_match_point_locs = hom_transform(src_match_point_locs, gt_transform)
    tgt_match_point_locs = hom_transform(tgt_match_point_locs, gt_transform)

    distances = np.sqrt(np.sum((src_match_point_locs - tgt_match_point_locs)**2, 1))
    n_inlier = (distances < tau1).sum()
    inlier_ratio = float(n_inlier) / distances.shape[0]

    ################################
    mid_tgt = np.argwhere(mutual_closest_ids)
    mid_src = tgt_src_nn_ids[mutual_closest_ids][:,None]
    select = distances < tau1

    kpt_mid_src = src_key_point_ids[mid_src[select]]
    kpt_mid_tgt = tgt_key_point_ids[mid_tgt[select]]

    kpts = np.concatenate((kpt_mid_src, kpt_mid_tgt), 1)
    ##############################################

    result_log.append(n_inlier)
    result_log.append(inlier_ratio)

    # print(" inlier distance min and mean: %f, %f" % (distances.min(), distances.mean()))

    print(" Frag %d, %d: Found %d, N_inlier is %d, Inlier_ratio is %f" % (src_frag_id, tgt_frag_id, distances.shape[0], 
                                                              n_inlier, inlier_ratio))


    return n_inlier, inlier_ratio, result_log, kpts


TAU_RANGE = [0.05, 0.1, 0.2]
    
def evaluate_scene(scene_dir, feature_dir, scene_name, suffix=None, num_thread=8, tau2=0.05):
    scene_dir = join(scene_dir, scene_name)

    if 'seq-01' in os.listdir(scene_dir):
        get_pc_path = lambda x: join(scene_dir, 'seq-01','cloud_bin_%d.ply'%x)
        get_kp_path = lambda x: join(scene_dir, 'seq-01','cloud_bin_%d.keypts.txt'%x)
    else:
        get_pc_path = lambda x: join(scene_dir, 'cloud_bin_%d.ply'%x)
        get_kp_path = lambda x: join(scene_dir, '01_Keypoints', 'cloud_bin_%dKeypoints.txt'%x)

    if suffix is None:
        descriptor = 'ours'
        get_feat_path = lambda x: join(feature_dir, 'feature%d.npy'%x)
    elif suffix == 'lmvd':
        descriptor = 'lmvd'
        get_feat_path = lambda x: join(feature_dir, 'cloud_bin_%d.desc.npy'%x)
    else:
        # used for 3DMatch eval e.g. _cloud_bin_59.ply_0.150000_16_1.750000_3DSmoothNet.npz
        descriptor = '3DSmooth'
        get_feat_path = lambda x: join(feature_dir, '_cloud_bin_%d.ply_%s.npz'%(x,suffix))

    if 'seq-01' in os.listdir(scene_dir):
        fragment_pairs, gt_transforms = read_gt_log(join(scene_dir, 'seq-01','gt.log'))
    else:
        fragment_pairs, gt_transforms = read_gt_log(join(scene_dir, 'gt.log'))
    n_inliers = []
    inlier_ratios = []

    mp_args = []
    for fragment_pair, gt_transform in zip(fragment_pairs, gt_transforms):
        src_frag_id, tgt_frag_id = fragment_pair[:2]
        ##### hack
        #if src_frag_id > 14 or tgt_frag_id > 14:
        #    continue
        ##### hack end

        srcp = get_feat_path(src_frag_id)
        tgtp = get_feat_path(tgt_frag_id)

        if not os.path.exists(srcp) or not os.path.exists(tgtp):
            print(f'Path at {srcp} does not exist!!')
            continue
        
        mp_args.append([src_frag_id, tgt_frag_id,
                        get_pc_path(src_frag_id),
                        get_pc_path(tgt_frag_id),
                        get_kp_path(src_frag_id),
                        get_kp_path(tgt_frag_id),
                        get_feat_path(src_frag_id),
                        get_feat_path(tgt_frag_id),
                        gt_transform, 0.1, descriptor])
    
        # evaluate_fragment_pair(*mp_args[-1])
    
    # import ipdb; ipdb.set_trace()
    pool = Pool(num_thread)
    rst = pool.starmap(evaluate_fragment_pair, mp_args)
    n_inliers, inlier_ratios, result_log, kpts = zip(*rst)

    #############################
    if suffix == 'lmvd':
        output_folder = join(scene_dir, "lmvd_test_kpts")
        os.makedirs(output_folder, exist_ok=True)
        for i, fp in enumerate(fragment_pairs):
            src_frag_id, tgt_frag_id = fp[:2]
            np.save(join(output_folder, f"cloud_bin_{src_frag_id}-cloud_bin_{tgt_frag_id}.keypts.npy"), kpts[i])

    ############################

    n_inliers = np.array(n_inliers)
    inlier_ratios = np.array(inlier_ratios)
    total_recall = np.mean(inlier_ratios > tau2)
    results = np.array(result_log)

    print("Total recall is %0.2f" % (total_recall * 100))
    np.savetxt(join(feature_dir, 'recall.txt'), results, fmt='%.2f', delimiter=',')

    return [(tau, 100 * np.mean(inlier_ratios > tau)) for tau in TAU_RANGE]


