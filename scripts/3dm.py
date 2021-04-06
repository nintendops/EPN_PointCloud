# (Used by 3DSmoothNet) procedure to generate traning data (keypoints in txt and voxels in csv)
import numpy as np
import os, sys
import subprocess
from parse import *
from core.util import *

def save_list(path, arr):
    with open(path, 'w') as f:
        for idx in arr:
            f.write('%d\n'%idx)

def parse_pc_file(filename):
    s1 = filename.split("points_")[1]
    idx = parse("{:d}.csv",s1)
    return idx[0]

POINTCLOUD_MODE = True

keypoint_sample = 1000

registration_file = '/home/ICT2000/chenh/Haiwei/MScenes/sun3d-home_at-home_at_scan1_2013_jan_1-evaluation/gt.log' 
data_path = '/home/ICT2000/chenh/Haiwei/MScenes/sun3d-home_at-home_at_scan1_2013_jan_1'

# output paths
output_root = 'data/train/sun3d-home_at-home_at_scan1_2013_jan_1'
keypoint_path = os.path.join(output_root, 'test_keypoints')
sdv_path = os.path.join(output_root, 'test_pointclouds')
create_dir(output_root)
create_dir(keypoint_path)
create_dir(sdv_path)

save_keypoints = True

# parse registration text
regs = parse_registration(registration_file)
# input data
scanfiles = [os.path.join(data_path,f) for f in os.listdir(data_path)]
scanfiles.sort(key=lambda s: extract_id_from_path(s,useint=True))

ctn = 0
search_radius = 0.3

for k,v in regs.items():
    k1, k2 = parse('{:d}n{:d}',k)

    # generation filter 
    if ctn < 419:
        ctn += 1
        continue
    
    print("Processing data pair", k)
    
    new_key = "pair%d_%dn%d"%(ctn,k1,k2)
    pc1 = load_ply(scanfiles[k1])
    pc2 = load_ply(scanfiles[k2])
    # sanity check
    # print(k1,k2,extract_id_from_path(scanfiles[k1]),extract_id_from_path(scanfiles[k2]))
    
    anchor_token = os.path.join(keypoint_path, "keypoints_%s_anchor.txt"%new_key)
    pos_token = os.path.join(keypoint_path, "keypoints_%s_pos.txt"%new_key)
    
    if save_keypoints:
        # extract and save overlap keypoints
        kpt1, kpt2, _ = find_scenes_overlap(pc1,pc2,v, k=70000)
        save_list(anchor_token, kpt1)
        save_list(pos_token, kpt2)
    continue
    # gather parameters for voxelization
    name_prefix = new_key
    point_cloud_files = [scanfiles[k1], scanfiles[k2]]
    keypoints_files = [anchor_token, pos_token]
    remove_command = "rm -rf " + sdv_path + '/*.csv'
            
    for i in range(2):
        args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] \
        + " -s " + name_prefix + " -o " + sdv_path + '/' \
        + " -r " + str(search_radius) 
        print(args)
        subprocess.call(args, shell=True)
        
        if POINTCLOUD_MODE:
            # resample and combine point clouds
            pc_files = list(filter(lambda n: n.startswith(new_key) and n.endswith('.csv'),os.listdir(sdv_path)))
            pc_files.sort(key=parse_pc_file)
            
            prefix = 'src' if i == 0 else 'tgt'
            all_pc = np.zeros((keypoint_sample, 4096,3), dtype=np.float32)
            eval_counter = 0        
            for pcf in pc_files:
                pcf = os.path.join(sdv_path, pcf)
                evalptidx = parse_pc_file(pcf)
                
                while evalptidx > eval_counter:
                    print("Found evaluation points %d missing, filing in zero points..."%eval_counter)
                    all_pc[eval_counter] = np.zeros((4096,3))
                    eval_counter += 1
                
                pc = np.loadtxt(pcf,delimiter=',')              
                if pc.shape[0] == 0:
                    print("Empty evaluation point found at", eval_counter)
                    pc = np.zeros((4096,3))
                else:
                    pc = resample_pc(pc, 4096)
                #save_path = os.path.join('data/visualization', 'pc_patch%d.ply'%pidx)
                #write_color_pointcloud_as_ply(pc, save_path)
                all_pc[eval_counter] = pc
                eval_counter += 1

            save_name = new_key + '_%s'%prefix
            np.save(os.path.join(sdv_path,save_name), all_pc)
            subprocess.call(remove_command, shell=True)
        
    ctn += 1
print('Done!')


# procedure to convert csvs into tfrecord
import tensorflow as tf

import numpy as np
import os, sys
import subprocess
from parse import *
from core.util import *

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_sdv_csv(path):
    voxels = np.fromfile(path, dtype=np.float32).reshape(-1,16,16,16,1)
    return voxels

TEST_WRITER = True
TEST_LOADER = False

output_root = 'data/train/redkitchen'
keypoint_path = os.path.join(output_root, 'keypoints')
sdv_path = os.path.join(output_root, 'sdv')
tfrecord_path = os.path.join(output_root, 'tfrecords')
create_dir(tfrecord_path)

# example: pair97_5n11_cloud_bin_5.ply_0.150000_16_1.750000.csv
def sdv_file_sort(s):
    pid, _ = parse("pair{:d}_{}.csv", s)
    return pid

sdv_files = os.listdir(sdv_path)
sdv_files.sort(key=sdv_file_sort)
sdv_files_ref = sdv_files
sdv_files = [os.path.join(sdv_path, f) for f in sdv_files]    

if TEST_WRITER:
    # tfrecords writer
    train_filename = 'redkitchen_train_0.tfrecord'
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path,train_filename))

    for i in range(0, len(sdv_files),2):
        print("Processing pair %d, %d"%(sdv_file_sort(sdv_files_ref[i]), sdv_file_sort(sdv_files_ref[i+1])))
        if i > 0 and i % 4 == 0:
            writer.close()
            train_filename = 'redkitchen_train_%d.tfrecord'%(i//4)
            writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path,train_filename))
        group_anc_voxels = read_sdv_csv(sdv_files[i])
        group_pos_voxels = read_sdv_csv(sdv_files[i+1])
        assert group_anc_voxels.shape[0] == group_pos_voxels.shape[0]
        npts = group_anc_voxels.shape[0]
        for nn in range(npts):
            anc_voxels = group_anc_voxels[nn]
            pos_voxels = group_pos_voxels[nn]
            features = tf.train.Features(feature=
                                         {"X":_floats_feature(anc_voxels.reshape(-1)),
                                          "Y":_floats_feature(pos_voxels.reshape(-1))}
                                        )
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())   
        if i > 40:
            break

    print("all sdv loaded into tfrecords")
    writer.close()

from core.ops import _parse_function
import glob
if TEST_LOADER:
    batch_size = 16
    # tfrecords loader
    training_data_files = glob.glob(tfrecord_path + '/' + '*.tfrecord')
    dataset = tf.data.TFRecordDataset(training_data_files)
    dataset = dataset.map(_parse_function)
    # Shuffle the data set
    dataset = dataset.shuffle(buffer_size=batch_size)

    # Repeat the input indefinitely
    dataset = dataset.repeat()

    # Generate batches
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 2)
    # Create a one-shot iterator
    iterator = dataset.make_one_shot_iterator()
    anc, pos = iterator.get_next()
    with tf.Session() as sess:
        anc_eval = sess.run(anc)
        import pdb; pdb.set_trace()
        pos_eval = sess.run(pos)
    print(anc.shape)

%matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sdv = read_sdv_csv(sdv_files[0])[22]
grid = sdv.reshape(16,16,16) > 1e-4

colors_b = sdv / sdv.max()
colors_r = np.zeros((16,16,16,1), dtype=np.float32) 
colors_a = np.zeros((16,16,16,1), dtype=np.float32) + 0.6
colors = np.concatenate((colors_r, colors_b,colors_b,colors_a),axis=3)

fig = plt.figure()
ax = plt.gca(projection='3d')
ax.voxels(grid, facecolors=colors)
plt.show()

print("END OF SCRIPT")
    