"""
    Init:
    Init data structure
    ./data/visceral
        /train
        /train
        /annotations
    
    Steps
    1. loop through train/train
    2. aggregate segmentation
    3. 
"""

from preprocessing import *
import os
import numpy as np

# Train vars
root_path = './data/visceral'
train_path = root_path + '/train'
dom_a_path = train_path + '/trainA' # CT
dom_b_path = train_path + '/trainB' # MR
train_seg_path = train_path + '/annotations'
seg_root_path = root_path + '/annotations'
nii_ext_name = '.nii.gz'
scan_paths_train = get_image_paths_given_substr(train_path, '.nii')
scan_names = [ p.split('/')[-1].strip('.nii.gz') for p in scan_paths_train ]

os.makedirs(train_seg_path, exist_ok=True)
os.makedirs(dom_a_path, exist_ok=True)
os.makedirs(dom_b_path, exist_ok=True)

print("Converting zipped nii to npz")
prepare_volume_as_npz(scan_paths_train, nii_ext_name, train_path)

print("Getting all segmentations")
prepare_seg_as_npz(seg_root_path, scan_names, train_seg_path)

print("Processing npz volume files to npz image slices")
npz_file_paths = get_image_paths_given_substr(train_path, '.npz')

for scan_path in npz_file_paths:
    scan_name = scan_path.replace(".npz", "").split('/')[-1]
    seg_path = train_seg_path + '/' + scan_name + '.npz'
    is_ct = is_ct_file(scan_path)
    is_mr = not is_ct

    print(scan_name, is_ct)
    scan = np.load(scan_path)['data']
    seg = np.load(seg_path)['data']

    # crop scan
    cropped_scan = crop_volume(scan, seg, is_mr)

    # get all patches
    all_patches = get_all_patches(cropped_scan)
    for i, patch in enumerate(all_patches):
        dom_path = dom_b_path
        
        if (is_ct):
            dom_path = dom_a_path

        save_path = dom_path + '/' + scan_name + '_' + str(i) + '.npz'
        np.savez(save_path, data=patch)

