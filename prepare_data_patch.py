"""
    DEPRECATED - DO NOT USE
"""

from preprocessing import *
import os
import numpy as np

# Train vars
root_path = './data'
train_path = root_path + '/ct_mr_nrad'
dom_a_path = train_path + '/testA' # CT
dom_b_path = train_path + '/testB' # MR
# train_seg_path = train_path + '/annotations'
# seg_root_path = root_path + '/annotations'
nii_ext_name = '.nii.gz'
scan_paths_trainA = get_image_paths_given_substr(dom_a_path, '.nii')
scan_paths_trainB = get_image_paths_given_substr(dom_b_path, '.nii')
scan_names_A = [ p.split('/')[-1].strip('.nii.gz') for p in scan_paths_trainA ]
scan_names_B = [ p.split('/')[-1].strip('.nii.gz') for p in scan_paths_trainB ]

# os.makedirs(train_seg_path, exist_ok=True)
# os.makedirs(dom_a_path, exist_ok=True)
# os.makedirs(dom_b_path, exist_ok=True)

print("Converting zipped nii to npz")
prepare_volume_as_npz(scan_paths_trainA, nii_ext_name, dom_a_path)
prepare_volume_as_npz(scan_paths_trainB, nii_ext_name, dom_b_path)

# print("Getting all segmentations")
# prepare_seg_as_npz(seg_root_path, scan_names, train_seg_path)

"""
print("Processing npz volume files to npz image slices")
npz_file_paths = get_image_paths_given_substr(train_path, '.npz')

crops = np.load('./crop_idx.npz', allow_pickle=True)['data']

for scan_path in npz_file_paths:
    scan_name = scan_path.replace(".npz", "").split('/')[-1]
    seg_path = train_seg_path + '/' + scan_name + '.npz'
    is_ct = is_ct_file(scan_path)
    is_mr = not is_ct

    print(scan_name, is_ct)
    scan = np.load(scan_path)['data']
    # seg = np.load(seg_path)['data']

    # crop scan
    # cropped_scan = crop_volume(scan, seg, is_mr)
    if crops.item().get(scan_name) is not None:    
        idx = crops.item().get(scan_name)
        cropped_scan = scan[idx[0]:idx[1], idx[2]:idx[3], idx[4]: idx[5]]
        # get all patches
        all_patches = get_all_patches(cropped_scan)
        print(all_patches.shape)
        for i, patch in enumerate(all_patches):
            dom_path = dom_b_path
        
            if (is_ct):
                dom_path = dom_a_path

            save_path = dom_path + '/' + scan_name + '_' + str(i) + '.npz'
        
            # patch = resize_img(patch, 128)
            np.savez(save_path, data=patch)

"""

def prepare_data(path, target_path, crops):
    npz_file_paths = get_image_paths_given_substr(path, '.npz')

    for scan_path in npz_file_paths:
        scan_name = scan_path.replace(".npz", "").split('/')[-1]
        # seg_path = train_seg_path + '/' + scan_name + '.npz'
        # is_ct = is_ct_file(scan_path)
        # is_mr = not is_ct

        print(scan_name)
        scan = np.load(scan_path)['data']
        # seg = np.load(seg_path)['data']

        # crop scan
        # cropped_scan = crop_volume(scan, seg, is_mr)
        if crops.item().get(scan_name) is not None:    
            idx = crops.item().get(scan_name)
            cropped_scan = scan[idx[0]:idx[1], idx[2]:idx[3], idx[4]: idx[5]]
            # get all patches
            print('cropped ', cropped_scan.shape)
            all_patches = get_all_patches(cropped_scan, side='a', dim=128)
            print(all_patches.shape)
            for i, patch in enumerate(all_patches):
                dom_path = target_path
          
                save_path = dom_path + '/' + scan_name + '_' + str(i) + '.npz'
          
                # patch = resize_img(patch, 128)
                np.savez(save_path, data=patch)

"""
crops = np.load('./crop_new_idx.npz', allow_pickle=True)['data']
# call function here!
prepare_data(dom_a_path, dom_a_path + '/trainA', crops)
prepare_data(dom_b_path, dom_b_path + '/trainB', crops)
"""

