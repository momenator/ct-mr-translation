from preprocessing import *
import os
import numpy as np

"""
    Before running the script, make sure to have the scans in the structure
    below. Names may vary:

    input:
    training scans - ./data/name_of_data/train/*.nii.gz
    test scans - ./data/name_of_data/test/*.nii.gz
    annotation masks - ./data/annotations/*.nii.gz (optional)
    crop_file - *.npz (optional, manual crops of scan of region of interest) 

    output:
    training - ./data/name_of_data/train/trainA
             - ./data/name_of_data/train/trainB
             - ./data/name_of_data/train/annotations
    
    testing  - ./data/name_of_data/train/testA
             - ./data/name_of_data/train/testB
             - ./data/name_of_data/train/annotations

    Then transfer trainA, trainB, testA, testB to CycleGAN/datasets/name_of_data
"""

def get_patches(scan_path, scan_name, side='c', patch_size=256, patch_step=(128, 128)):
    scan = np.load(scan_path)['data']

    # crop scan using segmentation - currently not in use
    # seg = np.load(seg_path)['data']
    # cropped_scan = crop_volume(scan, seg, is_mr)
     
    # get all patches
    all_patches = get_all_patches(scan, side=side, dim=patch_size, step=patch_step)
    
    print(all_patches.shape)

    return all_patches


def prepare_data(root_path, crops, is_train = True, is_prep_npz=True, is_prep_seg=False, side='c', patch_size=256, patch_step=(128, 128)):

    data_type = 'train'
    if is_train is False:
        data_type = 'test'

    # root_path = './data/visceral_full'
    
    train_path = root_path + '/' + data_type
    dom_a_path = train_path + '/{}A'.format(data_type) # CT
    dom_b_path = train_path + '/{}B'.format(data_type) # MR
    
    train_seg_path = train_path + '/annotations'
    seg_root_path = root_path + '/annotations'
    
    nii_ext_name = '.nii.gz'
    scan_paths_train = get_image_paths_given_substr(train_path, '.nii')
    scan_names = [ p.split('/')[-1].strip('.nii.gz') for p in scan_paths_train ]

    # os.makedirs(train_seg_path, exist_ok=True)
    # os.makedirs(dom_a_path, exist_ok=True)
    # os.makedirs(dom_b_path, exist_ok=True)
    
    if is_prep_npz is True:
        print("Converting zipped nii to npz with crops")
        os.makedirs(dom_a_path, exist_ok=True)
        os.makedirs(dom_b_path, exist_ok=True)
        prepare_volume_as_npz(scan_paths_train, nii_ext_name, train_path, crops)
        
    if is_prep_seg is True:
        print("Getting all segmentations")
        os.makedirs(train_seg_path, exist_ok=True)
        prepare_seg_as_npz(seg_root_path, scan_names, train_seg_path, crops)

    # only generate slices when preparing training data!
    if is_train is True:
        
        print("Processing npz volume files to npz image slices")
        
        npz_file_paths = get_image_paths_given_substr(train_path, '.npz')

        for scan_path in npz_file_paths:
            scan_name = scan_path.replace(".npz", "").split('/')[-1]
            print(scan_name)
            # seg_path = train_seg_path + '/' + scan_name + '.npz'
            is_ct = is_ct_file(scan_path)
            # is_ct = 'T1' not in scan_path - for ct_mr_nrad
            # is_ct = 'T1' not in scan_path
            is_mr = not is_ct

            # get all patches
            all_patches = get_patches(scan_path, scan_name, side=side, patch_size=patch_size, patch_step=patch_step)

            for i, patch in enumerate(all_patches):
                dom_path = dom_b_path
            
                if (is_ct):
                    dom_path = dom_a_path

                save_path = dom_path + '/' + scan_name + '_' + str(i) + '.npz'
            
                # patch = resize_img(patch, 128)
                np.savez(save_path, data=patch)


if __name__ == '__main__':
    data_path = './data/visceral_full'
    crop_path = './visceral_crops.npz'
    
    # data_path = './data/ct_mr_nrad'
    # crop_path = './ct_mr_nrad_crops.npz'
    
    crops = np.load(crop_path, allow_pickle=True)['data']

    # prepare train data here
    # prepare_data(data_path, crops, is_train=True, is_prep_npz=True, is_prep_seg=False, side='c', patch_size=256, patch_step=(64, 64))

    # # prepare test data here
    prepare_data(data_path, crops, is_train=False, is_prep_npz=False, is_prep_seg=True)

