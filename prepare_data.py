"""
  given zipped nii files, generate a folder structure
  
  - data
  -- trainA
  -- trainB
  -- testA
  -- testB
"""
from preprocessing import *


# variables
root_path = './data/visceral'
npz_path = root_path + '/npz'
npz_seg_path = npz_path + '/annotations'
seg_root_path = root_path + '/annotations'
nii_ext_name = '.nii.gz'
scan_paths = get_image_paths_given_substr(root_path, '.nii')
scan_names = [ p.split('/')[-1].strip('.nii.gz') for p in scan_paths ]
path_ct = './data/visceral/npz/ct'
path_mr = './data/visceral/npz/mr'

print("Converting zipped nii to npz")
prepare_npz_from_volume(scan_paths, nii_ext_name, npz_path)

print("Getting all segmentations")
prepare_seg_as_npz(seg_root_path, scan_names, npz_seg_path)

print("Processing npz volume files to npz image slices")
npz_file_paths = get_image_paths_given_substr(npz_path, '.npz')

for p in npz_file_paths:

    filename = p.replace(".npz", "").split('/')[-1]
    full_scan = np.load(p)['data']
    for i, sl in enumerate(full_scan):
        slice_path = (path_ct if is_ct_file(filename) else path_mr) + '/' + filename + '_' + str(i)
        np.savez(slice_path, data=sl)

# prepare trainA and testA
# domain A - CT, B - MRI

print("Put them in cycleGAN folder structure")

trainA_path = npz_path + '/final/trainA'
testA_path = npz_path + '/final/testA'
trainB_path = npz_path + '/final/trainB'
testB_path = npz_path + '/final/testB'

ct_slice_paths = get_image_paths_given_substr(path_ct, '.npz')
mr_slice_paths = get_image_paths_given_substr(path_mr, '.npz')

prepare_train_test_set(ct_slice_paths, trainA_path, testA_path, npz_seg_path)
prepare_train_test_set(mr_slice_paths, trainB_path, testB_path, npz_seg_path)

