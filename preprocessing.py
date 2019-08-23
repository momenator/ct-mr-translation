"""
    Preprocessing functions for the medical scans
"""

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import os
import shutil
import math
from patchify import *


def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    """
        itk_image = itk scan (3d volume) object
        out_spacing = desired spacing of itk_image
        @returns resampled itk_image with spacing = out_spacing
    """

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    # 2. resample grid 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    
    # 3. transformation
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    # 4. interpolate
    # resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def combine_segmentations(segs):
    """
        segs: list of images
        @return one image containing all segmentation
    """
    if len(segs) == 0:
        return None
    
    original_spacing = segs[0].GetSpacing()
    original_origin = segs[0].GetOrigin()
    original_direction = segs[0].GetDirection()
    
    combined_seg = sitk.GetArrayFromImage(segs[0])
    
    for seg in segs:
        seg = sitk.GetArrayFromImage(seg)
        combined_seg = np.where(combined_seg != 0, combined_seg, seg)
    
    combined_seg = sitk.GetImageFromArray(combined_seg)

    # combined segmentation may lose the original spacing, etc
    # make sure to reset it back to the original one!
    combined_seg.SetSpacing(original_spacing)
    combined_seg.SetOrigin(original_origin)
    combined_seg.SetDirection(original_direction)
    
    return combined_seg


def print_volume_details(volume):
    """
        Print details of volume
    """
    print("--------------------")
    print("Origin ", volume.GetOrigin())
    print("Size ", volume.GetSize())
    print("Spacing ", volume.GetSpacing())
    print("Direction ", volume.GetDirection())
    print("--------------------")


def prepare_segmentations(path):
    """
        TODO: Loop though the data folder, group all 
        annotations/segmentation for one scan together
        and save them in one file!
    """
    pass

def prepare_dataset(path, side='axial'):
    """
        TODO: given path to dataset, 
        1. apply segmentation
        2. then resampled them,
        3. finally create image slices from the scans

        output a single
    """

    pass


def ls_given_path(path):
    ls = os.listdir(path)
    return ls


def get_image_paths_given_substr(root_path, substr):
    ls = os.listdir(root_path)
    filtered_ls = list(filter(lambda x: substr in x, ls))
    # append root_path
    filtered_ls = [ root_path + '/' + path for path in filtered_ls ]
    return filtered_ls


def get_image_given_path(path, data_type=None):
    if data_type is not None:
        return sitk.ReadImage(path, data_type)
    return sitk.ReadImage(path)


def get_images_given_path(paths, data_type):
    images = []
    for path in paths:
        if data_type is not None:
            curr = get_image_given_path(path, data_type)
        else:
            curr = get_image_given_path(path)
        images.append(curr)
    return images


def normalise_scan(scan):
    return (scan - scan.mean()) / scan.std()


def prepare_volume_as_npz(scan_paths=[], ext_name='', save_dir=''):
    """
        prepare npzs file from volumes
    """
    for scan_path in scan_paths:
        scan_path_stripped = str(scan_path).strip(ext_name)
        scan_name = scan_path_stripped.split('/')[-1]

        print(scan_name)

        current_scan = get_image_given_path(scan_path)
        
        # resample
        resampled_scan = resample_img(current_scan, [1, 1, 1])

        # normalise - Gaussian with zero mean and unit variance
        normalised_scan = normalise_scan(sitk.GetArrayFromImage(resampled_scan))

        # save array
        save_image_as_npz(normalised_scan, save_dir + '/' + scan_name, is_numpy_arr=True)


def save_image_as_npz(image, output_dir, is_numpy_arr=False):
    if is_numpy_arr: 
        image_arr = image
    else:
        image_arr = sitk.GetArrayFromImage(image)
    
    np.savez(output_dir, data=image_arr)



def combine_segmentations(segs):
    """
        segs: list of images (simple itk format)
        @return one image containing all segmentation
    """
    if len(segs) == 0:
        return None
    
    original_spacing = segs[0].GetSpacing()
    original_origin = segs[0].GetOrigin()
    original_direction = segs[0].GetDirection()
    
    combined_seg = sitk.GetArrayFromImage(segs[0])
    
    for seg in segs:
        seg = sitk.GetArrayFromImage(seg)
        combined_seg = np.where(combined_seg != 0, combined_seg, seg)
    
    combined_seg = sitk.GetImageFromArray(combined_seg)
    combined_seg.SetSpacing(original_spacing)
    combined_seg.SetOrigin(original_origin)
    combined_seg.SetDirection(original_direction)
    return combined_seg


def is_ct_file(file_path):
    return 'ct' in file_path.lower()


def prepare_seg_as_npz(seg_path, scan_names, output_dir):
    for scan_name in scan_names:
        seg_paths = get_image_paths_given_substr(seg_path, scan_name)
        segs = get_images_given_path(seg_paths, data_type=sitk.sitkUInt8)
        combined_segs = combine_segmentations(segs)
        combined_segs = resample_img(combined_segs, [1, 1, 1])
        save_image_as_npz(combined_segs, output_dir + '/' + scan_name)


def prepare_train_test_set(data_paths, train_path, test_path, seg_path, train_ratio=0.8):
    num_imgs = len(data_paths)
    num_train = int(math.floor(num_imgs * train_ratio))
    num_test = int(math.ceil(num_imgs * (1 - train_ratio)))
    train_indices = np.random.choice(num_train, num_train, replace=False)
    test_indices = np.random.choice(num_test, num_test, replace=False)

    for i, data_path in enumerate(data_paths):
        print(i, data_path)
        filename = data_path.split('/')[-1]
        target_path = (train_path if i in train_indices else test_path) + '/' + filename
        
        # preprocess here: adjust FOV -> resize to 256
        try:
            img = np.load(data_path)['data']
            img_seg = get_img_segmentation(seg_path, filename)
            processed_img = preprocess_img(img, img_seg, 256)
            
            # save target here!
            np.savez(target_path, data=processed_img)
        except:
            continue


def get_img_segmentation(seg_path, img_name):
    im_idx = int(img_name.strip('.npz').split('_')[-1])
    img_name_formatted = img_name.strip('_{}.npz'.format(im_idx))
    seg = get_image_paths_given_substr(seg_path, img_name_formatted)
    # should be in npz format so use np.load!
    seg = np.load(seg[0])['data']
    return seg[im_idx]


def get_y_axis_segmentation_bound(seg_img):
    # 0 is the axial axis
    seg_idx_y_axis = np.where(seg_img != 0)[0]
    
    # in case there is no segmentation!
    if len(seg_idx_y_axis) == 0:
        midpoint = seg_img.shape[0] // 2
        return (midpoint, midpoint)
    return (np.min(seg_idx_y_axis), np.max(seg_idx_y_axis))


def preprocess_img(img, img_seg, size):
    # is image square?
    x, y = img.shape
    is_sqr_image = x == y
    
    # if square
    if is_sqr_image:
        return preprocess_square_img(img, img_seg, size)
    else:
        return preprocess_rectangle_img(img, size)


def resize_img(img, size=256):
    x, y = img.shape
    return zoom(img, (size/x, size/y))


def preprocess_rectangle_img(img, size):
    min_val = np.min(img)
    sqr_img = pad_img_to_square(img, min_val)
    return resize_img(sqr_img, size)


def preprocess_square_img(img, img_seg, size):
    min_val = np.min(img)
    y_min, y_max = get_y_axis_segmentation_bound(img_seg)
    min_idx = y_min - (160 // 2) + 35
    max_idx = y_min + (160 // 2) + 35
    padded_img = pad_img_to_square(img[min_idx:max_idx,:], min_val)
    return resize_img(padded_img, size)


def pad_img_to_square(M,val=0):
    (a,b) = M.shape
    if a > b:
        padding=((0,0),(((a-b)//2),((a-b)//2)))
    else:
        padding=(((b-a)//2, (b-a)//2),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)


def crop_volume(volume, segmentation):
    seg_not_zero = np.where(segmentation != 0)

    x_bounds = (np.min(seg_not_zero[0]), np.max(seg_not_zero[0]))
    y_bounds = (np.min(seg_not_zero[1]), np.max(seg_not_zero[1]))
    z_bounds = (np.min(seg_not_zero[2]), np.max(seg_not_zero[2]))

    return volume[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1], z_bounds[0]: z_bounds[1]]


def get_all_patches(volume, side='c', dim=256):
    """
        side = either 'c', 'a', 's'
        a - axial
        c - coronal
        s - sagittal
    """
    a, c, s = volume.shape
    all_patches = []
    
    if side == 'a':
        count = a
    elif side == 'c':
        count = c
    else:
        count = s
    
    for i in range(count):
        if side == 'a':
            scan_slice = volume[i,:,:]
        elif side == 'c':
            scan_slice = volume[:,i,:]
        else:
            scan_slice = volume[:,:,i]
        patches = get_patches_from_2d_img(scan_slice)
        all_patches.append(patches)
    
    all_patches = np.array(all_patches).reshape(-1, dim, dim)
    
    return all_patches

def get_patches_from_2d_img(image, dim=256, step=64):
    patches = patchify(image, (dim, dim), step=step)
    _, _, x, y = patches.shape
    patches = patches.reshape(-1, x, y) 
    return patches