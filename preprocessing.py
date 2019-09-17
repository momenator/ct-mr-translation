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
import torch


def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkBSpline):
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
    resample.SetInterpolator(interpolator)

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


def normalise_zero_mean_unit_var(arr):
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / std, mean, std


def denormalise_zero_mean_unit_var(arr, mean, std):
    return arr * std + mean


# normalise to [-1 1]
def normalise_tanh(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return 2 * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) - 1, min_val, max_val


def denormalise_tanh(arr, min_val, max_val):
    return ((arr + 1) / 2) * (max_val - min_val) + min_val


# center data and normalise to -1 and 1
def normalise_scan(scan):
    # set data to zero mean and unit variance
    normalised_scan, mean, std = normalise_zero_mean_unit_var(scan)
    
    # normalise to -1 and 1
    # min_val and max_val are the min and max values of normalised numpy array
    normalised_scan, min_val, max_val = normalise_tanh(normalised_scan)
    return normalised_scan, mean, std, min_val, max_val


def denormalise_scan(nrml_scan, mean, std, min_val, max_val):
    denormalised = denormalise_tanh(nrml_scan, min_val, max_val)
    return denormalise_zero_mean_unit_var(denormalised, mean, std)


def get_numpy_scan(scan_img):
    # itk image gets converted to numpy array! order is (z, y, x) for some reason
    numpy_scan = sitk.GetArrayFromImage(scan_img)
    numpy_w_fixed_axes = numpy_scan.T
    numpy_w_fixed_axes = numpy_w_fixed_axes[:, :, ::-1]
    return numpy_w_fixed_axes


def prepare_volume_as_npz(scan_paths=[], ext_name='', save_dir='', crops=None):
    """
        prepare npzs file from volumes
    """
    for scan_path in scan_paths:
        scan_path_stripped = str(scan_path).strip(ext_name)
        scan_name = scan_path_stripped.split('/')[-1]

        print(scan_name)

        if crops is not None and crops.item().get(scan_name) is None:
            continue

        current_scan = get_image_given_path(scan_path)
        
        # resample
        resampled_scan = resample_img(current_scan, [1, 1, 1])

        # convert to numpy and fix axes!
        resampled_scan = get_numpy_scan(resampled_scan)

        # normalise - Gaussian with zero mean and unit variance then scaled to [-1, 1]
        normalised_scan, mean, std, min_val, max_val = normalise_scan(resampled_scan)

        final_scan = normalised_scan

        if crops is not None and crops.item().get(scan_name) is not None:
            idx = crops.item().get(scan_name)
            final_scan = final_scan[idx[4]: idx[5], idx[2]: idx[3], idx[0]: idx[1]]

        # save array
        np.savez(save_dir + '/' + scan_name, data=final_scan, mean=mean, std=std, min_val=min_val, max_val=max_val)


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
        combined_segs = resample_img(combined_segs, [1, 1, 1], sitk.sitkNearestNeighbor)
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


def crop_volume(volume, segmentation, remove_label_1=False):
    # There is a trace segmentation labelled 1
    if remove_label_1:
        seg_not_zero = np.where((segmentation != 0) & (segmentation != 1))
        
        # there is an issue where seq_not_zero return empty arrays
        # in which just get the indices of non 0 values
        try:
            assert(len(seg_not_zero) == 3)
            assert(seg_not_zero[0].shape[0] > 0)
            assert(seg_not_zero[1].shape[0] > 0)
            assert(seg_not_zero[2].shape[0] > 0)
        except:
            seg_not_zero = np.where(segmentation != 0)
    else:
        seg_not_zero = np.where(segmentation != 0)

    x_bounds = (np.min(seg_not_zero[0]), np.max(seg_not_zero[0]))
    y_bounds = (np.min(seg_not_zero[1]), np.max(seg_not_zero[1]))
    z_bounds = (np.min(seg_not_zero[2]), np.max(seg_not_zero[2]))

    cropped = volume[x_bounds[0]: x_bounds[1], y_bounds[0]: y_bounds[1], z_bounds[0]: z_bounds[1]]
    
    dim=256
    x_width = x_bounds[1] - x_bounds[0]
    z_width = z_bounds[1] - z_bounds[0]

    if (x_width < dim):
        # expand in x direction from original scan
        pad = ((dim-x_width) // 2) + 1
        cropped = volume[x_bounds[0] - pad : x_bounds[1] + pad, y_bounds[0]: y_bounds[1],:]
        
        x_width, _, _ = cropped.shape
        
        if (x_width < dim):
            # still smaller than desired width so pad till >= 256
            pad = ((dim-x_width) // 2) + 1
            padding=((pad, pad), (0,0), (0,0))
            cropped = np.pad(cropped, padding, mode='constant',constant_values=0)    
    
    if (z_width < dim):
        pad = ((dim - z_width) // 2) + 1
        
        cropped = cropped[:, :, (z_bounds[0] - pad): (z_bounds[1] + pad)]
        
        _, _, z_width = cropped.shape
        
        if (z_width < dim):
            pad = ((dim - z_width) // 2) + 1
            padding=((0,0), (0,0), (pad, pad))
            cropped = np.pad(cropped, padding, mode='constant',constant_values=0)
    
    return cropped


def get_all_patches(volume, side='c', dim=256, step=(128, 128)):
    """
        side = either 'c', 'a', 's'
        a - axial
        c - coronal
        s - sagittal
    """
    a, c, s = volume.shape
    all_patches = []
    
    if side == 's':
        count = s
    elif side == 'c':
        count = c
    else:
        count = a
    
    for i in range(count):
        if side == 's':
            scan_slice = volume[:,:,i]
        elif side == 'c':
            scan_slice = volume[:,i,:]
        else:
            scan_slice = volume[i,:,:]
        patches = get_patches_from_2d_img(scan_slice, dim, step)
        all_patches.append(patches)
    
    all_patches = np.array(all_patches).reshape(-1, dim, dim)
    
    return all_patches

def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches


def reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img

def get_patches_from_2d_img(image, dim=256, step=128):
    patches = patchify(image, (dim, dim), step=step)
    _, _, x, y = patches.shape
    patches = patches.reshape(-1, x, y) 
    return patches
