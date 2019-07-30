"""
    Preprocessing functions for the medical scans
"""

import SimpleITK as sitk
import numpy as np
import os


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


def get_segmentation_paths_given_filename(seg_path, filename):
    ls = os.listdir(seg_path)
    filtered_ls = list(filter(lambda x: filename in x, ls))
    return [ seg_path + path for path in filtered_ls ]


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

