import matplotlib.pyplot as plt


def plot_image(image):
    """
        Plot an image using matplotlib
    """
    plt.imshow(image)
    plt.show()


def plot_img_histogram(image):
    """
        Plot histogram of values given a slice 
        of scan
    """
    plt.hist(image.ravel())
    plt.show()


def display_mid_slices(volume, resample=True, spacing=[1,1,1]):
    
    if resample:
        volume = resample_img(volume, spacing)
    
    volume_arr = sitk.GetArrayFromImage(volume)
        
    x, y, z = volume_arr.shape
        
    sagittal_mid_slice = volume_arr[x//2,:,:]
    axial_mid_slice = volume_arr[:,y//2,:]
    coronal_mid_slice = volume_arr[:,:,z//2]
    
    plot_image(sagittal_mid_slice)
    plot_image(axial_mid_slice)
    plot_image(coronal_mid_slice)
    
    return None

