import logging
import numpy as np
import matplotlib.path


# initialize logger
logger = logging.getLogger(__name__)


def paint_roi(img,roi): 
    '''Give uniform color to pixels outside the ROI
    To assist the segmentation algorithm, pixels outside the ROI
    are given a uniform, odd color. This enhances the probability
    of superpixel boundaries coinciding with the ROI boundary and 
    therefore helps avoiding splitting of superpixels by the ROI.
    Parameters
    ----------
    img : np.ndarray
        NxMxC image matrix
    roi : np.ndarray
        Kx2 matrix containing ROI vertices in UV coordinates
    Returns
    -------
    np.ndarray
        NxMxC image matrix with colored pixels outside ROI
    '''

    logger.info('Adjusting pixel intensity outside ROI...')
    
    iroi = get_roi_mask(roi,img.shape)
    
    # cols = [251,110,82] # flamingo color
    cols = [0,0,0] # black has better performance

    for i in range(img.shape[-1]):
        img[:,:,i][iroi] = cols[i]

    return img    


def get_roi_mask(roi,imshape):
    '''Get mask for image based on ROI
    Parameters
    ----------
    roi : np.ndarray
        Kx2 matrix containing ROI vertices in UV coordinates
    imshape : tuple
        tuple containing image dimensions
    Returns
    -------
    np.ndarray
        NxM mask matrix
    '''

    proi = matplotlib.path.Path(roi)
    ny,nx = imshape[:2]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    imcoor = np.vstack((x,y)).T

    iroi = proi.contains_points(imcoor)
    iroi = np.invert(iroi.reshape((ny,nx)))
    
    return iroi
