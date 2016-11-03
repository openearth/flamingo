import numpy as np
import cv2


def find_homography(UV, XYZ, K, distortion=np.zeros((1,4)), z=0):
    '''Find homography based on ground control points

    Parameters
    ----------
    UV : np.ndarray
        Nx2 array of image coordinates of gcp's
    XYZ : np.ndarray
        Nx3 array of real-world coordinates of gcp's
    K : np.ndarray
        3x3 array containing camera matrix
    distortion : np.ndarray, optional
        1xP array with distortion coefficients with P = 4, 5 or 8
    z : float, optional
        Real-world elevation on which the image should be projected

    Returns
    -------
    np.ndarray
        3x3 homography matrix

    Notes
    -----
    Function uses the OpenCV image rectification workflow as described in
    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    starting with solvePnP.

    Examples
    --------
    >>> camera_id = 4
    >>> r = argus2.rest.get_rectification_data('kijkduin')
    >>> H = flamingo.rectification.find_homography(r[camera_id]['UV'],
                                                   r[camera_id]['XYZ'],
                                                   r[camera_id]['K'])
    '''

    UV = np.asarray(UV).astype(np.float32)
    XYZ = np.asarray(XYZ).astype(np.float32)
    K = np.asarray(K).astype(np.float32)
    
    # compute camera pose
    rvec, tvec = cv2.solvePnP(XYZ, UV, K, distortion)[-2:]
    
    # convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    
    # assume height of projection plane
    R[:,2] = R[:,2] * z

    # add translation vector
    R[:,2] = R[:,2] + tvec.flatten()

    # compute homography
    H = np.linalg.inv(np.dot(K, R))

    # normalize homography
    H = H / H[-1,-1]

    return H


def get_pixel_coordinates(img):
    '''Get pixel coordinates given an image

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing u-coordinates
    np.ndarray
        NxM matrix containing v-coordinates
    '''

    # get pixel coordinates
    U, V = np.meshgrid(range(img.shape[1]),
                       range(img.shape[0]))

    return U, V


def rectify_image(img, H):
    '''Get projection of image pixels in real-world coordinates
       given an image and homography

    Parameters
    ----------
    img : np.ndarray
        NxMx1 or NxMx3 image matrix
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    '''

    U, V = get_pixel_coordinates(img)
    X, Y = rectify_coordinates(U, V, H)

    return X, Y


def rectify_coordinates(U, V, H):
    '''Get projection of image pixels in real-world coordinates
       given image coordinate matrices and  homography

    Parameters
    ----------
    U : np.ndarray
        NxM matrix containing u-coordinates
    V : np.ndarray
        NxM matrix containing v-coordinates
    H : np.ndarray
        3x3 homography matrix

    Returns
    -------
    np.ndarray
        NxM matrix containing real-world x-coordinates
    np.ndarray
        NxM matrix containing real-world y-coordinates
    '''

    UV = np.vstack((U.flatten(),
                    V.flatten())).T

    # transform image using homography
    XY = cv2.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]
    
    # reshape pixel coordinates back to image size
    X = XY[:,0].reshape(U.shape[:2])
    Y = XY[:,1].reshape(V.shape[:2])

    return X, Y
