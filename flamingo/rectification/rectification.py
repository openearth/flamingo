import numpy as np
import cv2

def find_homography(UV, XYZ, K, distortion=0, z=0):

    UV = np.asarray(UV).astype(np.float32)
    XYZ = np.asarray(XYZ).astype(np.float32)
    K = np.asarray(K).astype(np.float32)
    
    # compute camera pose
    rvec, tvec = cv2.solvePnP(XYZ, UV, K, distortion)[1:]
    
    # convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    
    # assume height of projection plane
    R[:,2] *= z

    # add translation vector
    R[:,2] += tvec

    # compute homography
    H = np.linalg.inv(np.dot(K, R))

    # normalize homography
    H = H / H[-1,-1]

    return H

def rectify_image(img, H, K=None, distortion=0):

    # undistort image
    if K is not None:
        img = cv2.undistort(img, K, distortion)

    # get pixel coordinates
    U, V = np.meshgrid(range(img.shape[1]),
                       range(img.shape[0]))

    UV = np.vstack((U.flatten(),
                    V.flatten())).T

    # transform image using homography
    XY = cv2.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]
    
    # reshape pixel coordinates back to image size
    X = XY[:,0].reshape(img.shape[:2])
    Y = XY[:,1].reshape(img.shape[:2])

    return X, Y
