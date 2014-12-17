Rectification
=============

This module provides functions to project an image onto a
real-world coordinate system using ground control points. The
module is largely based on the *OpenCV Camera Calibration and 3D
Reconstruction* workflow and works nicely together with the *argus2*
toolbox for coastal image analysis.

A typical workflow consists of determining ground control points
by measuring the real-world coordinates of object visible in the
image and the image coordinates of these very same objects. Also
the camera matrix and lens distortion parameters should be
determined.

Subsequently, a homography can be determined using the
:func:`flamingo.rectification.rectification.find_homography()` function and a projection of the image can be
plotted using the accompanying :mod:`flamingo.rectification.plot` module.

.. seealso::
   http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

Rectification
-------------

.. automodule:: flamingo.rectification.rectification
   :members:

Visualization
-------------

.. automodule:: flamingo.rectification.plot
   :members:
