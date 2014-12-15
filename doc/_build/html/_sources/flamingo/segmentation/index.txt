Segmentation
============

This module provides functions to segmentate an image into
superpixels. It is largely based on the *scikit-image* toolbox.
Apart from the regular segmentation functions it provides
postprocessing functions to ensure connected segments in a regular
grid. It also provides various visualization tools for segmented
images.

.. seealso::
   http://scikit-image.org/docs/0.10.x/api/skimage.segmentation.html

Superpixels
-----------

.. automodule:: flamingo.segmentation.superpixels
   :members:

Postprocessing
--------------

.. automodule:: flamingo.segmentation.postprocess
   :members:

Visualization
-------------

.. automodule:: flamingo.segmentation.plot
   :members:
