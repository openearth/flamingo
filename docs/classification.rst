Classification
==============

This module provides functions to train image classification models, like Logistic Regressors and Conditional Random Fields.
It provides functions for feature extraction that are largely based on the *scikit-image* toolbox and
it provides functions for model training and optimization that are largely based on the *pystruct* and *scikit-learn* toolbox.

.. seealso::
   http://scikit-image.org/docs/0.10.x/api/skimage.feature.html
.. seealso::
   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
.. seealso::
   https://pystruct.github.io/references.html

Models
------

.. automodule:: classification.models
   :members:

Features
--------

.. automodule:: classification.features.features
   :members:

Blocks
^^^^^^

.. automodule:: classification.features.blocks
   :members:

Scale invariant features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: classification.features.scaleinvariant
   :members:

Normalizing features
^^^^^^^^^^^^^^^^^^^^

.. automodule:: classification.features.normalize
   :members:

Relative location prior
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: classification.features.relativelocation
   :members:

Channels
--------

.. automodule:: classification.channels
   :members:

Test
----

.. automodule:: classification.test
   :members:

Visualization
-------------

.. automodule:: classification.plot
   :members:

Utils
-----

.. automodule:: classification.utils
   :members:
