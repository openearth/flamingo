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

.. automodule:: flamingo.classification.models
   :members:

Features
--------

.. automodule:: flamingo.classification.features.features
   :members:

Blocks
^^^^^^

.. automodule:: flamingo.classification.features.blocks
   :members:

Scale invariant features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flamingo.classification.features.scaleinvariant
   :members:

Normalizing features
^^^^^^^^^^^^^^^^^^^^

.. automodule:: flamingo.classification.features.normalize
   :members:

Relative location prior
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flamingo.classification.features.relativelocation
   :members:

Channels
--------

.. automodule:: flamingo.classification.channels
   :members:

Test
----

.. automodule:: flamingo.classification.test
   :members:

Visualization
-------------

.. automodule:: flamingo.classification.plot
   :members:

Utils
-----

.. automodule:: flamingo.classification.utils
   :members:
