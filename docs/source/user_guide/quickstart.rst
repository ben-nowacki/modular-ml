:orphan:

.. _quickstart:

================
Quickstart
================

This page highlights the basic workflow in ModularML.

Create a FeatureSet
-------------------

.. code-block:: python

    import modularml as mml

    fs = mml.FeatureSet(
        label="PulseFeatures",
        features={"voltage": [[3.5, 3.6, 3.7], ...]},
        targets={"soh": [0.95, ...]}
    )

Build a ModelGraph
------------------

.. code-block:: python

    from modularml.models import SequentialMLP
    from modularml.core import ModelStage, ModelGraph

    stage = ModelStage(label="Regressor", model=SequentialMLP(output_shape=(1, 1)))
    mg = ModelGraph([fs, stage])

Run Training
------------

*Documentation in progress...*

Next steps
----------

- See the :doc:`../examples/index` for more detailed tutorials.
- Review the :doc:`../api/index` for all classes and functions.

