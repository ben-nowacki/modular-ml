:orphan:

.. _installation:

===============
Installation
===============

ModularML requires Python 3.10 or higher.

Basic install
-------------

To install from PyPI:

.. code-block:: bash

   pip install modularml

Development install
-------------------

If you plan to contribute or run the test suite, clone the repo and install with extras:

.. code-block:: bash

   git clone https://github.com/REIL-UConn/modular-ml.git
   cd modular-ml
   pip install -e .[all,dev]

Check installation
------------------

To verify ModularML is installed correctly:

.. code-block:: bash

   python -c "import modularml as mml; print(mml.__version__)"

