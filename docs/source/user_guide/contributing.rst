:orphan:

.. _contributing:

=================
Contributing
=================

We welcome contributions to ModularML!

Setup
-----

1. Fork and clone the repository.
2. Install development dependencies:
    
.. code-block:: bash

    pip install -e .[all,dev]

3. Verify the installation with tests:

.. code-block:: bash

    nox -s unit

Workflow
--------

- Use feature branches and create Pull Requests for review.
- Write descriptive commit messages.
- Run `nox -s pre-commit` before pushing changes.

Coding style
------------

- Follow PEP8 conventions.
- We use `ruff` for linting and formatting (via pre-commit hooks).

Tests
-----

- All new features should include unit tests.
- Run tests locally with:

.. code-block:: bash

    nox -s unit

Docs
----

- Docstrings are required for public functions and classes.
- Build docs locally with:

.. code-block:: bash

    nox -s docs
