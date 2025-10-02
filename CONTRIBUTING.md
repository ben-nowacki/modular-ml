# Contributing to ModularML

Thanks for your interest in contributing!
We welcome bug reports, feature requests, and code contributions.

## Installation for Development
Clone your fork and install with development dependencies:
```bash
git clone https://github.com/REIL-UConn/modular-ml.git
cd modular-ml
pip install -e .[dev,all]
```

## Workflow
1. **Open an issue** to discuss new features or bugs before coding.
2. **Create a branch** for development (e.g., `feature/my-new-features`)
3. **Write your code and tests:**
   - Follow PEP8
   - Use descriptive commit messages (remember that these are forever public)
4. **Run checks locally** before pushing:
   ```bash
   nox -s pre-commit   # formatting, linting
   nox -s unit         # unit tests
   nox -s integration  # integration tests (if applicable)
   nox -s examples     # notebook tests
   nox -s doctests     # doctests
   nox -s docs         # check that docs build
   ```
5. **Open a pull request** using the provided [template]([modul](https://github.com/REIL-UConn/modular-ml/pulls))



## Best Practices
* **Code style:** enforced by ruff (see the `ruff.toml` file for included/excluded styling checks)
* **Naming:** use descriptive, consisten names
  - Class are CamelCase, functions/variables are snake_case
* **Dependencies:**
   - Core library should remain lightweight. Do not directly import optional libraries (e.g., PyTorch, TensorFlow).
   - Backends should instead be wrapped in guard statement to ensure it is imported prior to calling it in any classes/method.
   - Users can install specific backends via: `pip install modularml[all-torch]`


## Testing

We use pytest + nox.
* **Unit tests**: fast, isolated (pytest -m unit).
* **Integration tests**: check workflows (pytest -m integration).
* **Notebooks**: tested with nbmake.

Every new feature should include a test in `tests/unit/` or `tests/integration/`.


## Documentation

Build docs locally with:

```bash
nox -s docs
```

Docstrings must follow PEP257.
