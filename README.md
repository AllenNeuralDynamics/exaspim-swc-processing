# exaspim-swc-processing

![Version](https://img.shields.io/badge/version-0.0.0-black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python](https://img.shields.io/badge/python->=3.11,<3.14-blue?logo=python)
![support](https://img.shields.io/badge/support-supported-brightgreen) 


Processing and packaging of exaSPIM neuron SWC reconstructions into per-cell AIND derived data assets.

## Installation

If you choose to clone the repository, you can install the package by running the following command from the root directory of the repository:

```bash
uv sync
```

## Development
Please test your changes using linting and testing:
```bash
uv run interrogate --verbose # Checks docstring coverage
uv run ruff check # Runs format checks
uv run pytest # Check tests and test coverage 
```

## License

This project is licensed under the  License - see the [LICENSE](LICENSE) file for details.
