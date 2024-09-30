# Avengers - Experimentalist Challenge

## Overview

This project is a Python-based application that includes a Mixed Disagreement Model for predicting values based on input
features. The project also includes a testing setup using GitHub Actions and supports multiple Python versions and
operating systems.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized into the following main parts:

### `challenge`

This folder contains the challenge-related scripts and data. It is used for testing and validating the model against
specific challenges or datasets.

### `docs`

This folder contains the documentation for the project. It includes detailed descriptions of the modules, and functions
used in the project.

### `src`

This folder contains the source code of the project. The main component is:

- `src/autora/experimentalist/autora_experimentalist_example/__init__.py`: Contains the implementation of the sampling
  method, which selects experimental conditions based on model disagreement.

## Installation

To install the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/parthapratimkalita/autora-experimentalist-challenge-Avengers.git
   cd autora-experimentalist-challenge-Avengers
   ```

2. Install the required dependencies:
   ```sh
   pip install -e .
   ```

## Usage

To use the Mixed Disagreement Experimentalist, you can import the relevant functions from the src folder. Here is an
example:

Example:
```
import pandas as pd
from autora.experimentalist.autora_experimentalist_mixed_disagreement import sample
```

Define all experimental conditions as a DataFrame or Numpy array:
```
all_conditions = pd.DataFrame({
'condition_1': [1, 2, 3, 4],
'condition_2': [10, 20, 30, 40],
})
```

Define models to evaluate the conditions (these models need to be compatible with the model_disagreement_sample
function):
```
models = [model_1, model_2, model_3]
```

Sample experimental conditions based on model disagreement:
```
sampled_conditions = sample(all_conditions, models, num=2)
```

Display the sampled conditions:
```
print(sampled_conditions)
```

In this example:

* all_conditions: A DataFrame or NumPy array representing all possible experimental conditions.

* models: A list of models to be used for evaluating disagreement among conditions.

* num: The number of conditions to sample based on the disagreement.

## Testing

The project uses `pytest` for testing. To run the tests, use the following command:

```sh
pytest
```

The project also includes a GitHub Actions workflow (`.github/workflows/test-pytest.yml`) to automatically run tests on
different Python versions and operating systems.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



