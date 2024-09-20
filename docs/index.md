# autora-experimentalist-example

Explain what your contribution is doing here

## Inputs

The experiment sampling method considers the following inputs:

- **X**: A numpy array or pandas DataFrame representing the experimental conditions. Each row corresponds to a different condition, and each column represents a different feature of the conditions.
- **models**: A list of models that are used to evaluate the conditions. These models help in determining the most informative conditions to sample.
- **reference\_conditions** (optional): A numpy array or pandas DataFrame representing reference conditions that the sampled conditions should be compared against.
- **num**: An integer specifying the number of conditions to sample.

These inputs are considered to ensure that the sampling method can effectively identify the most informative experimental conditions based on the provided data and models.

## Sampling Method

The sampling method used is an adapted model disagreement method. This hybrid approach combines random sampling with model disagreement. It returns selected samples for independent variables for which the models disagree the most in terms of their predictions and then takes random samples from it.

This approach is chosen because it allows for a more targeted and efficient sampling process, focusing on conditions that are likely to yield the most valuable insights for the experiment.