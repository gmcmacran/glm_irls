# What is this library?
An implementation of the [generlized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) 
using [iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)

The models implemented are:
* Gaussian model with identity, log, and reciprocal links.
* Binomial model with logit and probit links.
* Poisson model with natural log, identity, and square root links.
* Gamma model with inverse, identity, and log links.
* Inverse Gaussian model with mu^-2, inverse, identity, and log links.

# What does this library depend on?
* `numpy`: for matrices
* `scipy`: for a few mathematical operations

