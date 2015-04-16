Stochastic variational inference
================================

Stochastic variational inference (:cite:`Hoffman:2013`) learns shared latent
variables with stochastic optimization.  It uses subsets of the data to compute
noisy estimates of the gradients.  The method can be used to scale the inference
on large datasets, if the model has a specific structure.
