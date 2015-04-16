General black-box framework
===========================


"Black box"
-----------

In order to have minimal amount of model specific computations, one can use
black box variational inference (:cite:`Paisley:2013,Ranganath:2014`).  It uses
stochastic optimization and computes noisy gradients of the VB lower bound by
sampling from the approximate posterior distribution :math:`q(Z)` to estimate
the relevant expectations.  In principle, the method can be applied to any model
for which the (unnormalized) joint density :math:`p(Y,Z)` can be computed.


Variational approximation as linear regression
----------------------------------------------

:cite:`Salimans:2013`


Gradient-based approximation
----------------------------

Titsias:2014

continuous variable and differentiable density
