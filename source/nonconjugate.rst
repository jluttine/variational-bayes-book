Non-conjugate methods
=====================

* tilted VB

* general methods: black box variational inference etc


For conjugate exponential family models, it is possible to use collapsed
variational inference, which marginalises some of the variables (see, e.g.,
:cite:`Teh:2007,Hensman:2012`).  It is also possible to extend VB inference to a
wide range of non-conjugate models (see, e.g.,
:cite:`Knowles:2011,Hensman:2014:tvb`).  In order to have minimal amount of
model specific computations, one can use black box variational inference
(:cite:`Saliman:2013,Paisley:2013,Ranganath:2014`).  It uses stochastic
optimization and computes noisy gradients of the VB lower bound by sampling from
the approximate posterior distribution :math:`q(Z)` to estimate the relevant
expectations.  In principle, the method can be applied to any model for which
the (unnormalized) joint density :math:`p(Y,Z)` can be computed.
