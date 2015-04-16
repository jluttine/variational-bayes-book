Variational message passing
===========================

If the model is in the conjugate exponential family, the VB-EM algorithm can be
implemented as the variational message passing (VMP) algorithm
(:cite:`Winn:2005`).  The algorithm is based on local computations, in which
each factor is represented by a node.  When a node is updated, it receives
messages from its children and parents and uses those messages to compute the
new approximate distribution and relevant expectations.  The advantage of this
message passing formulation is that the computations are local and depend on
well-defined messages.  This makes it easy to modify the model by adding or
changing nodes.
