Variational Bayesian approximation
==================================

.. todo::
   
   * Kullback-Leibler divergence

   * lower bound

   * mean-field, fixed form

   * VB-EM (conjugate exponential family models)



In variational Bayesian (VB) methods, the idea is to find an approximate
distribution :math:`q(Z)` which is close to the true posterior distribution
:math:`p(Z|Y)` (see, e.g.,
:cite:`Jordan:1998,Attias:2000,Bishop:2006,Fox:2012`).  The dissimilarity is
defined as the Kullback-Leibler (KL) divergence of :math:`p(Z|Y)` from
:math:`q(Z)`:

.. math::

   \operatorname{KL}( q||p) &= - \int q(Z) \log \frac {p(Z|Y)} { q(Z)}
                             \mathrm{d} Z .

The divergence is always nonnegative and zero only when :math:`q(Z)=p(Z|Y)`.
However, the divergence cannot typically be evaluated because the true posterior
distribution is intractable.


The key idea in VB is that the divergence can be minimized indirectly by
maximizing another function which is tractable.  This function is a lower bound
of the log marginal likelihood.  It can be found by decomposing the log marginal
likelihood as

.. math::
   
   \begin{split}
    \log p(Y) &= \int q(Z) \log p(Y) \mathrm{d}Z
    \\
    &= \int q(Z) \log\frac{p(Y,Z)}{p(Z|Y)} \mathrm{d}Z
    \\
    &= \int q(Z) \log \frac {p(Y,Z) q(Z)} {p(Z|Y) q(Z)} \mathrm{d}Z
    \\
    &= \int q(Z) \log \frac {p(Y,Z)} {q(Z)} \mathrm{d}Z - \int q(Z) \log \frac
    {p(Z|Y)} {q(Z)} \mathrm{d}Z
    \\
    &= \int q(Z) \log \frac {p(Y,Z)} {q(Z)} \mathrm{d}Z + \mathrm{KL}( q \| p )
    \\
    &\geq \int q(Z) \log \frac {p(Y,Z)} {q(Z)} \mathrm{d}Z
    \\
    &\equiv \mathcal{L}( q).
   \end{split}

Because the KL divergence is always non-negative, :math:`\mathcal{L}(q)` is a
lower bound for :math:`\log p(Y)`.  Furthermore, because the sum of
:math:`\mathcal{L}(q)` and :math:`\mathrm{KL}(q \| p)` is constant with respect to
:math:`q`, maximizing :math:`\mathcal{L}( q)` is equivalent to minimizing
:math:`\mathrm{KL}(q \| p)`.  In some cases, even :math:`\mathcal{L}(q)` may be
intractable and further approximations are needed to find a tractable lower
bound.



Thus far, there is nothing approximate in the procedure, because the optimal
solution which minimizes the divergence is the true posterior distribution
:math:`q(Z) = p(Z|Y)`.  In order to find a tractable solution, the range of
functions needs to be restricted in some way.  However, the range must be as
rich and flexible as possible in order to find as good an approximation as
possible.  This can be achieved by assuming a fixed functional or factorial form
for the distribution.



This work restricts the class of approximate distributions by assuming that the
:math:`q` distribution factorizes with respect to some grouping of the variables:

.. math::
   :label: eq-vb-factorized
   
   q(Z) = \prod^M_{m=1}  q_m(Z_m),

where :math:`Z_1,\ldots,Z_M` form a partition of :math:`Z`.  The notation is kept less
cluttered by ignoring the subscript on :math:`q`.  The lower bound :math:`\mathcal{L}(q)`
can be maximized with respect to one factor at a time.  The optimal factor
:math:`q(Z_m)` can be found by inserting the approximate distribution
\eqref{eq:VB_factorized} to :math:`\mathcal{L}(q)` and maximizing :math:`\mathcal{L}(q)`
with respect to :math:`q(Z_m)`.  This yields

.. math::
   :label: eq-vb-update
   
   q(Z_m) = \exp\left( \langle \log p(Y,Z) \rangle_{\setminus m} \right) ,

where the expectation is taken over all the other factors except :math:`q(Z_m)`.
An iterative update of the factors until convergence is called the variational
Bayesian expectation maximization (VB-EM) algorithm
(:cite:`Attias:2000,Beal:2003`).  Alternatively, it is also possible to use, for
instance, gradient-based optimization methods to optimize the parameters of the
approximate :math:`q` distributions (see, e.g., :cite:`Honkela:2010` or
stochastic variational inference (:cite:`Hoffman:2013`) in order to improve
scaling to large datasets with stochastic optimization.


.. _fig-vb-illustration:

.. figure:: _images/vb-illustration.*
    :align: center

    An illustration of typical effects of the factorizing approximation.  A true
    posterior (in black) and the optimal factorizing VB posterior (in red).


:num:`Fig. #fig-vb-illustration` illustrates typical effects of the factorizing
VB approximation: only one mode of the true posterior is captured, dependencies
between variables are lost and marginal distributions are too narrow.

