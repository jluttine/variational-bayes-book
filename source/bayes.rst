Bayesian inference
==================


The Bayesian framework provides a principled way to model and analyze data.  The
framework uses probabilities to represent the knowledge of the modelled process
and the unknown quantities.  Thus, simple rules of probability theory can be
used for inference.  The same basic rules are used regardless of the complexity
or the application field of the problem.  The beauty of the Bayesian framework
is that it can be derived from simple axioms as a unique way of doing rational
reasoning.



Bayesian modelling has several advantages over ad hoc approaches: 1) The
probabilities account for the uncertainty in the results.  2) Missing values are
usually not a problem because the whole framework is about incomplete knowledge.
3) Model comparison can be done in a principled way.  4) Overfitting is
prevented by combining many models and taking into account their complexities.
5) Modelling assumptions and priors are expressed explicitly and can be
modified.  6) Existing models can be straightforwardly modified, extended or
used as building blocks for more complex models.




This chapter gives a brief introduction to Bayesian modelling.  Section
:ref:`sec-probability-theory` summarizes the foundation by explaining how the
probability theory can be interpreted as a unique system of consistent rational
reasoning under uncertainty.  Section :ref:`inference` shows how this theory is
applied in Bayesian modelling.

..  However, because modelling can rarely be performed exactly, some
    approximations are needed.  Sections :ref:`sec-variational-bayes` and
    :ref:`sec:mcmc` explain the variational Bayesian and Markov chain Monte Carlo
    methods, which are the relevant approximation methods in this thesis.


.. _sec-probability-theory:

Probability Theory
------------------


The Bayesian framework is based on probability theory by interpreting
probabilities as plausibility assignments.  This differs from the frequentist
approach, which interprets probabilities as frequencies in repeated experiments.
The Bayesian framework uses probability theory as an extension of logic to
handle uncertain propositions which do not need to be related to random events
or repeated experiments.  Thus, the rules of the probability theory can be
applied to a wide range of problems involving incomplete knowledge instead of
random events or repeated experiments.


The Bayesian interpretation of probability can be derived from desired
qualitative properties for rational reasoning :cite:`Jaynes:2003,Cox:1961`.  The
idea is that propositions have subjective plausibilities and the rules for
handling the plausibility assignments should have rational properties.  The
desired properties of rational reasoning can be roughly summarized as follows:

 * *Comparability:* degrees of plausibility can be compared and are represented
   by real numbers.
  
 * *Continuity:* an infinitesimally greater plausibility corresponds to an
   infinitesimally greater number.
  
 * *Logicality:* rules are consistent with Aristotelian logic.

 * *Rationality:* rules have qualitative correspondence with weak
   syllogisms.[#weak-syllogisms]_

 * *Consistency:* every possible way of reasoning must lead to the same result
   and equivalent plausibilities are represented by equal numbers.

 * *Neutrality:* all relevant evidence is taken into account without ignoring
   any information.
  
The list is a slightly rephrased and simplified version of the list presented by
:cite:`Jaynes:2003`.


From the properties of rational reasoning, one can derive a unique set of
quantitative rules.  Omitting the long and rigorous derivations
:cite:`Jaynes:2003`, the resulting rules are the well-known product rule

.. math::
   :label: eq-product-rule
   
   p(A,B) = p(A|B)p(B) = p(B|A)p(A)

and the sum rule

.. math::

  p(A) + p(\overline{A}) = 1,

where :math:`A` and :math:`B` are propositions, and :math:`\overline{A}` is the
complement of :math:`A`.  The probabilities :math:`p(\cdot)` represent the state
of knowledge, where certainty is represented by 1 and impossibility by 0.
Therefore, applying probability theory to inference problems means that one uses
common sense consistently.


From the product rule :eq:`eq-product-rule`, it follows that

.. math::
   :label: eq-bayes-theorem
   
   p(A|B) = \frac{p(A,B)}{p(B)}  = \frac{p(B|A)p(A)}{p(B)},

which is the Bayes' theorem.  This can be seen as a formula for updating the
beliefs about :math:`A` after given new evidence :math:`B`.  Thus, the
properties of rational reasoning determine how we should rationally change the
beliefs we have when given new evidence.  However, note that the rules do not
determine which beliefs are a priori rational.
kk

.. _sec-inference:

Inference
---------

In Bayesian modelling, the probability theory provides tools for constructing
generative models for data and obtaining knowledge about the models given some
data (see, e.g., :cite:`Bishop:2006,Barber:2012,Murphy:2012`).  This information
can be used to get insight into the data and to make predictions.  A generative
model :math:`\mathcal{M}` consists of a likelihood function
:math:`p(Y|Z,\mathcal{M})` explaining the data :math:`Y` with parameters
:math:`Z` and a prior function :math:`p(Z|\mathcal{M})` providing the prior
knowledge about the model parameters.  The goal is to find the posterior
distribution of the model parameters:

.. math::
   
   p(Z|Y,\mathcal{M}) = \frac {p(Y|Z,\mathcal{M})p(Z|\mathcal{M})}
   {p(Y|\mathcal{M})},
  
which can be used, for instance, to make predictions.  The denominator
:math:`p(Y|\mathcal{M})` is called the marginal likelihood, defined as

.. math::
   
   p(Y|\mathcal{M}) = \int p(Y|Z,\mathcal{M})p(Z|\mathcal{M}) dZ,

which is the probability (density) of the observations when the model
:math:`\mathcal{M}` is assumed to be true.  Typically, the conditioning on the
model is not explicitly shown if there is no risk of misunderstanding.  Thus, we
discard :math:`\mathcal{M}` from our notation.




Models usually have hierarchical structure, which means that the prior of a set
of unknown variables is defined in terms of another set of unknown variables.
This may lead to extremely complex posterior inference unless priors have
convenient forms to simplify calculations.  In particular, the prior for an
unknown variable can be chosen such that the resulting posterior distribution
conditioned on all other unknown variables is in the same family as the prior.
This type of prior distribution is called a conjugate prior for the likelihood.
In addition, if the distributions are from the exponential family, the model is
said to be from the conjugate exponential family.


The main challenge in Bayesian inference is that the posterior distribution
:eq:`eq-bayes-theorem` is often analytically intractable.  Therefore, one has to
resort to methods that approximate the posterior.  These methods can roughly be
divided into two categories: deterministic and stochastic techniques
(:cite:`Bishop:2006`).  Both of these techniques have their advantages and
disadvantages.



Deterministic methods use analytic approximations to the posterior.  The
resulting approximate distribution is often evaluated efficiently, but it
usually requires extra work because some formulas must be derived analytically.
The approximate distribution does not, in general, recover the true posterior
distribution exactly.  Important deterministic approximations include: maximum
likelihood and maximum a posteriori methods, which approximate the posterior
distribution with a point estimate; Laplace method, which fits a Gaussian
distribution to a mode of the posterior probability density function;
variational Bayes (:cite:`Jordan:1998`) and expectation propagation
(:cite:`Minka:2001b`), which find an approximate distribution by minimizing an
information-theoretic dissimilarity to the true distribution; and integrated
nested Laplace approximations for latent Gaussian models (:cite:`Rue:2009`).


Stochastic techniques approximate the posterior distribution with a finite
number of samples.  The samples from the intractable posterior may be obtained
in several ways depending on the problem.  These stochastic techniques are
covered comprehensively, for instance, in the book by :cite:`Gelman:2003`.  In
complex problems, sampling is often implemented with random-walk type
algorithms, called Markov chain Monte Carlo (MCMC).  In general, stochastic
methods have the property that the approximation approaches the true posterior
at the limit of infinite computation time.  However, for large and complex
problems, the convergence can be extremely slow.


.. rubric:: Footnotes
            
   .. [#weak-syllogisms] See \cite{Jaynes:2003} for a detailed discussion on
      weak syllogisms.
