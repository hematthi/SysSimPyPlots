Comparing catalogs
===================

Plotting histograms and CDFs are a great way of visually comparing different models and the data. However, a more quantitative way of comparing distributions is possible using various distance measures. A widely used and intuitively simple distance function is the `two-sample Kolmogorov-Smirnov (KS) distance <https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test>`_, which is simply defined as the maximum difference between two CDFs. We have two functions that compute the KS distance, one for discrete distributions and one for continuous distributions:

.. code-block:: python

   # Compute the KS distance between two multiplicity distributions:
   d_KS, x_KS = KS_dist_mult(sss_per_sys['Mtot_obs'], ssk_per_sys['Mtot_obs'])

   # Compute the KS distance between two period distributions:
   d_KS, x_KS = KS_dist(sss['P_obs'], ssk['P_obs'])

Both functions return the KS distance (``d_KS``) as well as the x-value corresponding to that distance (``x_KS``, i.e. where the difference in the CDFs is the greatest).

Another well known distance is the `two-sample Anderson-Darling (AD) distance <https://en.wikipedia.org/wiki/Anderson–Darling_test>`_, which computes an integral over the difference of two CDFs (weighted towards the tails). While this measure is more sensitive to differences in the extremes of the distributions, we found that samples with vastly different sizes (e.g., numbers of planets) can still produce low AD distances (see Section 2.4.2 of `He et al. 2019 <https://arxiv.org/pdf/1907.07773.pdf>`_ for further discussion). Thus, we also define a "modified" AD distance which re-normalizes by (divides out) the constant in front of the integral, `n*m/N` where `n` and `m` are the sample sizes (and `N=n+m`):

.. code-block:: python

   # Compute the standard AD distance between two period distributions:
   d_AD = AD_dist(sss['P_obs'], ssk['P_obs'])

   # Compute the modified AD distance between two period distributions:
   d_ADmod = AD_mod_dist(sss['P_obs'], ssk['P_obs'])
