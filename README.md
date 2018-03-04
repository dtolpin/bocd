Bayesian Online Changepoint Detection
=====================================

An algorithm to get the probability of a changepoint in a time series. The algorithm is described in:

[2] Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection, arXiv 0710.3742 (2007)                                                                                 

Inspired by https://github.com/hildensia/bayesian_changepoint_detection, and
significantly modified.  This implementation can run in fixed space, and the
tradeoff between time and space efficiency and accurancy is configurable.

`bocd/test_bocd.py` provides usage examples (along with serving as unit test
suite for the implementation).
