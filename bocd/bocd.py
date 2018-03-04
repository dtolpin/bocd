from __future__ import division
import numpy
import scipy.stats

# Copyright (c) 2014 Johannes Kulick
# Code borrowed from:
#    https://github.com/hildensia/bayesian_changepoint_detection
# under the MIT license.

class BOCD(object):
    def __init__(self, hazard_function, observation_likelihood):
        """Initializes th detector with zero observations.
        """
        self.t0 = 0
        self.t = -1
        self.growth_probs = numpy.array([1.])
        self.hazard_function = hazard_function
        self.observation_likelihood = observation_likelihood

    def update(self, x):
        """Updates changepoint probabilities with a new data point.
        """
        self.t += 1

        t = self.t - self.t0

        # allocate enough space
        if len(self.growth_probs) == t + 1:
            self.growth_probs = numpy.resize(self.growth_probs, (t + 1) * 2)

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        pred_probs = self.observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = self.hazard_function(numpy.array(range(t + 1)))

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        cp_prob = numpy.sum(self.growth_probs[0:t + 1] * pred_probs * H)

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        self.growth_probs[1:t + 2] = self.growth_probs[0:t + 1] * pred_probs * (1-H)
        # Put back changepoint probability
        self.growth_probs[0] = cp_prob

        # Renormalize the run length probabilities for improved numerical
        # stability.
        self.growth_probs[0:t + 2] = self.growth_probs[0:t + 2] / \
            numpy.sum(self.growth_probs[0:t + 2])

        # Update the parameter sets for each possible run length.
        self.observation_likelihood.update_theta(x)

    def prune(self, t0):
        """prunes memory before time t0. That is, pruning at t=0
        does not change the memory. One should prune at times
        which are likely to correspond to changepoints.
        """
        self.t0 = t0
        self.observation_likelihood.prune(self.t - t0 + 1)


def constant_hazard(lam, r):
    """Computes the "constant" hazard, that is corresponding
    to Poisson process.
    """
    return 1/lam * numpy.ones(r.shape)


class StudentT:
    """Student's t predictive posterior.
    """
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = numpy.array([alpha])
        self.beta0 = self.beta = numpy.array([beta])
        self.kappa0 = self.kappa = numpy.array([kappa])
        self.mu0 = self.mu = numpy.array([mu])

    def pdf(self, data):
        """PDF of the predictive posterior.
        """
        return scipy.stats.t.pdf(x=data,
                                 df=2*self.alpha,
                                 loc=self.mu,
                                 scale=numpy.sqrt(self.beta * (self.kappa+1) /
                                                  (self.alpha * self.kappa)))

    def update_theta(self, data):
        """Bayesian update.
        """
        muT0 = numpy.concatenate((self.mu0, (self.kappa * self.mu + data) /
                                            (self.kappa + 1)))
        kappaT0 = numpy.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = numpy.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = numpy.concatenate((self.beta0,
                                    self.beta +
                                    (self.kappa * (data - self.mu)**2) /
                                    (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

    def prune(self, t):
        self.mu = self.mu[:t + 1]
        self.kappa = self.kappa[:t + 1]
        self.alpha = self.alpha[:t + 1]
        self.beta = self.beta[:t + 1]

# Original single-shot code

def online_changepoint_detection(data, hazard_func, observation_likelihood):
    """Computes changepoint probabilities in the data.
    Returns the matrix of probabilities of all sequence lengths at each point.
    """
    R = numpy.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = self.observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(numpy.array(range(t+1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = numpy.sum( R[0:t+1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / numpy.sum(R[:, t+1])

        # Update the parameter sets for each possible run length.
        self.observation_likelihood.update_theta(x)

    return R
