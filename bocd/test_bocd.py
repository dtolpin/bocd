"""Unit tests for BOCD.
"""

import pytest
import numpy
import numpy.random
import numpy.linalg
from functools import partial
from clew.changepoint.bocd import * 

LAMBDA = 100
ALPHA = 0.1
BETA = 1.
KAPPA = 1.
MU = 0.
DELAY = 15
THRESHOLD = 0.5


@pytest.fixture
def series():
    xs = numpy.random.normal(size=1000)
    xs[len(xs) // 4:len(xs) // 2] += 10.
    xs[len(xs) // 2:3 * len(xs) // 4] -= 10. 
    return xs


def test_BOCD(series):
    """Tests that changepoints are detected.
    """
    bocd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU))
    changepoints = []
    for x in series[:DELAY]:
        bocd.update(x)
    for x in series[DELAY:]:
        bocd.update(x)
        if bocd.growth_probs[DELAY] >= THRESHOLD:
            changepoints.append(bocd.t - DELAY + 1)
    
    assert numpy.linalg.norm(numpy.array(changepoints) - 
                             numpy.array([250, 500, 750]),
                             ord=1) < 3

 
def test_BOCD_prune(series):
    """Tests that changepoints are detected while pruning.
    """
    bocd = BOCD(partial(constant_hazard, LAMBDA),
                StudentT(ALPHA, BETA, KAPPA, MU))
    changepoints = []
    for x in series[:DELAY]:
        bocd.update(x)
    for x in series[DELAY:]:
        bocd.update(x)
        if bocd.growth_probs[DELAY] >= THRESHOLD:
            changepoints.append(bocd.t - DELAY + 1)
            bocd.prune(bocd.t - DELAY)
    
    assert numpy.linalg.norm(numpy.array(changepoints) - 
                             numpy.array([250, 500, 750]),
                             ord=1) < 3
