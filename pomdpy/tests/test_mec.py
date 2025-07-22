import os
import os.path
import pickle

from pomdpy.beliefsuppaut import BeliefSuppAut
from pomdpy.parsers import pomdp


def test_mec():
    for example in next(os.walk("examples"), (None, None, []))[2]:
        with open(os.path.join("examples", example), "r") as f:
            with open(os.path.join("pomdpy/tests/mec", example), "rb") as d:
                env = pomdp.parse(f.read())
                aut = BeliefSuppAut(env)
                aut.setBuchi([len(env.states)-1], range(len(env.states)-1), areIds=True)
                calculated_mecs = aut.mecs()
                old_mecs = pickle.load(d)

                assert calculated_mecs == old_mecs
