import sys; sys.path.insert(0,'/home/totoro/Documents/thesis/source/METOR'); sys.path.insert(0,'/home/totoro/Documents/thesis/source/METOR/tests')
import cli.train_state as ts, pathlib, tempfile, pytest
orig=ts.restore_rng
def broken(st, gen):            # restore every stream except the DataLoader generator
    orig(st, None) if False else None
    import random, numpy as np, torch
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch_cpu"].cpu())
ts.restore_rng=broken
import test_rb1_train_state as t
try:
    t.test_two_epochs_equal_one_then_resume_then_one(pathlib.Path(tempfile.mkdtemp()))
    print("MUTANT SURVIVED (test is vacuous)")
except AssertionError as e:
    print("mutant killed: test detects a missing loader-generator restore")
