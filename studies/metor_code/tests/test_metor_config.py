"""Config layering, and that the shipped configs say what the paper says.

The second half matters more than it looks. This reconstruction's whole claim to
being checkable is that the paper's stated hyperparameters are separated from
the ones invented here. If `configs/metor_vidvrd.yaml` drifts from Sec. 4.2,
that separation silently stops being true, so the paper's values are pinned as
literals below -- change one and a test fails, which is the point.
"""
from __future__ import annotations

import os

import pytest

yaml = pytest.importorskip("yaml")

from cli.config import apply_configs, extract_config_paths, load_yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDVRD = os.path.join(REPO, "configs", "metor_vidvrd.yaml")
VIDOR = os.path.join(REPO, "configs", "metor_vidor.yaml")
DEFAULTS = os.path.join(REPO, "configs", "metor_defaults.yaml")


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------ layering
def test_extract_config_paths_handles_both_spellings():
    argv = ["--config", "a.yaml", "--lr", "1e-4", "--config=b.yaml", "--n_iter", "3"]
    import sys
    saved = sys.argv
    try:
        sys.argv = ["prog"] + argv
        paths = extract_config_paths(argv)
        assert paths == ["a.yaml", "b.yaml"]
        assert sys.argv[1:] == ["--lr", "1e-4", "--n_iter", "3"]
    finally:
        sys.argv = saved


def test_later_configs_win_over_earlier_ones():
    args = _NS(n_iter=2)
    apply_configs(args, [DEFAULTS, VIDOR])
    assert args.n_iter == 3          # VidOR's value, layered last


def test_explicit_flags_beat_config_files():
    """A config must never silently override something typed on the command
    line -- that would make an ablation sweep quietly not run."""
    args = _NS(n_iter=99)
    apply_configs(args, [VIDVRD], typed={"n_iter"})
    assert args.n_iter == 99


def test_missing_config_fails_loudly():
    with pytest.raises(SystemExit):
        load_yaml("/nonexistent/config.yaml")


# --------------------------------------------- the paper's stated values
@pytest.mark.parametrize("key,expected", [
    ("n_queries", 100),     # "The number of object queries N_q is set to 100."
    ("alpha", 0.9),         # "The hyperparameter alpha in Eq. 11 is set to 0.9."
    ("theta_traj", 1.0),    # "set to 1.0, 0.2, and 0.1, respectively"
    ("theta_ctx", 0.2),
    ("theta_cst", 0.1),
    ("max_epoch", 30),      # "trained for a total of 30 epochs"
    ("batch_size", 1),      # "batch size is set to 1"
    ("lr", 1e-4),           # "initial learning rate of 1e-4"
    ("frame_stride", 30),   # "sampled every 30 video frames"
    ("clip_len", 30),
    ("n_iter", 2),          # "two for VidVRD and three for VidOR"
])
def test_vidvrd_config_matches_the_paper(key, expected):
    assert load_yaml(VIDVRD)[key] == expected


def test_vidor_differs_only_in_the_iteration_count():
    """Sec. 4.2 names exactly one per-dataset difference. If a second appears,
    it came from somewhere other than the paper."""
    vidvrd, vidor = load_yaml(VIDVRD), load_yaml(VIDOR)
    differing = {k for k in set(vidvrd) | set(vidor)
                 if vidvrd.get(k) != vidor.get(k)}
    assert differing == {"n_iter", "dataset", "ps"}
    assert vidor["n_iter"] == 3


def test_lr_schedule_matches_the_paper():
    cfg = load_yaml(VIDVRD)
    assert cfg["lr_milestones"] == [15, 20, 25]
    assert cfg["lr_gamma"] == 0.1


def test_paper_and_invented_values_stay_in_separate_files():
    """The PAPER-GAP defaults must not leak into the dataset configs, or the
    distinction this repository rests on stops being visible."""
    invented = {"n_ctx_obj", "n_ctx_rel", "n_prompt_vec", "d_model",
                "max_pair_objects", "temperature", "residual_mapping"}
    stated = set(load_yaml(VIDVRD)) | set(load_yaml(VIDOR))
    assert not (invented & stated), f"invented values in a paper config: {invented & stated}"
    assert invented <= set(load_yaml(DEFAULTS))


def test_frozen_backbone_is_the_paper_s():
    assert load_yaml(VIDVRD)["clip_backbone"] == "ViT-L/14@336px"
