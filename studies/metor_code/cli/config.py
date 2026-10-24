"""YAML config layering on top of the argparse namespace.

The paper states some hyperparameters and leaves others unstated, and it matters
a great deal which is which. Splitting them across `configs/metor_vidvrd.yaml`
(theirs) and `configs/metor_defaults.yaml` (ours) only helps if loading them is
easy, so:

    python -m cli.train --config configs/metor_defaults.yaml \
                              --config configs/metor_vidvrd.yaml --n_iter 3

Precedence, lowest to highest:

    argparse defaults  <  --config files, in the order given  <  flags typed

That last step is the one worth getting right. A config file must not silently
override something the user typed on the command line, so the loader only
applies a key if the corresponding flag was left at its default.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ImportError:                                  # pragma: no cover
    yaml = None


#: Config keys that name something other than a parser flag. They are attached
#: to the namespace anyway so the training loop can read them, but they are not
#: checked against the parser.
_NON_FLAG_KEYS = {
    "optimizer", "lr_milestones", "lr_gamma", "seed", "temperature",
    "n_heads", "n_dec_layers", "dropout", "n_dec_layers",
    "traj_score_threshold", "box_filter_threshold",
}


def extract_config_paths(argv: List[str] | None = None) -> List[str]:
    """Pull `--config X` out of argv, returning the paths and leaving argv clean.

    Done by hand rather than with a pre-parser because `parse_args()` in
    utils/parser_func.py owns the real parser and calls `parse_args()` itself;
    threading a second parser through it would mean editing vendored code.
    """
    argv = sys.argv[1:] if argv is None else argv
    paths, rest = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config needs a path")
            paths.append(argv[i + 1])
            i += 2
        elif argv[i].startswith("--config="):
            paths.append(argv[i].split("=", 1)[1])
            i += 1
        else:
            rest.append(argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + rest
    return paths


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required for --config. Install it, or pass the settings "
            "as flags:  pip install pyyaml")
    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config {path} must be a mapping at the top level")
    return data


def _typed_flags(argv: Iterable[str]) -> set:
    """Flag names the user actually typed, as dest-style keys."""
    typed = set()
    for tok in argv:
        if tok.startswith("--"):
            typed.add(tok[2:].split("=", 1)[0].replace("-", "_"))
    return typed


def apply_configs(args: argparse.Namespace, config_paths: List[str],
                  typed: set | None = None) -> argparse.Namespace:
    """Layer config files onto `args` without clobbering explicit flags."""
    typed = typed if typed is not None else set()
    for path in config_paths:
        for key, value in load_yaml(path).items():
            if key in typed:
                continue                      # the command line wins
            setattr(args, key, value)
    args.config_files = list(config_paths)
    return args


def parse_args_with_config():
    """`utils.parser_func.parse_args()` plus `--config` layering."""
    config_paths = extract_config_paths()
    typed = _typed_flags(sys.argv[1:])
    from utils.parser_func import parse_args
    args = parse_args()
    return apply_configs(args, config_paths, typed)
