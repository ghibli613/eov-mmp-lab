"""Every live module must import, and nothing may reference a deleted one."""
import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

LIVE_MODULES = [
    "cli.common", "cli.train", "cli.evaluate",
    "data_loading.dataset",
    "models.methods", "models.object_classifier", "models.relation_classifier",
    "models.end2end_model", "models.gen_labels",
    "inference.post_process", "inference.video_relation_detection_openvoc",
    "utils.paths", "utils.parser_func", "utils.eov_utils",
    "vlm.text_encoder",
    "third_party.vidvrd_eval_api.visual_relation_detection",
]

# Removed during the 2026-08-22 cleanup; see docs/11_Port-status.md.
DELETED_MODULES = [
    "models.methods.detectors.ov_prompt.solq",
    "models.methods.detectors.ov_prompt.dct",
    "models.methods.segmentation",
    "inference.video_relation_detection",
    "inference.video_relation_detection_ab",
    "utils.arguments",
    "utils.video_transform",
    "vlm.ptm_encoder",
    "models.model",
    "models.model_zoo.model_tuing_plus_repro_copy_new_cross_dataset",
    "models.methods.detectors.ov_prompt.backbone",
    "models.util.scheduler",
]


@pytest.mark.parametrize("module", LIVE_MODULES)
def test_live_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", DELETED_MODULES)
def test_deleted_module_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def _absolute_imports(path: Path):
    """Every module this file imports, with relative imports resolved."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    pkg = path.relative_to(ROOT).parent.as_posix().replace("/", ".")
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parts = pkg.split(".") if pkg else []
                base = ".".join(parts[:len(parts) - node.level + 1])
                if node.module:
                    base = f"{base}.{node.module}" if base else node.module
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


def test_no_source_file_still_imports_a_deleted_module():
    """A stale `from x import y` would only fail when that line runs.

    Relative imports are resolved, so a live `.segmentation` next to a deleted
    `models.methods.segmentation` is not a false positive.
    """
    deleted = set(DELETED_MODULES)
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "ops/build/", "ops/dist/")) or "__pycache__" in rel:
            continue
        for module in _absolute_imports(path) & deleted:
            offenders.append(f"{rel} imports {module}")
    assert not offenders, "stale imports:\n" + "\n".join(offenders)
