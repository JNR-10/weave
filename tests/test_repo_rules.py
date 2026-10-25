"""AGENTS.md rules that a test can enforce; a failure message says how to fix it."""

import ast
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def m4_only_modules() -> set[str]:
    """Packages from the m4 extra (plus mlx itself), as import names; absent on Intel and CI."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    return {"mlx"} | {dist.replace("-", "_") for dist in extras["m4"]}


def module_level_imports(tree: ast.Module):
    """Imports that run when the module is imported: everything outside function bodies."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
        pending.extend(ast.iter_child_nodes(node))


def test_portable_code_never_imports_m4_packages_at_module_level():
    blocked = m4_only_modules()
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in sorted((ROOT / "src").rglob("*.py"))
        for name in module_level_imports(ast.parse(path.read_text()))
        if name.split(".")[0] in blocked
    ]
    assert not offenders, (
        "These imports run at module level, so `import weave...` fails on Intel Macs and in CI:\n"
        + "\n".join(offenders)
        + "\nMove each import inside the function that needs it (see weave.telemetry.baseline)."
    )


def test_raw_evidence_and_weights_are_never_tracked():
    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], stderr=subprocess.DEVNULL, text=True
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    offenders = [
        f
        for f in tracked
        if (f.startswith("results/") and f != "results/.gitkeep")
        or f.startswith("models/")
        or f.endswith((".safetensors", ".gguf", ".npz"))
    ]
    assert not offenders, (
        "Raw evidence or model weights are tracked:\n"
        + "\n".join(offenders)
        + "\nUntrack them with `git rm --cached <path>`. Raw runs go to the shared evidence archive"
        " recorded in experiments/phase1/README.md; weights are re-downloaded from the model lock."
    )
