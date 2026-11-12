#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick the tests, the NPU lanes and the E2E cases a PR needs.

Five wiring rules, any hit selects. Keys are what `--explain` / `layers` /
`reasons` print:

  reverse_deps  module <- importers <- tests. A package `__init__.py` expands to
                its submodules first (`import pkg.sub` runs the init).
  mirror        src/<x>/a/b.py -> tests/<x>/a/b/, same depth only. A parent
                fallback turns one file into a whole layer.
  convention    no static edge: `pkgutil.walk_packages` providers and
                `importlib` plugins wired by name, plus discovery tests.
  path_ref      `spec_from_file_location` loads found by literal path search.
  test_tree     conftest is auto-loaded per directory (no edge) so it is wired
                by scope; helpers use the test-tree import graph.

`deleted` is a post-pass for tests still importing a removed module.
`full_reasons` explains `mode=all`: size threshold, dependency / CI-config
change, unknown blast radius, or `ci:full`.

No "name looks like test name" rule on purpose: at 260 modules it added 24
tests, none unique, and mapped `packed.py` to four unrelated `test_packed_*`.

The import graph is stdlib-only; E2E facts reuse `scripts/ci/facts.py` and the
lint-time PyYAML dependency. A call stays around 0.6 s, so no mapping table rots.

    lint_select/select_tests.py --changed-file a.py [--changed-file b.py] [--explain]
    git diff --name-only base...HEAD | lint_select/select_tests.py --changed-from -
    lint_select/select_tests.py --coverage | --verify
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

PKG = "torchturbo"
CHEAPEST_MODEL = "qwen3_5"   # 2B dense, runs untrimmed

# Huge PRs are refactors: that is when a full run pays off, and the selector
# would pick most files anyway.
MAX_CHANGED_FILES = 300

# Past this, "precise" costs as much as "all" and only "all" is trustworthy.
# Mirrors NUM_MODELS_TO_TRIGGER_FULL_CI in transformers tests_fetcher. Not lower:
# the `torchturbo.engine` facade legitimately selects 60+ files.
MAX_SELECTED_TESTS = 90

# Layer keys shared by `layers`, `reasons` and the coverage report.
LAYER_NAMES = ("reverse_deps", "mirror", "convention", "path_ref", "test_tree")

# Lane keys in output order. A new device appends its lanes here; nothing else
# repeats the list.
LANES = ("cpu_unit", "npu_unit", "npu_e2e")
# Lanes whose payload is a file list, so GITHUB_OUTPUT emits `<key>_files`.
UNIT_LANES = ("cpu_unit", "npu_unit")


# --------------------------------------------------------------------------
# Index: src module graph + test -> module dependencies
# --------------------------------------------------------------------------
class Index:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.src: dict[str, Path] = {}
        self.deps: dict[str, set[str]] = {}
        self.tests: dict[str, set[str]] = {}   # test path -> torchturbo modules it imports
        self.cases: dict[str, int] = {}         # test path -> static test_ functions
        self.test_files: list[str] = []
        # Raw test sources: path-based loading is invisible to the import graph.
        self.texts: dict[str, str] = {}
        # Test name + directory tokens, used by the provider naming convention.
        self.test_tokens: dict[str, set[str]] = {}
        # Test-tree nodes, including conftest / helpers / __init__; with only
        # `test_*.py`, editing a conftest selects nothing at all.
        self.tests_src: dict[str, Path] = {}
        self.test_deps: dict[str, set[str]] = {}
        self._rev: dict[str, set[str]] = {}
        self._trev: dict[str, set[str]] = {}
        self._build()

    # ---- build -----------------------------------------------------------
    def _build(self) -> None:
        src_root = self.root / "src"
        for path in sorted((src_root / PKG).rglob("*.py")):
            self.src[self._dotted(path, src_root)] = path

        for name, path in self.src.items():
            self.deps[name] = self._scan(self._parse(path), path, name)[0]

        tests_root = self.root / "tests"
        for path in sorted(tests_root.rglob("test_*.py")):
            rel = path.relative_to(self.root).as_posix()
            self.test_files.append(rel)
            text = path.read_text(encoding="utf-8", errors="replace")
            self.texts[rel] = text
            # One walk yields both edges and case count.
            seeds, cases = self._scan(self._parse_text(text), path, None)
            self.tests[rel] = seeds
            # Static count only: pytest expands parametrize on top. Used for the
            # selection percentage; the unit of selection is always the file.
            self.cases[rel] = cases
            stem = Path(rel)
            self.test_tokens[rel] = (_tokens(stem.stem)
                                     | _tokens("/".join(stem.parts[:-1])))

        # Test-tree graph. Without conftest / helpers as nodes, editing
        # a conftest or a shared fixture module selects nothing.
        for path in sorted(tests_root.rglob("*.py")):
            self.tests_src[self._dotted(path, self.root)] = path
        for name, path in self.tests_src.items():
            self.test_deps[name] = self._scan(self._parse(path), path, name,
                                              self.tests_src)[0]

        self._rev = self.reverse_deps()
        self._trev = _reverse(self.test_deps)

    @staticmethod
    def _dotted(path: Path, base: Path) -> str:
        parts = list(path.relative_to(base).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _resolve(self, cand: str, table: dict[str, Path] | None = None) -> str | None:
        """Trim a dotted name to a module that exists.

        In `from x.y import Foo`, `Foo` is a symbol; step back one part at a
        time. Pass `table` to resolve inside the test tree instead.
        """
        table = self.src if table is None else table
        parts = cand.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in table:
                return key
        return None

    def _pkg_of(self, modname: str, path: Path) -> str:
        if path.name == "__init__.py":
            return modname
        return modname.rsplit(".", 1)[0] if "." in modname else ""

    def text_needles(self, changed: list[str]) -> set[str]:
        """Path literals worth searching for in test sources.

    `scripts/` launchers load via `spec_from_file_location`, so a full path and a
    bare filename both count. `src/` modules load by absolute path, usually a
    package `__init__.py`, which is then a clue for the whole package. Full
    paths only for src: `base.py` appears everywhere.
    """
        needles: set[str] = set()
        for p in changed:
            if p.startswith("scripts/") and not p.startswith("scripts/ci/"):
                needles |= {p, Path(p).name}
            elif p.startswith(f"src/{PKG}/") and p.endswith(".py"):
                needles.add(p)
                parent = Path(p).parent.as_posix()
                if parent and parent != ".":
                    needles.add(f"{parent}/__init__.py")
        # `__init__.py` matches everywhere; a bare filename is pure noise.
        needles.discard("__init__.py")
        return needles

    def text_referrers(self, needles: set[str]) -> set[str]:
        """Test sources containing any of these strings."""
        if not needles:
            return set()
        return {rel for rel, text in self.texts.items()
                if any(n in text for n in needles)}

    @staticmethod
    def _parse(path: Path) -> ast.AST:
        return Index._parse_text(path.read_text(encoding="utf-8", errors="replace"))

    @staticmethod
    def _parse_text(text: str) -> ast.AST:
        try:
            return ast.parse(text)
        except SyntaxError:
            return ast.Module(body=[], type_ignores=[])

    def _scan(self, tree: ast.AST, path: Path, modname: str | None,
              table: dict[str, Path] | None = None) -> tuple[set[str], int]:
        """Single walk: (imported modules, static test_ function count)."""
        found: set[str] = set()
        cases = 0
        for node in _iter_statements(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hit = self._resolve(alias.name, table)
                    if hit:
                        found.add(hit)

            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # Relative import: back out of the package by `level`.
                    if modname is None:
                        continue
                    pkg = self._pkg_of(modname, path).split(".")
                    up = node.level - 1
                    if up:
                        pkg = pkg[:-up] if up <= len(pkg) else []
                    base = ".".join(pkg + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                if not base:
                    continue
                for alias in node.names:
                    hit = (self._resolve(f"{base}.{alias.name}", table)
                           or self._resolve(base, table))
                    if hit:
                        found.add(hit)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test"):
                # Async cases count too; missing them only skews `cases`.
                cases += 1
        return found, cases

    # ---- derived ---------------------------------------------------------
    # Cached at build time: `dependents()` is called once per discoverer.
    def reverse_deps(self) -> dict[str, set[str]]:
        return _reverse(self.deps)

    def dependents(self, seeds: set[str]) -> set[str]:
        """Seeds plus every module that (transitively) depends on them."""
        seen: set[str] = set()
        stack = [s for s in seeds if s in self.src]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self._rev.get(cur, ()))
        return seen

    def test_tree_dependents(self, seeds: set[str]) -> set[str]:
        """Seeds plus every test-tree file that (transitively) imports them."""
        seen: set[str] = set()
        stack = [s for s in seeds if s in self.tests_src]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self._trev.get(cur, ()))
        return seen

    def test_tree_impact(self, changed: list[str]) -> set[str]:
        """Test files affected by editing the test tree itself.

    conftest is auto-loaded per directory with no import edge, so it is wired by
    scope; other helpers use the test-tree import graph. Ignoring non-`test_*`
    files used to make a conftest edit select nothing - green CI, nothing run.

    Markers now live in the test files and `lint_check markers` rejects a
    conftest outright, so the conftest branch is a safety net: if one appears
    anyway, its edit must still select the directory it governs.
    """
        hit: set[str] = set()
        for rel in changed:
            if not (rel.startswith("tests/") and rel.endswith(".py")):
                continue
            if Path(rel).name == "conftest.py":
                scope = rel.rsplit("/", 1)[0]
                prefix = "" if scope == "tests" else scope + "/"
                hit |= {t for t in self.test_files if t.startswith(prefix)}
                continue
            if not (self.root / rel).exists():
                # Not a graph node any more, so nothing depends on it: search by
                # name instead. Over-selecting a deletion is safe, missing it is not.
                hit |= self.text_referrers({Path(rel).stem, rel})
                continue
            if rel in self.test_files:
                hit.add(rel)
            name = self._dotted(self.root / rel, self.root)
            for dep in self.test_tree_dependents({name}):
                path = self.tests_src.get(dep)
                if path is not None and path.name.startswith("test_"):
                    hit.add(path.relative_to(self.root).as_posix())
        return hit

    def dangling_tests(self, deleted: set[str]) -> set[str]:
        """Deleted modules -> tests that still import them (or their submodules).

        `self.src` is scanned *after* deletion, so reverse_deps finds nothing
        from a removed seed - and a removal is exactly what breaks a whole file
        at import time. A test seed that is a prefix of a deleted module counts.

        `import torchturbo.registry.core` collapses to `torchturbo.registry` once
        the file is gone, hence the prefix match.
        """
        if not deleted:
            return set()
        out: set[str] = set()
        for test, seeds in self.tests.items():
            for seed in seeds:
                if any(d == seed or d.startswith(seed + ".") for d in deleted):
                    out.add(test)
                    break
        return out

# --------------------------------------------------------------------------
# E2E case mapping
# --------------------------------------------------------------------------
def pick_e2e(changed: list[str], root: Path, *, all_cases: bool = False,
             force: bool = False) -> list[dict]:
    """Changed paths -> self-describing E2E cases."""
    ci_root = Path(__file__).resolve().parents[1]
    if str(ci_root) not in sys.path:
        sys.path.insert(0, str(ci_root))
    from facts import load_catalog, select_e2e_cases

    catalog = load_catalog(root)
    if catalog.errors:
        print("::warning::CI facts contain errors; E2E selection may be incomplete",
              file=sys.stderr)
    return select_e2e_cases(catalog, changed, all_cases=all_cases, force=force)


# Removed `wants_npu(changed)`: a second path-prefix guess that could start NPU
# with only CPU files picked, or skip it while NPU tests were picked.


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def _tokens(stem: str) -> set[str]:
    return {t for t in stem.replace("-", "_").split("_") if t and t != "test"}


def _reverse(edges: dict[str, set[str]]) -> dict[str, set[str]]:
    """{node: nodes it depends on} -> {node: nodes depending on it}."""
    rev: dict[str, set[str]] = {name: set() for name in edges}
    for name, deps in edges.items():
        for dep in deps:
            rev.setdefault(dep, set()).add(name)
    return rev


# `ast.walk` visits every literal and call: 2.3 s of the 3.4 s build went there,
# while what we need only appears at statement positions. Skip expressions.
_STMT_NODES = (ast.stmt, ast.ExceptHandler, ast.match_case)


def _iter_statements(tree: ast.AST):
    """Walk statement nodes only (expression subtrees skipped)."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _STMT_NODES):
                stack.append(child)


# Capacity is a code fact. Markers only choose the device lane; `multi_*`
# states that the lane needs more than one device, not a 2/4/8 job shape.
NPU_MARKERS = frozenset({"npu", "multi_npu"})
GPU_MARKERS = frozenset({"gpu", "multi_gpu"})
DEVICE_MARKERS = frozenset({"cpu", *NPU_MARKERS, *GPU_MARKERS})
KNOWN_MARKERS = DEVICE_MARKERS | {"integration"}


@dataclass(frozen=True)
class TestRequirement:
    marker_names: frozenset[str]
    module_markers: frozenset[str]
    min_npus: int
    npu_tests: int
    test_count: int
    unclassified_tests: tuple[str, ...]

    @property
    def npu(self) -> bool:
        return self.min_npus > 0

    @property
    def cpu(self) -> bool:
        return "cpu" in self.marker_names

    @property
    def pure_npu(self) -> bool:
        return self.test_count > 0 and self.npu and self.npu_tests == self.test_count


_REQUIREMENT_CACHE: dict[Path, TestRequirement] = {}
_NPU_CODE_CACHE: dict[Path, bool] = {}

_DEVICE_PATTERNS = (
    re.compile(r"\bimport\s+torch_npu\b"),
    re.compile(r"\bfrom\s+torch_npu\b"),
    re.compile(r"""importorskip\(\s*["']torch_npu["']"""),
    re.compile(r"\btorch\.npu\."),
    re.compile(r"\bfla_npu\b"),
    re.compile(r"""device\s*=\s*["']npu["']"""),
)


def _marker_names(node: ast.AST) -> set[str]:
    """Collect pytest marker names from a decorator or pytestmark value."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: set[str] = set()
        for item in node.elts:
            out |= _marker_names(item)
        return out
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute) and node.attr in KNOWN_MARKERS:
        base = node.value
        if (isinstance(base, ast.Attribute) and base.attr == "mark"
                and isinstance(base.value, ast.Name) and base.value.id == "pytest"):
            return {node.attr}
    return set()


def _module_markers(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in getattr(tree, "body", ()):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            out |= _marker_names(value)
    return out


def test_requirement(path: Path) -> TestRequirement:
    """Read explicit pytest markers for a test file; never import the file."""
    hit = _REQUIREMENT_CACHE.get(path)
    if hit is not None:
        return hit
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        hit = TestRequirement(frozenset(), frozenset(), 0, 0, 0, ())
    else:
        module = _module_markers(tree)
        tests = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test")]
        test_sets = [
            module | set().union(*(_marker_names(dec) for dec in node.decorator_list))
            for node in tests
        ]
        names = set().union(*test_sets) if test_sets else set(module)
        hit = TestRequirement(
            marker_names=frozenset(names),
            module_markers=frozenset(module),
            min_npus=(2 if "multi_npu" in names else
                      1 if "npu" in names else 0),
            npu_tests=sum(bool(markers & NPU_MARKERS) for markers in test_sets),
            test_count=len(tests),
            unclassified_tests=tuple(
                node.name for node, markers in zip(tests, test_sets)
                if not markers & DEVICE_MARKERS
            ),
        )
    _REQUIREMENT_CACHE[path] = hit
    return hit


def source_without_comments(path: Path) -> str:
    """Return source with comments blanked, preserving strings and layout."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        spans = [
            (tok.start, tok.end)
            for tok in tokenize.generate_tokens(io.StringIO(raw).readline)
            if tok.type == tokenize.COMMENT
        ]
        chars = [list(line) for line in raw.splitlines(keepends=True)]
        for (row0, col0), (row1, col1) in reversed(spans):
            if row0 == row1:
                chars[row0 - 1][col0:col1] = " " * (col1 - col0)
        return "".join("".join(line) for line in chars)
    except (OSError, tokenize.TokenError, IndentationError, SyntaxError):
        return raw


def uses_npu_code(path: Path) -> bool:
    """Whether a file contains real NPU APIs; used by lint, not selection."""
    hit = _NPU_CODE_CACHE.get(path)
    if hit is None:
        try:
            text = source_without_comments(path)
        except OSError:
            text = ""
        hit = any(pattern.search(text) for pattern in _DEVICE_PATTERNS)
        _NPU_CODE_CACHE[path] = hit
    return hit


def top_level_npu_import(path: Path) -> bool:
    """Whether importing this module requires torch_npu/fla_npu eagerly."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in {"torch_npu", "fla_npu"}
                   for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in {"torch_npu", "fla_npu"}:
                return True
    return False


# A former "leaf name match" layer is gone: at 260 modules it added 24 tests,
# none unique, and mapped `packed.py` to four unrelated `test_packed_*`.


def mirror_test_dir(root: Path, rel: str) -> str | None:
    """src-relative path -> same-depth mirror directory, or None.

    `models/qwen3_5_moe/model.py` -> `tests/models/qwen3_5_moe`. Never falls back
    to `tests/models`: that was the largest over-selection source (82 files).
    """
    parts = rel.split("/")[:-1]              # drop the filename
    if not parts:
        return None
    cand = "tests/" + "/".join(parts)
    return cand if (root / cand).is_dir() else None


# --------------------------------------------------------------------------
# Explicit wiring: real relations with no static import edge
# --------------------------------------------------------------------------
# Same idea as `preprocessor_patterns` in transformers/utils/tests_fetcher.py.
# Neither side has a static import, so isolated modules are wired by convention:
#
#   providers/<device>/<op>.py  ->  tests named after the <op> roots
#                               +  tests of the discovery mechanism itself
#
# No hit returns nothing, never a directory: an empty result is visible in
# `lanes.*.why`, a layer of unrelated tests is not.
PROVIDER_RE = re.compile(
    rf"^src/{PKG}/ops/providers/(?P<device>[^/]+)/(?P<op>[^/]+)\.py$"
)
# Discovery tests resolve providers via the registry / `importlib`, so match the
# file name (dispatch / providers), never a directory token:
# `tests/ops/providers/npu/test_gdn_kkt.py` tests one kernel.
PROVIDER_REGISTRY_HINTS = ("dispatch", "providers")

# Name / string resolved plugins: no plugin -> consumer edge, so the verifier is
# the test of the code that discovers the plugin.
#
#   models/<m>/lf_template.py|mm_plugin.py   scanned by llamafactory_v0
#   models/<m>/swift_template.py             imported by msswift.model_registry
#   adapters/msswift/trainer_factory.py      referenced by path in stages.py
DISCOVERER_RULES = (
    (re.compile(rf"^src/{PKG}/models/[^/]+/(?:lf_template|mm_plugin)\.py$"),
     ("torchturbo.adapters.llamafactory_v0",)),
    (re.compile(rf"^src/{PKG}/models/[^/]+/swift_template\.py$"),
     ("torchturbo.adapters.msswift.model_registry",)),
    (re.compile(rf"^src/{PKG}/adapters/msswift/trainer_factory\.py$"),
     ("torchturbo.adapters.msswift.stages",)),
)


def convention_wiring(index: Index, changed: list[str]) -> dict[str, list[str]]:
    """convention layer: naming / discovery relations, {changed file -> tests}."""
    stems = index.test_tokens
    registry = [t for t in index.test_files
                if any(h in Path(t).stem for h in PROVIDER_REGISTRY_HINTS)]
    of_discoverer: dict[str, set[str]] = {}

    def discoverer_tests(module: str) -> set[str]:
        if module not in of_discoverer:
            reach = index.dependents({module})
            of_discoverer[module] = {t for t, seeds in index.tests.items()
                                     if seeds & reach}
        return of_discoverer[module]

    hits: dict[str, list[str]] = {}
    for path in changed:
        found: set[str] = set()
        match = PROVIDER_RE.match(path)
        if match:
            want = _tokens(match.group("op"))
            if want:
                found |= {t for t, tokens in stems.items() if want <= tokens}
                found.update(registry)
        for pattern, discoverers in DISCOVERER_RULES:
            if pattern.match(path):
                found |= {t for d in discoverers for t in discoverer_tests(d)}
        if found:
            hits[path] = sorted(found)
    return hits


def select(index: Index, changed: list[str], force_all: bool = False,
           e2e_all: bool = False, e2e_force: bool = False,
           skip_devices: frozenset[str] = frozenset(),
           doc_only: bool = False) -> dict:
    changed = sorted(set(p for p in changed if p))
    total_tests = len(index.test_files)

    full_reasons: list[str] = []
    if force_all:
        full_reasons.append("force-all")
    # Empty `changed` usually means no information (shallow clone, merge-base
    # failure) -> fail open. A diff that only touched comments is an answer.
    if not changed and not doc_only:
        full_reasons.append("no-changed-files")
    if len(changed) > MAX_CHANGED_FILES:
        full_reasons.append(f"too-many-files({len(changed)})")
    # Dependency change: if the install breaks, everything is red anyway.
    if "pyproject.toml" in changed:
        full_reasons.append("deps-changed")
    # CI internals define *how* tests run: a silent under-selection is only
    # visible against the full set, so run it all. They must not burn an NPU.
    if any(p.startswith((".github/", "scripts/ci/")) for p in changed):
        full_reasons.append("ci-config-changed")

    src_changed = [p for p in changed if p.startswith(f"src/{PKG}/") and p.endswith(".py")]
    changed_modules: set[str] = set()
    for p in src_changed:
        mod = Index._dotted(index.root / p, index.root / "src")
        if mod in index.src:
            changed_modules.add(mod)

    # Package facade: `<pkg>/__init__.py` == every submodule, since `import
    # pkg.sub` runs the init (no graph edge). Feeds only reverse_deps: hundreds of
    # submodule names would create coincidental matches in name-based rules.
    expanded: set[str] = set()
    for p in src_changed:
        if not p.endswith("/__init__.py"):
            continue
        pkg = Index._dotted(index.root / p, index.root / "src")
        if pkg in index.src:
            expanded |= {m for m in index.src if m.startswith(pkg + ".")}

    # ---- reverse_deps: reverse import closure ----------------------------
    affected = index.dependents(changed_modules | expanded)
    reverse_dep_hits = {t for t, seeds in index.tests.items() if seeds & affected}

    # ---- convention: dynamic discovery / string-resolved modules ---------
    # Providers are enumerated by `pkgutil.walk_packages`; template plugins are
    # loaded by `importlib.import_module`. Neither has a static edge, so the
    # reverse closure above is necessarily empty for them.
    convention_hits = convention_wiring(index, src_changed)
    convention_tests_found = {t for tests in convention_hits.values() for t in tests}

    # ---- path_ref: path-loaded tests ------------------------------------
    # Launchers / package inits loaded via `spec_from_file_location` have no
    # import edge either; the only evidence is the path text in test sources.
    path_ref_hits = index.text_referrers(index.text_needles(changed))

    # ---- mirror: same-depth mirror directory ----------------------------
    # Same depth only, no parent fallback: a fallback turns one file into a whole
    # layer (`models/qwen3_5_moe_text` -> 72 tests). No "skip if another layer
    # hit" either: implicit rules make identical edits select different sets.
    mirror_hits: set[str] = set()
    for p in src_changed:
        rel = p[len(f"src/{PKG}/"):]
        probe = mirror_test_dir(index.root, rel)
        if probe is None:
            continue
        mirror_hits.update(t for t in index.test_files if t.startswith(probe + "/"))

    # ---- test_tree: edits inside the test tree itself --------------------
    # A conftest decides how its directory is collected, yet pytest auto-loads
    # it with no import edge - untreated, that edit is "green CI that verified
    # nothing".
    test_tree_hits = index.test_tree_impact(changed)

    # ---- deleted: removed sources ---------------------------------------
    # `changed_modules` keeps only modules that still exist, so a removal gives
    # the reverse closure nothing to start from. Add back whoever imported it.
    deleted_modules = {
        Index._dotted(index.root / p, index.root / "src")
        for p in src_changed
        if not (index.root / p).exists()
    }
    deleted_hits = index.dangling_tests(deleted_modules)

    precise = sorted(set(test_tree_hits) | reverse_dep_hits | convention_tests_found
                     | mirror_hits | path_ref_hits | deleted_hits)

    if full_reasons:
        mode = "all"
    elif total_tests and len(precise) > MAX_SELECTED_TESTS:
        # Too wide to be worth filtering; a full run is self-explanatory.
        full_reasons.append(f"too-many-tests({len(precise)})")
        mode = "all"
    else:
        mode = "precise"

    # A full run is really full: callers get every test file.
    targets = sorted(index.test_files) if mode == "all" else precise

    # ---- split into lanes by explicit pytest markers ---------------------
    # Pure NPU files go to the NPU lane; pure CPU files go to the CPU lane;
    # mixed files go to both and the device expression splits them.
    requirements = {t: test_requirement(index.root / t) for t in targets}
    npu_targets = [t for t in targets if requirements[t].npu]
    cpu_targets = [t for t in targets if requirements[t].cpu]

    e2e_cases = pick_e2e(changed, index.root)
    if e2e_force and not e2e_cases:
        # ci:e2e: run at least the cheapest profile, whatever the paths say.
        e2e_cases = pick_e2e(changed, index.root, force=True)
    if e2e_all:
        # ci:full / main: every declared case; the caller serializes them.
        e2e_cases = pick_e2e(changed, index.root, all_cases=True)

    # ---- lanes: the single decision output ------------------------------
    # Every lane carries a `why`, and a missing one fails the gate: "nobody knows
    # why it did not run" quietly becomes normal. The yaml reads only `run`.
    image_changed = any(p.startswith("docker/npu/") for p in changed)
    cpu_unit_run = len(cpu_targets) > 0

    # The NPU lane keys off whether the change really touches card code, not on
    # `mode`: `ci-config-changed` widens CPU but touches no operator or image.
    npu_touched = any(test_requirement(index.root / t).npu for t in precise)
    repo_wide = (force_all
                 or "no-changed-files" in full_reasons
                 or "deps-changed" in full_reasons
                 or any(v.startswith("too-many-") for v in full_reasons))
    npu_skipped = "npu" in skip_devices
    npu_unit_run = not npu_skipped and (npu_touched or image_changed or repo_wide)
    npu_scope = "all" if image_changed or repo_wide else "selected"
    npu_unit_files = (npu_targets if npu_unit_run and npu_scope == "selected"
                      else [])
    # A skip label closes every lane of that device: the point is to spend no
    # cards at all.
    npu_e2e_run = bool(e2e_cases) and not npu_skipped

    lanes = {
        "cpu_unit": {
            "run": cpu_unit_run,
            "why": ("comment / docstring-only change" if doc_only and not changed else
                    f"{len(cpu_targets)} CPU test files selected"
                    if cpu_unit_run else "no CPU test selected"),
            "files": cpu_targets,
        },
        "npu_unit": {
            "run": npu_unit_run,
            # Empty `files` = whole lane. Image / repo-wide changes use that;
            # targeted changes pass the selected NPU files.
            "files": npu_unit_files,
            "selected": len(npu_targets),
            "scope": npu_scope,
            "why": (
                "ci:skip-npu label" if npu_skipped else
                "image recipe change (whole lane)" if image_changed else
                "unknown blast radius / dependency change (whole lane)"
                if repo_wide else
                f"{len(npu_targets)} NPU test files selected"
                if npu_touched else
                "this change touches no NPU test"
            ),
        },
        "npu_e2e": {
            "run": npu_e2e_run,
            "why": (
                "ci:skip-npu label" if npu_skipped and e2e_cases else
                f"{len(e2e_cases)} case(s): "
                + ",".join(e["model_type"] for e in e2e_cases)
                if npu_e2e_run else
                "no matching E2E case"
            ),
            "cases": e2e_cases,
        },
    }

    reasons = {
        p: {"reverse_deps": len(reverse_dep_hits) if p in src_changed else 0,
            "mirror": len(mirror_hits) if p in src_changed else 0,
            "convention": len(convention_hits.get(p, [])),
            "path_ref": len(path_ref_hits)
                        if (p in src_changed or p.startswith("scripts/")) else 0,
            "test_tree": len(test_tree_hits) if p.startswith("tests/") else 0}
        for p in changed
    }

    return {
        "schema": 1,
        "mode": mode,
        "full_reasons": full_reasons,
        "doc_only": doc_only,
        "targets": targets,
        # `npu` is kept as a synonym for "run the npu-unit lane".
        "npu": npu_unit_run,
        "cpu_files": cpu_targets,
        "npu_files": npu_targets,
        "e2e_cases": e2e_cases,
        "lanes": lanes,
        "counts": {
            "files": len(changed),
            "tests": len(targets),
            "total": total_tests,
            "cases": sum(index.cases.get(t, 0) for t in targets),
            "cases_total": sum(index.cases.values()),
            "pct": round(100.0 * len(targets) / total_tests, 1) if total_tests else 0.0,
        },
        "reasons": reasons,
        "layers": {"reverse_deps": sorted(reverse_dep_hits),
                   "mirror": sorted(mirror_hits),
                   "convention": sorted(convention_tests_found),
                   "path_ref": sorted(path_ref_hits),
                   "test_tree": sorted(test_tree_hits),
                   "deleted": sorted(deleted_hits)},
    }


# --------------------------------------------------------------------------
# Changed-file discovery: git / stdin / file, minus comment-only edits
# --------------------------------------------------------------------------
def changed_from_git(base_ref: str = "origin/main", diff_filter: str = "ACDMT") -> tuple[list[str], int]:
    """Changed files: (list, count dropped as comment-only).

    Base is the merge-base, not main's tip, or a PR drifts towards full runs.
    `--no-renames` turns a rename into delete + add; `--diff-filter` keeps D,
    since a removal is what breaks a file at import time (lowercase `d` drops
    it). Any failure returns [], which `select()` turns into a full run.
    """
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], capture_output=True, text=True)

    def give_up(why: str) -> tuple[list[str], int]:
        print(f"::warning::changed_files: {why}; falling back to a full run",
              file=sys.stderr)
        return [], 0

    if git("rev-parse", "--git-dir").returncode != 0:
        return give_up("not a git repository")

    # The base branch may not be fetched yet; try once (works on shallow clones).
    if git("rev-parse", "--verify", "-q", base_ref).returncode != 0:
        git("fetch", "--no-tags", "--depth=200", "origin", base_ref.split("/", 1)[-1])

    base = git("merge-base", "HEAD", base_ref).stdout.strip() \
        or git("rev-parse", base_ref).stdout.strip()
    if not base:
        return give_up(f"cannot resolve base {base_ref}")

    print(f"changed_files: base={base} ref={base_ref}", file=sys.stderr)
    diff = git("diff", "--name-only", "--no-renames",
               f"--diff-filter={diff_filter}", f"{base}...HEAD")
    if diff.returncode != 0:
        return give_up(f"git diff failed: {diff.stderr.strip()[:200]}")
    files = [line for line in diff.stdout.splitlines() if line.strip()]
    kept = drop_doc_only(base, files)
    return kept, len(files) - len(kept)


def code_only(text: str) -> str | None:
    """Executable code only: serialized AST with docstrings stripped.

    Comments never reach the AST, so comment-only and docstring-only edits
    serialize identically. Same goal as transformers' `diff_is_docstring_only`.
    Returns None on a parse error (conservative: treat as a real change).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    holders = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))]
    for node in holders:
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.dump(tree)


def drop_doc_only(base: str, files: list[str]) -> list[str]:
    """Drop `.py` files whose only changes are comments or docstrings.

    Largest wasted-run class here: one comment line pulls a whole lane. Judge by
    comparing cleaned ASTs, not diff text (`#` inside strings fools the latter).
    Added / deleted / renamed / unreadable files always stay in.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for rel in files:
        path = Path(rel)
        if path.suffix != ".py" or not path.exists():
            kept.append(rel)
            continue
        old = subprocess.run(["git", "show", f"{base}:{rel}"],
                             capture_output=True, text=True)
        if old.returncode != 0:
            kept.append(rel)
            continue
        before = code_only(old.stdout)
        after = code_only(path.read_text(encoding="utf-8", errors="replace"))
        if before is not None and before == after:
            dropped.append(rel)
        else:
            kept.append(rel)
    if dropped:
        print(f"changed_files: ignoring {len(dropped)} comment/docstring-only "
              f"file(s): {', '.join(dropped)}", file=sys.stderr)
    return kept


def read_changed(args) -> tuple[list[str], int]:
    """(changed files, number of files dropped as comment-only)."""
    files: list[str] = []
    doc_dropped = 0
    if args.changed_from == "-":
        files += [line.strip() for line in sys.stdin if line.strip()]
    elif args.changed_from == "git":
        got, doc_dropped = changed_from_git(args.base_ref)
        files += got
    elif args.changed_from:
        files += [line.strip() for line in Path(args.changed_from).read_text(encoding="utf-8").splitlines() if line.strip()]
    files += args.changed_file or []
    return sorted(set(files)), doc_dropped


# --------------------------------------------------------------------------
# Event input -> flags, then the plan table
# --------------------------------------------------------------------------
def flags_from_event(mode: str, labels: str, with_e2e: bool,
                     force_all: bool = False) -> dict:
    """Event input -> the `select()` flags. **The only place with rules.**

    Was a bash `case` in the workflow, so labels lived in two places and drifted.
    mode=full: everything, E2E only with with_e2e. mode=pr: labels first
    (ci:full > ci:e2e > ci:skip-<device>), then dispatch's force_all / with_e2e;
    a PR event never sets with_e2e.
    """
    want = {"force_all": False, "e2e_all": False, "e2e_force": False,
            "skip_devices": frozenset()}

    if mode == "full":
        want["force_all"] = True
        want["e2e_all"] = bool(with_e2e)
        return want

    tags = {t.strip() for t in labels.split(",") if t.strip()}
    if "ci:full" in tags:
        want["force_all"] = True
        want["e2e_all"] = True
    elif "ci:e2e" in tags:
        want["e2e_force"] = True
    else:
        # ci:skip-<device>, e.g. ci:skip-npu. One label per device, not per lane.
        want["skip_devices"] = frozenset(
            tag[len("ci:skip-"):] for tag in tags if tag.startswith("ci:skip-"))

    # force_all means "all unit tests", never "all E2E": the 8-card lane is the
    # expensive one and stays opt-in.
    if force_all:
        want["force_all"] = True
    if with_e2e:
        want["e2e_force"] = True
    return want


def render_plan(result: dict, mode: str, image: str,
                changed_files: list[str], limit: int = 100) -> str:
    """The plan table in the job summary; it renders `select()` output."""
    counts = result["counts"]
    lanes = result["lanes"]
    runs = " / ".join(str(lanes[k]["run"]).lower() for k in LANES)
    e2e = ", ".join(p["model_type"] for p in result["e2e_cases"]) or "-"
    head = "\n".join([
        "## Plan",
        "",
        "| item | value |",
        "|---|---|---|",
        f"| mode | `{mode}` → `{result['mode']}` |",
        f"| lanes ({'/'.join(LANES)}) | `{runs}` |",
        f"| e2e | `{e2e}` |",
        f"| changed | {counts['files']} files |",
        f"| selected | {counts['tests']}/{counts['total']} files "
        f"({counts['pct']}%), ~{counts['cases']} cases |",
        f"| image | `{image}` |",
        f"| full run because | {', '.join(result['full_reasons']) or '-'} |",
    ])
    shown = changed_files[:limit]
    more = len(changed_files) - len(shown)
    body = "\n".join(shown) + (f"\n... (+{more} more)" if more > 0 else "")
    return (head
            + "\n\n<details><summary>changed files</summary>\n\n```\n"
            + body + "\n```\n</details>\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _bool(raw: str | bool) -> bool:
    """'true' / 'false' (a GitHub expression result) -> bool.

    Lets the yaml pass `--with-e2e "$WITH_E2E"` without shell conditionals.
    """
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "pyproject.toml").is_file())
    ap = argparse.ArgumentParser(description="Select tests / NPU lanes / E2E cases from changed files")
    ap.add_argument("--root", default=str(root),
                    help="repository root (default: inferred from this file)")
    ap.add_argument("--changed-file", action="append", default=[],
                    help="a changed file; repeatable")
    ap.add_argument("--changed-from", default=None,
                    help="read the changed list from a file; '-' = stdin, 'git' = compute the merge-base diff")
    ap.add_argument("--base-ref", default="origin/main",
                    help="base ref for --changed-from git (default: origin/main)")
    ap.add_argument("--dump-changed", default=None,
                    help="write the resolved changed list here (summary / full-run triggers)")
    ap.add_argument("--force-all", nargs="?", const=True, default=False, type=_bool,
                    help="force a full run (ci:full); accepts an explicit true/false")
    ap.add_argument("--e2e-all", action="store_true", help="run E2E for every family that has a case")
    ap.add_argument("--e2e-force", action="store_true", help="run the cheapest case even if no path matches")
    ap.add_argument("--no-npu", action="store_true", help="never start NPU (ci:skip-npu)")
    # ---- event input interpretation (replaces the bash case in the workflow) --
    ap.add_argument("--mode", choices=("pr", "full"), default=None,
                    help="pr = select from the diff; full = everything. Flags are derived from --mode")
    ap.add_argument("--labels", default="",
                    help="comma separated PR labels; only ci:full / ci:e2e / ci:skip-npu count")
    ap.add_argument("--with-e2e", nargs="?", const=True, default=False, type=_bool,
                    help="pr = the cheapest case; full = every case")
    ap.add_argument("--summary-file", default=None,
                    help="append the plan table here (GITHUB_STEP_SUMMARY)")
    ap.add_argument("--image-ref", default="-",
                    help="displayed in --summary-file only")
    ap.add_argument("--explain", action="store_true", help="print per-file layer hits")
    ap.add_argument("--coverage", "--audit", dest="coverage", action="store_true",
                    help="print the coverage report (JSON); never fails")
    ap.add_argument("--verify", "--self-test", dest="verify", action="store_true",
                    help="assert the selector's rules did not regress; exit 1 on failure")
    ap.add_argument("--output", default=None, help="write the JSON result here")
    ap.add_argument("--github-output", default=None,
                    help="append key=value lines here (GITHUB_OUTPUT)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "src" / PKG).is_dir():
        print(f"::error::cannot find {root}/src/{PKG}; pass --root", file=sys.stderr)
        return 2

    if args.verify:
        return verify(root)

    index = Index(root)

    if args.coverage:
        print(json.dumps(coverage(index), ensure_ascii=False, indent=2))
        return 0

    changed, doc_dropped = read_changed(args)
    if args.dump_changed:
        Path(args.dump_changed).write_text(
            "\n".join(changed) + ("\n" if changed else ""), encoding="utf-8"
        )

    # --mode wins when given (rules live in flags_from_event); otherwise use the
    # individual --force-all / --e2e-* flags (used by non-CI callers and tests).
    if args.mode:
        flags = flags_from_event(args.mode, args.labels, args.with_e2e,
                                 args.force_all)
    else:
        flags = {"force_all": args.force_all, "e2e_all": args.e2e_all,
                 "e2e_force": args.e2e_force,
                 "skip_devices": frozenset({"npu"}) if args.no_npu else frozenset()}
    # doc_only only when the diff was non-empty but every file was comment-only;
    # a missing diff still means a full run.
    flags["doc_only"] = bool(doc_dropped) and not changed
    result = select(index, changed, **flags)

    if args.explain:
        print(f"mode={result['mode']} files={result['counts']['files']} "
              f"files-hit={result['counts']['tests']}/{result['counts']['total']} "
              f"({result['counts']['pct']}%) cases~{result['counts']['cases']}")
        width = max(len(k) for k in LANES)
        for name in LANES:
            lane = result["lanes"][name]
            flag = "RUN " if lane["run"] else "skip"
            print(f"  {flag} {name:<{width}} {lane['why']}")
        if result["full_reasons"]:
            print(f"full run because: {', '.join(result['full_reasons'])}")
        if doc_dropped:
            print(f"  ignored {doc_dropped} comment/docstring-only file(s)")
        # Print only layers that hit: all-zero rows (docs, test-only edits)
        # would bury the signal.
        for p in changed:
            r = result["reasons"][p]
            hit = " ".join(f"{k}={v}" for k, v in r.items() if v) or "-"
            print(f"  {p}  {hit}")
        for e in result["e2e_cases"]:
            print(f"  E2E case {e['case_id']} ({e['framework']}/{e['env']}, "
                  f"{e['layers']} layers)"
                  f" <- {e['reason']}")

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)

    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as fh:
            fh.write(render_plan(result, args.mode or result["mode"],
                                 args.image_ref, changed))

    if args.github_output:
        lanes = result["lanes"]
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"mode={result['mode']}\n")
            for name in LANES:
                fh.write(f"{name}_run={str(lanes[name]['run']).lower()}\n")
            # Distinguishes "cpu-unit really got files" from "got everything":
            # an empty list makes run_tests.sh glob the whole lane.
            for name in UNIT_LANES:
                fh.write(f"{name}_files="
                         + json.dumps(lanes[name]["files"], ensure_ascii=False) + "\n")
            fh.write(f"npu={str(result['npu']).lower()}\n")
            fh.write(f"targets={json.dumps(result['targets'], ensure_ascii=False)}\n")
            fh.write("e2e_cases="
                     + json.dumps(result["e2e_cases"], ensure_ascii=False) + "\n")
            # Fallback: run_tests.sh globs everything when TEST_FILES is empty.
            fh.write(f"test_files={json.dumps(index.test_files, ensure_ascii=False)}\n")
            fh.write(f"summary={result['mode']} {result['counts']['files']}f "
                     f"{result['counts']['tests']}/{result['counts']['total']}f "
                     f"{result['counts']['pct']}% cases~{result['counts']['cases']}\n")
    return 0


# --------------------------------------------------------------------------
# Diagnostics. `--coverage` reports, `--verify` gates on the rules; neither
# feeds `select()`. `--verify` runs in the select job, so keep it cheap.
# --------------------------------------------------------------------------
def coverage(index: Index) -> dict:
    """Per-module coverage report: change one src module, see what gets selected.

    The verdict is `select()` itself. A second implementation here used to drift
    from the real one, making both the reported precision and the behaviour
    untrustworthy.
    """
    untested: list[str] = []
    strict = 0
    for mod, path in index.src.items():
        rel = path.relative_to(index.root).as_posix()
        r = select(index, [rel])
        if not r["targets"]:
            untested.append(mod)
            continue
        # "strict" = selected without mirror (a directory grain: it cannot miss,
        # but it does not prove the wiring is right).
        if any(r["layers"][k] for k in LAYER_NAMES if k != "mirror"):
            strict += 1

    # mirror is informational only: it is already one of the five layers.
    mirrored = 0
    for mod, path in index.src.items():
        rel = path.relative_to(index.root / "src").as_posix()
        rel = rel[len(PKG) + 1:] if rel.startswith(PKG + "/") else rel
        mirrored += bool(mirror_test_dir(index.root, rel))

    reachable = len(index.src) - len(untested)
    orphans = [t for t, seeds in index.tests.items() if not seeds]
    return {
        "modules": len(index.src),
        "test_files": len(index.test_files),
        "reachable": reachable,
        "mirrored": mirrored,
        # Share of modules whose edit selects at least one file (mirror included).
        "precision": round(100.0 * reachable / len(index.src), 1) if index.src else 0.0,
        # Same without directory mirroring: a drop here means a broken wire.
        "precision_strict": round(100.0 * strict / len(index.src), 1) if index.src else 0.0,
        "orphans": len(orphans),
        "orphan_files": orphans[:20],
        "untested": len(untested),
        "untested_sample": sorted(untested)[:20],
    }


def verify(root: Path) -> int:
    """Assert the selector's rules still hold. Exit 1 means the PR is red.

    Only decision boundaries belong here: lane switches, each wire's existence,
    full-run triggers, the docstring rule, event input -> flags. Per-layer counts
    are not pinned; they change with any legitimate refactor.
    """
    index = Index(root)
    total = len(index.test_files)
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
        if not cond:
            failures.append(name)

    def lane_runs(r: dict) -> dict[str, bool]:
        return {k: r["lanes"][k]["run"] for k in LANES}

    print(f"select_tests.py --verify ({len(index.src)} modules / {total} test files)")

    def sel(*paths: str) -> dict:
        return select(index, list(paths))

    # ---- lane switches (the yaml reads only these three) ------------------
    r = sel("docs/ci/README.md")
    check("docs -> 0 cases, no lane runs",
          r["counts"]["tests"] == 0 and not any(lane_runs(r).values()), str(lane_runs(r)))

    r = sel("src/torchturbo/ops/providers/npu/cross_entropy_loss.py")
    check("ops operator -> NPU runs with selected files",
          r["npu"] and r["targets"]
          and r["lanes"]["npu_unit"]["scope"] == "selected"
          and r["lanes"]["npu_unit"]["files"] == r["npu_files"]
          and r["lanes"]["npu_unit"]["files"] != [],
          str(r["lanes"]["npu_unit"]))
    check("ops operator -> E2E case = cheapest family",
          [e["model_type"] for e in r["e2e_cases"]] == [CHEAPEST_MODEL],
          str(r["e2e_cases"]))

    # engine.core is imported by the `torchturbo.engine` facade, so every test
    # doing `from torchturbo.engine import ...` is pulled in.
    r = sel("src/torchturbo/engine/core.py")
    check("engine -> precise with a realistic selection size",
          r["mode"] == "precise" and r["counts"]["tests"] >= 20, str(r["counts"]))
    # Layer names are defined once in LAYER_NAMES and reused by `layers`,
    # `reasons` and the coverage report; one drift means two vocabularies.
    check("layers and reasons share LAYER_NAMES",
          set(r["layers"]) - {"deleted"} == set(LAYER_NAMES)
          and all(set(v) == set(LAYER_NAMES) for v in r["reasons"].values()),
          f"{sorted(r['layers'])} / {sorted(next(iter(r['reasons'].values())) or {})}")

    r = sel("src/torchturbo/models/qwen3_5/model.py")
    check("family dir -> that family case, no npu_unit, npu_e2e runs",
          [e["model_type"] for e in r["e2e_cases"]] == ["qwen3_5"]
          and not r["lanes"]["npu_unit"]["run"] and r["lanes"]["npu_e2e"]["run"],
          str(lane_runs(r)))
    check("MoE family -> trimmed case",
          [e["layers"] for e in sel("src/torchturbo/models/qwen3_5_moe/model.py")["e2e_cases"]] == [4])
    check("package root -> cheapest E2E case",
          [e["model_type"] for e in sel("src/torchturbo/__init__.py")["e2e_cases"]]
          == [CHEAPEST_MODEL])

    ci_root = Path(__file__).resolve().parents[1]
    if str(ci_root) not in sys.path:
        sys.path.insert(0, str(ci_root))
    from facts import CiCatalog, E2eCase, select_e2e_cases

    def synthetic_case(case_id: str, triggers: set[str]) -> E2eCase:
        return E2eCase(
            model="bailing_hybrid", run_dir="bailing_v3", case_id=case_id,
            framework="lf", env="lf", image="default",
            install="install_lf_deps.sh", train="lf_sft.sh", hf_repo="org/model",
            layers=0, dataset="gsm8k", cards=8, sp=2, ep=4,
            triggers=frozenset(triggers),
        )

    catalog = CiCatalog(
        root=index.root, envs={}, lanes={}, models={"bailing_hybrid": {}},
        cases=(synthetic_case("case-pr-a", {"pr"}),
               synthetic_case("case-pr-b", {"pr"}),
               synthetic_case("case-nightly", {"nightly"})),
        model_envs={}, test_envs={},
    )
    picked = select_e2e_cases(
        catalog, ["src/torchturbo/models/bailing_hybrid/model.py"],
        all_cases=False, force=False,
    )
    check("multiple PR cases in one family are all selected",
          [case["case_id"] for case in picked] == ["case-pr-a", "case-pr-b"],
          str(picked))
    picked = select_e2e_cases(catalog, [], all_cases=True, force=False)
    check("full run selects nightly cases only",
          [case["case_id"] for case in picked] == ["case-nightly"], str(picked))

    r = sel("docker/npu/Dockerfile_aarch64")
    check("image recipe -> npu_unit runs the whole lane",
          r["lanes"]["npu_unit"]["run"]
          and r["lanes"]["npu_unit"]["scope"] == "all"
          and r["lanes"]["npu_unit"]["files"] == [])
    r = sel("tests/ops/npu/test_fused_rms_norm.py")
    check("npu test -> only that file in npu_unit",
          r["lanes"]["npu_unit"]["files"] == ["tests/ops/npu/test_fused_rms_norm.py"],
          str(r["lanes"]["npu_unit"]))
    r = sel("tests/models/qwen4_exp/test_qsa_distributed.py")
    check("mixed CPU/NPU file -> both lanes",
          r["lanes"]["cpu_unit"]["files"] == ["tests/models/qwen4_exp/test_qsa_distributed.py"]
          and r["lanes"]["npu_unit"]["files"] == ["tests/models/qwen4_exp/test_qsa_distributed.py"],
          str(lane_runs(r)))

    # Regression: CI edits need the full CPU suite and one tiny E2E smoke.
    r = sel("scripts/ci/lint_select/select_tests.py")
    check("CI edit -> full CPU, tiny E2E fallback",
          r["mode"] == "all" and r["cpu_files"]
          and not r["lanes"]["npu_unit"]["run"]
          and [e["case_id"] for e in r["e2e_cases"]] == ["bailing_hybrid-lf-sft"],
          f"{r['mode']} {lane_runs(r)}")
    r = sel(".github/workflows/ascend-pr.yml")
    check("workflow edit -> tiny E2E fallback",
          [e["case_id"] for e in r["e2e_cases"]] == ["bailing_hybrid-lf-sft"],
          str(r["e2e_cases"]))
    r = sel(".github/workflows/model_map.json")
    check("E2E map edit -> tiny E2E fallback",
          [e["case_id"] for e in r["e2e_cases"]] == ["bailing_hybrid-lf-sft"],
          str(r["e2e_cases"]))
    r = sel("scripts/ci/envs/common.txt")
    check("shared env edit -> one tiny E2E fallback",
          [e["case_id"] for e in r["e2e_cases"]] == ["bailing_hybrid-lf-sft"],
          str(r["e2e_cases"]))

    # Regression: a non-.py path that merely shares a prefix must not start E2E.
    check("docs under src/ and top-level docs -> no E2E",
          sel("src/torchturbo/ops/README.md")["e2e_cases"] == []
          and sel("README.md")["e2e_cases"] == [])

    # ---- scripts/ path loading: source-text search only -------------------
    # Directory mapping `scripts/runs/**` -> `tests/scripts/**` was wrong: the
    # latter covers scripts/{checkpoint,dataset,dequant}, a pure word collision.
    check("launcher edit -> only tests that really reference it",
          sel("scripts/runs/run_swift.py")["lanes"]["cpu_unit"]["files"]
          == ["tests/adapters/msswift/test_launcher.py"])
    r = sel("scripts/runs/qwen3_5_dense/install_lf_deps.sh")
    check("unreferenced launcher -> no CPU test (E2E covers it)",
          r["lanes"]["cpu_unit"]["files"] == [] and bool(r["e2e_cases"]),
          str(r["counts"]))

    # ---- full-run triggers (any hit -> mode=all) --------------------------
    r = sel("pyproject.toml")
    check("pyproject -> full run, every lane on, CPU files only",
          r["mode"] == "all" and all(lane_runs(r).values())
          and all(test_requirement(index.root / f).cpu
                  for f in r["lanes"]["cpu_unit"]["files"]),
          f"{r['mode']} {lane_runs(r)}")
    # When filtering is no longer worth it, a full run beats a partial set.
    check("huge PR / wide blast radius -> full run",
          sel(*[f"src/torchturbo/engine/f{i}.py" for i in range(400)])["mode"] == "all"
          and sel("src/torchturbo/runtime/base.py")["mode"] == "all")

    # ---- one assertion per wire: it really connects ---------------------
    # Providers are registered after `pkgutil.walk_packages`; no static edge.
    r = sel("src/torchturbo/ops/providers/native/cross_entropy_loss.py")
    check("convention -> same-op tests + discovery tests, no providers/ layer",
          {"tests/ops/test_npu_cross_entropy_loss.py",
           "tests/ops/test_dispatch.py"} <= set(r["targets"])
          and not any(t.startswith("tests/ops/providers/") for t in r["targets"]),
          str(r["targets"]))
    # Mirror: only when the test tree really mirrors the source directory.
    check("mirror -> family tests incl. test_model.py, which imports nothing",
          "tests/models/llada2_moe/test_model.py"
          in sel("src/torchturbo/models/llada2_moe/bd.py")["targets"])
    # spec_from_file_location: no import edge, source text only.
    check("path-loaded test -> selected",
          "tests/models/modules/test_cpu_resident.py"
          in sel("src/torchturbo/models/modules/cpu_resident/base.py")["targets"])
    # No parent fallback: without a same-depth test dir, do not go up a level.
    check("no same-depth test dir -> no whole-layer selection",
          len(sel("src/torchturbo/models/qwen3_5_moe_text/model.py")["targets"]) <= 3)
    # Package facade: `import pkg.sub` executes the init; no graph edge.
    check("package init -> covers every submodule's tests",
          len(sel("src/torchturbo/models/qwen4_exp/__init__.py")["targets"]) >= 10)
    # `Index.src` is scanned after deletion, so removals need a name lookup.
    check("module deletion -> tests that imported it",
          len(index.dangling_tests({"torchturbo.registry.registry"})) > 0)

    # ---- regression: the test tree itself (conftest / helpers) -----------
    # No conftest.py exists today (lint forbids them, markers live in the test
    # files); the branch is kept so adding one cannot silently select nothing.
    r = sel("tests/models/qwen4_exp/conftest.py")
    check("conftest (lint-forbidden) still selects only its directory",
          bool(r["targets"]) and all(t.startswith("tests/models/qwen4_exp/")
                                     for t in r["targets"]), str(r["targets"]))
    check("test helper -> reverse import graph of the test tree",
          "tests/models/test_attention_interface.py"
          in sel("tests/models/flash_attention_fakes.py")["targets"])
    check("deleted / moved test helper -> name fallback",
          "tests/models/test_attention_interface.py"
          in sel("tests/models/_moved_/flash_attention_fakes.py")["targets"])

    # ---- comment / docstring-only edits are not executable changes -------
    # Mirrors `diff_is_docstring_only` in transformers: comments never reach the
    # AST and docstrings are stripped, so both edits serialize identically.
    check("comments / docstrings only -> no change",
          code_only("x = 1  # old\n") == code_only("x = 1  # new\n")
          and code_only('def f():\n    """a"""\n    return 1\n')
          == code_only('def f():\n    """b"""\n    return 1\n'))
    check("real code / docstring turned into code -> a change",
          code_only("x = 1\n") != code_only("x = 2\n")
          and code_only('def f():\n    """a"""\n    return 1\n')
          != code_only('def f():\n    pass\n    return 1\n'))
    # Two meanings of an empty diff: no information -> full run; comments only
    # -> precise with nothing to run.
    r = select(index, [], doc_only=True)
    check("no diff -> full run; comments only -> precise, no lane, reason stated",
          select(index, [])["mode"] == "all"
          and r["mode"] == "precise" and not any(lane_runs(r).values())
          and "docstring" in r["lanes"]["cpu_unit"]["why"], str(lane_runs(r)))

    # Precision guard: a drop means a broken wire. The strict one excludes
    # directory mirroring.
    a = coverage(index)
    print(f"  info  precision {a['precision']}% (with mirror) / "
          f"{a['precision_strict']}% (strict) / {a['modules']} modules / "
          f"{a['test_files']} test files / {a['orphans']} tests with no import")
    check("precision >= 95% and strict precision >= 55%",
          a["precision"] >= 95.0 and a["precision_strict"] >= 55.0,
          f"{a['precision']}% / {a['precision_strict']}%")

    # ---- event input -> four flags (rules live only in flags_from_event) --
    def flags(mode: str, labels: str = "", with_e2e: bool = False,
              force_all: bool = False) -> tuple:
        f = flags_from_event(mode, labels, with_e2e, force_all)
        return (f["force_all"], f["e2e_all"], f["e2e_force"],
                "npu" in f["skip_devices"])

    check("full -> full unit tests; only with_e2e adds all cases",
          flags("full") == (True, False, False, False)
          and flags("full", with_e2e=True) == (True, True, False, False))
    check("pr without labels -> nothing; labels map to flags",
          flags("pr") == (False, False, False, False)
          and flags("pr", "ci:full") == (True, True, False, False)
          and flags("pr", "ci:e2e") == (False, False, True, False)
          and flags("pr", "ci:skip-npu") == (False, False, False, True))
    check("spaces inside labels are ignored",
          flags("pr", "bug, ci:e2e ,docs") == (False, False, True, False))
    # The label exists to spend no cards at all, so it closes the 8-card lane too.
    check("ci:skip-npu closes every NPU lane, not just the unit one",
          lane_runs(select(index, ["src/torchturbo/engine/core.py"],
                           skip_devices=frozenset({"npu"})))
          == {"cpu_unit": True, "npu_unit": False, "npu_e2e": False})
    # force_all means "all unit tests", never "all E2E": the 8-card lane is the
    # expensive one and stays opt-in.
    check("pr + force_all does not imply E2E",
          flags("pr", force_all=True) == (True, False, False, False))

    print(f"\n{'PASS' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except BrokenPipeError:
        # `select_tests.py --explain | head` closes the pipe early; stay quiet.
        sys.stdout = None
        sys.exit(0)
