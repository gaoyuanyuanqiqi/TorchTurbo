#!/usr/bin/env python3
"""Load CI facts and derive test requirements without importing test modules."""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - lint installs PyYAML
    yaml = None

DEVICE_MARKERS = frozenset({"cpu", "npu", "gpu"})
MULTI_DEVICE_MARKERS = frozenset({"multi_npu", "multi_gpu"})
MULTI_DEVICE_TARGETS = {"multi_npu": "npu", "multi_gpu": "gpu"}
REQUIRED_MARKERS = ("cpu", "npu", "multi_npu", "gpu", "multi_gpu", "integration")
KNOWN_MARKERS = frozenset(REQUIRED_MARKERS)
MODEL_PACKAGES = frozenset({"base", "modules", "registry", "weight_converter"})
FRAMEWORK_PACKAGES = {"llamafactory": "lf", "swift": "swift"}
CASE_TRIGGERS = frozenset({"pr", "nightly"})
CI_FALLBACK_CASE_ID = "bailing_hybrid-lf-sft"
CI_FALLBACK_PRIORITY = 3
CI_CONTROL_PREFIXES = (".github/workflows/", ".github/actions/", "scripts/ci/")


@dataclass(frozen=True)
class EnvSpec:
    name: str
    python: str
    requirements: str = ""
    overrides: tuple[str, ...] = ()
    extends: str = ""
    constraints: str = "auto"


@dataclass(frozen=True)
class LaneSpec:
    image: str
    device: str
    resource: str
    slots: int
    runner_variable: str
    runner: str

    @property
    def key(self) -> str:
        return f"{self.image}/{self.resource}"


@dataclass(frozen=True)
class E2eCase:
    model: str
    run_dir: str
    case_id: str
    framework: str
    env: str
    image: str
    install: str
    train: str
    hf_repo: str
    layers: int
    dataset: str
    cards: int
    sp: int
    ep: int
    triggers: frozenset[str]

    def as_selection(self, reason: str) -> dict[str, Any]:
        return {
            "model_type": self.model,
            "case_id": self.case_id,
            "framework": self.framework,
            "env": self.env,
            "layers": self.layers,
            "cards": self.cards,
            "sp": self.sp,
            "ep": self.ep,
            "triggers": sorted(self.triggers),
            "reason": reason,
        }


@dataclass(frozen=True)
class ModelEnv:
    env: str = "default"
    image: str = "default"


@dataclass(frozen=True)
class TestFacts:
    path: str
    devices: frozenset[str]
    device_processes: tuple[tuple[str, int], ...]
    families: frozenset[str]
    frameworks: frozenset[str]
    integration: bool
    env: str
    image: str
    process: str

    def processes_for(self, device: str) -> int:
        return dict(self.device_processes).get(device, 1)


@dataclass
class CiCatalog:
    root: Path
    envs: dict[str, EnvSpec]
    lanes: dict[str, LaneSpec]
    models: dict[str, dict[str, Any]]
    cases: tuple[E2eCase, ...]
    model_envs: dict[str, ModelEnv]
    test_envs: dict[str, str]
    e2e_defaults: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def images(self) -> frozenset[str]:
        return frozenset(lane.image for lane in self.lanes.values())

    def case(self, case_id: str) -> E2eCase | None:
        return next((case for case in self.cases if case.case_id == case_id), None)

    def cases_for(self, model: str, framework: str = "") -> list[E2eCase]:
        return [case for case in self.cases
                if case.model == model and (not framework or case.framework == framework)]

    def cases_for_run_dir(self, run_dir: str) -> list[E2eCase]:
        return [case for case in self.cases if case.run_dir == run_dir]

    def env_for_family(self, family: str) -> ModelEnv:
        return self.model_envs.get(family, ModelEnv())

    def lane_for(self, image: str, device: str, slots: int) -> LaneSpec | None:
        matches = [lane for lane in self.lanes.values()
                   if lane.image == image and lane.device == device and lane.slots >= slots]
        return min(matches, key=lambda lane: lane.slots, default=None)


@dataclass(frozen=True)
class PlanError:
    path: str
    message: str


def _read_toml(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            doc = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"{path}: {exc}")
        return {}
    if not isinstance(doc, dict):
        errors.append(f"{path}: top level must be a table")
        return {}
    return doc


def _read_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if yaml is None:
        errors.append("PyYAML is required to read lanes.yml")
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: {exc}")
        return {}
    if not isinstance(doc, dict):
        errors.append(f"{path}: top level must be a mapping")
        return {}
    return doc


def _load_envs(path: Path, errors: list[str]) -> dict[str, EnvSpec]:
    raw = _read_toml(path, errors).get("envs") or {}
    if not isinstance(raw, dict):
        errors.append(f"{path}: envs must be a table")
        return {}
    envs: dict[str, EnvSpec] = {}
    for name, row in raw.items():
        if not isinstance(row, dict):
            errors.append(f"{path}: env {name!r} must be a table")
            continue
        overrides = row.get("overrides") or []
        if isinstance(overrides, str):
            overrides = [overrides]
        if not isinstance(overrides, list) or not all(
                isinstance(item, str) for item in overrides):
            errors.append(f"{path}: env {name!r} overrides must be strings")
            overrides = []
        envs[str(name)] = EnvSpec(
            name=str(name),
            python=str(row.get("python") or ""),
            requirements=str(row.get("requirements") or ""),
            overrides=tuple(overrides),
            extends=str(row.get("extends") or ""),
            constraints=str(row.get("constraints") or "auto"),
        )
    for env in envs.values():
        if not env.python:
            errors.append(f"{path}: env {env.name!r} has no python version")
        if env.extends and env.extends not in envs:
            errors.append(f"{path}: env {env.name!r} extends unknown env {env.extends!r}")
        if env.requirements and not (path.parents[2] / env.requirements).is_file():
            errors.append(f"{path}: env {env.name!r} requirement file is missing: "
                          f"{env.requirements}")
        for override in env.overrides:
            if not (path.parents[2] / override).is_file():
                errors.append(f"{path}: env {env.name!r} override file is missing: "
                              f"{override}")
    return envs


def _load_lanes(path: Path, errors: list[str]) -> dict[str, LaneSpec]:
    raw = _read_yaml(path, errors).get("lanes") or {}
    if not isinstance(raw, dict):
        errors.append(f"{path}: lanes must be a mapping")
        return {}
    lanes: dict[str, LaneSpec] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            errors.append(f"{path}: lane {key!r} must be a mapping")
            continue
        try:
            slots = int(row["slots"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}: lane {key!r} has no integer slots")
            continue
        image = str(row.get("image") or "")
        lane = LaneSpec(
            image=image,
            device=str(row.get("device") or ""),
            resource=str(row.get("resource") or f"{row.get('device')}-{slots}"),
            slots=slots,
            runner_variable=str(row.get("runner_variable") or ""),
            runner=str(row.get("runner") or ""),
        )
        if not lane.image:
            errors.append(f"{path}: lane {key!r} has no image")
        if lane.device not in {"cpu", "npu", "gpu"}:
            errors.append(f"{path}: lane {key!r} has unknown device {lane.device!r}")
        if slots < 1:
            errors.append(f"{path}: lane {key!r} must have at least one slot")
        if not lane.runner:
            errors.append(f"{path}: lane {key!r} has no runner")
        lanes[str(key)] = lane
    seen_keys: dict[str, str] = {}
    for key, lane in lanes.items():
        if lane.key in seen_keys:
            errors.append(
                f"{path}: lanes {seen_keys[lane.key]!r} and {key!r} share {lane.key!r}"
            )
        else:
            seen_keys[lane.key] = key
    for device in ("cpu", "npu"):
        if not any(lane.device == device for lane in lanes.values()):
            errors.append(f"{path}: no {device} lane is defined")
    return lanes


def _load_e2e_defaults(path: Path, errors: list[str]) -> dict[str, int]:
    raw = _read_yaml(path, errors).get("e2e_defaults")
    if not isinstance(raw, dict):
        errors.append(f"{path}: e2e_defaults must be a mapping")
        return {}
    missing = [key for key in ("fetch_timeout", "case_timeout") if key not in raw]
    if missing:
        errors.append(f"{path}: e2e_defaults missing {', '.join(missing)}")
        return {}
    try:
        fetch_timeout = int(raw["fetch_timeout"])
        case_timeout = int(raw["case_timeout"])
    except (TypeError, ValueError):
        errors.append(f"{path}: e2e_defaults values must be integers")
        return {}
    if fetch_timeout < 1 or case_timeout <= fetch_timeout:
        errors.append(
            f"{path}: e2e_defaults requires 0 < fetch_timeout < case_timeout"
        )
    return {"fetch_timeout": fetch_timeout, "case_timeout": case_timeout}


def _load_model_envs(path: Path, envs: dict[str, EnvSpec],
                     images: frozenset[str], errors: list[str]) -> tuple[
                         dict[str, ModelEnv], dict[str, str]]:
    doc = _read_toml(path, errors)
    models: dict[str, ModelEnv] = {}
    tests: dict[str, str] = {}
    for name, row in (doc.get("models") or {}).items():
        if not isinstance(row, dict):
            errors.append(f"{path}: model {name!r} must be a table")
            continue
        env = str(row.get("env") or "default")
        image = str(row.get("image") or "default")
        if env not in envs:
            errors.append(f"{path}: model {name!r} uses unknown env {env!r}")
        if image not in images:
            errors.append(f"{path}: model {name!r} uses unknown image {image!r}")
        models[str(name)] = ModelEnv(env=env, image=image)
    for name, row in (doc.get("tests") or {}).items():
        if not isinstance(row, dict) or not isinstance(row.get("env"), str):
            errors.append(f"{path}: test override {name!r} needs an env")
            continue
        if row["env"] not in envs:
            errors.append(f"{path}: test override {name!r} uses unknown env {row['env']!r}")
        if not (path.parents[2] / name).is_file():
            errors.append(f"{path}: test override {name!r} does not exist")
        tests[str(name)] = row["env"]
    return models, tests


def registered_models(root: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted((root / "src" / "torchturbo" / "models").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not ((isinstance(func, ast.Name) and func.id == "register_model")
                        or (isinstance(func, ast.Attribute) and func.attr == "register_model")):
                    continue
                values = decorator.args[:1] + [
                    item.value for item in decorator.keywords if item.arg == "name"
                ]
                names.update(value.value for value in values
                             if isinstance(value, ast.Constant) and isinstance(value.value, str))
    return names


def _load_cases(root: Path, models: dict[str, dict[str, Any]], envs: dict[str, EnvSpec],
                images: frozenset[str], errors: list[str]) -> tuple[E2eCase, ...]:
    cases: list[E2eCase] = []
    seen: set[str] = set()
    for model, row in sorted(models.items()):
        run_dir = str(row.get("run_dir") or "")
        raw_cases = row.get("cases") or []
        if not isinstance(raw_cases, list):
            errors.append(f"model_map.json: {model}.cases must be a list")
            continue
        for index, raw in enumerate(raw_cases):
            where = f"model_map.json: {model}.cases[{index}]"
            if not isinstance(raw, dict):
                errors.append(f"{where} must be a mapping")
                continue
            case_id = str(raw.get("id") or "")
            framework = str(raw.get("framework") or "")
            env = str(raw.get("env") or "")
            image = str(raw.get("image") or "default")
            install = str(raw.get("install") or "")
            train = str(raw.get("train") or "")
            dataset = str(raw.get("dataset") or "")
            raw_triggers = raw.get("triggers") or []
            download = raw.get("download") or {}
            required = {
                "id": case_id, "framework": framework, "env": env,
                "install": install, "train": train, "dataset": dataset,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                errors.append(f"{where}: missing {', '.join(missing)}")
                continue
            if case_id in seen:
                errors.append(f"{where}: duplicate case id {case_id!r}")
                continue
            seen.add(case_id)
            if framework not in {"lf", "swift"}:
                errors.append(f"{where}: framework must be lf or swift")
            if env not in envs:
                errors.append(f"{where}: unknown env {env!r}")
            if image not in images:
                errors.append(f"{where}: unknown image {image!r}")
            if framework and framework not in env:
                errors.append(f"{where}: env {env!r} does not match framework {framework!r}")
            if not isinstance(raw_triggers, list) or any(
                    not isinstance(trigger, str) for trigger in raw_triggers):
                errors.append(f"{where}: triggers must be a list of strings")
                triggers = frozenset()
            else:
                triggers = frozenset(raw_triggers)
                unknown = triggers - CASE_TRIGGERS
                if unknown:
                    errors.append(
                        f"{where}: unknown triggers {', '.join(sorted(unknown))}"
                    )
            if not isinstance(download, dict):
                errors.append(f"{where}: download must be a mapping")
                download = {}
            hf_repo = str(download.get("hf_repo") or "")
            layers = download.get("layers", 0)
            if not hf_repo:
                errors.append(f"{where}: download.hf_repo is required")
            if not isinstance(layers, int) or layers < 0:
                errors.append(f"{where}: download.layers must be a non-negative integer")
                layers = 0
            try:
                cards, sp, ep = (int(raw[key]) for key in ("cards", "sp", "ep"))
            except (KeyError, TypeError, ValueError):
                errors.append(f"{where}: cards/sp/ep must be integers")
                continue
            if cards < 1 or sp < 1 or ep < 1 or cards % sp or (cards // sp) % ep:
                errors.append(f"{where}: invalid cards/sp/ep {cards}/{sp}/{ep}")
            run_path = root / "scripts" / "runs" / run_dir
            for label, script in (("install", install), ("train", train)):
                if not (run_path / script).is_file():
                    errors.append(f"{where}: {label} script is missing: {run_path / script}")
            cases.append(E2eCase(
                model=model, run_dir=run_dir, case_id=case_id, framework=framework,
                env=env, image=image, install=install, train=train, hf_repo=hf_repo,
                layers=layers, dataset=dataset, cards=cards, sp=sp, ep=ep,
                triggers=triggers,
            ))
    return tuple(cases)


def _load_models(path: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{path}: {exc}")
        return {}
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        errors.append(f"{path}: missing top-level models mapping")
        return {}
    for name, row in models.items():
        if not isinstance(row, dict):
            errors.append(f"{path}: model {name!r} must be a mapping")
        elif not isinstance(row.get("run_dir"), str):
            errors.append(f"{path}: model {name!r} needs a string run_dir")
    return {str(name): row for name, row in models.items() if isinstance(row, dict)}


@lru_cache(maxsize=4)
def load_catalog(root: Path) -> CiCatalog:
    errors: list[str] = []
    lanes_path = root / "scripts" / "ci" / "lanes.yml"
    envs = _load_envs(root / "scripts" / "ci" / "envs.toml", errors)
    lanes = _load_lanes(lanes_path, errors)
    e2e_defaults = _load_e2e_defaults(lanes_path, errors)
    images = frozenset(lane.image for lane in lanes.values())
    model_envs, test_envs = _load_model_envs(
        root / "scripts" / "ci" / "model_env.toml", envs, images, errors
    )
    models = _load_models(root / ".github" / "workflows" / "model_map.json", errors)
    cases = _load_cases(root, models, envs, images, errors)
    fallback = next((case for case in cases if case.case_id == CI_FALLBACK_CASE_ID), None)
    if fallback is None:
        errors.append(f"model_map.json: CI fallback case {CI_FALLBACK_CASE_ID!r} is missing")
    elif "pr" not in fallback.triggers:
        errors.append(f"model_map.json: CI fallback case {CI_FALLBACK_CASE_ID!r} must trigger pr")
    registered = registered_models(root)
    for name in sorted(registered - set(models)):
        errors.append(f"model_map.json: registered model {name!r} is missing")
    for name in sorted(set(models) - registered):
        errors.append(f"model_map.json: model {name!r} is not registered")
    for model in model_envs:
        if model not in models:
            errors.append(f"model_env.toml: model {model!r} is not in model_map.json")
    return CiCatalog(root=root, envs=envs, lanes=lanes, models=models, cases=cases,
                     model_envs=model_envs, test_envs=test_envs,
                     e2e_defaults=e2e_defaults, errors=errors)


def _markers(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result: set[str] = set()
        for item in node.elts:
            result |= _markers(item)
        return result
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute) and node.attr in KNOWN_MARKERS:
        base = node.value
        if (isinstance(base, ast.Attribute) and base.attr == "mark"
                and isinstance(base.value, ast.Name) and base.value.id == "pytest"):
            return {node.attr}
    return set()


def _module_markers(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in targets):
            result |= _markers(value)
    return result


def _imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules, {name for name in modules if name.split(".", 1)[0] in FRAMEWORK_PACKAGES}


def _literal_int(call: ast.Call, name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, int) else None
    return None


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_string(node: ast.AST) -> str:
    return node.value.strip().lower() if isinstance(node, ast.Constant) and isinstance(
        node.value, str
    ) else ""


def _normalized_device(value: str) -> str:
    return "gpu" if value == "cuda" else value


def _device_evidence(tree: ast.AST) -> set[str]:
    """Return devices with direct tensor/runtime evidence below ``tree``."""
    # Mocked runtime configuration does not execute device code.
    configured_devices: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func).rsplit(".", 1)[-1]
        if name not in {"SimpleNamespace", "Namespace"}:
            continue
        for keyword in node.keywords:
            if keyword.arg == "device":
                configured_devices.update(id(child) for child in ast.walk(keyword.value))

    devices: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in configured_devices:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([alias.name for alias in node.names]
                       if isinstance(node, ast.Import) else [node.module or ""])
            for module in modules:
                root = module.split(".", 1)[0]
                if root == "fla_npu":
                    devices.add("npu")
            continue
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if _dotted_name(base).rsplit(".", 1)[-1] == "CpuRuntime":
                    devices.add("cpu")
            continue
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        name = dotted.rsplit(".", 1)[-1]
        if name in {"npu", "cuda"} and dotted.count(".") >= 1:
            devices.add(_normalized_device(name))
        elif name in {"to", "device", "init_device_mesh"} and node.args:
            value = _literal_string(node.args[0])
            device = _normalized_device(value)
            if device in DEVICE_MARKERS:
                devices.add(device)
        elif name in {"require_npu", "require_npus"}:
            devices.add("npu")
        elif dotted == "CpuRuntime":
            devices.add("cpu")
        for keyword in node.keywords:
            if keyword.arg == "device":
                value = _literal_string(keyword.value)
                device = _normalized_device(value)
                if device in DEVICE_MARKERS:
                    devices.add(device)
    return devices


def _module_device_guards(tree: ast.Module) -> set[str]:
    """Return devices required by module-level runtime guards."""
    devices: set[str] = set()
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        if not isinstance(value, ast.Call):
            continue
        name = _dotted_name(value.func).rsplit(".", 1)[-1]
        if name in {"require_npu", "require_npus"}:
            devices.add("npu")
        elif name == "importorskip" and value.args:
            module = _literal_string(value.args[0]).split(".", 1)[0]
            if module in {"torch_npu", "fla_npu"}:
                devices.add("npu")
    return devices


def _parametrized_processes(tree: ast.AST) -> int:
    """Read world-size values from pytest parametrization, if present."""
    required = 1
    names = {"world_size", "nprocs", "num_processes", "rank_size"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _dotted_name(node.func).rsplit(
                ".", 1)[-1] != "parametrize" or len(node.args) < 2:
            continue
        try:
            parameters = ast.literal_eval(node.args[0])
            values = ast.literal_eval(node.args[1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(parameters, str):
            parameters = [part.strip() for part in parameters.split(",")]
        if not isinstance(parameters, (list, tuple)):
            continue
        indexes = [index for index, name in enumerate(parameters)
                   if isinstance(name, str) and name in names]
        if not indexes or not isinstance(values, (list, tuple)):
            continue
        for row in values:
            row = row if isinstance(row, (list, tuple)) else (row,)
            for index in indexes:
                if index < len(row) and isinstance(row[index], int):
                    required = max(required, row[index])
    return required


def _multiprocess_rank(tree: ast.Module) -> int:
    required = 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        if name not in {"spawn", "init_process_group", "require_npus"}:
            continue
        if name == "require_npus":
            size = node.args[0].value if node.args and isinstance(
                node.args[0], ast.Constant
            ) and isinstance(node.args[0].value, int) else None
        else:
            key = "world_size" if name == "init_process_group" else "nprocs"
            size = _literal_int(node, key)
        required = max(required, size if size is not None else 2)
    return max(required, _parametrized_processes(tree))


def _process_unit(path: str, families: set[str]) -> str:
    parts = Path(path).parts
    if parts[:2] == ("tests", "models"):
        if len(parts) == 3:
            return "layer:tests/models"
        if len(parts) >= 4 and parts[2] not in MODEL_PACKAGES:
            return f"family:{parts[2]}"
    if families:
        return "import:" + ",".join(sorted(families))
    parent = "/".join(parts[:-1])
    return f"layer:{parent}"


def _resolve_env(catalog: CiCatalog, path: str, families: set[str],
                 frameworks: set[str]) -> str:
    if path in catalog.test_envs:
        return catalog.test_envs[path]
    model_envs = {catalog.env_for_family(family).env for family in families}
    if len(model_envs) > 1:
        return "invalid:multiple-model-envs"
    model_env = next(iter(model_envs), "default")
    if frameworks == {"lf", "swift"}:
        return "invalid:multiple-frameworks"
    if frameworks:
        framework_env = next(iter(frameworks))
        if model_env == "default":
            return framework_env
        if model_env == "qwen4_exp" and framework_env == "swift":
            return "qwen4_exp_swift"
        return f"invalid:{model_env}+{framework_env}"
    return model_env


def scan_tests(catalog: CiCatalog) -> tuple[dict[str, TestFacts], list[PlanError]]:
    facts: dict[str, TestFacts] = {}
    errors: list[PlanError] = []
    for path in sorted((catalog.root / "tests").rglob("test_*.py")):
        rel = path.relative_to(catalog.root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError) as exc:
            errors.append(PlanError(rel, f"cannot parse test file: {exc}"))
            continue
        module_markers = _module_markers(tree)
        tests = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test")]
        devices: set[str] = set()
        multi_devices: set[str] = set()
        device_processes: dict[str, int] = {}
        integration = "integration" in module_markers
        module_guards = _module_device_guards(tree)
        for test in tests:
            markers = module_markers | set().union(
                *(_markers(item) for item in test.decorator_list)
            )
            test_devices = set(markers & DEVICE_MARKERS)
            test_multi = markers & MULTI_DEVICE_MARKERS
            test_devices.update(MULTI_DEVICE_TARGETS[name] for name in test_multi)
            integration |= "integration" in markers
            if not test_devices:
                errors.append(PlanError(rel, f"{test.name} has no device marker"))
            elif "cpu" in test_devices and len(test_devices) > 1:
                errors.append(PlanError(
                    rel, f"{test.name} mixes cpu with accelerator markers"
                ))
            evidence = _device_evidence(test) | module_guards
            if evidence and test_devices.isdisjoint(evidence):
                errors.append(PlanError(
                    rel, f"{test.name} markers {sorted(test_devices)} disagree with "
                    f"code evidence {sorted(evidence)}"
                ))
            devices |= test_devices
            multi_devices |= {MULTI_DEVICE_TARGETS[name] for name in test_multi}
            test_processes = _multiprocess_rank(test)
            for device in test_devices:
                device_processes[device] = max(
                    device_processes.get(device, 1), test_processes
                )
        if not tests:
            continue
        processes = _multiprocess_rank(tree)
        evidence = _device_evidence(tree)
        if evidence and devices.isdisjoint(evidence):
            errors.append(PlanError(
                rel, f"device markers {sorted(devices)} disagree with code evidence "
                f"{sorted(evidence)}"
            ))
        if len(devices) == 1:
            only_device = next(iter(devices))
            device_processes[only_device] = max(
                device_processes.get(only_device, 1), processes
            )
        for device in sorted(multi_devices):
            if device_processes.get(device, 1) < 2:
                errors.append(PlanError(
                    rel, f"multi_{device} marker needs at least two processes"
                ))
        for device in sorted(devices - {"cpu"} - multi_devices):
            if device_processes.get(device, 1) >= 2:
                errors.append(PlanError(
                    rel, f"{device} facts need multi_{device}, found "
                    f"{device_processes[device]} processes"
                ))
        modules, external = _imports(tree)
        families = {name.split(".")[2] for name in modules
                    if name.startswith("torchturbo.models.")
                    and len(name.split(".")) > 2
                    and name.split(".")[2] in catalog.models
                    and name.split(".")[2] not in MODEL_PACKAGES}
        frameworks = {FRAMEWORK_PACKAGES[name.split(".", 1)[0]] for name in external}
        if rel.startswith("tests/adapters/llamafactory_v0/"):
            frameworks.add("lf")
        if rel.startswith("tests/adapters/msswift/"):
            frameworks.add("swift")
        model_images = {catalog.env_for_family(family).image for family in families}
        image = next(iter(model_images), "default") if len(model_images) <= 1 else "invalid"
        env = _resolve_env(catalog, rel, families, frameworks)
        if env not in catalog.envs:
            errors.append(PlanError(rel, f"resolved env {env!r} is not defined"))
        if image not in catalog.images:
            errors.append(PlanError(rel, "test imports model families with different images"))
        facts[rel] = TestFacts(
            path=rel,
            devices=frozenset(devices),
            device_processes=tuple(sorted(device_processes.items())),
            families=frozenset(families),
            frameworks=frozenset(frameworks),
            integration=integration,
            env=env,
            image=image,
            process=_process_unit(rel, families),
        )
    return facts, errors


def pr_cases(cases: list[E2eCase]) -> list[E2eCase]:
    return [case for case in cases if "pr" in case.triggers]


def is_ci_control_path(path: str) -> bool:
    if path.startswith((".github/workflows/", ".github/actions/")):
        return True
    return path.startswith("scripts/ci/") and not path.endswith((".md", ".rst"))


def ci_fallback_case(catalog: CiCatalog, env: str = "") -> E2eCase | None:
    case = catalog.case(CI_FALLBACK_CASE_ID)
    if case is None or "pr" not in case.triggers:
        return None
    if env and env not in {"envs", "common"} and case.env != env:
        return None
    return case


def select_e2e_cases(catalog: CiCatalog, changed: list[str], *, all_cases: bool,
                     force: bool) -> list[dict[str, Any]]:
    by_model = {model: catalog.cases_for(model) for model in catalog.models}
    if all_cases:
        return [case.as_selection("all cases (full run)") for case in catalog.cases
                if "nightly" in case.triggers]
    if force:
        changed = ["pyproject.toml"]

    picks: dict[str, tuple[int, E2eCase, str]] = {}

    def add(case: E2eCase | None, priority: int, reason: str) -> None:
        if case is None:
            return
        current = picks.get(case.case_id)
        if current is None or priority < current[0]:
            picks[case.case_id] = (priority, case, reason)

    def add_pr(cases: list[E2eCase], priority: int, reason: str) -> None:
        for case in pr_cases(cases):
            add(case, priority, reason)

    for path in changed:
        if path.startswith("scripts/runs/"):
            parts = path.split("/")
            if len(parts) >= 4:
                run_dir, script = parts[2], parts[3]
                exact = [case for case in catalog.cases
                         if case.run_dir == run_dir
                         and script in {case.train, case.install}]
                if exact:
                    add_pr(exact, 0, f"{path} (exact case)")
                    continue
                for case in catalog.cases_for_run_dir(run_dir):
                    add_pr([case], 1, f"{path} (run_dir={run_dir})")
            elif len(parts) == 3:
                for case in catalog.cases_for_run_dir(parts[2]):
                    add_pr([case], 1, f"{path} (run_dir)")
        elif path.startswith("scripts/ci/envs/") or path == "scripts/ci/envs.toml":
            env = Path(path).stem
            scope = "all envs" if env in {"envs", "common"} else f"env={env}"
            add(ci_fallback_case(catalog, env), CI_FALLBACK_PRIORITY,
                f"{path} ({scope} CI fallback)")
        elif path == ".github/workflows/model_map.json":
            add(ci_fallback_case(catalog), CI_FALLBACK_PRIORITY,
                f"{path} (E2E map fallback)")
        elif is_ci_control_path(path):
            add(ci_fallback_case(catalog), CI_FALLBACK_PRIORITY,
                f"{path} (CI fallback)")
        elif path.startswith("src/torchturbo/adapters/"):
            if not path.endswith(".py"):
                continue
            framework = "swift" if "swift" in path else "lf"
            for model, cases in by_model.items():
                add_pr([case for case in cases if case.framework == framework],
                       1, f"{path} (framework={framework})")
        elif path.startswith("src/torchturbo/models/"):
            if not path.endswith(".py"):
                continue
            parts = path.split("/")
            family = parts[3] if len(parts) > 3 else ""
            if family in by_model and family not in MODEL_PACKAGES:
                add_pr(by_model[family], 0, f"{path} (family={family})")
            else:
                add_pr(by_model.get("qwen3_5", []), 2, f"{path} (shared model layer)")
        elif path.startswith("src/torchturbo/") and path.endswith(".py"):
            add_pr(by_model.get("qwen3_5", []), 2, f"{path} (startup path)")
        elif path.startswith("docker/npu/") or path == "pyproject.toml":
            add_pr(by_model.get("qwen3_5", []), 2, f"{path} (image/deps)")
    if picks and min(item[0] for item in picks.values()) < CI_FALLBACK_PRIORITY:
        picks = {case_id: item for case_id, item in picks.items()
                 if item[0] < CI_FALLBACK_PRIORITY}
    ordered = sorted(picks.values(), key=lambda item: (item[0], item[1].case_id))
    return [case.as_selection(reason) for _, case, reason in ordered]
