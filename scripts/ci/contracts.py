#!/usr/bin/env python3
"""Serializable contracts shared by the CI planner, executor, and gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
JOB_RESULT_FILENAME = "job-result.json"
SUITES = ("core", "model", "adapter")
JOB_KINDS = ("pytest", "model_e2e")
DEVICES = ("cpu", "npu", "gpu")
JOB_STATUSES = ("passed", "failed", "skipped", "cancelled", "infra_error")
PLAN_ARTIFACT_STATUSES = ("ready", "failed")


class ContractError(ValueError):
    """Raised when a CI contract is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class Source:
    repository: str
    ref: str
    sha: str

    def to_dict(self) -> dict[str, str]:
        return {"repository": self.repository, "ref": self.ref, "sha": self.sha}


@dataclass(frozen=True)
class ImageRef:
    name: str
    kind: str
    ref: str
    digest: str = ""

    def to_dict(self) -> dict[str, str]:
        value = {"name": self.name, "kind": self.kind, "ref": self.ref}
        if self.digest:
            value["digest"] = self.digest
        return value


@dataclass(frozen=True)
class VenvSpec:
    name: str
    python: str
    requirements: tuple[str, ...]
    overrides: tuple[str, ...]
    extends: str | None
    constraints: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "python": self.python,
            "requirements": list(self.requirements),
            "overrides": list(self.overrides),
            "extends": self.extends,
            "constraints": self.constraints,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class Runtime:
    device: str
    ranks: int
    lane: str
    image: ImageRef
    venv: VenvSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "ranks": self.ranks,
            "lane": self.lane,
            "image": self.image.to_dict(),
            "venv": self.venv.to_dict(),
        }


@dataclass(frozen=True)
class PytestSpec:
    suite: str
    frameworks: tuple[str, ...]
    model_families: tuple[str, ...]
    marker: str
    process_group: str
    tags: tuple[str, ...]
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "frameworks": list(self.frameworks),
            "model_families": list(self.model_families),
            "marker": self.marker,
            "process_group": self.process_group,
            "tags": list(self.tags),
            "files": list(self.files),
        }


@dataclass(frozen=True)
class CommandSpec:
    script: str
    args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"script": self.script, "args": list(self.args)}


@dataclass(frozen=True)
class ModelE2ESpec:
    case_id: str
    model: str
    framework: str
    run_dir: str
    workdir: str
    install: CommandSpec
    train: CommandSpec
    hf_repo: str
    layers: int
    dataset: str
    ranks: int
    parallelism_sp: int
    parallelism_ep: int
    seq_len: int
    finetuning_type: str
    fetch_timeout_seconds: int
    case_timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "framework": self.framework,
            "run_dir": self.run_dir,
            "workdir": self.workdir,
            "install": self.install.to_dict(),
            "train": self.train.to_dict(),
            "download": {"hf_repo": self.hf_repo, "layers": self.layers},
            "dataset": self.dataset,
            "ranks": self.ranks,
            "parallelism": {"sp": self.parallelism_sp, "ep": self.parallelism_ep},
            "seq_len": self.seq_len,
            "finetuning_type": self.finetuning_type,
            "fetch_timeout_seconds": self.fetch_timeout_seconds,
            "case_timeout_seconds": self.case_timeout_seconds,
        }


@dataclass(frozen=True)
class Step:
    id: str
    kind: str
    ranks: int
    timeout_seconds: int
    spec: PytestSpec | ModelE2ESpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "ranks": self.ranks,
            "timeout_seconds": self.timeout_seconds,
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    allowed_skip_reasons: tuple[str, ...]
    timeout_minutes: int
    runtime: Runtime
    steps: tuple[Step, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "allowed_skip_reasons": list(self.allowed_skip_reasons),
            "timeout_minutes": self.timeout_minutes,
            "runtime": self.runtime.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class Plan:
    schema_version: int
    plan_id: str
    policy: str
    source: Source
    inputs: dict[str, str]
    coverage: dict[str, Any]
    jobs: tuple[Job, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "policy": self.policy,
            "source": self.source.to_dict(),
            "inputs": dict(sorted(self.inputs.items())),
            "coverage": self.coverage,
            "jobs": [job.to_dict() for job in self.jobs],
        }


@dataclass(frozen=True)
class PlanArtifact:
    schema_version: int
    artifact_id: str
    status: str
    policy: str
    source: Source
    errors: tuple[Issue, ...] = ()
    plan: Plan | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "status": self.status,
            "policy": self.policy,
            "source": self.source.to_dict(),
            "errors": [issue.to_dict() for issue in self.errors],
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }
        return value


@dataclass(frozen=True)
class PytestEvidence:
    executed_files: int
    tests_total: int
    tests_failed: int
    tests_errors: int
    tests_skipped: int

    def to_dict(self) -> dict[str, int]:
        return {
            "executed_files": self.executed_files,
            "tests_total": self.tests_total,
            "tests_failed": self.tests_failed,
            "tests_errors": self.tests_errors,
            "tests_skipped": self.tests_skipped,
        }


@dataclass(frozen=True)
class ModelE2EEvidence:
    case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"case_ids": list(self.case_ids)}


@dataclass(frozen=True)
class StepResult:
    id: str
    status: str
    exit_code: int
    spec_digest: str
    case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "status": self.status,
            "exit_code": self.exit_code,
            "spec_digest": self.spec_digest,
        }
        if self.case_ids:
            value["case_ids"] = list(self.case_ids)
        return value


@dataclass(frozen=True)
class JobResult:
    schema_version: int
    plan_id: str
    job_id: str
    kind: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: int
    runtime: Runtime
    items: PytestEvidence | ModelE2EEvidence
    steps: tuple[StepResult, ...]
    skip_reason: str | None = None
    skip_message: str | None = None
    failure: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "runtime": self.runtime.to_dict(),
            "items": self.items.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "skip": (
                {"reason": self.skip_reason, "message": self.skip_message or ""}
                if self.skip_reason is not None
                else None
            ),
            "failure": self.failure,
            "artifacts": dict(sorted(self.artifacts.items())),
        }
        return value


@dataclass(frozen=True)
class Result:
    schema_version: int
    plan_id: str
    jobs: tuple[JobResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "jobs": [job.to_dict() for job in self.jobs],
        }


@dataclass(frozen=True)
class Decision:
    schema_version: int
    plan_id: str
    status: str
    summary: dict[str, int]
    checks: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "status": self.status,
            "summary": dict(sorted(self.summary.items())),
            "checks": list(self.checks),
            "failures": list(self.failures),
        }


def canonical_json(value: object) -> str:
    """Return deterministic JSON for hashing and artifact comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(
    *,
    policy: str,
    source: Source,
    inputs: dict[str, str],
    coverage: dict[str, Any],
    jobs: list[Job] | tuple[Job, ...],
) -> Plan:
    payload = _plan_payload(
        policy=policy,
        source=source,
        inputs=inputs,
        coverage=coverage,
        jobs=jobs,
    )
    return Plan(
        schema_version=SCHEMA_VERSION,
        plan_id=digest(payload),
        policy=policy,
        source=source,
        inputs=dict(inputs),
        coverage=coverage,
        jobs=tuple(jobs),
    )


def build_plan_artifact(
    *,
    policy: str,
    source: Source,
    errors: list[Issue] | tuple[Issue, ...] = (),
    plan: Plan | None = None,
) -> PlanArtifact:
    status = "ready" if plan is not None and not errors else "failed"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "policy": policy,
        "source": source.to_dict(),
        "errors": [issue.to_dict() for issue in errors],
        "plan": plan.to_dict() if plan is not None else None,
    }
    return PlanArtifact(
        schema_version=SCHEMA_VERSION,
        artifact_id=digest(payload),
        status=status,
        policy=policy,
        source=source,
        errors=tuple(errors),
        plan=plan,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_payload(
    *,
    policy: str,
    source: Source,
    inputs: dict[str, str],
    coverage: dict[str, Any],
    jobs: list[Job] | tuple[Job, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "source": source.to_dict(),
        "inputs": dict(sorted(inputs.items())),
        "coverage": coverage,
        "jobs": [job.to_dict() for job in jobs],
    }


def plan_digest(plan: Plan) -> str:
    """Return the digest that must match ``plan.plan_id``."""
    return digest(_plan_payload(
        policy=plan.policy,
        source=plan.source,
        inputs=plan.inputs,
        coverage=plan.coverage,
        jobs=plan.jobs,
    ))


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{path} must be a mapping")
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{path} must be a list")
    return value


def _expect_str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _expect_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(f"{path} must be an integer >= {minimum}")
    return value


def _expect_str_list(value: Any, path: str) -> list[str]:
    items = _expect_list(value, path)
    if not all(isinstance(item, str) for item in items):
        raise ContractError(f"{path} must be a list of strings")
    return list(items)


def _expect_keys(value: dict[str, Any], path: str, *, required: set[str],
                 optional: set[str] = frozenset()) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ContractError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{path} has unknown fields: {', '.join(unknown)}")


def _parse_source(value: Any, path: str) -> Source:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"repository", "ref", "sha"})
    return Source(
        repository=_expect_str(row["repository"], f"{path}.repository"),
        ref=_expect_str(row["ref"], f"{path}.ref"),
        sha=_expect_str(row["sha"], f"{path}.sha"),
    )


def _parse_image(value: Any, path: str) -> ImageRef:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"name", "kind", "ref"}, optional={"digest"})
    kind = _expect_str(row["kind"], f"{path}.kind")
    if kind not in {"host", "container"}:
        raise ContractError(f"{path}.kind must be host or container")
    return ImageRef(
        name=_expect_str(row["name"], f"{path}.name"),
        kind=kind,
        ref=_expect_str(row["ref"], f"{path}.ref"),
        digest=_expect_str(row.get("digest", ""), f"{path}.digest", allow_empty=True),
    )


def _parse_venv(value: Any, path: str) -> VenvSpec:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "name", "python", "requirements", "overrides", "extends",
            "constraints", "digest",
        },
    )
    extends = row["extends"]
    if extends is not None:
        extends = _expect_str(extends, f"{path}.extends")
    return VenvSpec(
        name=_expect_str(row["name"], f"{path}.name"),
        python=_expect_str(row["python"], f"{path}.python"),
        requirements=tuple(_expect_str_list(row["requirements"], f"{path}.requirements")),
        overrides=tuple(_expect_str_list(row["overrides"], f"{path}.overrides")),
        extends=extends,
        constraints=_expect_str(row["constraints"], f"{path}.constraints"),
        digest=_expect_str(row["digest"], f"{path}.digest"),
    )


def _parse_runtime(value: Any, path: str) -> Runtime:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={"device", "ranks", "lane", "image", "venv"},
    )
    device = _expect_str(row["device"], f"{path}.device")
    if device not in DEVICES:
        raise ContractError(f"{path}.device has unknown device {device!r}")
    return Runtime(
        device=device,
        ranks=_expect_int(row["ranks"], f"{path}.ranks", minimum=1),
        lane=_expect_str(row["lane"], f"{path}.lane"),
        image=_parse_image(row["image"], f"{path}.image"),
        venv=_parse_venv(row["venv"], f"{path}.venv"),
    )


def _parse_pytest_spec(value: Any, path: str) -> PytestSpec:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "suite", "frameworks", "model_families", "marker",
            "process_group", "tags", "files",
        },
    )
    suite = _expect_str(row["suite"], f"{path}.suite")
    if suite not in SUITES:
        raise ContractError(f"{path}.suite has unknown suite {suite!r}")
    files = _expect_str_list(row["files"], f"{path}.files")
    if not files:
        raise ContractError(f"{path}.files must not be empty")
    return PytestSpec(
        suite=suite,
        frameworks=tuple(_expect_str_list(row["frameworks"], f"{path}.frameworks")),
        model_families=tuple(_expect_str_list(row["model_families"], f"{path}.model_families")),
        marker=_expect_str(row["marker"], f"{path}.marker"),
        process_group=_expect_str(row["process_group"], f"{path}.process_group"),
        tags=tuple(_expect_str_list(row["tags"], f"{path}.tags")),
        files=tuple(files),
    )


def _parse_command(value: Any, path: str) -> CommandSpec:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"script", "args"})
    return CommandSpec(
        script=_expect_str(row["script"], f"{path}.script"),
        args=tuple(_expect_str_list(row["args"], f"{path}.args")),
    )


def _parse_model_e2e_spec(value: Any, path: str) -> ModelE2ESpec:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "case_id", "model", "framework", "run_dir", "workdir", "install",
            "train", "download", "dataset", "ranks", "parallelism", "seq_len",
            "finetuning_type", "fetch_timeout_seconds", "case_timeout_seconds",
        },
    )
    download = _expect_mapping(row["download"], f"{path}.download")
    _expect_keys(download, f"{path}.download", required={"hf_repo", "layers"})
    parallelism = _expect_mapping(row["parallelism"], f"{path}.parallelism")
    _expect_keys(parallelism, f"{path}.parallelism", required={"sp", "ep"})
    framework = _expect_str(row["framework"], f"{path}.framework")
    if framework not in {"lf", "swift"}:
        raise ContractError(f"{path}.framework must be lf or swift")
    return ModelE2ESpec(
        case_id=_expect_str(row["case_id"], f"{path}.case_id"),
        model=_expect_str(row["model"], f"{path}.model"),
        framework=framework,
        run_dir=_expect_str(row["run_dir"], f"{path}.run_dir"),
        workdir=_expect_str(row["workdir"], f"{path}.workdir"),
        install=_parse_command(row["install"], f"{path}.install"),
        train=_parse_command(row["train"], f"{path}.train"),
        hf_repo=_expect_str(download["hf_repo"], f"{path}.download.hf_repo"),
        layers=_expect_int(download["layers"], f"{path}.download.layers"),
        dataset=_expect_str(row["dataset"], f"{path}.dataset"),
        ranks=_expect_int(row["ranks"], f"{path}.ranks", minimum=1),
        parallelism_sp=_expect_int(parallelism["sp"], f"{path}.parallelism.sp", minimum=1),
        parallelism_ep=_expect_int(parallelism["ep"], f"{path}.parallelism.ep", minimum=1),
        seq_len=_expect_int(row["seq_len"], f"{path}.seq_len", minimum=1),
        finetuning_type=_expect_str(row["finetuning_type"], f"{path}.finetuning_type"),
        fetch_timeout_seconds=_expect_int(
            row["fetch_timeout_seconds"], f"{path}.fetch_timeout_seconds", minimum=1
        ),
        case_timeout_seconds=_expect_int(
            row["case_timeout_seconds"], f"{path}.case_timeout_seconds", minimum=1
        ),
    )


def _parse_step(value: Any, path: str) -> Step:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"id", "kind", "ranks", "timeout_seconds", "spec"})
    kind = _expect_str(row["kind"], f"{path}.kind")
    if kind not in JOB_KINDS:
        raise ContractError(f"{path}.kind has unknown kind {kind!r}")
    if kind == "pytest":
        spec: PytestSpec | ModelE2ESpec = _parse_pytest_spec(row["spec"], f"{path}.spec")
    else:
        spec = _parse_model_e2e_spec(row["spec"], f"{path}.spec")
    return Step(
        id=_expect_str(row["id"], f"{path}.id"),
        kind=kind,
        ranks=_expect_int(row["ranks"], f"{path}.ranks", minimum=1),
        timeout_seconds=_expect_int(
            row["timeout_seconds"], f"{path}.timeout_seconds", minimum=1
        ),
        spec=spec,
    )


def _parse_job(value: Any, path: str) -> Job:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "id", "kind", "allowed_skip_reasons", "timeout_minutes",
            "runtime", "steps",
        },
    )
    kind = _expect_str(row["kind"], f"{path}.kind")
    if kind not in JOB_KINDS:
        raise ContractError(f"{path}.kind has unknown kind {kind!r}")
    steps = tuple(
        _parse_step(item, f"{path}.steps[{index}]")
        for index, item in enumerate(_expect_list(row["steps"], f"{path}.steps"))
    )
    if not steps:
        raise ContractError(f"{path}.steps must not be empty")
    return Job(
        id=_expect_str(row["id"], f"{path}.id"),
        kind=kind,
        allowed_skip_reasons=tuple(
            _expect_str_list(row["allowed_skip_reasons"], f"{path}.allowed_skip_reasons")
        ),
        timeout_minutes=_expect_int(
            row["timeout_minutes"], f"{path}.timeout_minutes", minimum=1
        ),
        runtime=_parse_runtime(row["runtime"], f"{path}.runtime"),
        steps=steps,
    )


def _plan_from_dict(value: Any, path: str = "plan") -> Plan:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "schema_version", "plan_id", "policy", "source", "inputs",
            "coverage", "jobs",
        },
    )
    schema_version = _expect_int(row["schema_version"], f"{path}.schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ContractError(
            f"{path}.schema_version is {schema_version}, expected {SCHEMA_VERSION}"
        )
    inputs_raw = _expect_mapping(row["inputs"], f"{path}.inputs")
    inputs = {
        _expect_str(key, f"{path}.inputs key"): _expect_str(
            item, f"{path}.inputs.{key}"
        )
        for key, item in inputs_raw.items()
    }
    plan = Plan(
        schema_version=schema_version,
        plan_id=_expect_str(row["plan_id"], f"{path}.plan_id"),
        policy=_expect_str(row["policy"], f"{path}.policy"),
        source=_parse_source(row["source"], f"{path}.source"),
        inputs=inputs,
        coverage=_expect_mapping(row["coverage"], f"{path}.coverage"),
        jobs=tuple(
            _parse_job(item, f"{path}.jobs[{index}]")
            for index, item in enumerate(_expect_list(row["jobs"], f"{path}.jobs"))
        ),
    )
    expected = plan_digest(plan)
    if plan.plan_id != expected:
        raise ContractError(
            f"{path}.plan_id does not match plan contents; expected {expected}"
        )
    return plan


def load_plan(path: Path) -> Plan:
    """Load a Plan and verify its schema, fields, and content digest."""
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read plan {path}: {exc}") from exc
    return _plan_from_dict(value)


def _parse_issue(value: Any, path: str) -> Issue:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"code", "path", "message"})
    return Issue(
        code=_expect_str(row["code"], f"{path}.code"),
        path=_expect_str(row["path"], f"{path}.path"),
        message=_expect_str(row["message"], f"{path}.message"),
    )


def load_plan_artifact(path: Path) -> PlanArtifact:
    """Load and verify a Planner artifact, including failed-plan output."""
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read plan artifact {path}: {exc}") from exc
    row = _expect_mapping(value, "artifact")
    _expect_keys(
        row,
        "artifact",
        required={
            "schema_version", "artifact_id", "status", "policy", "source",
            "errors", "plan",
        },
    )
    schema_version = _expect_int(row["schema_version"], "artifact.schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ContractError(
            f"artifact.schema_version is {schema_version}, expected {SCHEMA_VERSION}"
        )
    status = _expect_str(row["status"], "artifact.status")
    if status not in PLAN_ARTIFACT_STATUSES:
        raise ContractError(f"artifact.status has unknown status {status!r}")
    source = _parse_source(row["source"], "artifact.source")
    errors = tuple(
        _parse_issue(item, f"artifact.errors[{index}]")
        for index, item in enumerate(_expect_list(row["errors"], "artifact.errors"))
    )
    plan = _plan_from_dict(row["plan"], "artifact.plan") if row["plan"] is not None else None
    if status == "ready" and (plan is None or errors):
        raise ContractError("artifact.status=ready requires a plan and no errors")
    if status == "failed" and not errors:
        raise ContractError("artifact.status=failed requires at least one error")
    payload = {
        "schema_version": schema_version,
        "status": status,
        "policy": _expect_str(row["policy"], "artifact.policy"),
        "source": source.to_dict(),
        "errors": [issue.to_dict() for issue in errors],
        "plan": plan.to_dict() if plan is not None else None,
    }
    artifact_id = _expect_str(row["artifact_id"], "artifact.artifact_id")
    expected = digest(payload)
    if artifact_id != expected:
        raise ContractError(
            f"artifact.artifact_id does not match contents; expected {expected}"
        )
    return PlanArtifact(
        schema_version=schema_version,
        artifact_id=artifact_id,
        status=status,
        policy=payload["policy"],
        source=source,
        errors=errors,
        plan=plan,
    )


def _parse_pytest_evidence(value: Any, path: str) -> PytestEvidence:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "executed_files", "tests_total", "tests_failed", "tests_errors",
            "tests_skipped",
        },
    )
    return PytestEvidence(
        executed_files=_expect_int(row["executed_files"], f"{path}.executed_files"),
        tests_total=_expect_int(row["tests_total"], f"{path}.tests_total"),
        tests_failed=_expect_int(row["tests_failed"], f"{path}.tests_failed"),
        tests_errors=_expect_int(row["tests_errors"], f"{path}.tests_errors"),
        tests_skipped=_expect_int(row["tests_skipped"], f"{path}.tests_skipped"),
    )


def _parse_model_e2e_evidence(value: Any, path: str) -> ModelE2EEvidence:
    row = _expect_mapping(value, path)
    _expect_keys(row, path, required={"case_ids"})
    case_ids = tuple(_expect_str_list(row["case_ids"], f"{path}.case_ids"))
    if not case_ids:
        raise ContractError(f"{path}.case_ids must not be empty")
    return ModelE2EEvidence(case_ids=case_ids)


def _parse_step_result(value: Any, path: str) -> StepResult:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={"id", "status", "exit_code", "spec_digest"},
        optional={"case_ids"},
    )
    status = _expect_str(row["status"], f"{path}.status")
    if status not in JOB_STATUSES:
        raise ContractError(f"{path}.status has unknown status {status!r}")
    case_ids = tuple(
        _expect_str_list(row.get("case_ids", []), f"{path}.case_ids")
    )
    return StepResult(
        id=_expect_str(row["id"], f"{path}.id"),
        status=status,
        exit_code=_expect_int(row["exit_code"], f"{path}.exit_code", minimum=-255),
        spec_digest=_expect_str(row["spec_digest"], f"{path}.spec_digest"),
        case_ids=case_ids,
    )


def _job_result_from_dict(value: Any, path: str = "job_result") -> JobResult:
    row = _expect_mapping(value, path)
    _expect_keys(
        row,
        path,
        required={
            "schema_version", "plan_id", "job_id", "kind", "status",
            "started_at", "finished_at", "duration_seconds", "runtime", "items",
            "steps", "skip", "failure", "artifacts",
        },
    )
    schema_version = _expect_int(row["schema_version"], f"{path}.schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ContractError(
            f"{path}.schema_version is {schema_version}, expected {SCHEMA_VERSION}"
        )
    kind = _expect_str(row["kind"], f"{path}.kind")
    if kind not in JOB_KINDS:
        raise ContractError(f"{path}.kind has unknown kind {kind!r}")
    status = _expect_str(row["status"], f"{path}.status")
    if status not in JOB_STATUSES:
        raise ContractError(f"{path}.status has unknown status {status!r}")
    skip_reason: str | None = None
    skip_message: str | None = None
    if row["skip"] is not None:
        skip = _expect_mapping(row["skip"], f"{path}.skip")
        _expect_keys(skip, f"{path}.skip", required={"reason", "message"})
        skip_reason = _expect_str(skip["reason"], f"{path}.skip.reason")
        skip_message = _expect_str(
            skip["message"], f"{path}.skip.message", allow_empty=True
        )
    if status == "skipped" and skip_reason is None:
        raise ContractError(f"{path}.status=skipped requires skip.reason")
    if status != "skipped" and skip_reason is not None:
        raise ContractError(f"{path}.skip is only valid for skipped status")
    failure = row["failure"]
    if failure is not None:
        failure = _expect_mapping(failure, f"{path}.failure")
        if not all(isinstance(key, str) for key in failure):
            raise ContractError(f"{path}.failure keys must be strings")
    artifacts = _expect_mapping(row["artifacts"], f"{path}.artifacts")
    artifacts = {
        _expect_str(key, f"{path}.artifacts key"): _expect_str(
            item, f"{path}.artifacts.{key}"
        )
        for key, item in artifacts.items()
    }
    if kind == "pytest":
        items: PytestEvidence | ModelE2EEvidence = _parse_pytest_evidence(
            row["items"], f"{path}.items"
        )
    else:
        items = _parse_model_e2e_evidence(row["items"], f"{path}.items")
    return JobResult(
        schema_version=schema_version,
        plan_id=_expect_str(row["plan_id"], f"{path}.plan_id"),
        job_id=_expect_str(row["job_id"], f"{path}.job_id"),
        kind=kind,
        status=status,
        started_at=_expect_str(row["started_at"], f"{path}.started_at"),
        finished_at=_expect_str(row["finished_at"], f"{path}.finished_at"),
        duration_seconds=_expect_int(row["duration_seconds"], f"{path}.duration_seconds"),
        runtime=_parse_runtime(row["runtime"], f"{path}.runtime"),
        items=items,
        steps=tuple(
            _parse_step_result(item, f"{path}.steps[{index}]")
            for index, item in enumerate(_expect_list(row["steps"], f"{path}.steps"))
        ),
        skip_reason=skip_reason,
        skip_message=skip_message,
        failure=failure,
        artifacts=artifacts,
    )


def load_job_result(path: Path) -> JobResult:
    """Load one JobResult JSON file."""
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read job result {path}: {exc}") from exc
    return _job_result_from_dict(value)


def load_result(paths: list[Path]) -> Result:
    """Load JobResult files and aggregate them into a Result."""
    if not paths:
        raise ContractError("at least one job result is required")
    jobs = [load_job_result(path) for path in paths]
    plan_ids = {job.plan_id for job in jobs}
    if len(plan_ids) != 1:
        raise ContractError("job results have different plan_id values")
    job_ids = [job.job_id for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise ContractError("job results contain duplicate job_id values")
    return Result(
        schema_version=SCHEMA_VERSION,
        plan_id=next(iter(plan_ids)),
        jobs=tuple(jobs),
    )

