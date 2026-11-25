"""Base contracts shared by the CI production and consumption domains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

SCHEMA_VERSION = 21
DEVICES = ("cpu", "npu", "gpu")
SUITES = ("core", "model", "adapter")


class ContractError(ValueError):
    """Raised when a CI contract is malformed or internally inconsistent."""


class Contract:
    """Dataclass mixin that keeps the public ``to_dict`` API."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Diagnostic(Contract):
    code: str       # Machine-readable error code
    location: str   # File path or configuration field path
    message: str    # Human-readable explanation


@dataclass(frozen=True)
class GitSource(Contract):
    repository: str  # Repository name or remote URL
    git_ref: str     # Git branch, tag, or pull-request ref
    commit_sha: str  # Exact Git commit SHA


class DiagnosticError(ValueError):
    """Raised when a CI step reports one or more structured diagnostics."""

    def __init__(self, issues: Iterable[Diagnostic]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))
