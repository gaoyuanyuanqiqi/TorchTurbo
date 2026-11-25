from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.contracts.base import GitSource  # noqa: E402
from scripts.ci.execution.producer.config import (  # noqa: E402
    CiConfig,
    Scope,
    bind_image_refs,
    load_config,
)
from scripts.ci.execution.producer.facts import (  # noqa: E402
    TestFacts,
    load_facts,
)
from scripts.ci.execution.producer.planner import build_plan, source_for  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config() -> CiConfig:
    unbound = load_config(ROOT)
    bindings = {
        name: f"test.invalid/{name}@sha256:{hashlib.sha256(name.encode()).hexdigest()}"
        for name, image in unbound.images.items()
        if image.kind == "container"
    }
    return bind_image_refs(unbound, bindings)


@pytest.fixture(scope="session")
def facts() -> TestFacts:
    return load_facts(ROOT)


@pytest.fixture(scope="session")
def source(repo_root: Path) -> GitSource:
    return source_for(
        repo_root,
        repository="example/torchturbo",
        ref="refs/pull/123/head",
        sha="0123456789abcdef0123456789abcdef01234567",
    )


@pytest.fixture(scope="session")
def pr_scope(config: CiConfig) -> Scope:
    return config.scope("pull_request")


@pytest.fixture(scope="session")
def nightly_scope(config: CiConfig) -> Scope:
    return config.scope("nightly")


@pytest.fixture(scope="session")
def pr_plan(config: CiConfig, facts: TestFacts, source: GitSource):
    return build_plan(config, facts, "pull_request", source)


@pytest.fixture(scope="session")
def nightly_plan(config: CiConfig, facts: TestFacts, source: GitSource):
    return build_plan(config, facts, "nightly", source)
