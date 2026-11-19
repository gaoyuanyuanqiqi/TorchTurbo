import pytest


pytestmark = pytest.mark.cpu


def test_pr_ci_intentional_failure() -> None:
    assert False, "intentional PR CI failure probe"
