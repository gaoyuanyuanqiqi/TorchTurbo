# TorchTurbo CI

Current design and test scope:
[`scripts/ci/CI-V3.md`](../../scripts/ci/CI-V3.md).

Local development, commands, and contracts:
[`scripts/ci/CI-V3_DEVELOPMENT.md`](../../scripts/ci/CI-V3_DEVELOPMENT.md).

The execution path is:

```text
lint -> Planner -> PlannerOutput -> Executor -> JobResult -> Gate
```

- `scripts/ci/contracts/base.py` defines shared contract primitives.
- `scripts/ci/contracts/planner_contract.py` defines planner output and execution plans.
- `scripts/ci/contracts/executor_contract.py` defines job results.
- `scripts/ci/execution/producer/planner.py` builds execution plans and planner output.
- `scripts/ci/config/ci.yml` is the only CI configuration. It declares targets,
  images, runners, venvs, test areas, and pull-request/nightly scopes.
- `scripts/ci/execution/producer/facts.py` derives pytest facts from explicit
  markers and integration cases from the suite runner manifest.
- Pull requests run all core pytest plus the two lightweight CPU/NPU integration
  cases; nightly runs all pytest suites and all integration cases.
- `.github/workflows/ascend-pr.yml` is the main PR and reusable workflow entry.
- `.github/workflows/ascend-manual-test.yml` reduces the normal Plan to one exact job.
- `.github/workflows/ascend-manual-image.yml` is the manual image build/reuse entry.
- `.github/workflows/ascend-nightly.yml` reuses the main workflow with the
  nightly scope.
- `.github/workflows/ascend-image-pr.yml` maps changed Dockerfiles to logical images and
  runs one independent `build -> tests` pipeline per affected image.
- `.github/workflows/ascend-image-promote.yml` reuses the validated candidate per affected
  image and promotes its alias.
- `.github/workflows/ascend-image-pipeline.yml` is the reusable single-image build and
  validation unit.
