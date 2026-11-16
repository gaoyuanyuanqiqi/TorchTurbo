# TorchTurbo CI

The current CI architecture and local validation commands live in
[`scripts/ci/README.md`](../../scripts/ci/README.md).

The execution path is:

```text
lint -> Planner -> PlanArtifact -> Executor -> JobResult -> Gate
```

- `scripts/ci/catalog.yml` declares images, venvs, lanes, pytest suites, and
  model E2E cases.
- `scripts/ci/policies.yml` selects the pull-request or nightly scope.
- `.github/workflows/ascend-pr.yml` is the main PR/workflow-call entry.
- `.github/workflows/ascend-manual.yml` runs one exact Plan job through the
  same reusable workflow.
- `.github/workflows/ascend-nightly.yml` reuses the main workflow with the
  nightly policy.
