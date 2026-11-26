#!/usr/bin/env python3
"""Command line entry point for the CI planner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from ...contracts.base import Diagnostic, GitSource
from ...contracts.planner_contract import CHANGE_MODES
from ...image.build_image import resolve_image
from ...image.image_config import ImageError
from ...image.registry import RegistryError
from ...tools.cli import append_github_output, common_parser, repository_path, write_json_output
from ...tools.json_io import canonical_json
from .config import CiConfigError, bind_image_refs, load_config
from .facts import FactsError, load_facts
from .matrix import github_outputs
from .planner import (
    PlanError,
    PlanOptions,
    build_planner_output,
    planner_output_for,
    requested_images,
    source_for,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", parents=[common_parser()])
    plan.add_argument("--ci-type", default="pull_request")
    plan.add_argument("--change-mode", choices=CHANGE_MODES, default="code")
    plan.add_argument("--image-ref", action="append", default=[], metavar="NAME=REF")
    plan.add_argument("--image-filter", default="")
    plan.add_argument("--job-filter", default="")
    plan.add_argument("--target-filter", default="")

    failed = commands.add_parser("failed-output", parents=[common_parser()])
    failed.add_argument("--code", required=True)
    failed.add_argument("--location", default="ci")
    failed.add_argument("--message", required=True)

    return parser


def _image_refs(values: list[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for value in values:
        name, separator, ref = value.partition("=")
        if not separator or not name or not ref:
            raise ValueError(f"bad --image-ref {value!r}; expected NAME=REF")
        refs[name] = ref
    return refs


def _source(root: Path) -> GitSource:
    return source_for(
        root,
        repository=os.environ.get("GITHUB_REPOSITORY", ""),
        ref=os.environ.get("CI_SOURCE_REF") or os.environ.get("GITHUB_REF", ""),
        sha=os.environ.get("CI_SOURCE_SHA", ""),
    )


def _print_diagnostics(issues: Iterable[Diagnostic]) -> None:
    for issue in issues:
        print(f"{issue.location}: {issue.message}", file=sys.stderr)


def _fail_plan(args: argparse.Namespace, issues: Iterable[Diagnostic]) -> int:
    output = build_planner_output(errors=issues)
    write_json_output(args.output, output.to_dict(), emit_stdout=False)
    _print_diagnostics(output.errors)
    return 1


def _failed_output(args: argparse.Namespace) -> int:
    output = build_planner_output(errors=(Diagnostic(
        code=args.code,
        location=args.location,
        message=args.message,
    ),))
    write_json_output(args.output, output.to_dict())
    _print_diagnostics(output.errors)
    return 0


def _plan(args: argparse.Namespace, source: GitSource) -> int:
    if args.job_filter and args.target_filter:
        raise PlanError([Diagnostic(
            code="invalid-selection",
            location="selection",
            message="--job-filter cannot be combined with --target-filter",
        )])
    root = repository_path(args.root)
    image_refs = _image_refs(args.image_ref)
    config = load_config(root)
    options = PlanOptions(
        ci_type=args.ci_type,
        git_source=source,
        change_mode=args.change_mode,
        image_filter=args.image_filter,
        target_filter=args.target_filter,
        job_filter=args.job_filter,
    )
    if args.change_mode == "code":
        for image_id in requested_images(config, options):
            if image_id in image_refs:
                continue
            try:
                image_refs[image_id] = resolve_image(root, image_id).ref
            except (ImageError, RegistryError) as exc:
                raise PlanError([Diagnostic(
                    code="image-resolution",
                    location=f"resources.images.{image_id}.ref",
                    message=str(exc),
                )]) from exc
    config = bind_image_refs(config, image_refs)
    facts = load_facts(root)
    output = planner_output_for(
        config,
        facts,
        options,
    )
    write_json_output(args.output, output.to_dict(), emit_stdout=False)
    if output.status != "ready" or output.plan is None:
        _print_diagnostics(output.errors)
        return 1
    append_github_output(github_outputs(output.plan, config), args.github_output)
    if not args.output and not args.github_output:
        print(canonical_json(output.plan.to_dict()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "failed-output":
        return _failed_output(args)
    source = _source(repository_path(args.root))
    try:
        return _plan(args, source)
    except (CiConfigError, FactsError, PlanError) as exc:
        return _fail_plan(args, exc.issues)
    except ValueError as exc:
        return _fail_plan(args, [Diagnostic(
            code="invalid-cli",
            location="plan",
            message=str(exc),
        )])


if __name__ == "__main__":
    sys.exit(main())
