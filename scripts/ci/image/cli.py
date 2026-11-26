"""Command-line adapter for the image domain."""

from __future__ import annotations

import argparse
import json
import sys

from ..tools.cli import (
    append_github_output,
    common_parser,
    read_lines,
    repository_path,
    write_json_output,
)
from .build_image import BuildError, get_or_build_image, promote_image, sync_alias
from .image_config import ImageError
from .select_images import ImageSelectionError, select_images
from .registry import RegistryError


def _read_paths(path: str) -> list[str]:
    try:
        return read_lines(path)
    except OSError as exc:
        raise ImageSelectionError(f"cannot read changed-file list: {exc}") from exc


def cmd_plan(args: argparse.Namespace) -> int:
    changed = _read_paths(args.changed_files) if args.changed_files else None
    specs = select_images(
        repository_path(args.root), image_ids=args.images, changed_files=changed
    )
    matrix = {"include": [
        {"image_id": spec.image_id, "target": spec.target} for spec in specs]}
    write_json_output(args.output, matrix)
    append_github_output(
        {
            "image_jobs": json.dumps(matrix, ensure_ascii=False, separators=(",", ":")),
            "image_count": len(specs),
            "images": ",".join(spec.image_id for spec in specs),
        },
        args.github_output,
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    result = get_or_build_image(repository_path(args.root), args.image,
                                use_cache=not args.no_cache,
                                builder=args.builder, jobs=args.max_jobs)
    image = result.image
    write_json_output(args.output, result.to_dict())
    append_github_output(
        {
            "ref": image.ref,
            "digest": image.digest,
            "built": str(result.built).lower(),
            "cache_hit": str(not result.built).lower(),
            "elapsed_seconds": result.elapsed_seconds,
            "image_id": image.image_id,
            "content_hash": result.content_hash,
        },
        args.github_output,
    )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    result = promote_image(repository_path(args.root), args.image)
    write_json_output(args.output, result.to_dict())
    append_github_output(
        {
            "alias": result.alias,
            "ref": result.image.ref,
            "digest": result.image.digest,
            "promoted": "true",
        },
        args.github_output,
    )
    return 0


def cmd_sync_aliases(args: argparse.Namespace) -> int:
    root = repository_path(args.root)
    try:
        specs = select_images(root, image_ids=args.images)
    except ImageError as exc:
        print(f"::warning::cannot select images: {exc}", file=sys.stderr)
        return 0 if args.best_effort else 2
    failed = 0
    for spec in specs:
        try:
            result = sync_alias(root, spec.image_id)
        except (ImageError, RegistryError, BuildError) as exc:
            failed += 1
            print(f"::warning::{spec.image_id}: {exc}", file=sys.stderr)
            continue
        print(f"{spec.image_id}: alias {result.alias}={result.image.digest}")
    if failed and not args.best_effort:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = common_parser()

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", parents=[common])
    plan.add_argument("--changed-files", default="")
    plan.add_argument("--images", default="")
    plan.set_defaults(handler=cmd_plan)

    build = commands.add_parser("build", parents=[common])
    build.add_argument("--image", required=True)
    build.add_argument("--max-jobs", default="auto")
    build.add_argument("--builder", default="")
    build.add_argument("--no-cache", action="store_true")
    build.set_defaults(handler=cmd_build)

    promote = commands.add_parser("promote", parents=[common])
    promote.add_argument("--image", required=True)
    promote.set_defaults(handler=cmd_promote)

    sync = commands.add_parser("sync-aliases", parents=[common])
    sync.add_argument("--images", default="all")
    sync.add_argument("--best-effort", action="store_true")
    sync.set_defaults(handler=cmd_sync_aliases)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ImageError, RegistryError, BuildError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
