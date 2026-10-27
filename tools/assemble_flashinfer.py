#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Rebuild the two pinned FlashInfer trees without importing FlashInfer."""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile


VERSION = "0.6.17"
VARIANTS = ("vt", "vtoea")


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(c in value for c in "\x00\r\n")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or PurePosixPath(value).is_absolute()):
        raise ValueError("unsafe relative path: {!r}".format(value))
    return value


def sha256_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256 value: {!r}".format(value))
    return value


def installed_distribution(site_packages):
    if site_packages is None:
        distribution = importlib.metadata.distribution("flashinfer-python")
    else:
        directory = Path(site_packages).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("--site-packages must name a directory")
        matches = [
            item for item in importlib.metadata.distributions(path=[str(directory)])
            if re.sub(r"[-_.]+", "-", item.metadata.get("Name", "")).lower()
            == "flashinfer-python"
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one flashinfer-python distribution in {}".format(directory))
        distribution = matches[0]
    if distribution.version != VERSION:
        raise ValueError("flashinfer-python version must be {}, found {}".format(VERSION, distribution.version))
    return distribution, Path(distribution.locate_file("")).resolve(strict=True)


def read_record(distribution):
    text = distribution.read_text("RECORD")
    if text is None:
        raise ValueError("flashinfer-python has no RECORD")
    files = {}
    for row in csv.reader(io.StringIO(text), strict=True):
        if not row or not row[0].startswith("flashinfer/"):
            continue
        path = relative_path(row[0][len("flashinfer/"):])
        if "__pycache__" in PurePosixPath(path).parts:
            continue
        if len(row) != 3 or path in files:
            raise ValueError("invalid or duplicate RECORD entry: {}".format(path))
        if not row[1].startswith("sha256=") or not row[2].isdigit():
            raise ValueError("RECORD requires SHA-256 and size for {}".format(path))
        encoded = row[1][len("sha256="):]
        try:
            digest = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        except ValueError as exc:
            raise ValueError("invalid RECORD SHA-256 for {}".format(path)) from exc
        if len(digest) != 32:
            raise ValueError("invalid RECORD SHA-256 for {}".format(path))
        files[path] = (digest.hex(), int(row[2]))
    if not files:
        raise ValueError("RECORD contains no FlashInfer package files")
    return files


def read_manifest(path, record):
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise ValueError("manifest version must be {}".format(VERSION))
    trees = manifest.get("trees")
    if not isinstance(trees, dict) or set(trees) != set(VARIANTS):
        raise ValueError("manifest must describe vt and vtoea trees")
    for variant in VARIANTS:
        tree = trees[variant]
        if not isinstance(tree, dict):
            raise ValueError("invalid manifest tree: {}".format(variant))
        sha256_value(tree.get("tree_sha256"))
        overlays = tree.get("files")
        if not isinstance(overlays, dict):
            raise ValueError("manifest files must be a mapping")
        for path, entry in overlays.items():
            relative_path(path)
            if "__pycache__" in PurePosixPath(path).parts:
                raise ValueError("manifest must not contain __pycache__ files")
            if not isinstance(entry, dict):
                raise ValueError("invalid overlay entry: {}".format(path))
            sha256_value(entry.get("sha256"))
            wheel_hash = record[path][0] if path in record else "added"
            if entry.get("wheel_sha256") != wheel_hash:
                raise ValueError("manifest/RECORD mismatch for {}: {}".format(variant, path))
    return trees


def copy_verified(source, destination, expected=None, size=None):
    if source.resolve(strict=True) != source or not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("source must be a regular file without symlinks: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    length = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        mode = stat.S_IMODE(os.fstat(reader.fileno()).st_mode)
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
            length += len(block)
            writer.write(block)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("SHA-256 mismatch: {}".format(source))
    if size is not None and length != size:
        raise ValueError("RECORD size mismatch: {}".format(source))
    destination.chmod(mode)
    return actual, mode


def directory_modes(root):
    result = {"": stat.S_IMODE(root.stat().st_mode)}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("symlink in tree: {}".format(path))
            result[path.relative_to(root).as_posix()] = stat.S_IMODE(path.stat().st_mode)
    return result


def inventory(root):
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("non-regular file in target: {}".format(path))
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.relative_to(root).as_posix()] = (digest.hexdigest(), stat.S_IMODE(mode))
    return files, directory_modes(root)


def tree_sha256(files):
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update("{}  {}\n".format(files[path][0], path).encode("utf-8"))
    return digest.hexdigest()


def build_tree(stage, overlay, source_root, record, tree):
    stage.mkdir()
    files = {"LICENSE": copy_verified(overlay / "LICENSE", stage / "LICENSE")}
    for path, (digest, size) in sorted(record.items()):
        if path not in tree["files"]:
            relative = "flashinfer/" + path
            files[relative] = copy_verified(source_root / relative, stage / relative, digest, size)
    for path, entry in sorted(tree["files"].items()):
        relative = "flashinfer/" + path
        files[relative] = copy_verified(overlay / relative, stage / relative, entry["sha256"])
    if tree_sha256(files) != tree["tree_sha256"]:
        raise ValueError("assembled tree SHA-256 mismatch for {}".format(overlay.name))
    # Directory modes matter too, but do not enter the content-only tree hash.
    directories = [stage] + [path for path in stage.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        relative = directory.relative_to(stage)
        original = overlay / relative
        if not original.is_dir():
            original = source_root / relative
        if original.resolve(strict=True) != original or not original.is_dir():
            raise ValueError("invalid source directory: {}".format(original))
        directory.chmod(stat.S_IMODE(original.stat().st_mode))
    return files, directory_modes(stage)


def target_state(target, expected=None):
    if target.is_symlink():
        raise ValueError("refusing symlink target: {}".format(target))
    if not target.exists():
        return "missing"
    if not target.is_dir():
        raise ValueError("target is not a directory: {}".format(target))
    if not any(target.iterdir()):
        return "empty"
    if expected is not None and inventory(target) == expected:
        return "identical"
    raise ValueError("refusing non-empty target that is not already identical: {}".format(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", metavar="DIR", help="read flashinfer-python from this site-packages directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    distribution, source_root = installed_distribution(args.site_packages)
    record = read_record(distribution)
    overlay_root = root / "flashinfer-overlays"
    trees = read_manifest(overlay_root / "MANIFEST.json", record)
    for variant in VARIANTS:
        target = root / ("flashinfer-" + variant)
        if source_root == target or source_root in target.parents or target in source_root.parents:
            raise ValueError("assembly target overlaps site-packages: {}".format(target))
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError("unsafe target: {}".format(target))
    with tempfile.TemporaryDirectory(prefix=".assemble-flashinfer-", dir=str(root)) as temporary:
        stages = {}
        for variant in VARIANTS:
            stage = Path(temporary) / variant
            expected = build_tree(stage, overlay_root / variant, source_root, record, trees[variant])
            target = root / ("flashinfer-" + variant)
            state = target_state(target, expected)
            stages[variant] = (stage, target, expected, state)
        # All inputs and both destinations are checked before publishing either tree.
        for variant in VARIANTS:
            stage, target, expected, state = stages[variant]
            if state != "identical":
                # Recheck before removing even an empty pre-existing directory.
                state = target_state(target, expected)
                if state == "empty":
                    target.rmdir()
                if state != "identical":
                    stage.rename(target)
            print("[assemble_flashinfer] {}: {}".format(target.name, "already identical" if state == "identical" else "assembled"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error, importlib.metadata.PackageNotFoundError) as exc:
        print("[assemble_flashinfer] error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
