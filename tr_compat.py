# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Relocatable, authenticated prebuilt loading for the approved TR policy."""
import hashlib
import os
from pathlib import Path
import tempfile

# The approved policy is intentionally not shipped; the launcher rejects tactic rows.
# Retain the digest pin for this dormant compatibility loader.
_POLICY_SHA256 = "5f2eb929cbf6b33dc05f98a093a736f9d14555d8464ddb17b627718812d05c1c"
_POLICY_FILE = Path(__file__).resolve().parent / "policies/tactic_rows/normal.json"


def _paths(edition=None):
    if os.environ.get("SGLANG_TACTIC_ROWS", "0") != "1":
        raise ValueError("SGLANG_TACTIC_ROWS must be 1")
    if (os.environ.get("FLASHINFER_DISABLE_JIT") != "1"
            or os.environ.get("FLASHINFER_P2_NO_NINJA") != "1"):
        raise ValueError("TR requires FLASHINFER_DISABLE_JIT=1 and FLASHINFER_P2_NO_NINJA=1")
    selected = os.environ.get("FLASH_NEXT_EDITION")
    if selected != "normal" or (edition is not None and edition != selected):
        raise ValueError("TR policy is approved only for the normal edition")
    paths = []
    for name in ("SGLANG_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"):
        value = os.environ.get(name, "")
        if not value or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(name + " must be an absolute path without traversal")
        paths.append(Path(value).resolve())
    cache, workspace = paths
    root = cache.parent
    if cache == root or not workspace.is_relative_to(root) or workspace == root:
        raise ValueError("TR cache and workspace must share a cache root")
    if os.environ.get("SGLANG_TACTIC_ROWS_APPROVED_SHA256") != _POLICY_SHA256:
        raise ValueError("TR requires the shipped approved policy SHA256")
    return cache, workspace, root


def _library(value, workspace):
    path = Path(value).resolve()
    if not path.is_relative_to(workspace) or path.suffix != ".so":
        raise ValueError("TR libraries must be .so files in the configured workspace")
    return path


def install_policy_paths(api):
    if getattr(api.load_policy, "_flash_next_compat", False):
        return
    try:
        cache, workspace, root = _paths(api.EDITION)
    except ValueError as error:
        raise api.PolicyError(str(error)) from error
    # Replace only the cache path boundary, not any authentication.
    api._PRIVATE_ROOT = root
    original_load = api.load_policy
    original_verify = api.verify_runtime

    def load_policy(*, capturing=False):
        # Policy/configuration checks and installation happen only on first load
        # in each process. The original loader retains its lock and capture guard.
        with api._LOCK:
            if api._POLICY is None:
                if capturing:
                    raise api.PolicyError("load approved policy before CUDA graph capture")
                try:
                    if _paths(api.EDITION) != (cache, workspace, root):
                        raise ValueError("TR cache selection changed; restart required")
                    raw = _POLICY_FILE.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != _POLICY_SHA256:
                        raise ValueError("shipped TR policy SHA256 mismatch")
                    directory = cache / "tactic_rows"
                    if not directory.resolve().is_relative_to(cache):
                        raise ValueError("TR policy directory escapes configured cache")
                    directory.mkdir(parents=True, exist_ok=True)
                    target = directory / "policy.json"
                    if not target.resolve().is_relative_to(cache):
                        raise ValueError("TR policy file escapes configured cache")
                    if target.exists():
                        if target.read_bytes() != raw:
                            raise ValueError("cached TR policy differs; refusing to overwrite")
                    else:
                        temporary = None
                        try:
                            fd, temporary = tempfile.mkstemp(prefix=".policy-", dir=directory)
                            with os.fdopen(fd, "wb") as stream:
                                stream.write(raw)
                            try:
                                os.link(temporary, target)
                            except FileExistsError:
                                if target.read_bytes() != raw:
                                    raise ValueError("cached TR policy differs; refusing to overwrite")
                        finally:
                            if temporary is not None and os.path.exists(temporary):
                                os.unlink(temporary)
                except (OSError, ValueError) as error:
                    raise api.PolicyError(str(error)) from error
            return original_load(capturing=capturing)

    def verify_runtime(library_path, **kwargs):
        try:
            library = _library(library_path, workspace)
        except ValueError as error:
            raise api.PolicyError(str(error)) from error
        # Preserve exact GPU, source, toolchain and library hashes from approval.
        return original_verify(library, **kwargs)

    load_policy._flash_next_compat = True
    api.load_policy = load_policy
    api.verify_runtime = verify_runtime


def install():
    if os.environ.get("SGLANG_TACTIC_ROWS", "0") == "0":
        return
    cache, workspace, root = _paths()
    from flashinfer.jit.core import JitSpecNvcc
    from flashinfer.fused_moe import tactic_rows_policy
    install_policy_paths(tactic_rows_policy)
    if getattr(JitSpecNvcc.try_load, "_flash_next_prebuilt_only", False):
        return

    def try_load(self):
        path = _library(self.get_library_path(), workspace)
        if not path.is_file():
            return None
        try:
            return self.load(path)
        except Exception:
            return None  # DISABLE_JIT fails closed rather than rebuilding.

    # Do not use the AOT fast path, which can select a different artifact.
    try_load._flash_next_prebuilt_only = True
    JitSpecNvcc.try_load = try_load
