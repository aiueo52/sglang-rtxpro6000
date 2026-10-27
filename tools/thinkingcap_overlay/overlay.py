#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Build a ThinkingCap dense-trunk overlay with the original Qwen MTP head.

CPU-only, Python standard library; no model framework, downloads or GPU needed.
Supply local checkpoints (including their indexes/configs/tokenizers):
  TC:   bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4
  ORIG: RadixArk/Qwen3.8-Flash-Next-NVFP4
Use the stock ORIG checkpoint as PROD for the documented recipe. Obtain the
weights separately and comply with each model's license; no weights are bundled.

  python3 tools/thinkingcap_overlay/overlay.py plan --tc "$TC" --orig "$ORIG" --prod "$ORIG" --head orig
  python3 tools/thinkingcap_overlay/overlay.py build --tc "$TC" --orig "$ORIG" --prod "$ORIG" --head orig --out "$OUT"
  python3 tools/thinkingcap_overlay/overlay.py verify --tc "$TC" --orig "$ORIG" --prod "$ORIG" --head orig --dir "$OUT/model-overlay-orig"

For hyper pro, set TARGET_MODEL to the verified model-overlay-orig directory.
Keep PROD in place: unchanged files are symlinked to its resolved source files.
All sources are read-only. Only same-name, same-dtype/shape changed dense tensors
are copied from TC; routed trunk experts and PLE stay PROD's, without requantizing.
Every mtp.* tensor comes from ORIG. Inspect plan's absent_in_tc before building.
Writes stream in 64 MiB chunks; failed writes remove their own partial file.
Build requires a new destination directory (no overwrite or automatic resume).
--pause is disabled and --min-free-gb is zero unless explicitly supplied.
--files creates an incomplete trial, not a checkpoint suitable for serving.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __package__:
    from . import convert as cv
else:
    import convert as cv  # noqa: E402

NEVER = (".mlp.experts.", ".ngram_embedding.shard_")


def same_bytes(pa, a0, a1, pb, b0, b1):
    if a1 - a0 != b1 - b0:
        return False
    for x, y in zip(cv.iter_range(pa, a0, a1), cv.iter_range(pb, b0, b1)):
        if x != y:
            return False
    return True


def sha(path, a, b):
    h = hashlib.sha256()
    for c in cv.iter_range(path, a, b):
        h.update(c)
    return h.hexdigest()


def plan(tc, prod, orig, head, pause=None):
    """{'replace': {name: (src_kind, src_file)}, 'pending': [...], 'files': {prod_file: [replaced names]}, ...}."""
    tc, prod, orig = Path(tc), Path(prod), Path(orig)
    hc = cv.HeaderCache()
    pr_wm, tc_wm = cv.load_index(prod), cv.load_index(tc)
    if head != "orig":
        raise ValueError("only the original Qwen MTP head is supported")
    or_wm = cv.load_index(orig)
    cv.check_index(prod, pr_wm, hc)
    cv.check_index(orig, or_wm, hc)
    ready = cv.check_index(tc, tc_wm, hc, allow_missing=True)
    if {n for n in pr_wm if n.startswith("mtp.")} != {n for n in or_wm if n.startswith("mtp.")}:
        raise ValueError("original and production MTP tensor names differ")
    replace, pending, mismatch, absent_in_tc, equal = {}, [], [], [], 0
    for n, pf in sorted(pr_wm.items()):
        if n.startswith("mtp."):
            pd, ps, _, _ = hc.tensor(prod / pf, n)
            od, os_, _, _ = hc.tensor(orig / or_wm[n], n)
            if (pd, ps) != (od, os_):
                raise SystemExit(f"orig {n} is {od} {os_}, production {pd} {ps}")
            replace[n] = ("orig", or_wm[n])
            continue
        if any(s in n for s in NEVER):
            continue
        if n not in tc_wm:
            absent_in_tc.append(n)
            continue
        if tc_wm[n] not in ready:
            pending.append(n)
            continue
        if pause and cv.check_pause(pause):
            raise SystemExit(3)
        pd, ps, pa, pb = hc.tensor(prod / pf, n)
        td, ts, ta, tb = hc.tensor(tc / tc_wm[n], n)
        if (pd, ps) != (td, ts):
            mismatch.append([n, [pd, ps], [td, ts]])
            continue
        if same_bytes(tc / tc_wm[n], ta, tb, prod / pf, pa, pb):
            equal += 1
        else:
            replace[n] = ("tc", tc_wm[n])
    # dense tensors ThinkingCap has and production lacks would be silently dropped: report them
    tc_only = sorted(n for n in tc_wm if n not in pr_wm and not n.startswith("mtp.")
                     and not any(s in n for s in NEVER))
    files = {}
    for n in replace:
        files.setdefault(pr_wm[n], []).append(n)
    # a production file that may hold a pending tensor is not decided yet
    undecided = sorted({pr_wm[n] for n in pending})
    return {"replace": replace, "pending": pending, "undecided_files": undecided, "mismatch": mismatch,
            "absent_in_tc": absent_in_tc, "tc_only": tc_only, "equal": equal, "files": {f: sorted(v) for f, v in sorted(files.items())}}


def pattern_counts(names):
    c = {}
    for n in names:
        k = re.sub(r"layers\.\d+\.", "layers.N.", n)
        c[k] = c.get(k, 0) + 1
    return dict(sorted(c.items()))


def cmd_plan(a):
    p = plan(a.tc, a.prod, a.orig, a.head, a.pause)
    prod = Path(a.prod)
    sizes = {f: (prod / f).stat().st_size for f in p["files"]}
    rep = {"head": a.head, "replaced": len(p["replace"]), "equal_dense": p["equal"], "pending": len(p["pending"]),
           "undecided_files": p["undecided_files"], "mismatch": p["mismatch"], "absent_in_tc": p["absent_in_tc"],
           "tc_only": p["tc_only"], "rewrite_files": {f: {"replaced": len(v), "bytes": sizes[f]} for f, v in p["files"].items()},
           "rewrite_gb": sum(sizes.values()) / 1e9, "patterns": pattern_counts(p["replace"])}
    print(json.dumps({k: v for k, v in rep.items() if k != "patterns"}, indent=1))
    print(json.dumps(rep["patterns"], indent=1))
    if a.report:
        with open(a.report, "x", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0 if not p["pending"] and not p["mismatch"] and not p["tc_only"] else 4


def cmd_build(a):
    prod, tc, orig = Path(a.prod), Path(a.tc), Path(a.orig)
    dst = Path(a.out) / f"model-overlay-{a.head}{a.suffix}"
    only = set(a.files.split(",")) if a.files else None
    if os.path.lexists(dst):
        raise SystemExit(f"destination already exists (will not overwrite): {dst}")
    if os.path.lexists(prod / "OVERLAY.json"):
        raise SystemExit("production contains reserved side file OVERLAY.json")
    p = plan(tc, prod, orig, a.head, a.pause)
    if p["mismatch"] or p["tc_only"]:
        raise SystemExit(f"dtype/shape mismatch {p['mismatch'][:3]} or TC-only dense {p['tc_only'][:3]}")
    todo = sorted(p["files"]) if only is None else sorted(only)
    if only is None and p["undecided_files"]:
        raise SystemExit(f"TC download not finished: {len(p['pending'])} tensors pending in {p['undecided_files']}")
    for f in todo:
        if f in p["undecided_files"]:
            raise SystemExit(f"{f} may hold a tensor of an unfinished TC shard")
        if f not in p["files"]:
            raise SystemExit(f"{f} has nothing to replace")
    total = sum((prod / f).stat().st_size for f in todo)
    if a.max_gb is not None and total > a.max_gb * 1e9:
        raise SystemExit(f"planned write {total/1e9:.2f} GB > --max-gb {a.max_gb}")
    floor = a.min_free_gb * 2**30  # GiB; free space available to this user
    dst.parent.mkdir(parents=True, exist_ok=True)
    left = total
    if shutil.disk_usage(dst.parent).free - left < floor:
        raise SystemExit(f"free {shutil.disk_usage(dst.parent).free/2**30:.1f} GiB - planned {left/2**30:.1f} GiB "
                         f"< --min-free-gb {a.min_free_gb} GiB")
    dst.mkdir()
    hc = cv.HeaderCache()
    print(f"overlay {a.head}: rewrite {len(todo)} files {total/1e9:.2f} GB, {len(p['replace'])} tensors replaced",
          flush=True)
    for f in todo:
        if cv.check_pause(a.pause):
            return 3
        out = dst / f
        h, meta, ds = cv.read_header(prod / f)
        names = sorted(h, key=lambda n: h[n]["data_offsets"][0])
        w = cv.Writer(out, [(n, h[n]["dtype"], h[n]["shape"]) for n in names], meta)
        t0 = time.time()
        try:
            for n in names:
                kind, sf = p["replace"].get(n, ("prod", f))
                src = {"tc": tc, "orig": orig, "prod": prod}[kind] / sf
                _, _, sa, sb = hc.tensor(src, n)
                w.begin(n)
                for c in cv.iter_range(src, sa, sb):
                    if not cv.mem_ok():
                        raise SystemExit("insufficient available memory for streaming buffers; stopping")
                    if shutil.disk_usage(dst).free - len(c) < floor:
                        raise SystemExit(f"free space under --min-free-gb {a.min_free_gb} GiB; stopping")
                    w.write(c)
                w.end()
            w.close()
        except BaseException:
            w.abort()
            raise
        print(f"wrote {out} {w.total/1e9:.3f} GB in {time.time()-t0:.0f}s", flush=True)
    # every other production file (safetensors and side files) is a symlink to production's real file
    rewritten = set(p["files"]) if only is None else only
    for q in sorted(prod.iterdir()):
        if q.name in rewritten or q.name == "model.safetensors.index.json" or q.is_dir():
            continue
        if q.name in p["files"]:  # partial trial: a file that should be rewritten is left out entirely
            continue
        link = dst / q.name
        if not link.exists() and not link.is_symlink():
            os.symlink(q.resolve(), link)
    with open(dst / "model.safetensors.index.json", "xb") as fh:
        fh.write((prod / "model.safetensors.index.json").read_bytes())
    partial = sorted(set(p["files"]) - rewritten)
    with open(dst / "OVERLAY.json", "x", encoding="utf-8") as fh:
        json.dump({"head": a.head, "prod": str(prod.resolve()), "tc": str(tc.resolve()), "orig": str(orig.resolve()),
                   "partial_missing_files": partial,
                   "replaced": {n: list(v) for n, v in sorted(p["replace"].items())}}, fh, indent=1)
    print(f"built {dst}{' (PARTIAL: ' + ','.join(partial) + ' left out)' if partial else ''}")
    return 0


def cmd_verify(a):
    d, prod, tc, orig = Path(a.dir), Path(a.prod), Path(a.tc), Path(a.orig)
    info = json.loads((d / "OVERLAY.json").read_text(encoding="utf-8"), object_pairs_hook=cv.unique_object)
    partial = set(info["partial_missing_files"])
    problems, checked = [], {"replaced_sha": 0, "kept_sha": 0, "symlinks": 0, "headers": 0}
    pr_wm = cv.load_index(prod)
    if cv.load_index(d) != pr_wm:
        problems.append("index weight_map differs from production")
    if info["head"] != a.head:
        problems.append(f"built for head {info['head']}, verifying {a.head}")
    # recompute the plan: the replaced set must be what the data says now
    p = plan(tc, prod, orig, a.head, a.pause)
    replaced = p["replace"]  # do not trust source paths in the output manifest
    if {n: list(v) for n, v in replaced.items()} != info["replaced"]:
        problems.append("replaced set differs from a fresh plan")
    if p["pending"] or p["mismatch"] or p["tc_only"]:
        problems.append("fresh plan has pending, mismatched or TC-only tensors")
    if not partial <= set(p["files"]):
        problems.append("partial manifest lists files not requiring replacement")
    hc = cv.HeaderCache()
    files = sorted(set(pr_wm.values()))
    k = 0
    for f in files:
        if f in partial:
            if os.path.lexists(d / f):
                problems.append(f"partial file unexpectedly present: {f}")
            continue
        q = d / f
        if not q.exists():
            problems.append(f"missing {f}")
            continue
        if q.is_symlink():
            if q.resolve() != (prod / f).resolve():
                problems.append(f"{f} links to {q.resolve()}, not production's {(prod / f).resolve()}")
            checked["symlinks"] += 1
            if any(pr_wm[n] == f for n in replaced):
                problems.append(f"{f} holds a replaced tensor but is a symlink")
            continue
        h, _, _ = cv.read_header(q)
        ph, _, _ = cv.read_header(prod / f)
        if {n: (t["dtype"], t["shape"]) for n, t in h.items()} != {n: (t["dtype"], t["shape"]) for n, t in ph.items()}:
            problems.append(f"{f}: header set/dtype/shape differs from production")
            continue
        for n in sorted(h):
            if cv.check_pause(a.pause):
                return 3
            _, _, oa, ob = hc.tensor(q, n)
            if n in replaced:
                kind, sf = replaced[n]
                src = {"tc": tc, "orig": orig}[kind] / sf
                checked["replaced_sha"] += 1
            else:
                k += 1
                if a.sample and k % max(1, a.sample) != 0:
                    continue
                src = prod / f
                checked["kept_sha"] += 1
            _, _, sa, sb = hc.tensor(src, n)
            if sha(q, oa, ob) != sha(src, sa, sb):
                problems.append(f"{f}:{n} bytes differ from {src.name}")
        checked["headers"] += 1
    for q in sorted(prod.iterdir()):
        if q.suffix == ".safetensors" or q.name in ("model.safetensors.index.json",) or q.is_dir():
            continue
        x = d / q.name
        if not x.is_symlink() or x.resolve() != q.resolve():
            problems.append(f"side file {q.name} is not production's")
    rep = {"dir": str(d), "head": a.head, "partial_missing_files": sorted(partial), "checked": checked,
           "replaced": len(replaced), "patterns": pattern_counts(replaced), "pending_now": len(p["pending"]),
           "problems": problems, "ok": not problems}
    print(json.dumps(rep, indent=1))
    if a.report:
        with open(a.report, "x", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0 if not problems else 1


def protect_sources(path, sources):
    """Reject writes overlapping source roots or their resolved linked assets."""
    target = Path(path).resolve()
    for source in sources:
        if target == source or source in target.parents or target in source.parents:
            raise ValueError(f"write path overlaps a source: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "build", "verify"])
    ap.add_argument("--tc", required=True, help="local ThinkingCap checkpoint directory")
    ap.add_argument("--prod", required=True, help="local production-layout checkpoint; normally the stock original")
    ap.add_argument("--orig", required=True, help="local original Qwen checkpoint (MTP source)")
    ap.add_argument("--head", default="orig", choices=["orig"], help="MTP head source (default: orig)")
    ap.add_argument("--out", help="build: parent of the new model-overlay-orig directory")
    ap.add_argument("--dir", help="verify: overlay directory")
    ap.add_argument("--suffix", default="", help="build: optional output directory name suffix")
    ap.add_argument("--files", help="build: comma-separated production files (incomplete trial)")
    ap.add_argument("--max-gb", type=float, default=None, help="build: maximum planned write in decimal GB")
    ap.add_argument("--min-free-gb", type=float, default=0, help="build: free-space floor in GiB (default: 0)")
    ap.add_argument("--sample", type=int, default=None, help="verify: hash every Nth kept tensor (default: all)")
    ap.add_argument("--pause", default=None, help="optional pause-file path; disabled by default")
    ap.add_argument("--report", help="plan/verify: new JSON report path (never overwritten)")
    a = ap.parse_args(argv)
    if a.command == "build" and not a.out:
        ap.error("--out is required for build")
    if a.command == "verify" and not a.dir:
        ap.error("--dir is required for verify")
    if not re.fullmatch(r"[A-Za-z0-9._-]*", a.suffix):
        ap.error("--suffix must contain only letters, digits, dot, underscore or hyphen")
    for flag in ("min_free_gb", "max_gb"):
        value = getattr(a, flag)
        if value is not None and (not math.isfinite(value) or value < 0):
            ap.error(f"--{flag.replace('_', '-')} must be finite and nonnegative")
    if a.sample is not None and a.sample < 1:
        ap.error("--sample must be positive")
    try:
        sources = set()
        for name in ("tc", "prod", "orig"):
            root = Path(getattr(a, name)).resolve(strict=True)
            if not root.is_dir():
                raise ValueError(f"--{name} must be a checkpoint directory")
            setattr(a, name, str(root))
            sources.add(root)
            sources.update(q.resolve() for q in root.iterdir())
        if a.command == "build":
            protect_sources(Path(a.out) / f"model-overlay-{a.head}{a.suffix}", sources)
        if a.report:
            if a.command == "build":
                ap.error("--report is supported only for plan and verify")
            protect_sources(a.report, sources)
        if cv.check_pause(a.pause):
            return 3
        return {"plan": cmd_plan, "build": cmd_build, "verify": cmd_verify}[a.command](a)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
