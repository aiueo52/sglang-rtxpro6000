"""CPU-only routing/capture checks. Never initialize CUDA or start a server.

Load the production modules under lightweight package shells, bypassing the
SGLang public API and its optional serving dependencies. Only rank logging is
stubbed. Torch-dependent confidence checks explicitly skip if torch cannot load.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "python/sglang/srt/speculative"
GRAPH_BS = list(range(1, 9))
EXPECTED = {
    1: (1, [3, 7, 15]),
    2: (2, [3, 7]),
    3: (2, [3, 7]),
    4: (4, [3, 7]),
    5: (5, [3]),
    6: (5, [3]),
    7: (5, [3]),
    8: (5, [3]),
}


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adaptive(monkeypatch):
    # Keep policy selection deterministic even under a serving shell environment.
    monkeypatch.setenv("SGLANG_RND_DL_HOSTW", "0")
    monkeypatch.setenv("SGLANG_ADAPTIVE_POLICY", "ema")
    for name, path in (
        ("sglang", ROOT / "python/sglang"),
        ("sglang.srt", ROOT / "python/sglang/srt"),
        ("sglang.srt.speculative", SPEC),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    utils = types.ModuleType("sglang.srt.utils")
    utils.log_info_on_rank0 = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    load_module(monkeypatch, "sglang.srt.speculative.dl_policy", SPEC / "dl_policy.py")
    module = load_module(
        monkeypatch,
        "sglang.srt.speculative.adaptive_spec_params",
        SPEC / "adaptive_spec_params.py",
    )
    # The EMA controller imports the confidence selector too; allow all pure
    # routing checks to run on machines lacking torch.
    selector = types.ModuleType("sglang.srt.speculative.adaptive_confidence")
    selector.confidence_policy_enabled = lambda: False
    selector.ConfidenceStepSlot = type("UnusedConfidenceSlot", (), {})
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    return module


def params(module):
    result = module.AdaptiveSpeculativeParams(
        initial_steps=15, cfg_path=str(ROOT / "adaptive/w16_c8.json")
    )
    result.set_cuda_graph_bs(GRAPH_BS)
    return result


@pytest.mark.parametrize("bs", GRAPH_BS)
def test_route_and_row_budget(adaptive, bs):
    controller = params(adaptive)
    bucket, candidates = EXPECTED[bs]
    assert controller.get_bucket_for_batch(bs) == bucket
    assert controller._route(bs) is controller._slots[bucket]
    assert controller._route(bs).candidate_steps == candidates
    assert controller.get_steps_for_batch(bs) in candidates
    for steps in candidates:
        assert bs * (steps + 1) <= 32
        assert bs * (steps + 1) * 10 <= 320
    assert controller._slots[1].current_steps == 15


def test_capture_sets_include_initial_state(adaptive):
    controller = params(adaptive)
    assert controller.candidate_steps == [3, 7, 15]
    expected = {3: GRAPH_BS, 7: [1, 2, 3, 4], 15: [1]}
    for steps, batches in expected.items():
        captured = adaptive.adaptive_capture_bs_for_step(
            capture_bs=GRAPH_BS,
            step=steps,
            cfg_path=str(ROOT / "adaptive/w16_c8.json"),
        )
        assert captured == batches == controller.cuda_graph_bs_for_step(steps)
        for bs in captured:
            assert bs * (steps + 1) <= 32  # verify and draft extend
            assert bs <= 32  # each draft decode forward (topk=1)
    assert adaptive.adaptive_capture_bs_for_step(
        capture_bs=GRAPH_BS, step=31,
        cfg_path=str(ROOT / "adaptive/w16_c8.json"),
    ) == []


def test_route_pads_before_selecting_slot(adaptive):
    controller = params(adaptive)
    controller.set_cuda_graph_bs([1, 4, 8])
    assert controller.get_bucket_for_batch(2) == 4
    assert controller.get_bucket_for_batch(3) == 4
    assert controller.get_bucket_for_batch(5) == 5
    assert controller.get_bucket_for_batch(7) == 5


def test_c4_settings_preserved(adaptive):
    _, cfg = adaptive._load_adaptive_config(str(ROOT / "adaptive/w16_c8.json"))
    base = json.loads((ROOT / "adaptive/w16_c4.json").read_text())
    assert set(cfg) == {1, 2, 4, 5}
    for bucket in (1, 2, 4):
        assert cfg[bucket] == base[str(bucket)]
    assert cfg[5] == {
        **base["4"], "candidate_steps": [3],
        "step_time_ms": {"3": base["4"]["step_time_ms"]["3"]},
    }


def test_confidence_policy_loads_table(adaptive, monkeypatch):
    try:
        import torch  # noqa: F401 -- import only; no CUDA calls
    except (ImportError, OSError) as exc:
        pytest.skip(f"torch cannot be imported; confidence policy unavailable: {exc}")
    load_module(
        monkeypatch, "sglang.srt.speculative.adaptive_confidence",
        SPEC / "adaptive_confidence.py",
    )
    monkeypatch.setenv("SGLANG_ADAPTIVE_POLICY", "confidence")
    controller = params(adaptive)
    for bs, (bucket, candidates) in EXPECTED.items():
        slot = controller._route(bs)
        assert controller.get_bucket_for_batch(bs) == bucket
        assert slot.candidate_steps == candidates
        for steps in candidates:
            assert slot.step_time_ms(steps) > 0
            assert bs * (steps + 1) <= 32
    assert controller._route(8).step_time_ms(3) == 15.209


def test_all_three_runners_prune_before_capture():
    # Static wiring check supplements the real helper tests without importing
    # GPU runner dependencies. The guard must remain adaptive-only.
    for relative in (
        "model_executor/runner/decode_cuda_graph_runner.py",
        "speculative/eagle_draft_cuda_graph_runner.py",
        "speculative/eagle_draft_extend_cuda_graph_runner.py",
    ):
        tree = ast.parse((ROOT / "python/sglang/srt" / relative).read_text())
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "get_spec().speculative_adaptive"]
        assert any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "adaptive_capture_bs_for_step"
            for guard in guards for statement in guard.body
            for call in ast.walk(statement)
        ), relative


def test_launcher_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "serve-fast.sh")], check=True)


@pytest.mark.parametrize("concurrency", [1, 4, 6, 8])
def test_launcher_print_only(concurrency):
    # --print exits before server startup and before cache directory creation.
    # Use a clean environment and paths inside hb, with no model or GPU access.
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(ROOT),
        "XDG_CACHE_HOME": str(ROOT / ".test-c8-cache"),
        "PYTHON": sys.executable,
        "TARGET_MODEL": str(ROOT),
        "TOKEN_MAP": "none",
        "CONC": str(concurrency),
        "CUDA_VISIBLE_DEVICES": "",
        "FLASH_NEXT_ADMIN_API_KEY": "cpu-test-placeholder",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "serve-fast.sh"), "wa", "--print"],
        env=env, capture_output=True, text=True, check=True,
    )
    launch = json.loads(result.stdout)
    suffix = "c8" if concurrency == 8 else "c4"
    assert launch["env"]["ADAPTIVE_CONFIG"] == str(ROOT / f"adaptive/w16_{suffix}.json")
    args = launch["argv"]

    def final_value(flag):
        index = len(args) - 1 - args[::-1].index(flag)
        return args[index + 1]

    assert final_value("--speculative-num-steps") == "15"
    assert final_value("--speculative-num-draft-tokens") == "16"
    assert final_value("--speculative-adaptive-config") == launch["env"]["ADAPTIVE_CONFIG"]
    if concurrency != 1:
        assert final_value("--max-running-requests") == str(concurrency)
        start = args.index("--cuda-graph-bs-decode") + 1
        assert args[start:start + concurrency] == [str(bs) for bs in range(1, concurrency + 1)]
        assert args[start + concurrency].startswith("--")
