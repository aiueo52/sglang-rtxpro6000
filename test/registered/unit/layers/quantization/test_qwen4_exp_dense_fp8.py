import argparse
import unittest

import torch

from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    Qwen4ExpDenseFp8LinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.qwen4_exp_dense_fp8 import QWEN4_EXP_DENSE_FP8_DEFAULT
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4ExpDenseFp8Routing(unittest.TestCase):
    _EXCLUDED_MODULES = [
        "*self_attn*",
        "*linear_attn*",
        "*shared_expert*",
        "*hyper_connection*",
        "*ple*",
        "*embed_tokens*",
        "mtp*",
        "gate",
        "lm_head",
    ]

    def _method_for(self, prefix: str, categories: str | None):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=self._EXCLUDED_MODULES,
            packed_modules_mapping={},
        )
        with get_context().override_server_args(
            model_path="dummy", qwen4_exp_dense_fp8=categories
        ):
            layer = ReplicatedLinear(
                16,
                16,
                bias=False,
                quant_config=config,
                prefix=prefix,
            )
        return layer.quant_method

    def test_category_contract_and_hard_exclusions(self):
        categories = "default,mlp_gates,lm_head,indexer"
        selected = {
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            "model.layers.0.mlp.shared_expert.down_proj",
            "model.layers.3.self_attn.qkv_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.2.linear_attn.in_proj_qkvz",
            "model.layers.2.linear_attn.out_proj",
            "model.layers.0.attn_hyper_connection.input_mix_weight_down",
            "model.hyper_connection_mixer.input_mix_weight_up",
            "model.layers.0.mlp.gate",
            "model.layers.0.mlp.shared_expert_gate",
            "lm_head",
        }
        unquantized = {
            "model.embed_tokens",
            "model.layers.3.self_attn.q_norm",
            "model.layers.3.self_attn.indexer.index_qk_proj",
            "model.layers.2.linear_attn.in_proj_ba",
            "model.layers.0.attn_hyper_connection.block_inject_weight",
            "model.layers.2.ple.key_proj",
            "model.mtp.layers.0.self_attn.qkv_proj",
        }

        for prefix in selected:
            with self.subTest(prefix=prefix):
                self.assertIsInstance(
                    self._method_for(prefix, categories),
                    Qwen4ExpDenseFp8LinearMethod,
                )
        for prefix in unquantized:
            with self.subTest(prefix=prefix):
                self.assertIsInstance(
                    self._method_for(prefix, categories),
                    UnquantizedLinearMethod,
                )

        expert_method = self._method_for(
            "model.layers.0.mlp.experts.3.gate_up_proj", categories
        )
        self.assertIsInstance(expert_method, ModelOptFp4LinearMethod)

    def test_feature_off_and_disabled_categories_are_unquantized(self):
        prefix = "model.layers.0.mlp.shared_expert.down_proj"
        self.assertIsInstance(
            self._method_for(prefix, None), UnquantizedLinearMethod
        )
        self.assertIsInstance(
            self._method_for(prefix, "attn"), UnquantizedLinearMethod
        )

    def test_indexer_is_unquantized_even_when_explicitly_listed(self):
        method = self._method_for(
            "model.layers.3.self_attn.indexer.index_qk_proj",
            "indexer",
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)


class TestQwen4ExpDenseFp8Flag(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(
            parser,
            ServerArgs,
            fields=["model_path", "qwen4_exp_dense_fp8"],
        )
        return parser

    def test_default_is_off_and_bare_flag_enables_default_categories(self):
        parser = self._parser()
        disabled = parser.parse_args(["--model-path", "dummy"])
        enabled = parser.parse_args(
            ["--model-path", "dummy", "--qwen4-exp-dense-fp8"]
        )

        self.assertIsNone(disabled.qwen4_exp_dense_fp8)
        self.assertEqual(enabled.qwen4_exp_dense_fp8, QWEN4_EXP_DENSE_FP8_DEFAULT)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--qwen4-exp-dense-fp8",
                    "attn,not_a_category",
                ]
            )


class TestQwen4ExpDenseFp8Numerics(unittest.TestCase):
    def test_e4m3_per_tensor_quant_dequant_relative_error_bound(self):
        torch.manual_seed(0)
        magnitude = torch.rand((64, 128), dtype=torch.float32) * 0.75 + 0.25
        sign = torch.where(torch.rand_like(magnitude) < 0.5, -1.0, 1.0)
        weight = (magnitude * sign).to(torch.bfloat16)
        quantized, scale = input_to_float8(
            weight, dtype=torch.float8_e4m3fn
        )
        restored = quantized.float() * scale
        max_relative_error = (
            (restored - weight.float()).abs() / weight.float().abs()
        ).max()

        self.assertLessEqual(max_relative_error.item(), 1 / 16)


if __name__ == "__main__":
    unittest.main()
