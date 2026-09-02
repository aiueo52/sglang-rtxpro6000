from __future__ import annotations

from typing import Optional


DEFAULT_QWEN4_EXP_DENSE_FP8_CATEGORIES = (
    "shared_expert",
    "attn",
    "linear_attn",
    "hyper_connection",
)
QWEN4_EXP_DENSE_FP8_CATEGORIES = (
    *DEFAULT_QWEN4_EXP_DENSE_FP8_CATEGORIES,
    "mlp_gates",
    "lm_head",
    "mtp_dense",
    "indexer",
)
QWEN4_EXP_DENSE_FP8_DEFAULT = ",".join(
    DEFAULT_QWEN4_EXP_DENSE_FP8_CATEGORIES
)


def normalize_qwen4_exp_dense_fp8_categories(value: str) -> str:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if "default" in requested:
        requested.remove("default")
        requested.update(DEFAULT_QWEN4_EXP_DENSE_FP8_CATEGORIES)

    unknown = requested.difference(QWEN4_EXP_DENSE_FP8_CATEGORIES)
    if unknown:
        choices = ", ".join(QWEN4_EXP_DENSE_FP8_CATEGORIES)
        raise ValueError(
            "unknown Qwen4-Exp dense FP8 categories: "
            f"{', '.join(sorted(unknown))}; expected: {choices}"
        )
    if not requested:
        raise ValueError("Qwen4-Exp dense FP8 requires at least one category")

    return ",".join(
        category
        for category in QWEN4_EXP_DENSE_FP8_CATEGORIES
        if category in requested
    )


def parse_qwen4_exp_dense_fp8_categories(
    value: Optional[str],
) -> frozenset[str]:
    if value is None:
        return frozenset()
    return frozenset(normalize_qwen4_exp_dense_fp8_categories(value).split(","))


def select_qwen4_exp_dense_fp8_category(
    prefix: str, enabled_categories: frozenset[str]
) -> Optional[str]:
    if not enabled_categories:
        return None

    parts = tuple(part for part in prefix.split(".") if part)
    if (
        "ple" in parts
        or "indexer" in parts
        or "experts" in parts
        or "embed_tokens" in parts
        or "ngram_embedding" in parts
    ):
        return None
    is_mtp = "mtp" in parts

    category = None
    if len(parts) >= 3 and parts[-3:-1] == ("mlp", "shared_expert"):
        if parts[-1] in {"gate_up_proj", "down_proj"}:
            category = "shared_expert"
    elif len(parts) >= 2 and parts[-2] == "self_attn":
        if parts[-1] in {"qkv_proj", "o_proj"}:
            category = "attn"
    elif len(parts) >= 2 and parts[-2] == "linear_attn":
        if parts[-1] in {"in_proj_qkvz", "out_proj"}:
            category = "linear_attn"
    elif any(
        part
        in {
            "attn_hyper_connection",
            "mlp_hyper_connection",
            "hyper_connection_mixer",
        }
        for part in parts
    ):
        if parts[-1] in {"input_mix_weight_down", "input_mix_weight_up"}:
            category = "hyper_connection"
    elif len(parts) >= 2 and parts[-2] == "mlp":
        if parts[-1] in {"gate", "shared_expert_gate"}:
            category = "mlp_gates"
    elif parts and parts[-1] == "lm_head":
        category = "lm_head"
    elif is_mtp and parts and parts[-1] in {"fc_embedding", "fc_hidden"}:
        category = "mtp_dense"

    if is_mtp:
        # The MTP draft layer's dense projections (attn q/k/v/o, fc_embedding,
        # fc_hidden, shared expert, router) are one opt-in category; its
        # hyper-connection mix stays on the fused BF16 kernel.
        if category in {"attn", "shared_expert", "mlp_gates", "mtp_dense"}:
            category = "mtp_dense"
        else:
            return None

    return category if category in enabled_categories else None
