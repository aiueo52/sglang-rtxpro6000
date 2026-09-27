# flash-next-fast branch

This branch is a fork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) (itself a fork of [SGLang](https://github.com/sgl-project/sglang)). It adds a series of 105 commits on top of `16e5682aad` (`pennyroyal-main-sm120-final`, 2026-08-27) that speed up single-stream decoding of Qwen3.8-Flash-Next (NVFP4) on one NVIDIA RTX PRO 6000 Blackwell Max-Q.

Many thanks to jpezzulli: this work builds directly on that fork.

The method, results, benchmark harness, FlashInfer patches, measurement notes and the licence/NOTICE details are in the companion repository **[aiueo52/flash-next-rtxpro6000](https://github.com/aiueo52/flash-next-rtxpro6000)**. Start there; `docs/reproduce.md` explains how to run this branch.

Some commits contain code derived from flash-linear-attention (MIT), causal-conv1d (BSD-3-Clause) and the Triton tutorials (MIT); their licence texts are in `third_party_licenses/flash-next-fast/`. `bench/glue_e/gdn_wy_T16_strided.patch` modifies FlashInfer (Apache-2.0) code; FlashInfer's NOTICE is reproduced in that directory as `NOTICE.flashinfer`. See the companion repository's `NOTICE` for the per-commit list.

Every upstream file this branch modifies is listed in `MODIFICATIONS.md` and carries a one-line "Modified by aiueo52 for flash-next-fast (2026)" header comment (Apache-2.0 §4(b)).

The private MTP head and draft token map used for the published numbers are not included, so results with this branch alone will be somewhat lower.
