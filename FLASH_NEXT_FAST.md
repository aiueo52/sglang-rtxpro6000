# flash-next-fast branch

This branch is a fork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) (itself a fork of [SGLang](https://github.com/sgl-project/sglang)), based on `16e5682aad6e3335f38ec8d5711176278f12a086` (`pennyroyal-main-sm120-final`, 2026-08-27).

- **The flash-next-fast series** (121 commits on top of the base): commits 1–105 are the 2026-09-08 measured state; 106–121 (2026-10-01/02) add the sampling-mode work and the 2026-10-02 production build. Two comment-only commits (after 105 and after 121) add the modification headers and `MODIFICATIONS.md`. The 2026-10-02 build is commit `5105985`.
- **The 2026-10-27 update** by aiueo52 and contributors makes this tree the new release, with two editions: **normal** (Qwen3.8-Flash-Next NVFP4) and **hyperpro** (ThinkingCap with OEA). See [README.md](README.md) for requirements, setup, launch and measured results.

Many thanks to jpezzulli: this work builds directly on that fork.

The FlashInfer 0.6.17 changes ship as source overlays in `flashinfer-overlays/`, not as complete dependency trees. Assemble the normal and hyperpro trees locally with [tools/assemble_flashinfer.py](tools/assemble_flashinfer.py).

Every upstream file this branch modifies is listed in [MODIFICATIONS.md](MODIFICATIONS.md) and carries a one-line "Modified by aiueo52 for flash-next-fast (2026)" header comment (Apache-2.0 §4(b)). Retained third-party notices are indexed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Some commits contain code derived from flash-linear-attention (MIT), causal-conv1d (BSD-3-Clause) and the Triton tutorials (MIT); their licence texts are in `third_party_licenses/flash-next-fast/`. `bench/glue_e/gdn_wy_T16_strided.patch` modifies FlashInfer (Apache-2.0) code; FlashInfer's NOTICE is reproduced in that directory as `NOTICE.flashinfer`.

The method, measurement notes and history are in the companion repository **[aiueo52/hyperburst-rtxpro6000](https://github.com/aiueo52/hyperburst-rtxpro6000)**; its `docs/reproduce.md` describes how the 2026-10-02 build was run.
