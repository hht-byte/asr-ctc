# Expanded CTC pilot implementation plan

**Goal:** Move from tiny-subset memorization to a larger speaker-disjoint pilot with measurable vocabulary coverage and explicit optimizer-step budgets.

**Architecture:** Keep the frozen encoder and linear CTC head. Extend the existing pinned AISHELL recipe and builder, freeze the original dev/test selections, audit recorded train/dev token IDs before training, and extend train.py with step budgets and sampled training metrics. Do not use test metrics for selection.

**Tech stack:** Python standard library, PyTorch / torch_npu, existing Qwen loader.

- [x] Dataset: add a pinned expanded recipe, preserve the original held-out recordings, reject leakage, and test with synthetic archives.
- [x] Coverage: add a stdlib train/dev recorded-ID audit, report OOV and exact-prefix representability, integrate a configurable dev-OOV gate before encoder loading, and test leakage/unknown IDs.
- [x] Training: add max optimizer steps and fixed-step validation, report global_step plus sampled train TER, preserve epoch-only behavior, test real tiny CPU training and exact stopping.
- [x] Delivery preparation: write executable stage-two instructions, run the full suite, and complete review. Publish this verified change to the authorized remote.

Ruling: provide a reproducible download/build recipe rather than claim new audio or teacher traces were generated locally. The user's model/NPU environment is unavailable. Exact expanded audio counts and recorded-ID coverage are reported by the tools after preparation/collection.

Ruling: retain the original holdout by reconstructing its selection with the original pinned recipe before selecting expanded training records; exclude its speakers, transcripts and audio hashes from training.

Verification (2026-10-14): 50 CPU tests passed, including real tiny-head optimization with validation at steps 2/4/5 and OOV rejection before encoder loading. Frozen holdout manifests are byte-identical in synthetic archive tests. Expanded recipe sizes and SHA256 values match pinned source metadata (64 train speakers, 4 held-out speakers, 70 source files, 2,533,381,795 bytes). Final independent code review found no concrete correctness issues.

Limits: Windows two-rank execution could not be validated: torchrun failed before workers because the local PyTorch build lacks libuv, and a FileStore rendezvous alternative timed out. No Ascend/HCCL run or expanded dataset/teacher generation was performed locally. Those remain target-environment acceptance checks, documented in STAGE2.md.
