# 08_drug_asr_rules: moved

The fix-rule build stage (scripts 01 to 07, the hand-written TSVs, the v1 and v2 rule files, the rejected candidates and the tests) moved on 2026-11-15 to its own repository, [UltiMed-ASR-FR-v2-regex-fixes](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v2-regex-fixes), which also adds a lookup-table form of the rules for apps whose regex engine has no lookbehind. The history up to the move is in this repository's git log.

Those scripts still read a few things from this repository (the corpus manifests in `99_hf_release/data/NeMO_files`, `utils/drug_casing.json`, the stage-02 drug lexicon, `utils/nemo_manifest.py`), so they expect a checkout of it next to theirs, or at `$ULTIMED_SCRIPTS_DIR`. Keep those paths stable or update the new repository with them.

Written with Claude Code.
