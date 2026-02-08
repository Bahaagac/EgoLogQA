# Validation Sampling Plan

1. List dataset files with `egologqa.io.hf_fetch.list_mcap_files` using fixed `repo_id`, `revision`, and `prefix`.
2. Sort by `hf_path` lexicographically (already guaranteed by list helper).
3. Select the first `N` files (`N` defaults to 50).
4. Persist selected `hf_path` values in `validation/results/run_index.json` sorted by `hf_path`.

This protocol is deterministic for a fixed dataset revision.
