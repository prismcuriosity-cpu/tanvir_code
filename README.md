# tanvir_code

Breast DCE-MRI pipeline: 5-channel preprocessing → LightUNet segmentation →
tumor ROI → Attention-MIL + clinical features → pCR prediction.

`CGMRNet_BreastDCEDL_finetuned_seg.ipynb` is the runnable notebook. Requires
a local GPU, PyTorch, and the BreastDCEDL NIfTI dataset (paths configured at
the top of the notebook) — none of that is available in this repo/session,
so changes here are reviewed by inspection, not by execution.

## Improvement roadmap status

Following the project's experiment hierarchy (EXP-00 → EXP-18, one
independent change at a time):

- **EXP-00 (baseline freeze)** — done. A `## 8.5 PHASE 0` section saves
  `config.json`, `segmentation_model.pt`, `classification_model.pt`,
  `segmentation_predictions.csv` (per-slice Dice/areas), and
  `classification_predictions.csv`, plus `metrics.json`, to `baseline/` —
  evaluated on the validation split only.
- **EXP-01 (classification data audit)** — done. A `## 3.1 PHASE 1` section
  audits every patient in the manifest for `missing_pcr` /
  `missing_pre` / `missing_early` / `missing_late` / `missing_mask` /
  `missing_roi`, reports non-exclusive reason counts plus a primary
  reason, and separates a looser `classification_eligible` flag (mask not
  required, since the current classifier crops the metadata ROI directly).
- **EXP-02/EXP-03 (fix the eligible cohort + verify the stratified
  split)** — done. `## 3.2 PHASE 2` checks the pCR balance per split on the
  eligible cohort (the existing `candidate_strata` cohort+pCR
  stratification was kept, not replaced). `classification_df` (PHASE 3,
  inside "## 7") is now built from the audited eligible cohort with
  explicit `assert`s instead of the old
  `manifest['phase_ready'] & pCR.notna()` filter, which silently dropped
  every patient missing a usable ROI.
- **EXP-04 (segmentation threshold)** — done. `tune_segmentation_threshold`
  now sweeps 0.10–0.70 (was 0.30–0.70) and reports Dice, IoU, precision,
  and recall per threshold, selected on validation only.
- **Immediate dev-speed fix** — done. `force_full_epochs=False` /
  `early_stopping_patience=7` are now the live `CFG` defaults, with the
  original 30-epoch/`force_full_epochs=True` values kept in
  `BASELINE_CFG_SNAPSHOT` for reference. `train_segmentation` now actually
  stops early (it previously ran the full `seg_epochs` unconditionally);
  `train_classifier` already had this behavior.

Not yet implemented (each is its own experiment, deliberately left for a
follow-up pass once EXP-00–EXP-04 have been run and reviewed on real data):
segmentation loss ablation (EXP-06/PHASE 5), augmentation (PHASE 6), error
analysis (PHASE 7), input-channel ablation (EXP-07/PHASE 8), classification
threshold/imbalance/augmentation (EXP-08–10/PHASE 13–14), MRI/clinical/
attention ablation (EXP-11/PHASE 15), MIL bag-size sweep (EXP-12/PHASE 16),
automatic slice selection (EXP-13/PHASE 17), predicted-ROI vs ground-truth
ROI (EXP-14–15/PHASE 18–20), regularization sweep (PHASE 21), multi-seed
reporting (EXP-16/PHASE 22), and the final locked test (EXP-17–18/PHASE
23–24).
