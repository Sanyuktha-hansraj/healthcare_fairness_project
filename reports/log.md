#  EDA Log
Generated: 2025-11-24 01:03

## Dataset Summary
- Total images: 656
- Unique patients: 656
- Malignant cases: 171 (26.1%)
- Benign cases: 485 (73.9%)

## Skin Tone Distribution
skin_tone_label
Fitzpatrick III-IV    241
Fitzpatrick I-II      208
Fitzpatrick V-VI      207

## Malignancy Rates by Group
      skin_tone_label  total  malignant_count  malignancy_rate
0    Fitzpatrick I-II    208               49            23.56
1  Fitzpatrick III-IV    241               74            30.71
2    Fitzpatrick V-VI    207               48            23.19

## Feature Engineering
- Total features extracted: 12
- Feature types: metadata + image statistics
- Output format: Parquet

## Fairness Observations
- Outcome disparity: 7.52 percentage points
- All groups have adequate sample sizes (>30)
- Ready for fairness-aware modeling in Week 3

## Deliverables Completed
- EDA notebook with visualizations
- Subgroup definitions
- Feature engineering pipeline
- Fairness metrics framework
