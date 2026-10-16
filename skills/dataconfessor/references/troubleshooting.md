# Troubleshooting

## The self-consistency check failed

The two independent halves of the proxy models disagree about who is atypical, so the
ranking is not reproducible and the flags mean nothing. In order:

1. **Longer proxies.** `--epochs 3`, then `--epochs 5`. The usual cause is proxies too weak
   to rank anyone: a model that has learned nothing scores every record alike. If a utility
   curve for the task exists, pick the epoch at which held-out performance first approaches
   its peak.
2. **More proxies.** `--models 8`, which doubles the cost.
3. **The data.** Is `patient_id` really a person? Are many records near-duplicates of each
   other? Does every class have enough records for one epoch to learn anything? Are the
   images so small that the signal is gone?
4. **Stop.** Some datasets cannot be screened this way — in the source study a per-example
   benchmark with no patient structure scored 0.50–0.64 and never passed. Say so, and point
   at a full shadow-model audit instead.

Never lower the threshold. It exists because a failed ranking is indistinguishable from a
good one by eye.

## It is slow

- The first run decodes every record into a cache; later runs on the same data skip that.
- `--size 128` halves the pixels of a 224 px run. Check self-consistency afterwards.
- `--arch smallcnn` is 3–7x cheaper than the ResNet default and recall was a wash on real
  patient-level data — but its self-consistency margin was much thinner (0.76 against 0.82
  and 0.89), and that is the one guard rail that works on an unseen dataset. Watch it.
- The cache lives in `.dataconfessor_cache/` beside the manifest. On a machine with a small
  system drive, put it elsewhere with `--cache-dir`.
- If the cache fits in memory it is read in once and the run becomes GPU-bound; if it does
  not, every batch is random disk I/O and the screen can take three times as long. The
  budget is half of free memory when `psutil` is installed and 4 GiB otherwise; set
  `DATACONFESSOR_RAM_GB` to raise it. The tool prints `loading N GiB of records into
  memory` when it takes this path.

## Errors

| Message | Cause |
|---|---|
| `missing column(s)` | the manifest is not the three-column format; run `init` |
| `sampled path(s) do not exist` | paths are resolved relative to the manifest file |
| `mixed or unsupported file types` | one manifest holds one format; split it, or use `--loader` |
| `every record belongs to one patient_id` | the grouping key is wrong |
| `only one label value` | atypicality is defined against a task; a constant label is not one |
| `has shape (...), but the first record has (...)` | signals must share a length; resample first |
| CUDA out of memory | lower `--size`, or `--device cpu` for a small dataset |

## The flags look wrong

- **Everything flagged is one class.** Expected, and not a bug: atypicality is defined
  relative to the task, and a rare class is poorly predicted by models that never saw it.
  In the source study's dermoscopy cohort the flagged patients were almost exactly the
  melanoma patients. This is the reason not to delete them.
- **The ranking is flat** (panel a of the report shows no tail). The cut is then closer to
  a budget allocation than to a finding. Say so.
- **Patients with many records dominate.** Risk is the maximum over a patient's records, so
  more records means more chances to be unpredictable. If the median is above ~50, group
  records at series or study level.

## Reproducibility

Same manifest, same flags, same `--seed` gives the same ranking. Repeated runs with
different seeds agreed to within 1.8 percentage points of recall in the source study.
