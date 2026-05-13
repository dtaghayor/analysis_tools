# Analysis Examples

This directory contains examples of how to use common analysis tools and functions.

## DataLoader

`DataLoader` loads processed WCTE ROOT files and applies data quality cuts.

```python
from analysis_tools import DataLoader

loader = DataLoader("/path/to/WCTE_merged_production_RXXXX.root")
```

### Data quality cuts

Enable any combination of cuts before iterating — they are applied automatically per batch:

```python
loader.apply_mPMT_data_quality_cuts()   # window/hit-level mPMT quality masks
loader.apply_vme_event_quality_cuts()   # VME digitisation and event quality
loader.apply_t5_event_quality_cuts()    # T5 event quality cuts
```

### Iterating over data

```python
for batch in loader.iterate(step_size="100 MB"):
    # batch is an awkward array of events passing all enabled cuts
    print(len(batch), "windows")
```

Pass `verbose=True` to print event counts after each cut.

### Metadata helpers

```python
loader.get_configuration()              # run configuration record
loader.get_data_quality_metrics()       # DQM summary
loader.get_vme_analysis_scalar_results()
loader.get_vme_analysis_run_info()
loader.get_good_wcte_pmts()             # returns (slot_ids, position_ids)
```
