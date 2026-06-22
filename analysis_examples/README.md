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

### Particle ID based on VME analysis

`BeamSelection` defines particle selections using series of cuts on WCTE beam monitor (VME) data. The examples in the code show the nominal selection cuts, they should be adapted to suit your own analysis needs.


```python
#read in the scalar results (i.e. nominal cut lines) of the VME analysis 
vme_scalar_results = loader.get_vme_analysis_scalar_results()

# --- Define your particle selections ---
# Every cut is a [variable, operator, value] triplet.
# You can apply a cut to any VME variable 
# Operators: ">", "<", ">=", "<=", "between" (value must be [low, high] for "between").
# Omit the TOF cut entirely if proton_tof_cut is 0 
# This case of TOF separation unavailable happensfor negative polarity and low momentum runs in production 1.0,
# To be improved in production 1.1 

tof_cut    = vme_scalar_results['proton_tof_cut']
if tof_cut == 0:
    print("WARNING: TOF separation unavailable for this run, setting TOF cut to default value of 999 ns.")
    tof_cut = 999

eveto_cut  = vme_scalar_results['act_eveto_cut']
tagger_cut = vme_scalar_results['act_tagger_cut']

# PIONS: fast particles that do not produce Cherenkov light in either ACT.
pion_sel = BeamSelection.pion(
    ["vme_act_eveto",  "<", eveto_cut],
    ["vme_act_tagger", "<", tagger_cut],
    ["vme_tof_corr",        "<", tof_cut],
)

# PROTONS: slow particles identified by their TOF falling in a window above the
#          fast/slow separation value. Only meaningful when proton_tof_cut > 0.

proton_sel = BeamSelection.proton(
    ["vme_tof_corr", "between", [tof_cut, tof_cut + 10]],
)
```

The selections are applied within the iterating loop described above and the `SelectionMonitor` is used to monitor visually the selections. The code write out the selected batches to an external .parquet file, choose this if you want to work with large datasets. You can do quick analysis on individual batches within the 'loader.iterate' loop. 

```python
# Enable parquet output for the selections you want to save.
# Default filename is "<particle>.parquet". Pass a path to override.
pion_sel.enable_parquet_output(f"run{run_number}_pions.parquet")
muon_sel.enable_parquet_output(f"run{run_number}_muons.parquet")
ele_sel.enable_parquet_output(f"run{run_number}_electrons.parquet")

# Decide which selections you want to monitor live during loading. This is optional but useful for understanding cut lines.
selections = [pion_sel, muon_sel, ele_sel, proton_sel] 
monitor    = SelectionMonitor(selections, update_every=10, vme_run_info=vme_run_info)
```

```python
for i_batch, batch in enumerate(loader.iterate(verbose=False, step_size="100 MB")):
    n_windows_passing += len(batch)

    monitor.update(batch)
    for sel in selections:
        sel._write_to_parquet(batch[sel.mask(batch)])

    #alternatively, do analysis on individual bath
    pion_batch = batch[pion_sel.mask(batch)]
    #do something with pion_batch...

for sel in selections:
    sel.close_parquet_writer()
```

## Multilaterator example 

### Installing submodules 

This example (and others in this repo) use git submodules (extern/Geometry, extern/TimeCal, extern/T5_analysis) which are private repos cloned over SSH. Before cloning, make sure your GitHub account has an SSH key set up and added under Settings → SSH and GPG keys (test with `ssh -T git@github.com`). 

For details on installation, please refer to the [analysis_tools README](https://github.com/WCTE/analysis_tools/blob/main/README.md). Make sure you install `analysis_tools` in your Python environment as a package so you can easily import `DataLoader` and `BeamSelection` used in this example. 

If cloning for the first time you can run the following to install the submodules:

`git clone --recurse-submodules`

If you already cloned without --recurse-submodules you can run the following to install the submodules:

`git submodule update --init --recursive`


### Multilaterating Michel electrons 

The Jupyter notebook [multilaterator_example.ipynb](https://github.com/WCTE/analysis_tools/blob/main/analysis_examples/multilaterator_example.ipynb) provides an example of a Michel electron search using WCTE data. 

#### 1. Load Run Data

Opens a merged production ROOT file for a given run (e.g. run 1478, −410 MeV/c). Cherenkov thresholds are printed to confirm which particle species are above threshold in each ACT detector.

#### 2. Filter and Save Muon Events

Apply default `DataLoader` data quality cuts. Select muons using `BeamSelection` VME cuts. 
Selected muon events are saved to a new ROOT file:
 
```
muon_filtered_R{run_number}_v1_0_{date}.root
```

#### 3. Find the Prompt Muon Time
 
A histogram of all calibrated PMT hit times is built across all selected events. The global peak (the prompt muon signal) is identified, and a ±50 ns window is defined around it. We will require all prompt muons to be within this window. In run 1478 we find:
 
```
Global peak time: 1709.0 ns
Prompt event window: [1659.0, 1759.0] ns
```
 
#### 4. Identify Michel Electron Candidates
 
For each muon event:
- The prompt muon arrival time is found from the peak in the prompt window
- A Michel electron search is performed in the range [mu_time + 200, 7500] ns
- The largest peak in this secondary window is taken as the Michel candidate
- A ±8/+4 ns window around the peak is used to collect hits for reconstruction
- Events with fewer than 30 hits are skipped

#### 5. Multilaterate
 
The `TC_Multilaterator` reconstructs the (x, y, z, t) vertex of each Michel candidate using PMT hit times and positions. Reconstruction uses a Huber-loss TRF fit with bounds of ±4000 mm and ±100 ns around the candidate time. 
 
Results are saved to:
```
multilaterator_res_run{run_number}_muon_{date}.pkl
```
 
#### 6. Apply Selection Cuts and Plot
 
After multilateration, we apply cuts on `nhits`, `total_charge`, `p-value`, and reject events with secondary hits in T5 to select a clean Michel electron sample. We produce diagnostic plots containing: 
- χ², ndof, χ²/ndof distributions
- Muon lifetime distribution with exponential decay fit 
- Reconstructed x, y, z positions with Gaussian fits