This repository contains some common WCTE analysis tools as a python package, the tools for processing the data are stored here as well as tools for analysing processed data. If you follow the installation instructions you can use these tools running import analysis_tools.

Examples for using some of these tools can be found in the analysis_examples directory and are also listed https://wcte.hyperk.ca/wg/simulation-and-analysis/data-analysis-1 .

The scripts used in the WCTE data production are included in the scripts repository. 

The layout of this repository is as follows:

```
analysis_tools/
├── analysis_examples/ #contains examples on using the analysis tools
├── analysis_tools/ #contains the python package for analysis - can be installed using instructions below
├── cpp_merger/ #contains the cpp merging code for the data production
├── scripts/ #contains the python scripts for running the production
├── data/ #contains data needed for the package functions
├── extern/ #contains external packages linked to this repository as submodules
├── include/ #contains data used by the beam analysis scripts 
├── notebooks/ #contains older example notebooks (from August 2025 workshop)
```

# Installation
Installation:

```
git clone <git repo location>
cd analysis_tools
pip install -e .
```

The `-e` flag allows you to edit the package 
If using on lxplus you will need to setup this in a python virtual environment 

Some of the analysis examples use external packages linked to this repository as submodules. To install the external packages run:
```
git submodule update --init --recursive
```
Since some of the modules are private you need to set up ssh access to github for this to work - see https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account . Note you can check your ssh is working by running ssh -T git@github.com. 

# Contribution Rules
Main branch on WCTE/analysis_tools is protected - please open a pull request (either from your own branch or fork) to push changes to main branch

# Package classes and functions

##  WaveformProcessing

Waveform processing contains a copy of the CFD used in the test stand repository

`WaveformProcessingTeststand.cfd_teststand_method()` processes the CFD using that method returning 
the charge and the time for a pulse in that waveform - including non-linearity corrections 
for both

Additionally the same CFD and charge calculation method used online by the mPMT is included 
in WaveformProcessingmPMT the versions to run on single waveforms and vectorised versions to run on arrays of waveforms are given 

## Pulse finding

Finds pulses in the waveforms using the same method as run online on the mPMT.

`do_pulse_finding` is a slower version processing one waveform at a time and returning a list of found pulses

`do_pulse_finding_vect` is a mostly vectorised version, returning a list of lists of found pulses for each waveform

`do_pulse_finding_fast` is a more optimised, fully-vectorised version that returns results as arrays of rows (waveforms)
and columns (samples) of all pulses found in all waveforms

## CalibrationDBInterface

Interfaces with the calibration database see more instruction here

https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation

Currently processed for the test database - to be updated when the production database 
is ready. The authentication requires a credential text file ./.wctecaldb.credential 
to be in the current working directory - more details in the database interface above

## PMTMapping 

PMTMapping is a class containing the mapping of the WCTE PMTs slot and position ids (position in the detector) to the channel and mPMT card ids (electronics numbering) and vice versa. This is generally not needed as processed data already has the mPMTs mapped to their position and slot numbers. 
Usage:
```
mapping = PMTMapping()
mapping.get_slot_pmt_pos_from_card_pmt_chan(card_id,pmt_channel)
```
 returns the slot and pmt position
and 
```
mapping.get_card_pmt_chan_from_slot_pmt_pos(slot_id,pmt_position)
```
returns the card and channel

The mapping json is located in the package

## DetectorGeometry

Class to load PMT positions, directions and calculate time of flight.

# Data Production Scripts

## production v_0
### add_timing_constants.py

The earliest version of production script for production v_0 which added timing constants to self-trigger data

## production v_0_5
### process_data_v0_5.py

Self-trigger data production v 0_5 (see https://wcte.hyperk.ca/wg/simulation-and-analysis/data-production-2) 
for self-trigger data which includes timing constants and data quality flags

## production v_1

Production v_1 processes both self-trigger and hardware-trigger data through a multi-step pipeline.
A top-level orchestrator script (`run_pipeline.py`) reads run metadata from the run info JSON,
determines the trigger type and beam analysis mode automatically, and calls the appropriate
pipeline scripts. Intermediate outputs are written to step-specific subdirectories under
`<output_base>/<run_number>/`.

### Top-level orchestrator

#### `run_pipeline.py`

The recommended entry point for production. For a given run it:
1. Reads trigger type and beam analysis mode from the run info JSON via `get_run_info()`.
2. Runs the appropriate trigger-specific pipeline (`run_hw_trigger_pipeline.py` or `run_self_trigger_pipeline.py`).
3. Conditionally runs beam monitor PID analysis (`WCTE_beam_analysis.py`) depending on the beam configuration.
4. Runs the T5 beam monitor analysis 

Usage:
```bash
python run_pipeline.py \
  -r <run_number> \
  -i <input_file(s)> \
  -o <output_base_dir> \
  [--steps wf calibrate dq beam] \
  [--debug]
```

**`--steps`** controls which parts of the pipeline run. If not provided all steps run by default.
If `wf` or `calibrate` is specified, the corresponding downstream steps are also re-run
(e.g. `--steps calibrate` also runs `dq`).

#### Beam analysis mode

The decision to run VME beam processing is determined automatically from the run info JSON.
If the run is in a tagged gamma configuration or downstream ACTs (act3–5) are missing,
the VME processing will not run.

### Pipeline runner scripts

These can also be run standalone (e.g. to reprocess a single step).

#### `run_self_trigger_pipeline.py`

Orchestrates self-trigger data processing through two steps:

```
Step 1 (calibrate) : calibrate_hits.py        → calibrated_hits/
Step 2 (dq)        : self_trigger_dq_flags.py → dq_flags/
```

Usage:
```bash
python run_self_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--steps calibrate dq] [--debug]
```

#### `run_hw_trigger_pipeline.py`

Orchestrates hardware-trigger data processing through three steps:

```
Step 1 (wf)        : hw_trigger_wf_processing.py → processed_waveforms/
Step 2 (calibrate) : calibrate_hits.py            → calibrated_hits/
Step 3 (dq)        : hw_trigger_dq_flags.py       → dq_flags/
```

Usage:
```bash
python run_hw_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--steps wf calibrate dq] [--debug]
```

Both pipeline scripts support `--steps` (see above). The dependency expansion is applied
automatically — requesting `calibrate` will also run `dq`.

### Intermediate scripts

#### `hw_trigger_wf_processing.py`

Runs pulse finding and charge/time determination on hardware-trigger raw waveform data.
Reads `WCTEReadoutWindows` trees and writes a `ProcessedWaveforms` ROOT tree.

#### `calibrate_hits.py`

Applies timing constants from the calibration database to hit times.
Works on both self-trigger (`WCTEReadoutWindows`) and waveform-processed (`ProcessedWaveforms`) input files.
Writes a `CalibratedHits` ROOT tree and a `Configuration` tree recording the git hash,
timing constant revision, and list of PMTs with timing constants.

#### `self_trigger_dq_flags.py`

Applies data quality flags for self-trigger runs. Determines the good channel list from the
intersection of slow-control stable channels and channels with calibration constants.
Applies trigger-level bitmask flags for slow-control excluded periods and the 67 ms periodic issue.
Applies hit-level bitmask flags for missing timing constants and slow-control unstable channels.

#### `hw_trigger_dq_flags.py`

Applies data quality flags for hardware-trigger runs. In addition to the channel-level flags above,
applies trigger-level flags for missing waveforms, missing trigger signals, and mismatched waveform lengths.

### Output structure

```
<output_base>/<run_number>/
  processed_waveforms/   (hw-trigger only)
    <base>_processed_waveforms.root
  calibrated_hits/
    <base>_calibrated_hits.root
  dq_flags/
    <base>_[self|hw]_trigger_dq_flags.root
    <base>_[self|hw]_trigger_dq_flags_status.json
  beam_data/             (normal beam runs only)
    beam_analysis_output_R<run_number>.root
```


### Shared utilities

Common functions used across all production scripts are in `analysis_tools/production_utils.py`:
- `get_run_info` — reads trigger type and beam analysis mode from the run info JSON
- `get_git_descriptor` — git provenance for output Configuration trees
- `file_sha256` — hash of the slow-control input file used
- `get_run_database_data`, `get_stable_mpmt_list_slow_control` — slow-control data access
- `get_slow_control_trigger_mask`, `get_67ms_mask` — trigger-level DQ masks
- `slot_pos_from_card_chan_list` — PMT channel mapping


# Beam monitor PID

The `script/WCTE_beam_analysis.py` code performs the 1pe calibration of the ACT PMTs as well as the *basic* event PID and momentum measurement based on VME monitor information (TOF, charge deposited in ACTs, etc...).

The code uses the BeamAnalysis defined in `analysis_tools/beam_monitors_pid.py`. 

To call the script on the lxplus machines, run:
`python3  script/WCTE_beam_analysis.py --r <run_number>  -i <input_file>  -o <output_base_dir>`

the VME merged data used by the beam analysis code is located at:

`/eos/experiment/wcte/data/2025_commissioning/offline_data_vme_match/`

the code looks up the run information from the .json file of run information compiled by Laurence from the Google sheet. 


All of the information about this version of the code (made for the winter 2025-2026 data production is available here: 
https://www.overleaf.com/read/fshzdwxbjwnn#c0127f please read ! Email Alie or Bruno if you have any questions. 

All the plots needed for visualising the selection are saved in the same folder that you output to.

You should have a look at them to check that the selection behaves as expected, an example is provided in notebooks/plots.

The code also calculates the mean momentum for each particle type before exiting the CERN beam pipe (upstream of T0) 
and right after exiting the WCTE beam window (i.e. into the tank). These momenta are also estimated for each trigger,
based on the estimated PID (shown on plots but not shared due to the large error and general unreliability). 
Note the error on these momenta (propagated from the time of flight resultion, taken as the
standard deviation of the TOF distribution for electrons) is very large for slow particles. Protons and deuterons events are 
identified using their time of flight but 3He nuclei aren't. The total charge in the TOF detector is saved in the output file but
no cut is placed on it. The "is_kept" branch of the output file stores information on whether the trigger passes the basic beam
requirements (no hits above threshold in the hole counters, hits in all T0, T1, T2 PMTs etc...). 

More information about the quality of each trigger (if the particle reached all trigger scintillators, DAQ looked ok etc...) is available in the corresponding bitmask variables. Please refer to this presentation for more details on how to use the code and what it does: https://wcte.hyperk.ca/meetings/collaboration-meetings/20260216/meeting-5/beam-monitor-status-overview/plenary_talk.pdf

