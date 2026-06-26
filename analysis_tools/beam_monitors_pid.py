#this code holds the functions necessary for reading in the data and identifying the paticle types based on the WCTE beam monitors 

import os
import json
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import pandas as pd
import awkward as ak
import pyarrow as pa, pyarrow.parquet as pq
import gc

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import moyal
from typing import Dict

#for the nice progress bar
from tqdm import tqdm

import os, shutil, subprocess, time, hashlib

from collections import defaultdict

import sys
# sys.path.append("/eos/user/a/acraplet/analysis_tools/")
#the module for reading the yaml file with the detector distances and dimensions - from Bruno
from .read_beam_detector_distances import DetectorDB as db

#Helper functions for file reading, written by Sahar
def stage_local(src_eos_path: str, min_free_gb=20, min_bytes=1_000_000) -> str:
    st = shutil.disk_usage("/tmp")
    if st.free/1e9 < min_free_gb:
        print("Not enough /tmp space; will stream from EOS")
        return ""

    # Use a unique local name to avoid collisions across different directories
    h = hashlib.md5(src_eos_path.encode()).hexdigest()[:8]
    local = f"/tmp/{os.path.basename(src_eos_path)}.{h}"

    def good(p): return os.path.exists(p) and os.path.getsize(p) >= min_bytes

    if not good(local):
        if os.path.exists(local):
            print("Cached local copy is too small; re-staging…")
            os.remove(local)
        print("Staging ROOT file to local disk (xrdcp)…")
        subprocess.run(["xrdcp", "-f", to_xrootd(src_eos_path), local], check=True)
        if not good(local):
            raise OSError(f"Local file too small after xrdcp: {local}")

    print("Using local copy:", local)
    return local


def to_xrootd(path: str) -> str:
    assert path.startswith("/eos/")
    return "root://eosuser.cern.ch//eos" + path[4:]


def make_blocks(idx: np.ndarray, max_block: int):
    if idx.size == 0:
        return []
    blocks = []
    start = idx[0]
    last  = idx[0]
    for v in idx[1:]:
        # if extending the block stays ≤ max_block, keep extending
        if (v - start) < max_block:
            last = v
        else:
            blocks.append((int(start), int(last)+1))
            start = last = v
    blocks.append((int(start), int(last)+1))
    return blocks





def make_flag_map(flags):
    """
    Build a deterministic mapping from flag name -> bit index (0..).
    Uses sorted order of flag names so mapping is reproducible.
    """
    return {name: idx for idx, name in enumerate(sorted(flags))}

def write_event_quality_mask(flag_dict,
                             flag_map):
    """
    Pack flags into an integer bitmask.
    
    Parameters
    - flag_dict: mapping of flag name -> truthy/falsey value (bool, 0/1, etc.)
    - flag_map: optional mapping flag name -> bit index (int).
                If None, a deterministic mapping from sorted(flag_dict.keys()) is used.
                
    Returns
    - mask: int where bit i (value 2**i) is set when the corresponding flag is true.
    """
    if flag_map is None:
        flag_map = make_flag_map(flag_dict.keys())

    mask = 0
    for flag_name, bit_idx in flag_map.items():
        if flag_name not in flag_dict:
            # Missing flags are treated as False (not set). Change if you want an error.
            continue
        if bool(flag_dict[flag_name]):
            mask |= (1 << bit_idx)
    return mask

def read_event_quality_mask(mask,
                            flag_map):
    """
    Unpack an integer bitmask back into a flag dictionary.
    
    Parameters
    - mask: integer bitmask
    - flag_map: mapping flag name -> bit index used when the mask was created
    
    Returns
    - dict of flag name -> bool (True if that bit is set)
    """
    result: Dict[str, bool] = {}
    for flag_name, bit_idx in flag_map.items():
        result[flag_name] = bool(mask & (1 << bit_idx))
    return result

class NotTheFirstHit(Exception):
    """Raised when a duplicate TDC hit arrives earlier than the first stored hit,
    i.e. the hits are not time-ordered as the de-duplication logic assumes."""
    pass


def _deduplicate_tdc_hits(ids, times):
    "Function used to remove the later TDC hits and corresponding channel ids in case there are more than one"
    seen = set()
    keep_ids = []
    keep_times = []
    duplicates_removed = defaultdict(int)
    stored_times = defaultdict(int)
    
    for ch, t in zip(ids, times):
        if ch in seen:
            #here we are counting how many events are being removed per channel
            duplicates_removed[ch] += 1
            if stored_times[ch] > t:
                #in case they are not ordered correctly, this is important
                raise NotTheFirstHit(
                    f"TDC hits not time-ordered for channel {ch}: "
                    f"stored first hit at {stored_times[ch]} > later hit at {t}"
                )
            continue
        seen.add(ch)
        keep_ids.append(ch)
        keep_times.append(t)
        stored_times[ch] = t

    #in case we are not removing any channel, can give it back as is
    if not duplicates_removed:
        return ids, times, duplicates_removed

    #ensure that the format on the way out is the same as on the way in
    if isinstance(ids, np.ndarray):
        ids_clean = np.asarray(keep_ids, dtype=ids.dtype)
    else:
        ids_clean = np.asarray(keep_ids)
    if isinstance(times, np.ndarray):
        times_clean = np.asarray(keep_times, dtype=times.dtype)
    else:
        times_clean = np.asarray(keep_times)
    return ids_clean, times_clean, duplicates_removed


def _tdc_requirement_met(group, tdc_set):
    if group["mode"] == "any_pair":
        return any(all(ch in tdc_set for ch in pair) for pair in group["channels"])
    return all(ch in tdc_set for ch in group["channels"])


#these are replaced with more accurate estimates using the travel time of all particles
# proton_tof_cut = 17.5 #ad-hoc but works for most analyses
# deuteron_tof_cut = 35 #35 #ad-hoc but works for most analyses
helium3_tof_cut = 30 #30 #ad-hoc 

        

#Default informations
c = 0.299792458 #in m.ns^-1 do not change the units please as these are called throuhout
L =  444.03 #4.3431
L_t0t4 = 305.68 #2.9485
L_t4t1 = 143.38 #1.3946

# Particle masses in GeV/c^2
particle_masses = {
    "Electrons": 0.000511,
    "Muons": 0.105658,
    "Pions": 0.13957,
    "Protons": 0.938272
}

reference_ids = (31, 46)          # (TDC ref for IDs <31, ref1 for IDs >31)
t0_group     = [0, 1, 2, 3]       # must all be present
t1_group     = [4, 5, 6, 7]       # must all be present
t4_group     = [42, 43]           # must all be present
t4_qdc_cut   = 200                # Only hits above this value
ACT0_group   = (12, 13)                
ACT1_group   = (14, 15)
ACT2_group   = (16, 17)                

ACT3_group   = (18, 19)                
ACT4_group   = (20, 21)                 
ACT5_group   = (22, 23)                
act_eveto_group = [12, 13, 14, 15, 16, 17]   # ACT-eveto channels
act_tagger_group = [18, 19, 20, 21, 22, 23]
hc_group = [9, 10]
hc_charge_cut = 150

t5_b0_group = [48, 56]     #T5, also known as the TOF  detector
t5_b1_group = [49, 57]     #Pairs of siPMs on either side of a given bar
t5_b2_group = [50, 58]     #Both SiPMs need to be above threshold for at least 
t5_b3_group = [51, 59]     #One of the bars for the event to be kept
t5_b4_group = [52, 60]
t5_b5_group = [53, 61]
t5_b6_group = [54, 62]
t5_b7_group = [55, 63]
t5_total_group = [t5_b0_group, t5_b1_group, t5_b2_group, t5_b3_group,
                  t5_b4_group, t5_b5_group, t5_b6_group, t5_b7_group]
t5_group = np.arange(48, 64, 1)

#basic functions, necessary in general
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

#three gaussian fit, necessary for fitting the T4 TOF distributions
def three_gaussians(x, amp, mean, sigma,  amp1, mean1, sigma1,  amp2, mean2, sigma2):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2) + amp2 * np.exp(-0.5 * ((x - mean2) / sigma2) ** 2)


def landau_gauss_convolution(x, amp, mpv, eta, sigma):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-3)
    eta = max(float(eta), 1e-3)
    # Keep the integration domain within the physical (positive) region to avoid
    # numerical overflow in the Landau tail.
    t_min = max(mpv - 5.0 * eta - 5.0 * sigma, 0.0)
    t_max = mpv + 15.0 * eta + 5.0 * sigma
    if t_max <= t_min:
        t_max = t_min + max(eta, sigma, 1.0)
    t = np.linspace(t_min, t_max, 2000)
    with np.errstate(over="ignore", under="ignore"):
        log_pdf = moyal.logpdf(t, loc=mpv, scale=eta)
    # clip to keep exponentiation stable
    log_pdf = np.clip(log_pdf, -700, 50)
    landau_pdf = np.exp(log_pdf)
    gauss = np.exp(-0.5 * ((x[:, None] - t[None, :]) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    conv = np.trapz(landau_pdf * gauss, t, axis=1)
    return amp * conv



def fit_gaussian(entries, bin_centers, p0=None, bounds=(-np.inf, np.inf)):
    # Get bin centers from edges with a prior and bounds

    if p0 is None:
        amp_guess = np.max(entries)
        mean_guess = bin_centers[np.argmax(entries)]
        sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

        popt, pcov = curve_fit(gaussian, bin_centers, entries,
                            p0=[amp_guess, mean_guess, sigma_guess], bounds=bounds)
    else:
        popt, pcov = curve_fit(gaussian, bin_centers, entries,
                            p0=p0, bounds=bounds)
    
    return popt, pcov


def fit_three_gaussians(entries, bin_centers):
    # Get bin centers from edges

    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(three_gaussians, bin_centers, entries,
                           p0=[amp_guess, mean_guess, sigma_guess, amp_guess/100, mean_guess-4, sigma_guess,  amp_guess/100, mean_guess+4, sigma_guess])
    
    return popt, pcov




class BeamAnalysis:
    def __init__(self, run_number, run_momentum, n_eveto, n_tagger, there_is_ACT5, output_dir = None, pdf_name=None, is_beam_paper_analysis = False, is_kaon_run = False):
        #Store the run characteristics
        self.run_number, self.run_momentum = run_number, run_momentum
        self.n_eveto, self.n_tagger = n_eveto, n_tagger
        self.there_is_ACT5 = there_is_ACT5
        self.is_kaon_run = is_kaon_run
        self.output_dir = output_dir
        
        if pdf_name is None:
            pdf_name = f"PID_run{run_number}_p{run_momentum}.pdf"
        if output_dir is None:
            output_dir = "."
        pdf_path = os.path.join(output_dir, pdf_name)
        self.pdf_global = PdfPages(pdf_path)
        self.channel_mapping = {12: "ACT0-L", 13: "ACT0-R", 14: "ACT1-L", 15: "ACT1-R", 16: "ACT2-L", 17: "ACT2-R", 18: "ACT3-L", 19: "ACT3-R", 20: "ACT4-L", 21: "ACT4-R", 22: "ACT5-L", 23: "ACT5-R"}
        print("Initialised the BeamAnalysis instance")
        print(f"Plots will be saved to {pdf_path}")

        self.is_beam_paper_analysis = is_beam_paper_analysis

        #Read the CERN jaw and magnet setting for this run, sometimes the logged value from the google sheet are wrong, this is lifted from the CERN database and more reliable. Some runs are missing because the jaw setting was modified halfway through the run, this won't matter for Physics analyses but will for beam papers. 
        df_jaw = pd.read_csv(
            "/eos/experiment/wcte/configuration/run_info/beamline_equipment_settings.csv",
            engine="python",          
            on_bad_lines="warn"      
        )

        self.jaw_setting = "Missing momentum slit opening info"

        self.momentum_slit_negative = -999
        self.momentum_slit_positive = -999

        self.acceptance_slit_negative = -999
        self.acceptance_slit_positive = -999


        df_jaw_filtered = df_jaw[df_jaw["run"] == run_number]
        if  not df_jaw_filtered.empty:
            

            self.momentum_slit_negative = df_jaw_filtered['T09.XCHV026:POS_JAW1_REF'].iloc[0]
            self.momentum_slit_positive = df_jaw_filtered['T09.XCHV026:POS_JAW2_REF'].iloc[0]

            self.acceptance_slit_negative = df_jaw_filtered['T09.XCHV011:POS_JAW1_REF'].iloc[0]
            self.acceptance_slit_positive = df_jaw_filtered['T09.XCHV011:POS_JAW2_REF'].iloc[0]

            if abs(self.momentum_slit_negative) == abs(self.momentum_slit_positive):
                self.jaw_setting = r"Collimator slit: $\pm$"f"{self.momentum_slit_positive} mm"
            else:
                self.jaw_setting = f"Collimator slit: [{df_jaw_filtered['T09.XCHV026:POS_JAW1_REF'].iloc[0]} - {df_jaw_filtered['T09.XCHV026:POS_JAW2_REF'].iloc[0]}]"
            if abs(self.acceptance_slit_negative) == abs(self.acceptance_slit_positive):
                self.jaw_setting += r", Acceptance slit: $\pm$"f"{self.acceptance_slit_positive} mm"
            else:
                self.jaw_setting += f", Acceptance slit: [{df_jaw_filtered['T09.XCHV011:POS_JAW1_REF'].iloc[0]} - {df_jaw_filtered['T09.XCHV011:POS_JAW2_REF'].iloc[0]}]"


        print(f"Run {self.run_number} has jaw setting {self.jaw_setting} ")
        
        
        
        
        
    @property
    def _plabels(self):
        """Particle label strings adjusted for beam polarity and kaon-run flag."""
        pos = self.run_momentum > 0
        return {
            "e":    "Positron"         if pos else "Electron",
            "es":   "Positrons"        if pos else "Electrons",
            "mu":   "Kaon"             if self.is_kaon_run else "Muon",
            "mus":  "Kaons"            if self.is_kaon_run else "Muons",
            "pi":   "Pion",
            "pis":  "Pions",
            "p":    "Proton"           if pos else "Antiproton",
            "ps":   "Protons"          if pos else "Antiprotons",
            "D":    "Deuteron",
            "Ds":   "Deuterons",
            "He3":  "Helium-3",
            "He3s": "Helium-3 nuclei",
        }

    def end_analysis(self):
        self.pdf_global.close()
        
    def open_file(self, n_events = -1, require_t5 = False, first_tdc_only = True, enforce_tdc_qdc_match = True, input_file = None, output_file = None):
        '''Read in the data as a pandas dataframe, read in the TOF and the ACt information'''
        
        self.require_t5_hit = require_t5
        
        if input_file is None:
            raise ValueError("input_file must be specified")
        file_path = input_file
        tree_name    = "WCTEReadoutWindows"

        # Open the file and grab the tree
        f    = uproot.open(file_path)
        tree = f[tree_name]

        # Load all four branches into NumPy arrays
        branches = [
            "beamline_pmt_tdc_times",
            "beamline_pmt_tdc_ids",
            "beamline_pmt_qdc_charges",
            "beamline_pmt_qdc_ids",
            "spill_counter"
        ]

        #low number of entries for testing
        if n_events == -1:
            data = tree.arrays(branches, library="np")
        else:
            data = tree.arrays(branches, library="np", entry_start = 0, entry_stop = n_events)
        
        
        #read the calibration file
        with open('../include/1pe_calibration.json', 'r') as file:
            calib_constants = json.load(file)

        # Access the calibration constants
        calibration = calib_constants["BeamCalibrationConstants"][0]
       
        
        calib_map = {
            ch: (gain, ped)
            for ch, gain, ped in zip(
                calibration["channel_id"],
                calibration["gain_value"],
                calibration["pedestal_mean"]
            )
        }
        

        # here we define a factor*nominal momentum which sets where the cut lines will fall for the ehavier particles, they should be well between positions of TOF peaks
        if abs(self.run_momentum) > 1800:
            factor = 1.05
        elif abs(self.run_momentum) > 1400:
            factor = 1.075
        elif abs(self.run_momentum) > 1000:
            factor = 1.1
        else:
            factor = 1.15


        


        #Define the cut lines in terms of the expected tof
        _, _, proton_tof_cut_array, _, _ =self.give_theoretical_TOF("Protons", np.array([abs(self.run_momentum) * factor])) #the smallest TOF value for a proton, considered to be the TOF at 110% of the theoretical proton momentum
        _, _, deuteron_tof_cut_array, _, _ =self.give_theoretical_TOF("Deuteron", np.array([abs(self.run_momentum) * factor])) #the smallest TOF value for a deuteron, considered to be the TOF at 110% of the theoretical proton momentum



        #Implement a TOF cut based on theoretical values for Helium3 as well
        _, _, helium3_tof_cut_array, _, _ =self.give_theoretical_TOF("Helium3", np.array([abs(self.run_momentum) * 2 * factor])) #the smallest TOF value for a helium3, considered to be the TOF at 110% of the theoretical helium3 momentum which is twice the beamline momentum since the helium3 has twice the charge

        




        electron_tof_offset = 2*0.626 #ns

        self.proton_tof_cut = proton_tof_cut_array[0] - electron_tof_offset
        self.deuteron_tof_cut = deuteron_tof_cut_array[0] - electron_tof_offset
        self.helium3_tof_cut = helium3_tof_cut_array[0] - electron_tof_offset

        if np.isnan(self.proton_tof_cut):
            self.proton_tof_cut = 9999 #default value in case the theoretical TOF cannot be calculated, this will ensure that no protons are identified
        if np.isnan(self.deuteron_tof_cut):
            self.deuteron_tof_cut = 9999 #default value in case the theoretical TOF cannot be calculated, this will ensure that no deuterons are identified
        if np.isnan(self.helium3_tof_cut):
            self.helium3_tof_cut = 9999 #default value in case the theoretical TOF cannot be calculated, this will ensure that no helium3 are identified

        
        
        print(f"At {self.run_momentum} MeV/c, the T0-T1 TOF limit used to identify protons is {self.proton_tof_cut:.2f}ns, and  to ID deuterons is {self.deuteron_tof_cut:.2f}ns.")
        print(f"At {self.run_momentum} MeV/c, the T0-T1 TOF limit used to identify helium3 is {self.helium3_tof_cut:.2f}ns.")
        
        nEvents = len(data["beamline_pmt_qdc_ids"])
        
        #Read all the entries
        act0_l, act1_l, act2_l, act3_l, act4_l, act5_l =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        act0_r, act1_r, act2_r, act3_r, act4_r, act5_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        charge_t0_0, charge_t0_1, charge_t0_2, charge_t0_3, charge_t1_0, charge_t1_1 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        charge_t1_2, charge_t1_3, charge_t4_0, charge_t4_1 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        #In VME data the charge of T5 wasn't read out, it's therefore not meaningful
       

        time_t0_0, time_t0_1, time_t0_2, time_t0_3, time_t1_0, time_t1_1, time_t1_2, time_t1_3, time_t4_0, time_t4_1   =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),      np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        act0_time_l, act0_time_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        mu_tag_l, mu_tag_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        is_kept = np.full(nEvents, np.nan, dtype=float)
        is_kept_event_id =  np.full(nEvents, np.nan, dtype=float) #keeping track of which events we want to keep

        #also save the time of flight information
        t0_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t1_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t4_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t0_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t1_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t4_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t5_avgs  =  np.full(nEvents, np.nan, dtype=float)
        tof_vals =  np.full(nEvents, np.nan, dtype=float)
        t4_l_array =  np.full(nEvents, np.nan, dtype=float)
        t4_r_array =  np.full(nEvents, np.nan, dtype=float)
        t4_l_array_second_hit =  np.full(nEvents, np.nan, dtype=float)
        t4_r_array_second_hit =  np.full(nEvents, np.nan, dtype=float)
        tof_t0t4_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t4t1_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t0t5_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t1t5_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t4t5_vals =  np.full(nEvents, np.nan, dtype=float)
        
        act_eveto_sums =  np.full(nEvents, np.nan, dtype=float)
        act_tagger_sums =  np.full(nEvents, np.nan, dtype=float)
        event_id =  np.full(nEvents, np.nan, dtype=float)
        ref0_times =  np.full(nEvents, np.nan, dtype=float)
        ref1_times =  np.full(nEvents, np.nan, dtype=float)
        evt_quality_bitmask =  np.full(nEvents, np.nan, dtype=float)
        digi_issues_bitmask =  np.full(nEvents, np.nan, dtype=float)
        
        #Also save the spill number for that event
        spill_number =  np.full(nEvents, np.nan, dtype=float)
        
        #save the reference bitmap for event quality flags:
        self.reference_flag_map = {
            "event_q_t0_or_t1_missing_tdc": 0,
            "event_q_t4_missing_tdc": 1,
            "event_q_t5_missing_tdc": 2,
            "event_q_hc_hit": 3,
            "event_q_t4_missing_qdc": 4,
            "t5_more_than_one_hit": 5
        }
        
        self.digitisation_issues_flag_map = {
            "qdc_failure": 0,
            "missing_digitiser_times": 1,
        }
        
        
        
        #Modify the code to implement Bruno's T5 cut and general improvement to the event quality cuts, also adding comments
        #default dict is similar to a normal dict with some additional features
        channel_clean_event_counts = defaultdict(int)
        
        #This dict counts the amount of times the tdc/qdc failed 
        tdc_qdc_failure_counts = {
            "qdc_failure": 0,
            "events_skipped": 0,
        }
        
        #the list of required signals to pass the event selection code
        required_groups = [
            {"name": "t0_group", "channels": t0_group, "mode": "all", "check_qdc": True},
            {"name": "t1_group", "channels": t1_group, "mode": "all", "check_qdc": True},
            {"name": "t4_group", "channels": t4_group, "mode": "all", "check_qdc": True},
        ]
        
        #always require T5
        required_groups.append(
            {
                "name": "t5_group",
                "channels": t5_total_group,
                "mode": "any_pair",
                "check_qdc": False,
            }
        )

        #initialise a list that will hold which T5 group(s) are being hit
        t5_bar_multiplicity = [] #if require_t5 else None
        
        #Initialise the progress bar
        nEvents = len(data[branches[0]])
        
        pbar = tqdm(total=nEvents, desc="Reading in events", unit="evt")
        
        t4_qdc_samples = {ch: [] for ch in t4_group}

        for evt_idx in range(nEvents):
            if evt_idx % 100 == 0:
                pbar.update(100)
            
            keep_event = True
            
            #convert the info stored in qdc_ids and charge to be at least 1D (converting scalars to 1D lists)
            qdc_ids_evt = np.atleast_1d(data["beamline_pmt_qdc_ids"][evt_idx])
            qdc_vals_evt = np.atleast_1d(data["beamline_pmt_qdc_charges"][evt_idx])
            
            #if this event does not have any qdc entry, save that
            event_q_no_qdc_entry = (qdc_ids_evt.size == 0)
            
            #read in the data
            tdc_times = data["beamline_pmt_tdc_times"][evt_idx]
            tdc_ids   = data["beamline_pmt_tdc_ids"][evt_idx]
            qdc_charges = data["beamline_pmt_qdc_charges"][evt_idx]
            qdc_ids     = data["beamline_pmt_qdc_ids"][evt_idx]

            MAX_CH = 128  #modify the dictionary based data hadnling to array

            corrected = np.full(MAX_CH, np.nan)
            corrected_second_hit = np.full(MAX_CH, np.nan)
            qdc_vals  = np.full(MAX_CH, np.nan)
            pe_vals   = np.full(MAX_CH, np.nan)
            
            # reference-time subtraction & first-hit only
            mask0 = (tdc_ids == reference_ids[0])
            mask1 = (tdc_ids == reference_ids[1])
            missing_digitiser_times = False
            if not mask0.any() or not mask1.any():
                keep_event = False
                ref0 = 0
                ref1 = 0
                missing_digitiser_times = True
            else:
                ref0 = tdc_times[mask0][0]
                ref1 = tdc_times[mask1][0]
                
            ref0_times[evt_idx] = ref0
            ref1_times[evt_idx] = ref1


            for ch, t in zip(tdc_ids, tdc_times):
                #do not store the information for the reference PMT and do not add the time if we already have an entry for that specific channel (That should be taking care of the case where more than one TDC is recorded) 
#                 if ch in reference_ids or ch in corrected:
                if ch in reference_ids:
                    continue
            
                if not np.isnan(corrected[ch]) and np.isnan(corrected_second_hit[ch]):
                    #if we already have a hit in the corrected time 
                    reference_time = ref0 if ch <= reference_ids[0] else ref1
                    corrected_second_hit[ch] = t - reference_time
                    continue
        
                reference_time = ref0 if ch <= reference_ids[0] else ref1
                corrected[ch] = t - reference_time
            
            for ch, q in zip(qdc_ids, qdc_charges):
                if np.isnan(qdc_vals[ch]):
                    #keeping the first charge only
                    qdc_vals[ch] = q
                    if (ch in act_eveto_group) or (ch in act_tagger_group):  
                        #use mean gain and mean pedestal, find it in the calibration data base
#                         calib_index = calibration["channel_id"].index(ch)
#                         gain = calibration["gain_value"][calib_index]
#                         pedestal = calibration["pedestal_mean"][calib_index]
                        gain, pedestal = calib_map[ch]
                        pe_vals[ch] = (q-pedestal)/gain 
            

            
            event_q_t0_or_t1_missing_tdc = False
            # require all channels on T0/T1 before computing averages
#             if not all(ch in corrected for ch in t0_group+t1_group):
            if not all(not np.isnan(corrected[ch]) for ch in t0_group+t1_group):
                keep_event = False
                event_q_t0_or_t1_missing_tdc = True
                t0 = None
                t1 = None
                t0_second_hit = None
                t1_second_hit = None
                
            else:
#                 t0 = np.mean([corrected[ch] for ch in t0_group])
                #Only include in the average the channels for which the hits arrive within 20ns of each other, we need not to include the case where one of the PMTs's time is those of an afterpulse or second bunch about 300ns later   
                vals = [corrected[ch] for ch in t0_group]
                #this is the median time of the t0 group, we are using it as a reference to select which channels to include in the average
                t0_median = np.median(vals)
                t0_within_20ns = [corrected[ch] for ch in t0_group if abs(corrected[ch] - t0_median) < 20]
                t0 = sum(t0_within_20ns) / len(t0_within_20ns) if t0_within_20ns else None
#                 if t0 is None:
#                     print("t0 is none because the values are: ", vals, "the median is ", t0_median)

               #repeat for t1
                vals = [corrected[ch] for ch in t1_group]
                t1_median = np.median(vals)
                t1_within_20ns = [corrected[ch] for ch in t1_group if abs(corrected[ch] - t1_median) < 20]
                t1 = sum(t1_within_20ns) / len(t1_within_20ns) if t1_within_20ns else None
#                 if t1 is None:
#                     print("t1 is none because the values are: ", vals, "the median is ", t1_median)

                vals = [corrected_second_hit[ch] for ch in t0_group]
                t0_second_hit = sum(vals) / len(vals)
                
                vals = [corrected_second_hit[ch] for ch in t1_group]
                t1_second_hit = sum(vals) / len(vals)

                #in case there are no PMTs within 20ns of the median, we are considering that the t0/t1 time is not reliable and we are skipping the event
                if t0 is None or t1 is None:
                    keep_event = False
                    event_q_t0_or_t1_missing_tdc = True
                

           
            #require already that there is a hit in all t4 PMTs
            #and otherwise that both hits are above threshold
            event_q_t4_missing_tdc = False
            event_q_t4_below_thres = False
            if not all(not np.isnan(corrected[ch]) for ch in t4_group):
                keep_event = False
                event_q_t4_missing_tdc = True
                t4 = None
                t4_l = None
                t4_r = None
                
                t4_second_hit = None
                t4_l_second_hit = None 
                t4_r_second_hit = None
                
            else:
                vals = [corrected[ch] for ch in t4_group]
                t4 = sum(vals) / len(vals)
                t4_l = corrected[t4_group[0]] #if not np.isnan(corrected[t4_group[0]]) else None
                t4_r = corrected[t4_group[1]] #if not np.isnan(corrected[t4_group[1]]) else None
                
                vals = [corrected_second_hit[ch] for ch in t4_group]
                t4_second_hit = sum(vals) / len(vals)
                
                t4_l_second_hit = corrected_second_hit[t4_group[0]] #if not np.isnan(corrected[t4_group[0]]) else None
                t4_r_second_hit = corrected_second_hit[t4_group[1]]
                
                
            event_q_t4_missing_qdc = False
            if not all(not np.isnan(qdc_vals[ch]) for ch in t4_group):
                keep_event = False
                event_q_t4_missing_qdc = True
            
            #compute the T5 mean time based on the bar that we have selected
            t5_bar_means = []
            for ch0, ch1 in t5_total_group:
                v0 = corrected[ch0]
                v1 = corrected[ch1]
                #checking if we have a second hit in the T5
                w0 = corrected_second_hit[ch0]
                w1 = corrected_second_hit[ch1]
                #implementing a 10ns coincidence between the two PMTs on the same bar
                if not np.isnan(v0) and not np.isnan(v1) and abs(v0-v1) < 10:
                    t5_bar_means.append(0.5 * (v0 + v1))

                if not np.isnan(w0) and not np.isnan(w1) and abs(w0-w1) < 10:
                    t5_bar_means.append(0.5 * (w0 + w1))
                 
            #in case there is no pairs of t5 hits 
            event_q_t5_missing_tdc = (len(t5_bar_means) == 0)
            
            if require_t5 and event_q_t5_missing_tdc:
                keep_event = False
            
            # instead of using the mean, take the erliest hit. 
            t5_earliest_time = min(t5_bar_means) if t5_bar_means else None
            t5_more_than_one_hit = (len(t5_bar_means)>1)
            
            if t5_more_than_one_hit:
                keep_event = False
            #the multiplicity shows how many bars are being hit, this could be a useful way to identify multiple particle events
#             if t5_bar_multiplicity is not None:
            t5_bar_multiplicity.append(len(t5_bar_means))
                

            #--------- HC cut ----------
            event_q_hc_hit = False
            for ch in hc_group:
                if qdc_vals[ch] >= hc_charge_cut:
                    keep_event = False
                    event_q_hc_hit = True

            #Keep all of the entries but then df is only the ones that we keep 
            t0_avgs[evt_idx] = t0
            t1_avgs[evt_idx] = t1
            t4_avgs[evt_idx] = t4
            
            t0_avgs_second_hit[evt_idx] = t0_second_hit
            t1_avgs_second_hit[evt_idx] = t1_second_hit
            t4_avgs_second_hit[evt_idx] = t4_second_hit
            
            t4_l_array[evt_idx] = t4_l
            t4_r_array[evt_idx] = t4_r
            
            t4_l_array_second_hit[evt_idx] = t4_l_second_hit
            t4_r_array_second_hit[evt_idx] = t4_r_second_hit
            
            t5_avgs[evt_idx] = t5_earliest_time
            
            tof        = None
            tof_t0t5   = None
            tof_t1t5   = None
            tof_t0t4   = None
            tof_t4t1   = None
            tof_t4t5   = None

            if not event_q_t0_or_t1_missing_tdc:
                tof = t1 - t0

                if t5_earliest_time is not None:
                    tof_t0t5 = t5_earliest_time - t0
                    tof_t1t5 = t5_earliest_time - t1

                if not event_q_t4_missing_tdc:
                    tof_t0t4 = t4 - t0
                    tof_t4t1 = t1 - t4
                    tof_t4t5 = (
                        t5_earliest_time - t1
                        if t5_earliest_time is not None
                        else None
                    )

            tof_vals[evt_idx] = tof
            tof_t0t5_vals[evt_idx] = tof_t0t5
            tof_t1t5_vals[evt_idx] = tof_t1t5
            tof_t0t4_vals[evt_idx] = tof_t0t4
            tof_t4t1_vals[evt_idx] = tof_t4t1
            tof_t4t5_vals[evt_idx] = tof_t4t5
                

            #svae the charge 
            act0_l[evt_idx] = pe_vals[12]
            act0_r[evt_idx] = pe_vals[13]
            act1_l[evt_idx] = pe_vals[14]
            act1_r[evt_idx] = pe_vals[15]
            act2_l[evt_idx] = pe_vals[16]
            act2_r[evt_idx] = pe_vals[17]
            act3_l[evt_idx] = pe_vals[18]
            act3_r[evt_idx] = pe_vals[19]
            act4_l[evt_idx] = pe_vals[20]
            act4_r[evt_idx] = pe_vals[21]
            act5_l[evt_idx] = pe_vals[22]
            act5_r[evt_idx] = pe_vals[23]

            #add the charge for the T0, T1 and T4 PMTs 
            charge_t0_0[evt_idx] = qdc_vals[0]
            charge_t0_1[evt_idx] = qdc_vals[1]
            charge_t0_2[evt_idx] = qdc_vals[2]
            charge_t0_3[evt_idx] = qdc_vals[3]
            charge_t1_0[evt_idx] = qdc_vals[4]
            charge_t1_1[evt_idx] = qdc_vals[5]   
            charge_t1_2[evt_idx] = qdc_vals[6]
            charge_t1_3[evt_idx] = qdc_vals[7]
            charge_t4_0[evt_idx] = qdc_vals[42]
            charge_t4_1[evt_idx] = qdc_vals[43]

            time_t0_0[evt_idx], time_t0_1[evt_idx], time_t0_2[evt_idx], time_t0_3[evt_idx], time_t1_0[evt_idx], time_t1_1[evt_idx], time_t1_2[evt_idx], time_t1_3[evt_idx], time_t4_0[evt_idx], time_t4_1[evt_idx] =  corrected[0], corrected[1], corrected[2], corrected[3], corrected[4], corrected[5], corrected[6], corrected[7], corrected[42], corrected[43]


            
            spill_id = data["spill_counter"][evt_idx]
            spill_number[evt_idx] = spill_id

            event_id[evt_idx] = evt_idx


            mu_tag_l[evt_idx] = qdc_vals[24]        
            mu_tag_r[evt_idx] = qdc_vals[25]

            act0_time_l[evt_idx] = corrected[12]
            act0_time_r[evt_idx] = corrected[13]
                
            is_kept[evt_idx] = keep_event

            flags = {
                "event_q_t0_or_t1_missing_tdc": event_q_t0_or_t1_missing_tdc,
                "event_q_t4_missing_tdc": event_q_t4_missing_tdc,
                "event_q_t5_missing_tdc": event_q_t5_missing_tdc,
                "event_q_hc_hit": event_q_hc_hit,
                "event_q_t4_missing_qdc": event_q_t4_missing_qdc,
                "t5_more_than_one_hit": t5_more_than_one_hit
            }
            
            bitmask = write_event_quality_mask(flags, self.reference_flag_map)
            evt_quality_bitmask[evt_idx] = bitmask
            
            flags_digi = {
                "qdc_failure": event_q_no_qdc_entry,
                "missing_digitiser_times": missing_digitiser_times,
               
            }
            
            #the bitmask that handles issues related to digitisation 
            bitmask_digitisation = write_event_quality_mask(flags_digi, self.digitisation_issues_flag_map)
            digi_issues_bitmask[evt_idx] = bitmask_digitisation 
            
            
            
            #here we are obtaining the bitmask corresponding to this specific set of flags 

            #End of the loop over events  
        act_arrays = [act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
                      act0_r, act1_r, act2_r, act3_r, act4_r, act5_r]

        act_arrays = [np.array(arr, dtype=float) for arr in act_arrays]

        act0_time_l = np.array(act0_time_l)
        act0_time_r = np.array(act0_time_r)

        #store them back in their initial name
        (act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
         act0_r, act1_r, act2_r, act3_r, act4_r, act5_r) = act_arrays

        
        mu_tag_l, mu_tag_r = np.array(mu_tag_l), np.array(mu_tag_r) 
        
        is_kept = np.array(is_kept)

        
        data_dict = {
            "t0_time": t0_avgs,
            "t1_time": t1_avgs,
            "t4_time": t4_avgs,
            "t0_time_second_hit": t0_avgs_second_hit,
            "t1_time_second_hit": t1_avgs_second_hit,
            "t4_time_second_hit": t4_avgs_second_hit,
            "time_t0_0": np.array(time_t0_0),
            "time_t0_1": np.array(time_t0_1),
            "time_t0_2": np.array(time_t0_2),    
            "time_t0_3": np.array(time_t0_3),
            "time_t1_0": np.array(time_t1_0),
            "time_t1_1": np.array(time_t1_1),
            "time_t1_2": np.array(time_t1_2),
            "time_t1_3": np.array(time_t1_3),
            "time_t4_0": np.array(time_t4_0),
            "time_t4_1": np.array(time_t4_1),
            "t5_time": t5_avgs,
            "t4_l": t4_l_array,
            "t4_r": t4_r_array,
            "t4_l_second_hit": t4_l_array_second_hit,
            "t4_r_second_hit": t4_r_array_second_hit,
            "act0_l": act0_l,
            "act1_l": act1_l,
            "act2_l": act2_l,
            "act3_l": act3_l,
            "act4_l": act4_l,
            "act5_l": act5_l,
            "act0_r": act0_r,
            "act1_r": act1_r,
            "act2_r": act2_r,
            "act3_r": act3_r,
            "act4_r": act4_r,
            "act5_r": act5_r,
            "charge_t0_0": np.array(charge_t0_0),
            "charge_t0_1": np.array(charge_t0_1),
            "charge_t0_2": np.array(charge_t0_2),
            "charge_t0_3": np.array(charge_t0_3),
            "charge_t1_0": np.array(charge_t1_0),
            "charge_t1_1": np.array(charge_t1_1),
            "charge_t1_2": np.array(charge_t1_2),
            "charge_t1_3": np.array(charge_t1_3),
            "charge_t4_0": np.array(charge_t4_0),
            "charge_t4_1": np.array(charge_t4_1),
            "event_id":event_id,
#             "total_TOF_charge":total_TOF_charge,
            "act0_time_l": act0_time_l,
            "act0_time_r": act0_time_r,
            "tof": tof_vals,
            "tof_t0t4": tof_t0t4_vals,
            "tof_t4t1": tof_t4t1_vals,
            "tof_t0t5": tof_t0t5_vals,
            "tof_t1t5": tof_t1t5_vals,
            "tof_t4t5": tof_t4t5_vals,
            "mu_tag_l": mu_tag_l,
            "mu_tag_r": mu_tag_r,
            "is_kept": is_kept,
            "ref0_time":ref0_times,
            "ref1_time":ref1_times,
            "spill_number":spill_number,
            "evt_quality_bitmask":evt_quality_bitmask,
            "digi_issues_bitmask":digi_issues_bitmask
          
        }
        
        # create DataFrame, much more robust than having many arrays 
        self.df_all = pd.DataFrame(data_dict)

        
        #add the combined branches that can be useful
        self.df_all["mu_tag_total"] = self.df_all["mu_tag_l"] + self.df_all["mu_tag_r"]
        
        if self.there_is_ACT5:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r", "act5_l", "act5_r"]
           
        else:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r"]
            
        
        #here make the subset sample that we are keeping for analysis:
        if self.require_t5_hit:
            self.df = self.df_all[self.df_all["evt_quality_bitmask"] == 0].copy()
        else:
            self.df = self.df_all[(self.df_all["evt_quality_bitmask"] == 0) | (self.df_all["evt_quality_bitmask"] == 4)].copy()
            
        #need to make sure that the digitisation issues are also included !!
        self.df = self.df[self.df["digi_issues_bitmask"] == 0]
        
        print(len(self.df), sum(self.df_all["is_kept"]))            
        self.run_analysis = True
            
            
        print(f"\n \n When T5 requirement is {self.require_t5_hit} the fraction of events kept for analysis is {len(self.df)/len(self.df_all) * 100}% \n \n")
        
        if len(self.df) == 0:
            #implement default values so we do not have a crash when trying to run 
            self.eveto_cut = -999
            self.act35_cut_pi_mu = -999
            self.mu_tag_cut = -999
            self.using_mu_tag_cut = 0
            
            for is_particle in ["is_electron", "is_muon", "is_pion", "is_proton", "is_deuteron", "is_helium3"]:
                self.df[is_particle] = False
            #plot the bitmask errors
            self.plot_event_quality_bitmask()
            
            self.output_to_root(output_file, scalar_info = False)
            self.end_analysis()
            #avoid throwing an exception, exiting instead 
            #raise Exception("No triggers are accepted for analysis, exiting early")
            
            print("No triggers accepted for analysis, exiting the beam analysis code")
            self.run_analysis = False
        
        #this will be necessary for identifying events later
        self.is_kept = is_kept
        
        #store the id of the events of interest
        self.is_kept_event_id = is_kept_event_id                          
        pbar.close()
        
        
    def adjust_1pe_calibration(self):

        bins = np.linspace(-1,3, 100)
        
        self.PMT_value_ped = []
        self.PMT_1pe_scale = []
        for PMT in self.PMT_list:
            fig, ax = plt.subplots(figsize = (8, 6))    
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "Config-file 1pe calib")

            index_ped = np.argmax(h)
            value_ped = 0.5 * (bins[index_ped] + bins[index_ped+1])
            #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            #note that df and df_all are two separate dataframes so the fix must be applied to both 
            self.df[PMT] -= value_ped 
            self.df_all[PMT] -= value_ped 
#             self.PMT_value_ped.append(value_ped)
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ pedestal shift")
            #plot the pedestal and 1 pe peak, for now reading from the pre-made calibration, eventually, will have to be made run 
            #check the position of the 1pe peak 
            #restict the portion to fit
            xmin = 0.5
            xmax = 1.5
            bin_centers = (bins[:-1] + bins[1:]) / 2
            is_onepe_region = (bin_centers >= xmin) & (bin_centers <= xmax)
            bins_onepe = bin_centers[is_onepe_region]
            entries_onepe = h[is_onepe_region]

            #fit a gaussian 
            try:
                popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
            except:
                print("We did not manage to fit the 1pe peak, we are therefore not correcting its position")
                popt = np.array([0, 1, 0])
            #ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")

            self.df[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            self.df_all[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            
            
            self.PMT_1pe_scale.append(popt[1])
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ charge scale")
            entries_onepe = h[is_onepe_region]
            try:
                popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
                ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")
            except:
                #in case we do not manage to fit the 1pe position, still do not crash the whole thing
                print("We did not manage to fit the 1pe peak")
            

            #now that we have the correct 1pe calibration, make the combination of ACT charges 
            self.df["act_eveto"] = self.df["act0_l"]+self.df["act0_r"]+self.df["act1_l"]+self.df["act1_r"]+self.df["act2_l"]+self.df["act2_r"]
            
            if self.there_is_ACT5:
                self.df["act_tagger"] = self.df["act3_l"]+self.df["act3_r"]+self.df["act4_l"]+self.df["act4_r"]+self.df["act5_l"]+self.df["act5_r"]
            
            else:
                self.df["act_tagger"] = self.df["act3_l"]+self.df["act3_r"]+self.df["act4_l"]+self.df["act4_r"]

            ax.set_yscale("log")
            ax.legend(fontsize = 16)
            ax.set_xlabel("Charge collected (PE)", fontsize = 18)
            ax.set_ylabel("Number of entries", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {PMT}", fontsize = 18)
            ax.grid()
            self.pdf_global.savefig(fig)
            plt.close()
            
        print("One PE calibration finished, please don't forget to check that it is correct")
        
        
        
    def tag_electrons_ACT02(self, tightening_factor = 0):
        '''Tagging the electrons based on the charge deposited in the upstream ACTs, add an additional scale factor to tighten the cut some more '''
        bins = np.linspace(0, 40, 200)
        fig, ax = plt.subplots(figsize = (8, 6))    
        h, _, _ = ax.hist(self.df["act_eveto"], bins = bins, histtype = "step")
        

        #automatically find the middle of the least populated bin within 4 and 15 PE
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 0.2

        if abs(self.run_momentum) < 800:
            x_max = 23
        elif abs(self.run_momentum) < 1200:
            x_max = 15
        else:
            x_max = 7
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        self.eveto_cut = bin_centers[index[min_index]]

        ax.axvline(self.eveto_cut, linestyle = '--', color = 'black', label = f'Optimal electron rejection veto: {self.eveto_cut:.1f} PE')
        
        

        
        if tightening_factor!=0:
            self.eveto_cut = self.eveto_cut * (1 - tightening_factor/100)
            ax.axvline(self.eveto_cut, linestyle = '--', color = 'red', label = f'with tightening factor ({tightening_factor}%): {self.eveto_cut:.1f} PE')
        ax.set_yscale("log")
        ax.set_xlabel("ACT0-2 total charge (PE)", fontsize = 18)
        ax.set_ylabel("Number of entries", fontsize = 18)
        ax.legend(fontsize = 16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)- ACT0-2", fontsize = 20)
        self.pdf_global.savefig(fig)
        plt.close()
        
        #make sure that the particle is not already a slow travelling particle, proton or deuterium
        if abs(self.run_momentum) > 350:
            self.df["is_electron"] = np.where(self.df["act_eveto"]>self.eveto_cut, (self.df["tof"]<self.proton_tof_cut), False)
        else:
            self.df["is_electron"] = (self.df["act_eveto"]>self.eveto_cut)
                                               
        n_electrons = sum(self.df["is_electron"])
        n_triggers = len(self.df["act_eveto"])
        
        if n_triggers > 0:
            print(f"A total of {n_electrons} electrons are tagged with ACT02 out of {n_triggers}, i.e. {n_electrons/n_triggers * 100:.1f}% of the dataset")
        else:
             print(f"A total of {n_triggers} triggers have been found")
        
        
    def tag_multiple_particle_events(self):
        '''Tagging the events with more than one particle, based on the presence of a higher charge than the Mip charge in T0, T1, T4 or T5.
        A diagnostic plot of the total T0 charge vs TOF is saved to the global PDF.'''
        
        # diagnostic plot: total T0 charge (channels 0..3) vs TOF
        self.df["charge_t0"] = self.df["charge_t0_0"]+ self.df["charge_t0_1"]+ self.df["charge_t0_2"]+ self.df["charge_t0_3"]
        self.df["charge_t1"] = self.df["charge_t1_0"]+ self.df["charge_t1_1"]+ self.df["charge_t1_2"]+ self.df["charge_t1_3"]
        
        fast_particle_cut = 21
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x_bins = np.linspace(10, 50, 100)
        y_bins = np.linspace(500, 28000, 200)
        h = ax.hist2d(self.df["tof"],self.df["charge_t0"], bins=(x_bins, y_bins), norm=LogNorm())
        
    
        ax.set_ylabel("Total T0 charge (QDC)", fontsize=18)
        ax.set_xlabel("TOF (ns)", fontsize=18)
        ax.set_title(
            f"Run {self.run_number} ({self.run_momentum} MeV/c) - T0 charge vs TOF",
            fontsize=20,
        )
        self.pdf_global.savefig(fig)
        plt.close()

        #for particle with TOF less than 17ns, plot a 1d histogram of the total T0 charge
        fig, ax = plt.subplots(figsize=(8, 6))  
        bins = np.linspace(10,  28000, 200)
        h, _, _ = ax.hist(self.df[self.df["tof"]<fast_particle_cut]["charge_t0"], bins = bins, histtype = "step")
     
        ax.set_xlabel("Total T0 charge (QDC)", fontsize=18)
        ax.set_title(
            f"Run {self.run_number} ({self.run_momentum} MeV/c) \n Total T0 charge for particles with TOF < 17ns",
            fontsize=20,
        )
        ax.grid()
        self.pdf_global.savefig(fig)
        plt.close()
        
        #Here plot the triggers which have a lot of charge deposited
        
        thresh_tot_charge = 8000
        
        df_fast = self.df[self.df["tof"]<fast_particle_cut]
        df_fast_high_charge = df_fast[df_fast["charge_t0"]>thresh_tot_charge]
        
        #is the amount of charge deposited in both the T0 and T1 is very large or just one of them?
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))  
        x_bins = np.linspace(0, 5000, 100)
        
        for i, ax in enumerate(axes):
            for PMT in range(4):
                
                ax.hist(df_fast_high_charge[f"charge_t{i}_{PMT}"], bins = x_bins, histtype = "step", label = f"PMT {PMT}")
            
            ax.set_xlabel(f"T{i} PMT charge (QDC)", fontsize=18)
            ax.legend(fontsize = 18)
            ax.set_title(
                f"Run {self.run_number} ({self.run_momentum} MeV/c)\n T0 charge [TOF < 17ns; T0_tot_charge > {thresh_tot_charge}]",
                fontsize=20,
            )
            ax.grid()
        
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()
        
        #is the amount of charge deposited in both the T0 and T1 is very large or just one of them?
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))  
        x_bins = np.linspace(0, 5000, 100)
        
        for i, ax in enumerate(axes):
            for PMT in range(4):
                
                ax.hist(df_fast[f"charge_t{i}_{PMT}"], bins = x_bins, histtype = "step", label = f"PMT {PMT}")
            
            ax.set_xlabel(f"T{i} PMT charge (QDC)", fontsize=18)
            ax.legend(fontsize = 18)
            ax.set_title(
                f"Run {self.run_number} ({self.run_momentum} MeV/c)\n T0 charge [All fast particles]",
                fontsize=20,
            )
            ax.grid()
        
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()
        
        #Check the charge deposited in the ACTs
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))  
        x_bins = np.linspace(-2, 20, 50)
        
       
        axes[0].hist(df_fast_high_charge[f"act_eveto"], bins = x_bins, histtype = "step", label = f"Fast particles depositing more than {thresh_tot_charge} in T0")
        axes[0].hist(df_fast[f"act_eveto"], bins = x_bins, histtype = "step", label = f"All fast particles")
            
        axes[0].set_xlabel(f"ACT0-2 total charge (PE)", fontsize=18)
        axes[0].legend(fontsize = 18)
        axes[0].grid()
        
        axes[1].hist(df_fast_high_charge[f"act_tagger"], bins = x_bins, histtype = "step", label = f"Fast particles depositing more than {thresh_tot_charge} in T0")
        axes[1].hist(df_fast[f"act_tagger"], bins = x_bins, histtype = "step", label = f"All fast particles")
            
        axes[1].set_xlabel(f"ACT3-5 total charge (PE)", fontsize=18)
        axes[1].legend(fontsize = 18)
        axes[1].grid()
        
        axes[0].set_title(
            f"Run {self.run_number} ({self.run_momentum} MeV/c)",
            fontsize=20)
        
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()
        

    def tag_electrons_ACT35(self, cut_line = 0):
        '''Tagging the electrons based on the charge deposited in the downstream ACTs, to 'clean up the edges'''
        
        n_electrons_initial = sum(self.df["is_electron"])
        
        #Here plot visually the electron cutline to check that it is correct
        self.plot_ACT35_left_vs_right(cut_line, "muon/electron")
        
        ### identify the particles above the cut as electrons
        #make sure that the particle is not already a slow travelling particle, proton or deuterium and that it stays an electron if it has already been identified by ACT02 but isn't above the cutline
        self.df["is_electron"] = np.where(self.df["act_tagger"]>cut_line, (self.df["tof"]<self.proton_tof_cut), self.df["is_electron"])
        
        self.act35_e_cut = cut_line

        

        
        n_electrons = sum(self.df["is_electron"])
        n_triggers = len(self.df["is_electron"])
        print(f"A total of {n_electrons-n_electrons_initial} additional electrons are tagged with ACT35, on top of the {n_electrons_initial} that were tagged with ACT02")
        
        
    def plot_ACT35_left_vs_right(self, cut_line = None, cut_line_label = None):
        if cut_line_label is None:
            cut_line_label = "pion/kaon" if self.is_kaon_run else "pion/muon"
        bins = np.linspace(0, 70, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        act_tagger_l = self.df["act3_l"]+self.df["act4_l"]+self.df["act5_l"] * int(self.there_is_ACT5)
        act_tagger_r =self.df["act3_r"]+self.df["act4_r"]+self.df["act5_r"]* int(self.there_is_ACT5)
        
        h = ax.hist2d(act_tagger_l, act_tagger_r, bins = (bins, bins),norm=LogNorm())
        fig.colorbar(h[3], ax=ax)
        if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
        ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
        ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\nACT3-5 all particles", fontsize = 20)
        self.pdf_global.savefig(fig)

        plt.close()
        

        try:
            not_electrons = ~self.df["is_electron"]

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons], act_tagger_r[not_electrons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT35 After eveto", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, electrons should be in the top right corner")
            
        except:
            print("Please make the electron selection using tag_electrons_ACT02 before checking the ACT35 left vs right plot, otherwise you will have all of the entries")
            return 0
        
        try:
            not_protons = ~self.df["is_proton"]
            not_protons = not_protons&(~self.df["is_deuteron"])

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons&not_protons], act_tagger_r[not_electrons&not_protons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT3-5 After eveto and p removal", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, protons should be in the bottom left corner")
            
        except:
            print("Please make the proton selection using tag_protons_TOF before checking the ACT35 left vs right plot to get the most of out it")
            return 0
        
        
        
    def plot_ACT02_left_vs_right(self):
        cut_line = self.eveto_cut
        cut_line_label = "electron veto cut"
        bins = np.linspace(0, 70, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        act_tagger_l = self.df["act0_l"]+self.df["act1_l"]+self.df["act2_l"]
        act_tagger_r =self.df["act0_r"]+self.df["act1_r"]+self.df["act2_r"]
        
        h = ax.hist2d(act_tagger_l, act_tagger_r, bins = (bins, bins),norm=LogNorm())
        
        try:
            fig.colorbar(h[3], ax=ax)
        except:
            print("Not adding colorbar, probably empty data")
        
        if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT0-2 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
        ax.set_xlabel("ACT0-2 left (PE)", fontsize = 18)
        ax.set_ylabel("ACT0-2 right (PE)", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\nACT0-2 all particles", fontsize = 20)
        
        try:
            self.pdf_global.savefig(fig)
        except:
            print("Not outputing figure of ACT35")

        plt.close()
        
        
        try:
            not_protons = ~self.df["is_proton"]
            not_electrons = ~self.df["is_electron"]
            not_protons = not_protons&(~self.df["is_deuteron"])

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons&not_protons], act_tagger_r[not_electrons&not_protons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT0-2 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT0-2 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT0-2 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT0-2 After eveto and p removal", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT02 left vs right plots have been made please check that they are sensible, protons should be in the bottom left corner")
            
        except:
            print("Please make the proton selection using tag_protons_TOF before checking the ACT02 left vs right plot to get the most of out it")
            return 0
        
    def tag_protons_TOF(self):
        '''Simple identification of the protons based on the time of flight, cutline fixed at 17.5ns for now, can be modified later if needed'''
        
        
        if abs(self.run_momentum) < 250:
            self.df["is_proton"] = False
            
        else:
            self.df["is_proton"] = np.where(self.df["tof"]>self.proton_tof_cut, self.df["tof"]<self.helium3_tof_cut, False)      
                
        
        if self.run_momentum < 350:
            self.df["is_helium3"] = False
            self.df["is_deuteron"] = False

        else:
            self.df["is_helium3"] = np.where(self.df["tof"]>self.helium3_tof_cut, self.df["tof"]<self.deuteron_tof_cut, False)
            self.df["is_deuteron"] = (self.df["tof"]>=self.deuteron_tof_cut)
                
                
        n_protons = sum(self.df["is_proton"]==True)
        n_deuteron = sum(self.df["is_deuteron"]==True)
        
        n_helium3 = sum(self.df["is_helium3"]==True)
        n_triggers = len(self.df["act_eveto"])
        
       
        
        if n_triggers > 0:
            print(f"A total of {n_protons} protons and {n_deuteron} deuterons nuclei are tagged using the TOF out of {n_triggers}, i.e. {n_protons/n_triggers * 100:.1f}% of the dataset are protons and {n_deuteron/n_triggers * 100:.1f}% are deuteron")
            print(f"A total of {n_helium3} helium3 nuclei, i.e. {n_helium3/n_triggers * 100:.2f}% of the dataset")
        
        else:
            print(f"A total of {n_triggers} triggers have been found")
        
        

        
        
    def tag_muons_pions_ACT35(self):
        '''Function to identify the muons and pions based on the charge deposited in ACT35, potentially using the muon tagger'''

        lbl = self._plabels

        #step 1: find the optimal cut line in the muon tagger and decide if it is useful to implement it (that is, in case there are still some non-electrons left after the cut
        bins = np.linspace(120, 800, 100)
           
        
        #make sure they are boolean first
        self.df["is_proton"] = self.df["is_proton"].astype(bool)
        self.df["is_electron"] = self.df["is_electron"].astype(bool)
        self.df["is_deuteron"] = self.df["is_deuteron"].astype(bool)
        self.df["is_helium3"] = self.df["is_helium3"].astype(bool)

        mu_tag_tot = self.df["mu_tag_l"]+self.df["mu_tag_r"]
        #cannot be any other particle already
        
        if self.is_beam_paper_analysis == False: # tightend electron veto cut
            if abs(self.run_momentum) > 300:
                muons_pions = (self.df["tof"] < self.proton_tof_cut) & (~self.df["is_electron"]) 
            else:
                muons_pions = (~self.df["is_electron"]) 
                
        else:
            #optionally for the beam paper analysis we can tighten further the eveto cut to have a purer muon and pion sample, this is for momentum studies at higher momentum, not necessary for most analyses
            tighter_ACT35_cut = self.eveto_cut - (4 *(abs(self.run_momentum)/810)) 
            
            print("we are applying tighter cut: ", tighter_ACT35_cut, " PE")
                  
            if abs(self.run_momentum) > 300:
                muons_pions = (self.df["tof"] < self.proton_tof_cut) & (self.df["act_eveto"]<tighter_ACT35_cut) 
            else:
                muons_pions = (self.df["act_eveto"]<tighter_ACT35_cut) 
            
            
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(mu_tag_tot, bins = bins, label = 'All particles', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_electron"]], bins = bins, label = 'Electrons', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_proton"]], bins = bins, label = 'Protons', histtype = "step")
        h, _, _ = ax.hist(mu_tag_tot[muons_pions], bins = bins, label = f'{lbl["mus"]} and pions', histtype = "step")
        ax.set_xlabel(f"Total charge in muon-tagger (QDC)", fontsize = 18)
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - Muon Tagger charge", fontsize = 20)


        #implement automatic muon tagger cut
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 150
        x_max = 300
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        mu_tag_cut = bin_centers[index[min_index]]
        
        #minimum fraction of muons and pions that have to be above the mu tag threshold (set to 0.5%)
        min_fraction_above_cut = 0.005
        
        n_electrons_above_cut = np.sum(self.df["is_electron"][mu_tag_tot>mu_tag_cut])
        n_muons_pions_above_cut = np.sum(muons_pions[mu_tag_tot>mu_tag_cut])
        self.n_muons_pions = np.sum(muons_pions)

        ax.axvline(mu_tag_cut, color = "k", linestyle = "--", label = f"Muon tagger cut: {mu_tag_cut:.1f} QDC \n {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all {lbl['mus'].lower()} and pions are above cut")
        ax.legend(fontsize = 16)

        ax.set_yscale("log")
        self.pdf_global.savefig(fig)
        plt.close()
        
       
        print(f"The muon tagger charge has been plotted. The optimal cut line is at {mu_tag_cut:.1f} a.u., there are {n_electrons_above_cut} electrons above the cut line and {n_muons_pions_above_cut} muons and pions, i.e. {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all muons and pions ({self.n_muons_pions})...")
        
        
        self.mu_tag_cut = mu_tag_cut
        
        bins = np.linspace(0, 80, 100)
        bins_act35 = np.linspace(0, 50, 100)
        
        if n_muons_pions_above_cut/self.n_muons_pions> min_fraction_above_cut:
            print(f"there are more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are applying it. (Please verify this on the plots)")
            
            self.using_mu_tag_cut = True
                  
            #apply the cut on the muon tagger and then find the optimal ACT35 cut (based on pions and muons)
            fig, ax = plt.subplots(figsize = (8, 6))
            
            electron_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (self.df["is_electron"])
            muons_pions_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (muons_pions)
            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions_above_mu_tag], bins = bins, label = f"{lbl['mus']}/pions passing muon tagger cut", histtype = "step")
            ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = f"All {lbl['mus'].lower()}/pions", histtype = "step")
            
            
            ax.hist(self.df["act_tagger"][electron_above_mu_tag], 
                    bins = bins, label = "Electrons passing muon tagger cut", histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 0.5
            x_max = 10
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/{lbl['mu'].lower()} cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_yscale("log")
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 with mu-tag cut", fontsize = 20)
            self.pdf_global.savefig(fig)
            plt.close()
            
            ############## Estimate the likelihood using the muon tagger cut distribution

            fig, ax = plt.subplots(figsize = (8, 6))
            bins = np.linspace(0, 80, 110)

            #make the event masks to identify events which pass the mu tag cut and are muons and pions
            if abs(self.run_momentum) > 300:
                mask_muons_pions = (self.df["is_electron"] == 0) & (self.df["tof"] < self.proton_tof_cut)
            else:
                mask_muons_pions = (self.df["is_electron"] == 0)
            mask_pass_mu_tag = (self.df["mu_tag_total"] > self.mu_tag_cut)
            #both (muons or pion) and passing muon tag 
            mask_both = mask_muons_pions & mask_pass_mu_tag


            h, _, _ = ax.hist(self.df["act_tagger"][mask_both],  histtype = "step", bins = bins, label = f"All {lbl['mu']} or pions above mu_tag cut ({sum(mask_both)}) events")
            h_all, _, _ = ax.hist(self.df["act_tagger"][mask_muons_pions],  histtype = "step", bins = bins, label = f"All {lbl['mu']} or pions: ({sum(mask_muons_pions)}) events")
            

            #Weight up the events passing the muon tagger cut so the maximas align (the muon peak)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 10
            x_max = 20
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the maximum count in that range
            max_index = np.argmax(h_all[mask])

            index = np.where(mask)[0]
            idx_muon_peak = index[max_index]
            muon_scale = h_all[idx_muon_peak]/h[idx_muon_peak]

            #create a new histogram of the distibution passing the muon tagger cut scaled up to the muon peak
            h_muon_scaled = h * muon_scale
            
            
                  
            # plot the scaled histogram 
            ax.step(bin_centers, h_muon_scaled, where='mid', label=f"Mu/pi above cut scaled to {lbl['mu'].lower()} peak ({sum(h_muon_scaled):.1f}) events")
            
            ### look at electrons
            n_electrons = sum(self.df["is_electron"])
            h_electron, _, _ =  ax.hist(self.df["act_tagger"][self.df["is_electron"] == 1],  histtype = "step", bins = bins, label = f"Tagged electrons ({n_electrons:.1f}) events")
            
            
            #### here, get the difference between the all and the scaled one
            h_all_minus_h_scaled = h_all - h_muon_scaled
            
            
             #clip that difference to be positive
            h_all_minus_h_scaled = h_all_minus_h_scaled.clip(0)
            
            #split the leftovers into electron (after the mu peak) and pions (before the muon peak)
            h_all_minus_h_scaled_pion = np.where(bin_centers<bin_centers[idx_muon_peak], h_all_minus_h_scaled, 0)
            
            h_all_minus_h_scaled_electron = np.where(bin_centers>=bin_centers[idx_muon_peak], h_all_minus_h_scaled, 0)
            
            ax.step(bin_centers, h_all_minus_h_scaled_pion,  where = 'mid', label = f"{lbl['pi']}-like distribution")

            
            h_pion = h_all_minus_h_scaled_pion

            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (act_tagger) (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            ax.grid()
#             self.pdf_global.savefig(fig)
            plt.close()
#             return 0
            
            
            #################### Once we have the distribution, remove the pion contamination 
            fig, ax = plt.subplots(figsize = (8, 6))
            
            ax.step(bin_centers, h_muon_scaled, where='mid', color = "black", label=f"Mu/pi above cut scaled to {lbl['mu'].lower()} peak", linewidth = 4)
            
            ax.step(bin_centers, h_pion, where = 'mid', color = "magenta", label = f"{lbl['pi']}-like distribution")
            
            
            
            pion_scalling = h_muon_scaled[0]/h_pion[0] 
            h_pion_scaled = h_pion * pion_scalling
            
            ax.step(bin_centers, h_pion_scaled, where='mid', label=f"Pion distr. scaled to Mu/pi above mutag cut bin 0")
            
            h_muon = h_muon_scaled - h_pion_scaled
            h_muon = h_muon.clip(0)
            
            ax.step(bin_centers, h_muon, where='mid', label=f"Mu/pi above mutag cut bin 0 minus scaled pion distr. => {lbl['mu'].lower()} population")
            
            
            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            ###############################################################################
            ############ Calculate effiency and purity as a function of the cut line
            h_pion_tot = h_pion + h_pion_scaled
            h_muon_tot = h_muon
            
            fig, ax = plt.subplots(figsize = (8, 6))
            
            ax.step(bin_centers, h_pion_tot, where='mid', color = "magenta", label=f"Total number of pions")
            ax.step(bin_centers, h_muon_tot, where='mid', color = "green", label=f"Total number of {lbl['mus'].lower()}")
            
            ax.set_yscale("log")
            ax.set_xlabel("ACT35 charge (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            plt.close()
            
            ###### for my own sanity
            n_pions_left = [sum(h_pion_tot[:b]) for b in range(len(bin_centers))]
            n_muons_left = [sum(h_muon_tot[:b]) for b in range(len(bin_centers))]
            
            n_pions_right = [sum(h_pion_tot[b:]) for b in range(len(bin_centers))]
            n_muons_right = [sum(h_muon_tot[b:]) for b in range(len(bin_centers))]
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, n_pions_left, where='mid', color = "magenta", label=f"Number of pions on the left of the cut line")
            ax.step(bin_centers, n_muons_left, where='mid', color = "green", label=f"Number of {lbl['mus'].lower()} on the left of the cut line")

            ax.step(bin_centers, n_pions_right, where='mid', linestyle = "--", color = "magenta", label=f"Number of pions on the right of the cut line")
            ax.step(bin_centers, n_muons_right, where='mid', linestyle = "--", color = "green", label=f"Number of {lbl['mus'].lower()} on the right of the cut line")
            
            ax.set_yscale("log")
            ax.set_xlabel("ACT35 charge (PE)", fontsize = 12)
            ax.set_ylabel("Number of triggers", fontsize = 12)
            ax.legend()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            ax.grid()
            plt.close()
            
            ##### rejection factors function of efficiency 
            n_pions_left = np.array([sum(h_pion_tot[:b]) for b in range(len(bin_centers))])
            n_muons_left = np.array([sum(h_muon_tot[:b]) for b in range(len(bin_centers))])
            
            n_pions_right = np.array([sum(h_pion_tot[b:]) for b in range(len(bin_centers))])
            n_muons_right = np.array([sum(h_muon_tot[b:]) for b in range(len(bin_centers))])
            
            pion_efficiency = n_pions_left/sum(h_pion_tot)
            muon_efficiency = n_muons_right/sum(h_muon_tot)
            
            #number of pions rejected per muon accepted in the pion selection (i.e. left)
            muon_rejection = n_muons_right/n_muons_left
            #number of muons rejected per pion accepted in the in muon selection (i.e. right)
            pion_rejection = n_pions_left/n_pions_right
            
            #Purity calculations
            pion_purity = n_pions_left/(n_pions_left+n_muons_left)
            muon_purity = n_muons_right/(n_pions_right+n_muons_right)
            
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(pion_efficiency, muon_rejection, marker = "x", color = "magenta")
            
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            
            ax.set_xlabel(f"{lbl['pi']} selection efficiency", fontsize = 12)
            ax.set_ylabel("# mu rejected per mu in sample", fontsize = 12)
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            plt.close()
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(muon_efficiency, pion_rejection, marker = "x", color = "green")
            
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            ax.set_xlabel(f"{lbl['mu']} selection efficiency", fontsize = 12)
            ax.set_ylabel("# pi rejected per pi in sample", fontsize = 12)
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            plt.close()
            
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, pion_purity, where='mid', color = "blue", label = "pion purity")
            ax.step(bin_centers, pion_efficiency, where='mid', color = "red", label = "pion efficiency")
            
            ax.set_xlabel("Cut line in ACT35 (PE)", fontsize = 12)
            ax.set_ylabel("")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {lbl['pis']}", fontsize = 20)
            plt.close()
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, muon_purity, where='mid', color = "blue", label = f"{lbl['mu'].lower()} purity")
            ax.step(bin_centers, muon_efficiency, where='mid', color = "red", label = f"{lbl['mu'].lower()} efficiency")
            ax.grid()

            ax.set_xlabel("Cut line in ACT35 (PE)", fontsize = 12)
            ax.set_ylabel("")
            ax.legend()
            ax.grid()

            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {lbl['mus']}", fontsize = 20)
            plt.close()
            
            
            
            
            
            
            


            
        else:
            print(f"there are not more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are not applying it. (Please verify this on the plots)" )#
            
            self.using_mu_tag_cut = False
            
            fig, ax = plt.subplots(figsize = (8, 6))

            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = f"{lbl['mus']} and pions", histtype = "step")
            
#             print(sum(muons_pions))
            
            
            ax.hist(self.df["act_tagger"][self.df["is_electron"]],
                    bins = bins, label = lbl["es"], histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 1.5
            
            if abs(self.run_momentum) > 700: 
                x_max = 10
                
            else:
                x_max = 20
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/{lbl['mu'].lower()} cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\nACT3-5 without muon tagger cut", fontsize = 20)
            ax.set_yscale("log")
            self.pdf_global.savefig(fig)
            plt.close()
            
        #at the end check visually that things are ok  

       
        self.plot_ACT35_left_vs_right(self.act35_cut_pi_mu)
        
        # if self.is_beam_paper_analysis == False: # tightend electron veto cut
        if abs(self.run_momentum) > 300:
            self.df["is_muon"] = (~self.df["is_electron"]) & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
            self.df["is_pion"] = (~self.df["is_electron"]) & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
        else:
            self.df["is_muon"] = (~self.df["is_electron"]) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
            self.df["is_pion"] = (~self.df["is_electron"]) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
                
        # else:
        #     #optionally for the beam paper analysis we can tighten further the eveto cut to have a purer muon and pion sample, this is for momentum studies at higher momentum, not necessary for most analyses
        #     tighter_ACT35_cut = self.eveto_cut - (4 *(abs(self.run_momentum)/810)) 
            
        #     print("we are applying tighter cut: ", tighter_ACT35_cut, " PE")
                  
        #     if abs(self.run_momentum) > 300:
        #         muons_pions = (self.df["tof"] < self.proton_tof_cut) & (self.df["act_eveto"]<tighter_ACT35_cut) 
                
        #         self.df["is_muon"] = (self.df["act_eveto"]<tighter_ACT35_cut)  & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
        #         self.df["is_pion"] = (self.df["act_eveto"]<tighter_ACT35_cut)  & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
        #     else:
        #         self.df["is_muon"] = (self.df["act_eveto"]<tighter_ACT35_cut)  & (self.df["act_tagger"]>self.act35_cut_pi_mu)
        #         self.df["is_pion"] = (self.df["act_eveto"]<tighter_ACT35_cut)  & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
                
                
        
        
        f_muons = sum(self.df["is_muon"])/self.n_muons_pions * 100
        f_pions = sum(self.df["is_pion"])/self.n_muons_pions * 100
        
        
        self.pion_purity = np.nan #pion_eff_new
        self.pion_efficiency =np.nan # pion_purity_new

        self.muon_purity = np.nan #muon_eff_new
        self.muon_efficiency = np.nan #muon_purity_new
        
        
        print(f"The pion/muon separtion cut line in ACT35 is {self.act35_cut_pi_mu:.1f}, out of {self.n_muons_pions:.1f} pions and muons, {f_muons:.1f}% are muons and {f_pions:.1f} are pions")
            
            
        
        
    def write_output_particles(self, particle_number_dict, store_PID_info, filename = None):
        """This functions writes out the WCTE tank information as well as the additional beam variables (TOF, ACT charges) necessary for making the selection, This function also stores the particle type guess obtained from the beam data but we encourage each analyser to develop their own selection"""
        
        
        if store_PID_info:
            #get the particle identification from the beam analysis
            index_mu = np.array(self.is_kept_event_id) * np.array(self.df["is_muon"])
            index_pi =  np.array(self.is_kept_event_id) * np.array(self.df["is_pion"])
            index_electron =  np.array(self.is_kept_event_id) * np.array(self.df["is_electron"])
            index_proton =  np.array(self.is_kept_event_id) * np.array(self.df["is_proton"])

            #Remove the events that are not of the given particle type
            index_mu = index_mu[index_mu !=0][:particle_number_dict["muon"]]
            index_pi = index_pi[index_pi !=0][:particle_number_dict["pion"]]
            index_electron = index_electron[index_electron !=0][:particle_number_dict["electron"]]
            index_proton = index_proton[index_proton !=0][:particle_number_dict["proton"]]
            

        # ---- inputs ------------------------------------------------------
        file_path = f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/{self.run_number}/WCTE_offline_R{self.run_number}S0_VME_matched.root"
        tree_name = "WCTEReadoutWindows"

        #these are the WCTE tank info branches
        branches = [
            "hit_pmt_calibrated_times",
            "window_data_quality",
            "hit_mpmt_card_ids", "hit_pmt_readout_mask",
            "hit_mpmt_slot_ids", "hit_pmt_position_ids",
            "hit_pmt_channel_ids", "hit_pmt_charges",
        ]

        BLOCK_MAX_EVENTS = 500
        WRITE_PARQUET   = True
        if filename == None:
             filename = f"Beam_PID_R{self.run_number}.parquet"
        PARQUET_FILE = filename

        # choose source
        local_copy = stage_local(file_path)  # "" if not staged
        file_for_uproot = local_copy or to_xrootd(file_path)

        # Combine all indices and particle labels

        if store_PID_info:
            all_keep_idx = np.concatenate([
                index_electron,
                index_mu,
                index_pi,
                index_proton
            ])


            particle_labels = (
                ["electron"] * len(index_electron) +
                ["muon"] * len(index_mu) +
                ["pion"] * len(index_pi) +
                ["proton"] * len(index_proton)
            )
            
        else: #if we do not want to save the particle ID
            #only keep the data that is kept? no, all of the data would be better 
            all_keep_idx = self.is_kept_event_id[:particle_number_dict["triggers"]]
            
            #do not store the PID, just say that we keep them 
            particle_labels = (
                ["particle"] * len(all_keep_idx)
            )
            
            
            
            
        all_keep_idx = np.array(all_keep_idx)
        particle_labels = np.array(particle_labels)

        # Sort by index for block-wise reading
        sorted_order = np.argsort(all_keep_idx)
        all_keep_idx = all_keep_idx[sorted_order]
        particle_labels = particle_labels[sorted_order]

        # Open the file to get total entries
        with uproot.open(file_for_uproot) as f:
            tree = f[tree_name]
            n_entries = tree.num_entries

        # Keep only valid indices
        mask_valid = (all_keep_idx >= 0) & (all_keep_idx < n_entries)
        all_keep_idx = all_keep_idx[mask_valid]
        particle_labels = particle_labels[mask_valid]

        # Split into blocks
        blocks = make_blocks(all_keep_idx, BLOCK_MAX_EVENTS)
        print(f"{len(all_keep_idx)} selected entries → {len(blocks)} blocks (max {BLOCK_MAX_EVENTS} ev/block)")

        t0 = time.time()

        if WRITE_PARQUET:
            writer = None
            written = 0

            for (s, e) in blocks:
                # Read block
                with uproot.open(file_for_uproot) as f:
                    arr_block = f[tree_name].arrays(branches, library="ak", entry_start=s, entry_stop=e)

                # Select events in this block
                mask_block = (all_keep_idx >= s) & (all_keep_idx < e)
                if not np.any(mask_block):
                    del arr_block
                    gc.collect()
                    continue

                local_idx = all_keep_idx[mask_block] - s
                sel = arr_block[local_idx]

                # Add particle type column
                sel_particle = particle_labels[mask_block]
                sel = ak.with_field(sel, sel_particle, "beam_pid")    # Awkward Array

                # Convert to Arrow table
                tbl = ak.to_arrow_table(sel, list_to32=True, string_to32=True)

                # Add run number column
                run_arr = pa.array([self.run_number] * len(sel), type=pa.int32())
                tbl = tbl.append_column("run", run_arr)

                # Create Parquet writer on first block
                if writer is None:
                    meta = dict(tbl.schema.metadata or {})
                    meta.update({
                        b"wcte.run_number": str(self.run_number).encode(),
                        b"wcte.source_path": file_path.encode(),
                        b"wcte.tree_name":  tree_name.encode(),
                    })
                    schema_with_meta = tbl.schema.with_metadata(meta)
                    writer = pq.ParquetWriter(PARQUET_FILE, schema_with_meta, compression="snappy")

                writer.write_table(tbl)
                written += len(sel)

                # Clean memory
                del arr_block, sel, tbl, run_arr
                gc.collect()

            if writer is not None:
                writer.close()

            print(f"Wrote {written} rows to {PARQUET_FILE} in {time.time()-t0:.2f}s")
            
            
            
    def TOF_particle_in_ns(self, particle_name, momentum, L = 4.3431):
        ''' returns the TOF of particles of a given momentum'''
        momentum  = momentum #(give them in MeV/c)

        # masses in MeV/c^2
        masses = {
            "Electrons": 0.511,
            "Muons": 105.658,
            "Kaons": 493.677,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123,
            "Helium3": 2808.392,
        }

        if particle_name not in masses:
            raise ValueError(f"Unknown particle: {particle_name}")

        m = masses[particle_name]
        c = 2.99792458e8  # m/s

        # gamma and beta
        gamma = np.sqrt(1 + (momentum/m)**2)
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta * c

        tof_seconds = L / v
        return tof_seconds * 1e9  # ns

      
    def return_losses(self, n_step, dist, particle_name, momentum, total_tof, total_length, psp, verbose = False):
        '''This function takes in the material that we are crossing, the associated energy lost table and the number of steps that we want to divide the crossing in and returns the time taken to travel the whole material and the total energy lost within it'''
        
        masses = { #MeV/c
            "Electrons": 0.511,
            "Muons": 105.658,
            "Kaons": 493.677,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123,
            "Helium3": 2808.392,

        }
        
        
        factor = 1 #for all other particles we have the correct table, no need for a multiplicative factor

        #we do not have the energy loss tables for helium3, lithium6 or tritium, we are extrapolating from those of Deuteron crudely accounting for the mass
        
        psp = psp.reset_index(drop=True) #so we can use index entries instead of float ones
        
        g4_energy = psp["#Kinectic_energy [GeV]"].to_numpy() * 1e3 

        
        for step in range(n_step):
            delta_L = dist/n_step 
            delta_t = self.TOF_particle_in_ns(particle_name, momentum, delta_L)

            total_tof += delta_t
            total_length += delta_L


            #account for the momentum lost
            for i in range(len(momentum)):
                #We do not want to work with int (will reach unphysical values) 
                momentum[i] = float(momentum[i])
                p = np.argmin(np.abs(g4_energy - momentum[i]))  # index of closest value to the g4 energy
                if p > len(psp["Total_st_pw [MeV/mm]"])-2:
                    p = len(psp["Total_st_pw [MeV/mm]"])-2
                    

                particle_kinetic_energy = np.sqrt(momentum[i]**2 + masses[particle_name]**2) - masses[particle_name] 


                if verbose:
                    print(f"Particle type: {particle_name}")
                    print(f"Step {step+1}/{n_step}, particle kinetic energy: {particle_kinetic_energy * 1e-3:.2f} MeV, stopping power: {psp['Total_st_pw [MeV/mm]'].iloc[p] * factor:.4f} MeV/mm, delta_L: {delta_L:.4f} mm, delta_t: {delta_t:.4f} ns")
                

                #now modify the momentum 
                stoppingPower = (psp["Total_st_pw [MeV/mm]"].iloc[p+1] - psp["Total_st_pw [MeV/mm]"].iloc[p]) / (psp["#Kinectic_energy [GeV]"][p+1] - psp["#Kinectic_energy [GeV]"].iloc[p]) * (particle_kinetic_energy *10**(-3) - psp["#Kinectic_energy [GeV]"].iloc[p]) + psp["Total_st_pw [MeV/mm]"].iloc[p]

                particle_kinetic_energy -= stoppingPower * factor * delta_L * 1e3

                momentum[i] = np.sqrt((particle_kinetic_energy + masses[particle_name])**2 - masses[particle_name]**2)

        return momentum, total_tof, total_length
    
    def give_theoretical_TOF(self, particle, initial_momentum_guess, verbose = False):
        '''This function returns the T0-T1, T0-T4 and T4-T1 TOFs that a given particle would have  for a given initial momentum (which can be a scalar or an array). It is a stepper function that propagates the momentum at each step, adding up the travel time to form the total TOF and taking into account the momentum lost at each step based on pre-calculated G4 tables and accurate beam material budget surveys.'''
    
        #read the detector positions and dimensions from the yaml file 
        det_module = db.from_yaml("../include/wcte_beam_detectors.yaml")

    
        
        if self.run_momentum < 0:
            if particle == "Electrons":
                p_name = "electron"
            if particle == "Muons":
                p_name = "kaonMinus" if self.is_kaon_run else "muMinus"
            if particle == "Pions":
                p_name = "piMinus"
            if particle == "Protons":
                p_name = "antiproton"
            if particle == "Deuteron":
                p_name = "deuteron"
            if particle == "Helium3":
                p_name = "He3" #


        elif self.run_momentum > 0:
            if particle == "Electrons":
                p_name = "positron"
            if particle == "Muons":
                p_name = "kaonPlus" if self.is_kaon_run else "muPlus"
            if particle == "Pions":
                p_name = "piPlus"
            if particle == "Protons":
                p_name = "proton"
            if particle == "Deuteron":
                p_name = "deuteron"
            if particle == "Helium3":
                p_name = "He3" #

                
        str_n_eveto = str(self.n_eveto)
        str_n_tagger = str(self.n_tagger)
        
  
        # Read in the theoretical losses from G4 tables provided by Arturo
        losses_dataset_air = f"../include/{p_name}StoppingPowerAirGeant4.csv"
        losses_dataset_plasticScintillator = f"../include/{p_name}StoppingPowerPlasticScintillatorGeant4.csv"
        losses_dataset_mylar = f"../include/{p_name}StoppingPowerMylarGeant4.csv"
        #ACT tables
        losses_dataset_upstream = f"../include/{p_name}StoppingPowerAerogel1p{str_n_eveto[2:]}Geant4.csv"
        losses_dataset_downstream = f"../include/{p_name}StoppingPowerAerogel1p{str_n_tagger[2:]}Geant4.csv"
   
        

        #Open all the files
        with open(losses_dataset_air, mode = 'r') as file:
            psp_air = pd.read_csv(file) #psp = particle stopping power

        with open(losses_dataset_plasticScintillator, mode = 'r') as file:
            psp_plasticScintillator = pd.read_csv(file) 

        with open(losses_dataset_upstream, mode = 'r') as file:
            psp_upstreamACT = pd.read_csv(file)

        with open(losses_dataset_downstream, mode = 'r') as file:
            psp_downstreamACT = pd.read_csv(file)

        with open(losses_dataset_mylar, mode = 'r') as file:
            psp_mylar = pd.read_csv(file)
        
        #these are the reference particle tables for this specific particle
        reference_tables = {"mylar": psp_mylar,
                            "scintillator": psp_plasticScintillator,
                            "upstreamAerogel": psp_upstreamACT,
                            "downstreamAerogel": psp_downstreamACT,
                            "air": psp_air,
                            "vinyl": psp_plasticScintillator,
                            }
        
        

        #Here we are creating a large array holding all of the materials in the beamline and the distance between them.

        if self.there_is_ACT5:
            list_all_detectors = ["Mylar_beam_window","T0","T4","ACT0_"+str(self.n_eveto),"ACT1_"+str(self.n_eveto),"ACT2_"+str(self.n_eveto),"ACT3_"+str(self.n_tagger),"ACT4_"+str(self.n_tagger),"ACT5_"+str(self.n_tagger),"T1","T5","WCTE_window"]
        else:
            list_all_detectors = ["Mylar_beam_window","T0","T4","ACT0_"+str(self.n_eveto),"ACT1_"+str(self.n_eveto),"ACT2_"+str(self.n_eveto),"ACT3_"+str(self.n_tagger),"ACT4_"+str(self.n_tagger),"T1","T5","WCTE_window"]


        #These are the arrays holding all of the layers of material that the particles see
        array_layers_thickness = []
        array_layers_material = []
        array_layers_name = []

        for d, det in enumerate(list_all_detectors):

            all_layers_name, all_layers_thickness, all_layers_material = det_module.get_all_layers(det)

            #if it is not the last detector we need to add a layer of air to it
            if d < len(list_all_detectors)-1:
                det1 = det
                det2 = list_all_detectors[d+1]
                
                det1_name = det
                det2_name = list_all_detectors[d+1]
                
                #when looking at the distance we only care about the number of the detector
                if det1[0:3]=="ACT":
                    det1 = det1[0:4]
                    det1_name = f"{det[0:3]}{det[4::]}"
                if det2[0:3]=="ACT":
                    det2 = det2[0:4]
                    det2_name = f"{list_all_detectors[d+1][0:3]}{list_all_detectors[d+1][4:]}"
                    
           
                half_thickness_det1 = det_module.get_total_thickness_m(det1_name)/2
                half_thickness_det2 = det_module.get_total_thickness_m(det2_name)/2
                #account for the air gap between detectors
                distance_to_next_det = det_module.distance_m(det1, det2) - half_thickness_det1 - half_thickness_det2
                
            else:
                distance_to_next_det = 0
               
            
            gap_name = f"{det1}_{det2}_air"

            array_layers_thickness.extend(all_layers_thickness)
            array_layers_material.extend(all_layers_material)
            array_layers_name.extend(all_layers_name)
            
            array_layers_thickness.append(distance_to_next_det)
            array_layers_material.append("air")
            array_layers_name.append(gap_name)

#         print("\nThe list of all materials are: ", array_layers_name, "\n")
        #make a copy of the initial momentum guesses that will be continuously updated to reflect how the momentum is reduced by travelling through matter.
        live_momentum = initial_momentum_guess.copy()
        total_length = 0
        #for each of the momenta we have a TOF
        total_tof = np.zeros(len(live_momentum))

        # when is_kaon_run, the "Muons" selection actually contains kaons: use kaon mass for propagation
        particle_for_propagation = "Kaons" if (self.is_kaon_run and particle == "Muons") else particle

        #now propagate that momentum through all the layers
        for l, layer_name in enumerate(array_layers_name):
            #We need to chose the number of steps per layer
            #if larger than 20 cm then it's probably air and we want somewhat large
            if array_layers_thickness[l] > 20e-2:
                steps = 50
            else:
                steps = 10


            live_momentum, total_tof, total_length = self.return_losses(int(steps/2), array_layers_thickness[l]/2, particle_for_propagation, live_momentum, total_tof, total_length, reference_tables[array_layers_material[l]], verbose = False)

            #just starting crossing each of the scintillators we are saving the travel time so we can make the differences afterwards
            if layer_name == "T0_scintillator":
                T0_time = total_tof.copy()

            if layer_name == "T4_scintillator":
                T4_time = total_tof.copy()

            if layer_name == "T1_scintillator":
                T1_time = total_tof.copy()

            if layer_name == "T5_scintillator":
                T5_time = total_tof.copy()
                
            live_momentum, total_tof, total_length = self.return_losses(int(steps/2), array_layers_thickness[l]/2, particle_for_propagation, live_momentum, total_tof, total_length, reference_tables[array_layers_material[l]], verbose = False)

        #after we have gone through all of the materials, we output the initial guesses and each of the TOFs: T0-T1, T0-T4, T4-T1 (for now) Those are arrays corresponding to the theoretical tof for each of the initial momenta guesses
        #This will be useful to fix the T4 timing offset (maybe) we should see some jitter
        T0T1_theoretical_TOF = T1_time-T0_time
        T0T4_theoretical_TOF = T4_time-T0_time
        T4T1_theoretical_TOF = T1_time-T4_time

        return initial_momentum_guess, live_momentum, T0T1_theoretical_TOF, T0T4_theoretical_TOF, T4T1_theoretical_TOF


    def find_momentum(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof, particle, give_both_bounds = False):
        """
        Solve TOF(p) - measured_tof = 0
        """
        # Build smooth interpolation of TOF(p) which will help us get away from the linear interpolation assumed before
        tof_func = interp1d(
            initial_momentum,
            theoretical_tof,
            kind='quadratic',          # or 'quadratic'
            bounds_error=False,
            fill_value="extrapolate"
        )
        def give_guess(meas_tof):
            def f(p):
                return tof_func(p) - meas_tof

            #Make sure there is a phase change between the points that we are trying to interpolate between, otherwise it beaks
            for i in range(len(initial_momentum) - 1):
                p1 = initial_momentum[i]
                p2 = initial_momentum[i + 1]

                f1 = f(p1)
                f2 = f(p2)

                # Exact solution at grid point
                if f1 == 0:
                    return p1

                # Sign change detected
                if f1 * f2 < 0:
                    return brentq(f, p1, p2)
            return 0

            

        
        momentum_guess = give_guess(measured_tof)

        momentum_low = give_guess(measured_tof - err_measured_tof)

        try:
            momentum_high =  give_guess(measured_tof + err_measured_tof)
        except:
            momentum_high = 2 * momentum_guess-momentum_low

        if (particle == "Muons" or particle == "Pions") and abs(self.run_momentum) > 650:
            momentum_high = initial_momentum[np.argmin(abs(theoretical_tof-measured_tof + err_measured_tof))]

            momentum_low = initial_momentum[np.argmin(abs(theoretical_tof-measured_tof - err_measured_tof))]

        # except ValueError:
        #     print(
        #         "The measured TOF is", measured_tof,
        #         "the theoretical TOF is", theoretical_tof,
        #         "No valid solution found"
        #     )
        #     return 0, 0

        minus_bound = abs(momentum_guess - momentum_low)
        plus_bound  = abs(momentum_high - momentum_guess)

        if give_both_bounds:
            return momentum_guess, minus_bound, plus_bound
        else:
            return momentum_guess, max(minus_bound, plus_bound)
    
        
    def extrapolate_momentum(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof, give_both_bounds = False):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error ''' 
        #first ensure that we are only looking at the part of the curve that is not nan
        initial_momentum = np.array(initial_momentum)
        theoretical_tof = np.array(theoretical_tof)
        mask = ~np.isnan(theoretical_tof)
        initial_momentum = initial_momentum[mask]
        theoretical_tof = theoretical_tof[mask]


        diff_m_exp = list(abs(theoretical_tof-measured_tof)) 
        A = diff_m_exp.index(min(diff_m_exp)) 
       
        if A == len(diff_m_exp):
            B = diff_m_exp.index(diff_m_exp[A-1])
        elif A == 0:
            B = diff_m_exp.index(diff_m_exp[A+1])
        else:  
            try:
                #if we cannot find an intrercept, we do not include it
                B = diff_m_exp.index(min(diff_m_exp[A+1], diff_m_exp[A-1])) 
            except: 
                print("The measured TOF is ", measured_tof, "the theoretical TOF is", theoretical_tof, " We did not find an intercept, returning 0, 0 for the momentum guess and error")
                return 0, 0
        #simple linear extrapolation: m = ( y_b x_a - y_a x_b ) / (x_a-x_b) 
        x_a, y_a = initial_momentum[A], theoretical_tof[A] 
        x_b, y_b = initial_momentum[B], theoretical_tof[B] 
        intercept = (y_b*x_a - y_a*x_b) / (x_a-x_b) 
        gradient = (y_a - intercept)/x_a 
        momentum_guess = (measured_tof - intercept)/gradient 
        momentum_minus = (measured_tof - err_measured_tof - intercept)/gradient 
        momentum_plus = (measured_tof + err_measured_tof - intercept)/gradient 
        # err_mom = momentum_guess-momentum_plus 
        minus_bound = abs(momentum_guess-momentum_minus)
        plus_bound = abs(momentum_guess-momentum_plus)
        err_mom = max(minus_bound, plus_bound)
        if give_both_bounds:
            return momentum_guess, minus_bound, plus_bound
        else:
            return momentum_guess, err_mom
    
    
    def extrapolate_trigger_momentum_coarse(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error, same version as abopve but working with arrays instead of single values, re-written with help from generative AI'''
        
        # Make sure they're numpy arrays
        initial_momentum = np.asarray(initial_momentum)
        theoretical_tof = np.asarray(theoretical_tof)

        # For each measured TOF, find closest index in theoretical_tof
        idx_closest = np.abs(theoretical_tof[:, None] - measured_tof).argmin(axis=0)

        # Pick neighbor for interpolation (next or previous)
        # use np.clip to avoid going out of bounds
        idx_neighbor = np.clip(idx_closest + 1, 0, len(theoretical_tof)-1)

        x_a = initial_momentum[idx_closest]
        y_a = theoretical_tof[idx_closest]
        x_b = initial_momentum[idx_neighbor]
        y_b = theoretical_tof[idx_neighbor]

        # linear interpolation parameters
        intercept = (y_b * x_a - y_a * x_b) / (x_a - x_b)
        gradient = (y_a - intercept) / x_a

        momentum_guess = (measured_tof - intercept) / gradient
        momentum_minus = (measured_tof - err_measured_tof - intercept) / gradient
        momentum_plus = (measured_tof + err_measured_tof - intercept) / gradient
        err_mom = np.abs(momentum_guess - momentum_plus)

        return momentum_guess, err_mom
    
    
    def extrapolate_trigger_momentum(self, initial_momentum, theoretical_tof,
                                 measured_tof, err_measured_tof):
        """
        Vectorized: invert tof(p) by interpolation tof->p, compute propagated error
        sigma_p = sigma_t / |dt/dp|.
        initial_momentum and theoretical_tof must describe the same-length grid.
        """

        p_grid = np.asarray(initial_momentum)
        t_grid = np.asarray(theoretical_tof)

        # Ensure monotonic t_grid for interpolation: sort by t_grid
        sort_idx = np.argsort(t_grid)
        t_sorted = t_grid[sort_idx]
        p_sorted = p_grid[sort_idx]

        # Optionally remove duplicates in t_sorted (np.interp requires strictly increasing x)
        # We'll compress duplicates by keeping the first occurrence
        dif = np.diff(t_sorted)
        keep = np.concatenate(([True], dif != 0))
        t_unique = t_sorted[keep]
        p_unique = p_sorted[keep]

        # Invert by interpolation: p(t) via np.interp
        momentum_guess = np.interp(measured_tof, t_unique, p_unique,
                                   left=np.nan, right=np.nan)

        # Numerical derivative dt/dp on original grid (use p_grid order)
        # compute dt/dp as gradient(t_grid) / gradient(p_grid)
        dp = np.gradient(p_grid)
        dt = np.gradient(t_grid)
        dtdp = dt / dp  # same length as grid

        # Now map derivative to the momentum_guess values by interpolation
        # but we need dtdp as function of p: sort by p_grid and use p_sorted (already sorted by t earlier)
        # For safety, sort p_grid increasing:
        p_sort_idx = np.argsort(p_grid)
        p_for_deriv = p_grid[p_sort_idx]
        dtdp_for_deriv = dtdp[p_sort_idx]

        # interpolate dtdp at momentum_guess
        dtdp_at_guess = np.interp(momentum_guess, p_for_deriv, dtdp_for_deriv,
                                  left=np.nan, right=np.nan)

        # propagate TOF error to momentum error
        err_mom = np.abs(err_measured_tof / dtdp_at_guess)

        return momentum_guess, err_mom

            
    # def estimate_particle_momentum(self):    
        

    #     particles_tof_names = {"Muons": "muon",
    #                            "Pions": "pion",
    #                            "Protons": "proton",
    #                            "Deuteron": "deuteron",
    #                            "Helium3": "helium3"
    #     }
        
    #     #dictionary to keep the initial and final mean particle momenta, as estimated from the T0T1 tof
    #     self.particle_mom_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
    #     self.particle_mom_mean_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        
    #     self.particle_mom_final_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
    #     self.particle_mom_final_mean_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

    #     #same for T0T4 tof
    #     self.particle_mom_mean_t0t4 = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
    #     self.particle_mom_mean_t0t4_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

    #     self.particle_mom_final_mean_t0t4 = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
    #     self.particle_mom_final_mean_t0t4_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        
        
    #     for particle in ["Muons", "Pions", "Protons", "Deuteron", "Helium3"]:
    #         measured_tof_mean = self.particle_tof_mean[particles_tof_names[particle]]
    #         measured_tof_error = self.particle_tof_eom[particles_tof_names[particle]]
            
            
    #         measured_tof_t0t4_mean = self.particle_tof_t0t4_mean[particles_tof_names[particle]]
    #         measured_tof_t0t4_error = self.particle_tof_t0t4_eom[particles_tof_names[particle]]
            
    #         if measured_tof_mean >= 0.2:
    #             if particle == "Muons" or particle == "Pions":
    #                 momentum_guess=np.linspace(170, 1900, 56)
    #             elif particle == "Protons":
    #                 #heavier particles have to have more energy to reach T1 (and even more so, T5) 
    #                 momentum_guess=np.linspace(300, 1900, 46)

    #             elif particle == "Deuteron":
    #                 #heavier particles have to have more energy to reach T1 (and even more so, T5) 
    #                 momentum_guess=np.linspace(650, 1900, 36)

    #             elif particle == "Heium3":
    #                 #heavier particles have to have more energy to reach T1 (and even more so, T5) 
    #                 momentum_guess=np.linspace(300, 1900, 36)

    #             initial_momentum_th, final_momentum_th, T0T1_TOF_th, T0T4_TOF_th, T4T1_TOF_th = self.give_theoretical_TOF(particle, momentum_guess)

    #             fig, ax = plt.subplots(figsize = (8, 6))
    #             ax.plot(initial_momentum_th, T0T1_TOF_th, color = "g", marker = "+", label = f"Predicted T0T1 TOF")
    #             ax.plot(initial_momentum_th, T0T4_TOF_th, color = "b", marker = "x", label = f"Predicted T0T4 TOF")
    #             #ax.plot(initial_momentum_th, T4T1_TOF_th, color = "k", marker = "o", label = f"{particle}: theoretical T4T1 TOF")


  
                
    #             ax.axhline(measured_tof_mean, color = "g", linestyle = "--", label = f"Measured T0T1 TOF:\n{measured_tof_mean:.2f} "r"$\pm$" f" {measured_tof_error:.1e} ns")
    #             ax.axhspan(measured_tof_mean - measured_tof_error, measured_tof_mean + measured_tof_error, color = "g", alpha = 0.2)

    #             ax.axhline(measured_tof_t0t4_mean, color = "b", linestyle = "--", label = f"Measured T0T4 TOF:\n{measured_tof_t0t4_mean:.2f} "r"$\pm$" f"  {measured_tof_t0t4_error:.1e} ns")
    #             ax.axhspan(measured_tof_t0t4_mean - measured_tof_t0t4_error, measured_tof_t0t4_mean + measured_tof_t0t4_error, color = "g", alpha = 0.2)


    #             #extrapolated_mean_mom, extrapolated_err_mom = self.extrapolate_momentum(initial_momentum_th, T0T1_TOF_th, measured_tof_mean, measured_tof_error)

    #             extrapolated_mean_mom, extrapolated_err_mom_minus, extrapolated_err_mom_plus = self.find_momentum(initial_momentum_th, T0T1_TOF_th, measured_tof_mean, measured_tof_error, True)

    #             extrapolated_mean_final_mom, extrapolated_err_final_mom = self.extrapolate_momentum(final_momentum_th, T0T1_TOF_th, measured_tof_mean, measured_tof_error)
                
    #             extrapolated_mean_mom_t0t4, extrapolated_err_mom_t0t4 = self.extrapolate_momentum(initial_momentum_th, T0T4_TOF_th, measured_tof_t0t4_mean, measured_tof_t0t4_error)


    #             extrapolated_mean_final_mom_t0t4, extrapolated_err_final_mom_t0t4 = self.extrapolate_momentum(final_momentum_th, T0T4_TOF_th, measured_tof_t0t4_mean, measured_tof_t0t4_error)

    #             if extrapolated_mean_mom is np.nan or extrapolated_mean_final_mom is np.nan:
    #                 raise Exception(f"We could not extrapolate the momentum for {particle} from the T0T1 TOF, the measured TOF is probably out of the range of our theoretical predictions, we are skipping this particle")

    #             ax.axvline(extrapolated_mean_mom, color = "green", linestyle = "-.", label = f"T0T1 TOF-est. momentum \n Initial: {extrapolated_mean_mom:.1f} "r"$\pm$" f" {extrapolated_err_mom:.1f} MeV/c \n Final: {extrapolated_mean_final_mom:.1f} "r"$\pm$" f" {extrapolated_err_final_mom:.1f} MeV/c")
                
    #             ax.axvline(extrapolated_mean_mom_t0t4, color = "blue", linestyle = "-.", label = f"T0T4 TOF-est. momentum \n Initial: {extrapolated_mean_mom_t0t4:.1f} "r"$\pm$" f" {extrapolated_err_mom_t0t4:.1f} MeV/c \n Final: {extrapolated_mean_final_mom_t0t4:.1f} "r"$\pm$" f" {extrapolated_err_final_mom_t0t4:.1f} MeV/c")
                

    #             ax.set_ylabel("TOF (ns)", fontsize = 18)
    #             ax.set_xlabel("Initial momentum (MeV/c)", fontsize = 18)
    #             ax.legend(fontsize = 12)
    #             ax.grid()
    #             ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {particle}", fontsize = 20)

    #             self.pdf_global.savefig(fig)
    #             plt.close()

    #             self.particle_mom_mean[particles_tof_names[particle]] = extrapolated_mean_mom
    #             self.particle_mom_mean_err[particles_tof_names[particle]] = extrapolated_err_mom
    #             self.particle_mom_final_mean[particles_tof_names[particle]] = extrapolated_mean_final_mom
    #             self.particle_mom_final_mean_err[particles_tof_names[particle]] = extrapolated_err_final_mom

    #             #same thing for the T0T4 TOF    
    #             self.particle_mom_mean_t0t4[particles_tof_names[particle]] = extrapolated_mean_mom_t0t4
    #             self.particle_mom_mean_t0t4_err[particles_tof_names[particle]] = extrapolated_err_mom_t0t4
    #             self.particle_mom_final_mean_t0t4[particles_tof_names[particle]] = extrapolated_mean_final_mom_t0t4
    #             self.particle_mom_final_mean_t0t4_err[particles_tof_names[particle]] = extrapolated_err_final_mom_t0t4  
          
    #     print("Initial momentum reconstructed: ", self.particle_mom_mean)

    def estimate_particle_momentum(self):    

        particles_tof_names = {
            "Muons": "muon",
            "Pions": "pion",
            "Protons": "proton",
            "Deuteron": "deuteron",
            "Helium3": "helium3"
        }

        # dictionary to keep the initial and final mean particle momenta
        self.particle_mom_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        # asymmetric errors
        self.particle_mom_mean_err_minus = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        self.particle_mom_mean_err_plus  = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        self.particle_mom_final_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        self.particle_mom_final_mean_err_minus = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        self.particle_mom_final_mean_err_plus  = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        # same for T0T4 tof
        self.particle_mom_mean_t0t4 = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        self.particle_mom_mean_t0t4_err_minus = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        self.particle_mom_mean_t0t4_err_plus  = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        self.particle_mom_final_mean_t0t4 = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        self.particle_mom_final_mean_t0t4_err_minus = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}
        self.particle_mom_final_mean_t0t4_err_plus  = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0}

        #Make the TOF error a constant fraction of the sigma so the electron TOF uncertainty is 0.05ns (it should be the smallest)
        factor_uncertainty = min(0.05/self.particle_tof_std["electron"], 1)

        factor_uncertainty_t0t4 = min(0.05/self.particle_tof_t0t4_std["electron"], 1)

        if self.run_momentum > 0:
            list_particles = ["Muons", "Pions", "Protons", "Deuteron", "Helium3"]
        else:
            list_particles = ["Muons", "Pions", "Protons"]

        for particle in list_particles:

            if self.is_kaon_run and particle == "Muons":
                print("This is a kaon run, using the kaon mass for the momentum extrapolation of the 'Muons' selection")
                particle_for_title = "Kaons"
            else:
                particle_for_title = particle


            measured_tof_mean = self.particle_tof_mean[particles_tof_names[particle]]

            
            measured_tof_error = factor_uncertainty * self.particle_tof_std[particles_tof_names[particle]]

            measured_tof_t0t4_mean = self.particle_tof_t0t4_mean[particles_tof_names[particle]]
            measured_tof_t0t4_error = factor_uncertainty_t0t4 * self.particle_tof_t0t4_std[particles_tof_names[particle]]

            if measured_tof_mean >= 0.2:

                if particle == "Muons" or particle == "Pions":
                    if not self.is_kaon_run:
                        momentum_guess = np.linspace(190, 2000, 56)
                    else:
                        momentum_guess = np.linspace(300, 2000, 56)

                elif particle == "Protons":
                    momentum_guess = np.linspace(400, 2200, 66)

                elif particle == "Deuteron":
                    momentum_guess = np.linspace(650, 2200, 66)

                elif particle == "Helium3":
                    momentum_guess = np.linspace(1200, 4300, 66)

                initial_momentum_th, final_momentum_th, T0T1_TOF_th, T0T4_TOF_th, T4T1_TOF_th = \
                    self.give_theoretical_TOF(particle, momentum_guess)

                

                # print(f"{particle}: starting momentum {initial_momentum_th}, final momentum {final_momentum_th}")

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.plot(
                    initial_momentum_th,
                    T0T1_TOF_th,
                    color="g",
                    marker="+",
                    label=f"Predicted T0T1 TOF"
                )

                ax.plot(
                    initial_momentum_th,
                    T0T4_TOF_th,
                    color="b",
                    marker="x",
                    label=f"Predicted T0T4 TOF"
                )

                ax.axhline(
                    measured_tof_mean,
                    color="g",
                    linestyle="--",
                    label=f"Measured T0T1 TOF:\n"
                        f"{measured_tof_mean:.2f} "
                        r"$\pm$"
                        f" {measured_tof_error:.2f} ns"
                )

                ax.axhspan(
                    measured_tof_mean - measured_tof_error,
                    measured_tof_mean + measured_tof_error,
                    color="g",
                    alpha=0.2
                )

                ax.axhline(
                    measured_tof_t0t4_mean,
                    color="b",
                    linestyle="--",
                    label=f"Measured T0T4 TOF:\n"
                        f"{measured_tof_t0t4_mean:.2f} "
                        r"$\pm$"
                        f" {abs(measured_tof_t0t4_error):.2f} ns"
                )

                ax.axhspan(
                    measured_tof_t0t4_mean - measured_tof_t0t4_error,
                    measured_tof_t0t4_mean + measured_tof_t0t4_error,
                    color="b",
                    alpha=0.2
                )

                # --- T0T1 initial momentum ---
                extrapolated_mean_mom, extrapolated_err_mom_minus, extrapolated_err_mom_plus = \
                    self.find_momentum(
                        initial_momentum_th,
                        T0T1_TOF_th,
                        measured_tof_mean,
                        measured_tof_error,particle,
                        True
                    )
                # print(f"{particle}: extrapolated initial momentum from T0T1 TOF is {extrapolated_mean_mom} with error +{extrapolated_err_mom_plus} -{extrapolated_err_mom_minus}")

                # --- T0T1 final momentum ---
                extrapolated_mean_final_mom, extrapolated_err_final_mom_minus, extrapolated_err_final_mom_plus = \
                    self.find_momentum(
                        final_momentum_th,
                        T0T1_TOF_th,
                        measured_tof_mean,
                        measured_tof_error,particle,
                        True
                    )
                # print(f"{particle}: extrapolated final momentum from T0T1 TOF is {extrapolated_mean_final_mom} with error +{extrapolated_err_final_mom_plus} -{extrapolated_err_final_mom_minus}")


                # --- T0T4 initial momentum ---
                extrapolated_mean_mom_t0t4, extrapolated_err_mom_t0t4_minus, extrapolated_err_mom_t0t4_plus = \
                    self.find_momentum(
                        initial_momentum_th,
                        T0T4_TOF_th,
                        measured_tof_t0t4_mean,
                        measured_tof_t0t4_error,particle,
                        True
                    )


                # --- T0T4 final momentum ---
                extrapolated_mean_final_mom_t0t4, extrapolated_err_final_mom_t0t4_minus, extrapolated_err_final_mom_t0t4_plus = \
                    self.find_momentum(
                        final_momentum_th,
                        T0T4_TOF_th,
                        measured_tof_t0t4_mean,
                        measured_tof_t0t4_error,particle,
                        True
                    )
                
                if np.isnan(extrapolated_mean_mom) or np.isnan(extrapolated_mean_final_mom):
                    raise Exception(
                        f"We could not extrapolate the momentum for {particle} "
                        f"from the T0T1 TOF"
                    )

                ax.axvline(
                    extrapolated_mean_mom,
                    color="green",
                    linestyle="-.",
                    label=
                        f"T0T1 TOF-est. momentum\n"
                        f"Initial: "
                        rf"${extrapolated_mean_mom:.1f}^{{+{extrapolated_err_mom_plus:.1f}}}_{{-{extrapolated_err_mom_minus:.1f}}}$ MeV/c"
                        "\n"
                        f"Final: "
                        rf"${extrapolated_mean_final_mom:.1f}^{{+{extrapolated_err_final_mom_plus:.1f}}}_{{-{extrapolated_err_final_mom_minus:.1f}}}$ MeV/c"
                )

                ax.axvspan(
                    extrapolated_mean_mom - extrapolated_err_mom_minus,
                    extrapolated_mean_mom + extrapolated_err_mom_plus,
                    color="g",
                    alpha=0.2
                )

                ax.axvline(
                    extrapolated_mean_mom_t0t4,
                    color="blue",
                    linestyle="-.",
                    label=
                        f"T0T4 TOF-est. momentum\n"
                        f"Initial: "
                        rf"${extrapolated_mean_mom_t0t4:.1f}^{{+{extrapolated_err_mom_t0t4_plus:.1f}}}_{{-{extrapolated_err_mom_t0t4_minus:.1f}}}$ MeV/c"
                        "\n"
                        f"Final: "
                        rf"${extrapolated_mean_final_mom_t0t4:.1f}^{{+{extrapolated_err_final_mom_t0t4_plus:.1f}}}_{{-{extrapolated_err_final_mom_t0t4_minus:.1f}}}$ MeV/c"
                )

                ax.axvspan(
                    extrapolated_mean_mom_t0t4 - extrapolated_err_mom_t0t4_minus,
                    extrapolated_mean_mom_t0t4 + extrapolated_err_mom_t0t4_plus,
                    color="b",
                    alpha=0.2
                )

                ax.set_ylabel("TOF (ns)", fontsize=18)
                ax.set_xlabel("Initial momentum (MeV/c)", fontsize=18)

                ax.legend(fontsize=12)

                ax.grid()

                if particle == "Muons" or particle == "Pions":
                    ax.set_title(
                        f"Run {self.run_number} "
                        f"({self.run_momentum} MeV/c) - {particle_for_title}\n (Selection tightening {self.tightening_offset*100:.0f}%)",
                        fontsize=20
                    )
                else:
                    ax.set_title(
                        f"Run {self.run_number} "
                        f"({self.run_momentum} MeV/c) - {particle_for_title}",
                        fontsize=20
                    )


                self.pdf_global.savefig(fig)
                plt.close()

                # --- Save T0T1 initial ---
                self.particle_mom_mean[particles_tof_names[particle]] = extrapolated_mean_mom
                self.particle_mom_mean_err_minus[particles_tof_names[particle]] = extrapolated_err_mom_minus
                self.particle_mom_mean_err_plus[particles_tof_names[particle]] = extrapolated_err_mom_plus

                # --- Save T0T1 final ---
                self.particle_mom_final_mean[particles_tof_names[particle]] = extrapolated_mean_final_mom
                self.particle_mom_final_mean_err_minus[particles_tof_names[particle]] = extrapolated_err_final_mom_minus
                self.particle_mom_final_mean_err_plus[particles_tof_names[particle]] = extrapolated_err_final_mom_plus

                # --- Save T0T4 initial ---
                self.particle_mom_mean_t0t4[particles_tof_names[particle]] = extrapolated_mean_mom_t0t4
                self.particle_mom_mean_t0t4_err_minus[particles_tof_names[particle]] = extrapolated_err_mom_t0t4_minus
                self.particle_mom_mean_t0t4_err_plus[particles_tof_names[particle]] = extrapolated_err_mom_t0t4_plus

                # --- Save T0T4 final ---
                self.particle_mom_final_mean_t0t4[particles_tof_names[particle]] = extrapolated_mean_final_mom_t0t4
                self.particle_mom_final_mean_t0t4_err_minus[particles_tof_names[particle]] = extrapolated_err_final_mom_t0t4_minus
                self.particle_mom_final_mean_t0t4_err_plus[particles_tof_names[particle]] = extrapolated_err_final_mom_t0t4_plus

        print("Initial momentum reconstructed: ", self.particle_mom_mean)        
            
    def measure_particle_TOF(self):
        '''Measure the TOF for each of the particles accounting for any offsets between the electron TOF and L/c'''

        lbl = self._plabels

        there_is_proton = False
        if sum(self.df["is_proton"]) > 10:
            there_is_proton = True
        #Define the bounds inside which we will attempt the fits 
        if abs(self.run_momentum) > 350:
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [60, 50, 70 ]

        elif there_is_proton:
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [60, 50, 70 ]

        else:
            
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [25, 20, 70 ]
            
            
            
        ##### First do T0-T1 ###########
        time_of_flight = self.df["tof"]
        
        #Define the bins
        bins_tof = np.arange(times_of_flight_min[0], times_of_flight_max[0], 0.05)

        bin_centers = (bins_tof[1:] + bins_tof[:-1])/2
        
        #Fit the electron TOF
        electron_tof = time_of_flight[self.df["is_electron"] == 1]
        h, _ = np.histogram(electron_tof, bins = bins_tof)
        
    
        det_module = db.from_yaml("../include/wcte_beam_detectors.yaml")
        popt, pcov = fit_gaussian(h, bin_centers)
        L = det_module.distance_m("T0", "T1")*100 #need to be cm
        #L is in cm, c is in m.ns^-1
        t0 = L/(c * 10**2) - popt[1] #convert m.ns^-1 into cm.ns^-1
        
        
        
         ##### Second do T0-T4 ###########
        

        ### Implement the requirement that the T4 left and right are the same within 3ns (so we do not have the jump)
        self.df_goodT4 = self.df[np.abs(self.df["t4_l"] - self.df["t4_r"])< 3]

        time_of_flight_t0t4 = self.df_goodT4["tof_t0t4"]
        
       
        bins_tof_t0t4 = np.arange(times_of_flight_min[0]+10, times_of_flight_max[0]+10, 0.05)
        bin_centers_t0t4 = (bins_tof_t0t4[1:] + bins_tof_t0t4[:-1])/2
        
        #Fit the electron TOF
        electron_tof_t0t4 = time_of_flight_t0t4[self.df_goodT4["is_electron"] == 1]
        h_t0t4, _ = np.histogram(electron_tof_t0t4, bins = bins_tof_t0t4)
        
        
        
        
        popt_t0t4, pcov_t0t4 = fit_gaussian(h_t0t4, bin_centers_t0t4)
        L_t0t4 = det_module.distance_m("T0", "T4")*100 #need to be cm
        t0_t0t4 = L_t0t4/(c * 10**2) - popt_t0t4[1] #convert m.ns^-1 into cm.ns^-1 
        
        fig, ax = plt.subplots()
        ax.hist(electron_tof_t0t4, bins = bins_tof_t0t4, histtype = "step")
        ax.grid
        ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_t0t4[0], popt_t0t4[1], popt_t0t4[2]), "--", color = "k")
        ax.set_title(f"{lbl['e']} T0-T4 TOF", weight = "bold")
        ax.set_xlabel("T0-T4 TOF (ns)")
        self.pdf_global.savefig(fig)

        
        #For pions and muons, tighten the selection so the accuracy of the momentum estimate is better, this is especially important at higher momentum 
        offset = 0.7

        if self.is_kaon_run:
            offset = 0.0

        print(f"Reducing the cut lines by {offset * 100}%")

        if self.using_mu_tag_cut:
            self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset)) & (self.df["mu_tag_total"]>self.mu_tag_cut)
            self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) & (self.df["mu_tag_total"]<self.mu_tag_cut)
        else:
            self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset))
            self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) 
        

        while offset >= 0.05 and (sum(self.df["is_tighter_muon"])<20 or sum(self.df["is_tighter_pion"])<20 ):
            offset -= 0.1
            print(f"Reducing the cut lines by {offset * 100}%")

            if self.using_mu_tag_cut:
                self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset)) & (self.df["mu_tag_total"]>self.mu_tag_cut)
                self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) & (self.df["mu_tag_total"]<self.mu_tag_cut)

            else:
                self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset))
                self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) 
            
        self.tightening_offset = offset
        

        
        bins_tof_t0t4 += t0_t0t4
        bin_centers_t0t4 += t0_t0t4
        
        
        print(f"The time difference between the reconstructed electron TOF and L/c = {L/(c * 10**2):.3f} is {t0:.3f} ns")
        print(f"The time difference between the reconstructed electron TOF and L/c (T0T4) = {L_t0t4/(c * 10**2):.3f} is {t0_t0t4:.3f} ns")
        
        #Correct the TOF by this offset, decide to make a new column, cleaner
        self.df["tof_corr"] = self.df["tof"] + t0
        self.df["tof_t0t4_corr"] = self.df["tof_t0t4"] + t0_t0t4

        #there adding the corrected T4 T0 TOF
        self.df_goodT4 = self.df[np.abs(self.df["t4_l"] - self.df["t4_r"])< 3]


        # if it is a beam paper analysis then make another TOF plot for just the muons and just the pions, have a 2x2 grid with the t0t1 TOF on the left and the T0T4 (good only) on the right with the pions on top and the muons on the bottom and have both fitted with a gaussian and the mean and sigma in legend to verify if there are any differences between the pions and muons at high momentum 

        bins_tof_small = np.linspace(13, 17, 50) 
        bin_centers_tof_small = (bins_tof_small[1:] + bins_tof_small[:-1])/2

        bins_tof_small_t0t4 = np.linspace(8, 12, 50) 
        bin_centers_tof_small_t0t4 = (bins_tof_small_t0t4[1:] + bins_tof_small_t0t4[:-1])/2

        if self.is_beam_paper_analysis:
            fig, axs = plt.subplots(2, 2, figsize = (10, 10))
            for i, particle in enumerate(["muon", "pion"]):
                ax = axs[i, 0]
                
                mask = self.df[f"is_{particle}"]==1
               
                
                particle_tof = self.df["tof_corr"][mask]
                h_particle, _ = np.histogram(particle_tof, bins = bins_tof_small)
                popt_particle, pcov_particle = fit_gaussian(h_particle, bin_centers_tof_small)

                err_mean=np.sqrt(pcov_particle[1][1])
                err_std=np.sqrt(pcov_particle[2][2])

                ax.hist(particle_tof, bins = bins_tof_small, histtype = "step", label = f"{particle.capitalize()}: tof mean = {popt_particle[1]:.3f} "+ r"$\pm$"+ f" {err_mean:.2e} ns \n tof std: {popt_particle[2]:.2f} "+ r"$\pm$"+ f" {err_std:.2e} ns \n ACT02 cut: {self.eveto_cut:.2f}")
                ax.plot(bins_tof_small, gaussian(bins_tof_small, popt_particle[0], popt_particle[1], popt_particle[2]), "--", color = "k")
                
                
                #tighten the electron cut and see the impact on the TOF 
                tighter_cut = 2 *(abs(self.run_momentum)/810)
                mask = mask & (self.df["act_eveto"]<tighter_cut)
                particle_tof = self.df["tof_corr"][mask]
                h_particle, _ = np.histogram(particle_tof, bins = bins_tof_small)
                popt_particle, pcov_particle = fit_gaussian(h_particle, bin_centers_tof_small)

                err_mean=np.sqrt(pcov_particle[1][1])
                err_std=np.sqrt(pcov_particle[2][2])

                ax.hist(particle_tof, bins = bins_tof_small, histtype = "step", label = f"{particle.capitalize()}: tof mean = {popt_particle[1]:.3f} "+ r"$\pm$"+ f" {err_mean:.2e} ns \n tof std: {popt_particle[2]:.2f} "+ r"$\pm$"+ f" {err_std:.2e} ns \n ACT02 cut: {tighter_cut:.2f}")
                ax.plot(bins_tof_small, gaussian(bins_tof_small, popt_particle[0], popt_particle[1], popt_particle[2]), "--", color = "k")
                
                

                #add a vertical line with the theoretical TOF for this particle at this momentum
                _, _, theoretical_t0t1_tof_cut_array, theoretical_t0t4_tof_cut, _ =self.give_theoretical_TOF(particle.capitalize()+"s", np.array([abs(float(self.run_momentum))]), verbose = True)
                theoretical_t0t1_tof_cut = theoretical_t0t1_tof_cut_array[0]
                theoretical_t0t4_tof_cut = theoretical_t0t4_tof_cut[0]

                ax.axvline(theoretical_t0t1_tof_cut, color = "r", linestyle = "--", label = rf"Theor. T0T1 TOF: {theoretical_t0t1_tof_cut:.2f}ns "+"\n"+rf"({(popt_particle[1]-theoretical_t0t1_tof_cut)/popt_particle[2]:.2f} $\sigma$ away from meas.)")


                ax.set_title(f"{particle.capitalize()} T0-T1 TOF", weight = "bold")
                ax.set_xlabel("T0-T1 TOF (ns)")
                ax.set_xlim(theoretical_t0t1_tof_cut-2, theoretical_t0t1_tof_cut+6)
                ax.legend()
                ax.grid()   

                ax = axs[i, 1]
                particle_tof_t0t4 = self.df_goodT4["tof_t0t4_corr"][self.df_goodT4[f"is_{particle}"]==1]
                h_particle_t0t4, _ = np.histogram(particle_tof_t0t4, bins = bins_tof_small_t0t4)
                try:
                    popt_particle_t0t4, pcov_particle_t0t4 = fit_gaussian(h_particle_t0t4, bin_centers_tof_small_t0t4)
                    err_mean=np.sqrt(pcov_particle[1][1])
                    err_std=np.sqrt(pcov_particle[2][2])
                except:
                    popt_particle_t0t4 = [0, 0, 0]
                    err_mean=0
                    err_std=0
                ax.hist(particle_tof_t0t4, bins = bins_tof_small_t0t4, histtype = "step", label = f"{particle.capitalize()}: tof mean = {popt_particle_t0t4[1]:.2f} "+ r"$\pm$"+ f" {err_mean:.2e} ns \n tof std: {popt_particle_t0t4[2]:.2f} "+ r"$\pm$"+ f" {err_std:.2e} ns")   

                ax.plot(bins_tof_small_t0t4, gaussian(bins_tof_small_t0t4, popt_particle_t0t4[0], popt_particle_t0t4[1], popt_particle_t0t4[2]), "--", color = "k")
                ax.axvline(theoretical_t0t4_tof_cut, color = "r", linestyle = "--", label = f"Theor. T0T4 TOF: {theoretical_t0t4_tof_cut:.2f} ns "+"\n"+rf"({(popt_particle_t0t4[1]-theoretical_t0t4_tof_cut)/popt_particle_t0t4[2]:.2f} $\sigma$ away from meas.)")
                ax.set_title(f"{particle.capitalize()} T0-T4 TOF)", weight = "bold")
                ax.set_xlabel("T0-T4 TOF (ns)")
                ax.legend()
                ax.set_xlim(theoretical_t0t4_tof_cut-2, theoretical_t0t4_tof_cut+6)

                ax.grid()
            fig.suptitle(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {lbl['mu']} and Pion TOF \n  {self.jaw_setting}", fontsize = 20)
            self.pdf_global.savefig(fig)
            plt.close()
                

        #Refine the muon and pion tagger cut to ensure that we are not biased in the TOF 
        
        

       

        
        #Check TOF for each particle type
        h_mu, _ = np.histogram(self.df["tof_corr"][self.df["is_tighter_muon"]==1], bins = bins_tof)
        
        
        try:
            popt_mu, pcov = fit_gaussian(h_mu, bin_centers)
        except: 
            popt_mu  = [0,0,0]
        
        h_pi, _ = np.histogram(self.df["tof_corr"][self.df["is_tighter_pion"]==1], bins = bins_tof)
        try:
            popt_pi, pcov = fit_gaussian(h_pi, bin_centers)
        except: 
            popt_pi  = [0,0,0]
        
        h_mu_t0t4, _ = np.histogram(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_tighter_muon"]==1], bins = bins_tof_t0t4)
        try:
            popt_mu_t0t4, pcov_t0t4 = fit_gaussian(h_mu_t0t4, bin_centers_t0t4)
        except: 
            popt_mu_t0t4 = [0,0,0]
        
        
        h_pi_t0t4, _ = np.histogram(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_tighter_pion"]==1], bins = bins_tof_t0t4)
        
        try:
            popt_pi_t0t4, pcov_t0t4 = fit_gaussian(h_pi_t0t4, bin_centers_t0t4)
        except: 
            popt_pi_t0t4 = [0,0,0]
        
        if there_is_proton:
            h_p, _ = np.histogram(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof)
            popt_p, pcov = fit_gaussian(h_p, bin_centers)

            _, _, _, proton_t0t4_tof_cut_array, _ =self.give_theoretical_TOF("Protons", np.array([abs(self.run_momentum) * 1.1]))

            self.expected_T0T4_proton = proton_t0t4_tof_cut_array[0]

            h_p_t0t4, _ = np.histogram(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_proton"]==1], bins = bins_tof_t0t4)
            try:
                #     #constrain the fit to be around the expected value (based on the T0T1) so we do not select the wrong T4 peak (one is correct the other one is not)

                
                #force the T4 proton fit to only include "in time" events, the other ones would correspond to a particle speeding up (since not in agreement with T0T1) not physical due to T4 jumping issues. 
                

           

                mask = (bin_centers_t0t4 >= self.expected_T0T4_proton - 5) & (bin_centers_t0t4 <= self.expected_T0T4_proton + 5)

                popt_p_t0t4, pcov_t0t4 = fit_gaussian(h_p_t0t4[mask], bin_centers_t0t4[mask])
                
            except: 
                popt_p_t0t4 = [0, 0, 0]
            
        if sum(self.df["is_deuteron"])>10:
            h_D, _ = np.histogram(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof)
            popt_D, pcov = fit_gaussian(h_D, bin_centers)
            
            h_D_t0t4, _ = np.histogram(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_deuteron"]==1], bins = bins_tof_t0t4)
            popt_D_t0t4, pcov_t0t4 = fit_gaussian(h_D_t0t4, bin_centers_t0t4)
            
            try:
                popt_D_t0t4, pcov_t0t4 = fit_gaussian(h_D_t0t4, bin_centers_t0t4)
            except: 
                popt_D_t0t4 = [0, 0, 0]
            
            
        #Here, plot the TOF 
        fig, ax = plt.subplots(figsize = (8, 6))
        
        #plot the distributions
        ax.hist(self.df["tof_corr"][self.df["is_electron"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['es']}: tof = {popt[1]+t0:.2f} "+ r"$\pm$"+ f" {popt[2]:.2f} ns")
        ax.hist(self.df["tof_corr"][self.df["is_muon"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['mus']}: tof = {popt_mu[1]:.2f} "+ r"$\pm$"+ f" {popt_mu[2]:.2f} ns")
        ax.hist(self.df["tof_corr"][self.df["is_tighter_pion"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['pis']}: tof = {popt_pi[1]:.2f} "+ r"$\pm$"+ f" {popt_pi[2]:.2f} ns")

        if there_is_proton:
            ax.hist(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['ps']}: tof = {popt_p[1]:.2f} "+ r"$\pm$"+ f" {popt_p[2]:.2f} ns")
        else:
            ax.hist(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof, histtype = "step", label = lbl["ps"])

        if sum(self.df["is_deuteron"])>10:
            ax.hist(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['Ds']}: tof = {popt_D[1]:.2f} "+ r"$\pm$"+ f" {popt_D[2]:.2f} ns")
        else:
            ax.hist(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof, histtype = "step", label = lbl["Ds"])

        # else:
        #     mean = self.df["tof_corr"][self.df["is_deuteron"]==1].mean()
        #     std = self.df["tof_corr"][self.df["is_deuteron"]==1].std()
        #     ax.hist(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof, histtype = "step", label = f"Deuterons: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")

            
        if sum(self.df["is_helium3"])>5:
            
            try:
                h_He3, _ = np.histogram(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof)
                popt_He3, pcov = fit_gaussian(h_He3, bin_centers)
                ax.hist(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['He3s']}: tof = {popt_He3[1]:.2f} "+ r"$\pm$"+ f" {popt_He3[2]:.2f} ns")
                ax.plot(bins_tof, gaussian(bins_tof, popt_He3[0], popt_He3[1], popt_He3[2]), "--", color = "k")
            except:
                popt_He3 = [0, 0, 0]
                mean = self.df["tof_corr"][self.df["is_helium3"]==1].mean()
                std = self.df["tof_corr"][self.df["is_helium3"]==1].std()
                ax.hist(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['He3s']}: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")
        else:
            mean = self.df["tof_corr"][self.df["is_helium3"]==1].mean()
            std = self.df["tof_corr"][self.df["is_helium3"]==1].std()
            ax.hist(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = lbl["He3"])
            
        #plot the fits, for visual inspection
        
        ax.plot(bins_tof, gaussian(bins_tof, popt[0], popt[1]+t0, popt[2]), "--", color = "k")
        
        for p in [popt_mu, popt_pi]:
            ax.plot(bins_tof, gaussian(bins_tof, p[0], p[1], p[2]), "--", color = "k")
        
        
        if there_is_proton:
            ax.plot(bins_tof, gaussian(bins_tof, popt_p[0], popt_p[1], popt_p[2]), "--", color = "k")
            
        if sum(self.df["is_deuteron"])>10:
            ax.plot(bins_tof, gaussian(bins_tof, popt_D[0], popt_D[1], popt_D[2]), "--", color = "k")


        #add the proton, deuteron and deuterium cut lines as shaded areas:
        if abs(self.run_momentum) > 350:
            try:
                ax.axvline(x=self.proton_tof_cut+t0, color='red', linestyle='--', linewidth=2, label=f'{lbl["p"]} cut: {self.proton_tof_cut+t0:.2f} ns')
                ax.axvline(x=self.deuteron_tof_cut+t0, color='purple', linestyle='--', linewidth=2, label=f'{lbl["D"]} cut: {self.deuteron_tof_cut+t0:.2f} ns')
                ax.axvline(x=self.helium3_tof_cut+t0, color='brown', linestyle='--', linewidth=2, label=f'{lbl["He3"]} cut: {self.helium3_tof_cut+t0:.2f} ns')
            except: 
                print("TOF cuts for heavier particles not defined or nans, not plotting them")

        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Time of flight (ns)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e5)
        ax.set_title(f"Run {self.run_number} T0-T1 TOF ({self.run_momentum} MeV/c) \n{self.jaw_setting}", fontsize = 20)
        self.pdf_global.savefig(fig)
        
        
        if there_is_proton:
            ax.set_xlim(None, popt_p[1] * 1.2)
            
        if sum(self.df["is_deuteron"])>20:
            ax.set_xlim(None, popt_D[1] * 1.2)
            
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        ################################### Make a second clean plot for T0T4 TOFs ################################ 
        fig, ax = plt.subplots(figsize = (8, 6))
        
        #plot the distributions
        ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_electron"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['es']}: tof = {popt_t0t4[1]+t0_t0t4:.2f} "+ r"$\pm$"+ f" {popt_t0t4[2]:.2f} ns")
        ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_muon"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['mus']}: tof = {popt_mu_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_mu_t0t4[2]:.2f} ns")
        ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_pion"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['pis']}: tof = {popt_pi_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_pi_t0t4[2]:.2f} ns")
        
        
        # if we are making the plots for the beam paper analyses, discard the out of time events:
        if there_is_proton:
            if self.is_beam_paper_analysis:
                mask = (self.df_goodT4["tof_t0t4_corr"] >= self.expected_T0T4_proton - 5) & (self.df_goodT4["tof_t0t4_corr"] <= self.expected_T0T4_proton + 5)
                ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_proton"]==1][mask], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['ps']}: tof = {popt_p_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_p_t0t4[2]:.2f} ns")
            else:
                ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_proton"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['ps']}: tof = {popt_p_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_p_t0t4[2]:.2f} ns")

        if sum(self.df_goodT4["is_deuteron"])>10:
            ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_deuteron"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['Ds']}: tof = {popt_D_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_D_t0t4[2]:.2f} ns")
           
        
        popt_He3_t0t4 = [0, 0, 0]
        if sum(self.df_goodT4["is_helium3"])>5:
            
            try:
                h_He3_t0t4, _ = np.histogram(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_helium3"]==1], bins = bins_tof_t0t4)
                popt_He3_t0t4, pcov_t0t4 = fit_gaussian(h_He3_t0t4, bin_centers_t0t4)
                ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_helium3"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"{lbl['He3s']}: tof = {popt_He3_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_He3_t0t4[2]:.2f} ns")
                ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_He3_t0t4[0], popt_He3_t0t4[1], popt_He3_t0t4[2]), "--", color = "k")
                
                
            except:
                
                mean = self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_helium3"]==1].mean()
                std = self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_helium3"]==1].std()

                popt_He3_t0t4 = [0, mean, std]
                ax.hist(self.df_goodT4["tof_t0t4_corr"][self.df_goodT4["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"{lbl['He3s']}: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")

            
      
        #plot the fits, for visual inspection
        
        ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_t0t4[0], popt_t0t4[1]+t0_t0t4, popt_t0t4[2]), "--", color = "k")
        
        for p in [popt_mu_t0t4, popt_pi_t0t4]:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, p[0], p[1], p[2]), "--", color = "k")
        
        
        if there_is_proton:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_p_t0t4[0], popt_p_t0t4[1], popt_p_t0t4[2]), "--", color = "k")
            
            mean_proton_T0T4_tof = popt_p_t0t4[1]
            std_proton_T0T4_tof = popt_p_t0t4[2]
                
                
            
        if sum(self.df["is_deuteron"])>20:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_D_t0t4[0], popt_D_t0t4[1], popt_D_t0t4[2]), "--", color = "k")


        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Time of flight (ns)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e5)
        ax.set_title(f"Run {self.run_number} T0-T4 TOF ({self.run_momentum} MeV/c)\n{self.jaw_setting}", fontsize = 20)
        self.pdf_global.savefig(fig)
        
        
        if there_is_proton:
            ax.set_xlim(None, popt_p[1] * 1.2)
            
        if sum(self.df["is_deuteron"])>20:
            ax.set_xlim(None, popt_D[1] * 1.2)
            
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        #Here save the mean TOF and std for each particle population
        self.particle_tof_mean = {
            "electron": popt[1]+t0,
            "muon": popt_mu[1],
            "pion": popt_pi[1],
            "proton": popt_p[1] if there_is_proton else 0,
            "deuteron": popt_D[1] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[1] if sum(self.df["is_helium3"])>5 else 0,
       
        }
        
        self.particle_tof_std = {
            "electron": popt[2],
            "muon": popt_mu[2],
            "pion": popt_pi[2],
            "proton": popt_p[2] if there_is_proton else 0,
            "deuteron": popt_D[2] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2] if sum(self.df["is_helium3"])>5 else 0,
          
        }
        
        self.particle_tof_eom = {
            "electron": popt[2]/np.sqrt(sum(self.df["is_electron"])),
            "muon": popt_mu[2]/np.sqrt(sum(self.df["is_muon"])),
            "pion": popt_pi[2]/np.sqrt(sum(self.df["is_pion"])),
            "proton": popt_p[2]/np.sqrt(sum(self.df["is_proton"])) if there_is_proton else 0,
            "deuteron": popt_D[2]/np.sqrt(sum(self.df["is_deuteron"])) if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2]/np.sqrt(sum(self.df["is_helium3"])) if sum(self.df["is_helium3"])>5 else 0,
           
            
        }
        
        #################### same for T0T4 #####################
        self.particle_tof_t0t4_mean = {
            "electron": popt_t0t4[1]+t0_t0t4,
            "muon": popt_mu_t0t4[1],
            "pion": popt_pi_t0t4[1],
            "proton": mean_proton_T0T4_tof if there_is_proton else 0,
            "deuteron": popt_D_t0t4[1] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3_t0t4[1] if sum(self.df["is_helium3"])>5 else 0,
          
        }
        
        self.particle_tof_t0t4_std = {
            "electron": abs(popt_t0t4[2]),
            "muon": abs(popt_mu_t0t4[2]),
            "pion": abs(popt_pi_t0t4[2]),
            "proton": abs(std_proton_T0T4_tof) if there_is_proton else 0,
            "deuteron": abs(popt_D_t0t4[2]) if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": abs(popt_He3_t0t4[2]) if sum(self.df["is_helium3"])>20 else 0,
          
        }
        
        self.particle_tof_t0t4_eom = {
            "electron": popt_t0t4[2]/np.sqrt(sum(self.df["is_electron"])),
            "muon": popt_mu_t0t4[2]/np.sqrt(sum(self.df["is_muon"])),
            "pion": popt_pi_t0t4[2]/np.sqrt(sum(self.df["is_pion"])),
            "proton": std_proton_T0T4_tof/np.sqrt(sum(self.df["is_proton"])) if there_is_proton else 0,
            "deuteron": popt_D_t0t4[2]/np.sqrt(sum(self.df["is_deuteron"])) if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3_t0t4[2]/np.sqrt(sum(self.df["is_helium3"])) if sum(self.df["is_helium3"])> 5 else 0,
           
            
        }



    def refine_particle_momentum_using_TOF(self):
        "This function makes a cleaner selection of muons and pions at higher momentum and uses it to refine the measurement of the momentum."

        lbl = self._plabels

        #Step 1: make a tighter muon and pion selection

        if self.using_mu_tag_cut:
            self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * 0.5) & (self.df["act_tagger"]> self.act35_cut_pi_mu * 1.50) & (self.df["mu_tag_total"]>self.mu_tag_cut)
            self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * 0.5) & (self.df["act_tagger"]< self.act35_cut_pi_mu * 0.50) & (self.df["mu_tag_total"]<self.mu_tag_cut)
        else:

            offset = 0.7

            print(f"Reducing the cut lines by {offset * 100}%")
            self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset))
            self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) 
            

            while offset >= 0.05 and (sum(self.df["is_tighter_muon"])<20 or sum(self.df["is_tighter_pion"])<20 ):
                offset -= 0.1
                print(f"Reducing the cut lines by {offset * 100}%")

                self.df["is_tighter_muon"] = self.df["is_muon"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]> self.act35_cut_pi_mu * (1+offset))
                self.df["is_tighter_pion"] = self.df["is_pion"] & (self.df["act_eveto"]< self.eveto_cut * (1-offset)) & (self.df["act_tagger"]< self.act35_cut_pi_mu * (1-offset)) 
            




        mask_muon_nominal = self.df["is_muon"] == 1
        mask_muon_tighter = self.df["is_tighter_muon"] == 1
        mask_pion_nominal = self.df["is_pion"] == 1 
        mask_pion_tighter = self.df["is_tighter_pion"] == 1

        #plot the ACT02 vs ACT35 distributions showing the nominal selection and the tighter one
        fig, axes = plt.subplots(2, 1,figsize = (8, 6))
        ax = axes[0]
        ax1 = axes[1]

        #muons on one side
        ax.scatter(self.df["act_eveto"][mask_muon_nominal][0:10000], self.df["act_tagger"][mask_muon_nominal][0:10000], label = f"Nominal {lbl['mu'].lower()} events", alpha = 0.5)
        ax.scatter(self.df["act_eveto"][mask_muon_tighter][0:10000], self.df["act_tagger"][mask_muon_tighter][0:10000], label = "Tighter selection", alpha = 0.5)
        ax.set_xlabel("ACT02 charge (PE)")
        ax.set_ylabel("ACT35 charge (PE)")
        ax.legend()
        ax.set_title(f"{lbl['mu']} selection")
        ax.grid()
        #pions on the other side
        ax1.scatter(self.df["act_eveto"][mask_pion_nominal][0:10000], self.df["act_tagger"][mask_pion_nominal][0:10000], label = "Nominal pion events", alpha = 0.5)
        ax1.scatter(self.df["act_eveto"][mask_pion_tighter][0:10000], self.df["act_tagger"][mask_pion_tighter][0:10000], label = "Tighter selection", alpha = 0.5)
        ax1.set_xlabel("ACT02 charge (PE)")
        ax1.set_ylabel("ACT35 charge (PE)")
        ax1.legend()
        ax1.set_title(f"{lbl['pi']} selection")
        ax1.grid()
        fig.suptitle(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()

        # Plot the charge deposited in the muon tagger for the muons and the pions, in the nominal and tighter selection to see if they reach it, also indicate on the plot the cut line that we use to determine if a particle reached it. indicate in the legend what fraction of each selection is above that cut line.

        mu_tag_bins = np.linspace(0, max(self.df["mu_tag_total"]), 100)

        fig, axes = plt.subplots(1, 2,figsize = (12, 8))
        ax = axes[0]
        ax1 = axes[1]
        ax.hist(self.df["mu_tag_total"][mask_muon_nominal], bins = mu_tag_bins, label = f"Nominal muon events: {sum(mask_muon_nominal)} total\n {sum(self.df['mu_tag_total'][mask_muon_nominal]>self.mu_tag_cut)/sum(mask_muon_nominal) * 100:.2f} % above muon tagger cut", histtype = "step")
        ax.hist(self.df["mu_tag_total"][mask_muon_tighter], bins = mu_tag_bins, label = f"Tighter muon selection: {sum(mask_muon_tighter)} total\n {sum(self.df['mu_tag_total'][mask_muon_tighter]>self.mu_tag_cut)/sum(mask_muon_tighter) * 100:.2f} % above muon tagger cut", histtype = "step")
        ax.axvline(self.mu_tag_cut, color = "r", linestyle = "--", label = f"Muon tagger cut: {self.mu_tag_cut:.2f} PE")
        ax.set_xlabel("Total charge in muon tagger (PE)")
        ax.set_ylabel("Number of events")
        ax.set_yscale("log")

        ax.legend(fontsize =12)
        ax.set_title(lbl["mus"], fontsize =14)
        ax.grid()


        ax1.hist(self.df["mu_tag_total"][mask_pion_nominal], bins = mu_tag_bins, label = f"Nominal pion events: {sum(mask_pion_nominal)} total\n {sum(self.df['mu_tag_total'][mask_pion_nominal]>self.mu_tag_cut)/sum(mask_pion_nominal) * 100:.2f} % above muon tagger cut", histtype = "step")
        ax1.hist(self.df["mu_tag_total"][mask_pion_tighter], bins = mu_tag_bins, label = f"Tighter pion selection: {sum(mask_pion_tighter)} total\n {sum(self.df['mu_tag_total'][mask_pion_tighter]>self.mu_tag_cut)/sum(mask_pion_tighter) * 100:.2f} % above muon tagger cut", histtype = "step")
        ax1.axvline(self.mu_tag_cut, color = "r", linestyle = "--", label = f"Muon tagger cut: {self.mu_tag_cut:.2f} PE")
        ax1.set_xlabel("Total charge in muon tagger (PE)")
        ax1.set_ylabel("Number of events")
        ax1.legend(fontsize =12)
        ax1.set_yscale("log")
        ax1.set_title(lbl["pis"], fontsize =14)
        ax1.grid()
        fig.suptitle(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()

        #Step 2: plot the TOF of the whole distribution and of the tighter selection including checks of whether it is truly gaussian distribution  
        #make  a plot of the TOF fitted with a gaussian and have for each plot the difference between the histrogam and the fot to check the goodness of the fit, indicate the fit value in the legend as well as teh error on each parameter. have the pions on the right and the muons on the left, and in each plot the nominal selection and the tighter one.

        fig, axes = plt.subplots(2, 2,figsize = (12, 8), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [3, 1]}, sharex = True)
        ax = axes[0, 0]
        ax1 = axes[0, 1]
        ax_res_muon = axes[1, 0]
        ax_res_pion = axes[1, 1]

        #reconstruct the associated momentum so we can verify the value
        #make a very fine momentum space around the expected one to have as much precision as possible
        momentum_guess = np.linspace(abs(self.run_momentum) * 0.80, abs(self.run_momentum) * 1.2, 100)
        initial_momentum_th_muon_nominal, final_momentum_th_muon_nominal, T0T1_TOF_th_muon_nominal, T0T4_TOF_th_muon_nominal, T4T1_TOF_th_muon_nominal = self.give_theoretical_TOF("Muons", np.array([self.run_momentum*1.0]))


        initial_momentum_th_pion_nominal, final_momentum_th_pion_nominal, T0T1_TOF_th_pion_nominal, T0T4_TOF_th_pion_nominal, T4T1_TOF_th_pion_nominal = self.give_theoretical_TOF("Pions", np.array([self.run_momentum*1.0]))

        T0T1_TOF_th_muon_nominal = T0T1_TOF_th_muon_nominal[0]
        T0T1_TOF_th_pion_nominal = T0T1_TOF_th_pion_nominal[0]

        momentum_guess = np.linspace(abs(self.run_momentum) * 0.80, abs(self.run_momentum) * 1.2, 100)
        initial_momentum_th_muon, final_momentum_th_muon, T0T1_TOF_th_muon, T0T4_TOF_th_muon, T4T1_TOF_th_muon = self.give_theoretical_TOF("Muons", momentum_guess)
        initial_momentum_th_pion, final_momentum_th_pion, T0T1_TOF_th_pion, T0T4_TOF_th_pion, T4T1_TOF_th_pion = self.give_theoretical_TOF("Pions", momentum_guess)

        #extrapolate the momentum:


        bins = np.linspace(13.5, 20.5, 150)

        ax.axvline(T0T1_TOF_th_muon_nominal, color = "red", linestyle = ':', label = f"Nominal {lbl['mu'].lower()} TOF: {T0T1_TOF_th_muon_nominal:.3f}ns")
        ax.hist(self.df["tof_corr"][mask_muon_nominal], bins = bins, histtype = "step", label = f"Nominal: {sum(mask_muon_nominal)} events")
        #gaussian fit to the nominal muon TOF distribution
        h_mu, bin_edges_mu = np.histogram(self.df["tof_corr"][mask_muon_nominal], bins = bins)
        bin_centers_mu = (bin_edges_mu[:-1] + bin_edges_mu[1:]) / 2
        popt_mu, pcov_mu = fit_gaussian(h_mu, bin_centers_mu)

        error_muon_TOF = 0.05 #np.sqrt(pcov_mu[1, 1])

        #extrapolate the associated momentum:
        momentum_est_muons, momentum_est_muons_error_minus, momentum_est_muons_error_plus   = self.extrapolate_momentum(initial_momentum_th_muon, T0T1_TOF_th_muon,popt_mu[1], error_muon_TOF, give_both_bounds = True)
        
        print("Pevious method: ", momentum_est_muons, momentum_est_muons_error_minus, momentum_est_muons_error_plus, " New method: ")

        #check with the new extrapolation method:
        momentum_est_muons, momentum_est_muons_error_minus, momentum_est_muons_error_plus = self.find_momentum(initial_momentum_th_muon, T0T1_TOF_th_muon,popt_mu[1], error_muon_TOF, give_both_bounds = True)


        ax.plot(bins, gaussian(bins, popt_mu[0], popt_mu[1], popt_mu[2]), "--", color = "k", label = f"TOF (standard selection):\n"rf"$\mu$ = {popt_mu[1]:.3f}ns $\sigma$ = {popt_mu[2]:.3f}ns" "\n"rf"Momentum: ${momentum_est_muons:.1f}^{{+{momentum_est_muons_error_plus:.1f}}}_{{-{momentum_est_muons_error_minus:.1f}}}$  MeV/c")

     

        ax.hist(self.df["tof_corr"][mask_muon_tighter], bins = bins, histtype = "step", label = f"Tighter: {sum(mask_muon_tighter)} total")
        #gaussian fit to the tighter muon TOF distribution
        h_mu_tighter, bin_edges_mu_tighter = np.histogram(self.df["tof_corr"][mask_muon_tighter], bins = bins)
        bin_centers_mu_tighter = (bin_edges_mu_tighter[:-1] + bin_edges_mu_tighter[1:]) / 2
        popt_mu_tighter, pcov_mu_tighter = fit_gaussian(h_mu_tighter, bin_centers_mu_tighter)

         #extrapolate the associated momentum:
        # momentum_est_tight_muons, momentum_est_muons_tight_error  = self.extrapolate_momentum(initial_momentum_th_muon, T0T1_TOF_th_muon,popt_mu_tighter[1], error_muon_TOF)

        momentum_est_tight_muons, momentum_est_muons_tight_error_minus, momentum_est_muons_tight_error_plus = self.find_momentum(initial_momentum_th_muon, T0T1_TOF_th_muon,popt_mu_tighter[1], error_muon_TOF, give_both_bounds = True)

        



        ax.plot(bins, gaussian(bins, popt_mu_tighter[0], popt_mu_tighter[1], popt_mu_tighter[2]), "--", color = "r", label = f"TOF (tighter selection):\n" rf"$\mu$ = {popt_mu_tighter[1]:.3f} ns $\sigma$ = {popt_mu_tighter[2]:.3f} ns ""\n"rf"Momentum: ${momentum_est_tight_muons:.1f}^{{+{momentum_est_muons_tight_error_plus:.1f}}}_{{-{momentum_est_muons_tight_error_minus:.1f}}}$  MeV/c")  


        # ax.set_xlabel("Time of flight (ns)", fontsize = 14)
        ax.set_ylabel("Number of events", fontsize = 14)
        ax.legend(fontsize = 12)
        ax.set_title(f"T0-T1 TOF distribution for {lbl['mu'].lower()} selection", fontsize = 14)
        ax.grid()

        #plot the residuals for the muon fits
        ax_res_muon.plot(bin_centers_mu, (h_mu - gaussian(bin_centers_mu, *popt_mu))/h_mu, label = "Residuals (standard)", color = "k") 
        ax_res_muon.plot(bin_centers_mu_tighter, (h_mu_tighter - gaussian(bin_centers_mu_tighter, *popt_mu_tighter))/h_mu_tighter, label = "Residuals (tighter)", color = "r")
        ax_res_muon.set_xlabel("Time of flight (ns)", fontsize = 14)
        ax_res_muon.set_ylabel("(Data - Fit)/Data", fontsize = 14)
        ax_res_muon.legend()
        ax_res_muon.grid()
        #repeat the same for the pions
        ax1.axvline(T0T1_TOF_th_pion_nominal, color = "blue", linestyle = ':', label = f"Nominal pion TOF: {T0T1_TOF_th_pion_nominal:.3f}ns")

        ax1.hist(self.df["tof_corr"][mask_pion_nominal], bins = bins, histtype = "step", label = f"Standard: {sum(mask_pion_nominal)} events")

        #gaussian fit to the nominal pion TOF distribution
        h_pi, bin_edges_pi = np.histogram(self.df["tof_corr"][mask_pion_nominal], bins = bins)
        bin_centers_pi = (bin_edges_pi[:-1] + bin_edges_pi[1:]) / 2
        popt_pi, pcov_pi = fit_gaussian(h_pi, bin_centers_pi)

        error_pion_TOF = 0.1

         #extrapolate the associated momentum:
        #momentum_est_pions, momentum_est_pions_error  = self.extrapolate_momentum(initial_momentum_th_pion, T0T1_TOF_th_pion,popt_pi[1], error_pion_TOF)

        momentum_est_pions, momentum_est_pions_error_minus, momentum_est_pions_error_plus = self.find_momentum(initial_momentum_th_pion, T0T1_TOF_th_pion,popt_pi[1], error_pion_TOF, give_both_bounds = True)





        ax1.plot(bins, gaussian(bins, popt_pi[0], popt_pi[1], popt_pi[2]), "--", color = "k", label = f"TOF (standard selection):\n"rf"$\sigma$ = {popt_pi[1]:.3f} ns $\sigma$ = {popt_pi[2]:.3f} ns""\n"rf"Momentum: ${momentum_est_pions:.1f}^{{+{momentum_est_pions_error_plus:.1f}}}_{{-{momentum_est_pions_error_minus:.1f}}}$  MeV/c") 

        ax1.hist(self.df["tof_corr"][mask_pion_tighter], bins = bins, histtype = "step", label = f"Tighter: {sum(mask_pion_tighter)} events")
        #gaussian fit to the tighter pion TOF distribution
        h_pi_tighter, bin_edges_pi_tighter = np.histogram(self.df["tof_corr"][mask_pion_tighter], bins = bins)
        bin_centers_pi_tighter = (bin_edges_pi_tighter[:-1] + bin_edges_pi_tighter[1:]) / 2
        popt_pi_tighter, pcov_pi_tighter = fit_gaussian(h_pi_tighter, bin_centers_pi_tighter)

        #momentum_est_tight_pions, momentum_est_pions_tight_error  = self.extrapolate_momentum(initial_momentum_th_pion, T0T1_TOF_th_pion,popt_pi_tighter[1], error_pion_TOF)

        momentum_est_tight_pions, momentum_est_pions_tight_error_minus, momentum_est_pions_tight_error_plus = self.find_momentum(initial_momentum_th_pion, T0T1_TOF_th_pion,popt_pi_tighter[1], error_pion_TOF, give_both_bounds = True)

        ax1.plot(bins, gaussian(bins, popt_pi_tighter[0], popt_pi_tighter[1], popt_pi_tighter[2]), "--", color = "r", label = f"TOF (tighter selection):\n"rf"$\mu$ = {popt_pi_tighter[1]:.3f} ns $\sigma$ = {popt_pi_tighter[2]:.3f} ns""\n"rf"Momentum: ${momentum_est_tight_pions:.1f}^{{+{momentum_est_pions_tight_error_plus:.1f}}}_{{-{momentum_est_pions_tight_error_minus:.1f}}}$  MeV/c")



        # ax1.set_xlabel("Time of flight (ns)", fontsize = 14)
        ax1.set_ylabel("Number of events", fontsize = 14)
        ax1.legend(fontsize = 12)
        ax1.set_title("T0-T1 TOF distribution for pion selection")
        ax1.grid()
        #plot the residuals for the pion fits
        ax_res_pion.plot(bin_centers_pi, (h_pi - gaussian(bin_centers_pi, *popt_pi))/h_pi, label = "Residuals (nominal)", color = "k")
        ax_res_pion.plot(bin_centers_pi_tighter, (h_pi_tighter - gaussian(bin_centers_pi_tighter, *popt_pi_tighter))/h_pi_tighter, label = "Residuals (tighter)", color = "r")
        ax_res_pion.set_xlabel("Time of flight (ns)", fontsize = 14)
        ax_res_pion.set_ylabel("(Data - Fit)/Data", fontsize = 14)
        ax_res_pion.legend()
        ax_res_pion.grid()
        fig.suptitle(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()

    



        
        
        
    def plot_all_TOFs(self):
        "quick function to plot all the TOFs for checking"
        bins = [np.linspace(10, 55, 150), np.linspace(0, 25, 150), np.linspace(-40, 40, 150), np.linspace(20, 70, 150), np.linspace(10, 55, 150), np.linspace(10, 55, 150), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300)]
        tof_var = ["tof_corr", "tof_t0t4_corr", "tof_t4t1", "tof_t0t5", "tof_t1t5", "tof_t4t5", "t0_time", "t1_time", "t4_time", "t5_time"]
        tof_names = ["T0-T1 (etof corrected)","T0-T4(etof corrected)", "T4-T1", "T0-T5", "T1-T5", "T4-T5", "T0", "T1", "T4", "T5"]
        
        for i in range(len(tof_var)):
            fig, ax = plt.subplots(figsize = (8, 6))
            for population in ["is_electron", "is_muon", "is_pion", "is_proton", "is_deuteron", "is_helium3"]:
                ax.hist(self.df[self.df[population]==1][tof_var[i]], bins = bins[i], label = f"{population} total: {sum(self.df[population])}", histtype = "step")

            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("Time of flight (ns)", fontsize = 18)
            ax.legend(fontsize = 10)
            ax.grid()
            ax.set_yscale("log")
            ax.set_ylim(0.5, 5e5)
            ax.set_title(f"Run {self.run_number} {tof_names[i]} Time ({self.run_momentum} MeV/c) \n require T5 = {self.require_t5_hit}", fontsize = 20)
            self.pdf_global.savefig(fig)
            
        
        
        
    def plot_TOF_charge_distribution(self):
        '''Check visually the total charge deposited in the TOF detector, can be handy to identify events that do not actually cross the TOF'''

        lbl = self._plabels

        fig, ax = plt.subplots(figsize = (8, 6))

        bins = np.linspace( min(self.df_all["total_TOF_charge"]), max(self.df_all["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = lbl["e"], histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = lbl["mu"], histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = lbl["pi"], histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = lbl["p"], histtype = "step")

        if sum(self.df["is_helium3"] == 1) >1:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_helium3"] == 1], bins = bins, label = lbl["He3"], histtype = "step")

        if sum(self.df["is_deuteron"] == 1) >10:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = lbl["D"], histtype = "step")
            
        _ = ax.hist(self.df_all["total_TOF_charge"][self.df_all["is_kept"] == 0], bins = bins, label = 'Triggers not kept for analysis', color = "k", histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace( min(self.df["total_TOF_charge"]), max(self.df["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = lbl["e"], histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = lbl["mu"], histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = lbl["pi"], histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = lbl["p"], histtype = "step")
        if sum(self.df["is_deuteron"] == 1) >10:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = lbl["D"], histtype = "step")
        if sum(self.df["is_helium3"] == 1) >5:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_helium3"] == 1], bins = bins, label = lbl["He3"], histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
                   
    def output_beam_ana_to_root(self, output_name = None):
        ''''Output the results of the beam analysis as a root file with three branches, the 1D run information (number, nominal momentum, refractive index, whether ACT5 is in the beam line), the 1D results (mean measured [T0T1] TOF and momentum for each particle type with EOM and std for the TOF only), number of triggers kept, total number of triggers'''
        if output_name == None:
            output_name = f"beam_analysis_output_R{self.run_number}.root"
            
            
        for col in self.df.columns:
            if col not in self.df_all.columns:
                self.df_all[col] = np.nan  # create empty column first if you want aligned length
                self.df_all.loc[self.df_all["is_kept"], col] = self.df[col].values

                
                
        for is_particle in ["is_muon", "is_electron", "is_pion", "is_proton", "is_deuteron", "is_helium3"]:
            self.df_all[is_particle] = self.df_all[is_particle].fillna(0)
            self.df_all[is_particle] = self.df_all[is_particle].astype(np.int32)

        
        
        self.df_all["is_kept"] = self.df_all["is_kept"].astype(np.int32)
        
        

       # --- Convert DataFrame to dictionary of numpy arrays ---
        branches = {col: self.df_all[col].to_numpy() for col in self.df_all.columns}

        # --- Create ROOT file and save the tree ---
        with uproot.recreate(output_name) as f:
            # Write the main TTree
            f["beam_analysis"] = branches

            # Write scalar metadata as a separate tree (recommended)
            f["run_info"] = {
                "run_number": np.array([self.run_number], dtype=np.int32),
                "run_momentum": np.array([self.run_momentum], dtype=np.float64),
                "n_eveto": np.array([self.n_eveto], dtype=np.float64),
                "n_tagger": np.array([self.n_tagger], dtype=np.float64),
                "there_is_ACT5":np.array([self.there_is_ACT5], dtype = np.int32),
                "is_kaon_run": np.array([int(self.is_kaon_run)], dtype=np.int32),

            }


            #save as a separate branch the 1d results of interest

            results = {
                "act_eveto_cut":np.array([self.eveto_cut], dtype=np.float64),
                "act_tagger_cut":np.array([self.act35_cut_pi_mu], dtype=np.float64),
                "proton_tof_cut":np.array([self.proton_tof_cut], dtype=np.float64),
                "deuteron_tof_cut":np.array([self.deuteron_tof_cut], dtype=np.float64),
                "mu_tag_cut": np.array([self.mu_tag_cut], dtype=np.float64),
                "using_mu_tag_cut": np.array([self.using_mu_tag_cut], dtype=np.float64),
                
                "pion_purity":np.array([self.pion_purity], dtype=np.float64),
                "pion_efficiency":np.array([self.pion_efficiency], dtype=np.float64),
                "muon_purity":np.array([self.muon_purity], dtype=np.float64),
                "muon_efficiency":np.array([self.muon_efficiency], dtype=np.float64),
            }
            for prefix, d in [("tof_mean", self.particle_tof_mean),
                  ("tof_std", self.particle_tof_std),
                  ("tof_eom", self.particle_tof_eom),
                  ("momentum_at_CERN_pipe_mean", self.particle_mom_mean),
                  ("momentum_at_CERN_pipe_eom", self.particle_mom_mean_err),
                  ("momentum_at_WCTE_window_mean", self.particle_mom_final_mean),
                  ("momentum_at_WCTE_window_eom", self.particle_mom_final_mean_err)]:
                for key, value in d.items():
                    results[f"{prefix}_{key}"] = np.array([value], dtype=np.float64)

                    
            results["n_electrons"] = np.array([sum(self.df_all["is_electron"]) ], dtype=np.float64) 
            results["n_muons"] =  np.array([sum(self.df_all["is_muon"])], dtype=np.float64) 
            results["n_pions"] =  np.array([sum(self.df_all["is_pion"])], dtype=np.float64)   
            results["n_protons"] =  np.array([sum(self.df_all["is_proton"])], dtype=np.float64)   
            results["n_deuterons"] =   np.array([sum(self.df_all["is_deuteron"])], dtype=np.float64)   
            results["n_helium3"] =   np.array([sum(self.df_all["is_helium3"])], dtype=np.float64)   
            results["n_triggers_kept"] =  np.array([sum(self.df_all["is_kept"])], dtype=np.float64) 
            
            results["n_triggers_total"] =  np.array([len(self.df_all["is_proton"]) ], dtype=np.float64)  
            
            
            f["scalar_results"] = results 
            
            print(f"Saved output file to {output_name}")
            
            
            
            
    def output_to_root(self, output_name = None, scalar_info=True):
        ''''Output the results of the beam analysis as a root file with three branches, the 1D run information (number, nominal momentum, refractive index, whether ACT5 is in the beam line), the 1D results (cut lines, number of triggers kept, total number of triggers) and the relevant varaibles '''
        if output_name == None:
            output_name = f"beam_analysis_output_R{self.run_number}.root"
            
           
        for col in self.df.columns:
            if self.is_beam_paper_analysis == False:
                if col in ["is_muon", "is_electron", "is_pion", "is_proton",
                        "is_deuteron", "is_helium3", "is_tighter_muon", "is_tighter_pion"
                        "final_momentum", "final_momentum_error",
                        "initial_momentum", "initial_momentum_error"]:
                    continue
            

            values = self.df[col].copy()

            # --- dtype handling ---
            if pd.api.types.is_bool_dtype(values):
                print(f"Converting boolean column {col} to int for ROOT compatibility.")
                values = values.astype(np.int32)

            if col in ["evt_quality_bitmask", "digi_issues_bitmask", "spill_number", "event_id"]:
                print(f"Casting {col} to int32")
                values = values.astype(np.int32)

            # --- ensure column exists ---
            if col not in self.df_all.columns:
                self.df_all[col] = np.nan

            # --- assign (this overwrites existing dtype) ---
            self.df_all.loc[self.df.index, col] = values
            

       
        self.df_all["event_id"] = self.df_all["event_id"].astype(np.int32)
        self.df_all["evt_quality_bitmask"] = self.df_all["evt_quality_bitmask"].astype(np.int32)
        self.df_all["digi_issues_bitmask"] = self.df_all["digi_issues_bitmask"].astype(np.int32)
        self.df_all["spill_number"] = self.df_all["spill_number"].astype(np.int32)

        
        self.df_all["is_kept"] = self.df_all["is_kept"].astype(np.int32) 
        
        
        #changing the names of the variables so they are more clear 
        rename_map = {
            "tof": "tof_t0t1",
            "t4_l": "t4_l_time",
            "t4_r": "t4_r_time",
            "act0_time_l": "act0_l_time",
            "act0_time_r": "act0_r_time",
            "act0_l": "act0_l_charge",
            "act1_l": "act1_l_charge",
            "act2_l": "act2_l_charge",
            "act3_l": "act3_l_charge",
            "act4_l": "act4_l_charge",
            "act5_l": "act5_l_charge",
            "act0_r": "act0_r_charge",
            "act1_r": "act1_r_charge",
            "act2_r": "act2_r_charge",
            "act3_r": "act3_r_charge",
            "act4_r": "act4_r_charge",
            "act5_r": "act5_r_charge",
            "mu_tag_l": "mu_tag_l_charge",
            "mu_tag_r": "mu_tag_r_charge",
         }
        
        self.df_all = self.df_all.rename(columns=rename_map)
        nTriggers = np.array([len(self.df_all["is_kept"]) ], dtype=np.int64)
        self.df_all = self.df_all.drop("is_kept", axis = 1)

        

       # --- Convert DataFrame to dictionary of numpy arrays ---
        branches = {col: self.df_all[col].to_numpy() for col in self.df_all.columns}

        # --- Create ROOT file and save the tree ---
        with uproot.recreate(output_name) as f:
            # Write the main TTree
            f["beam_analysis"] = branches

            # Write scalar metadata as a separate tree (recommended)
            f["run_info"] = {
                "run_number": np.array([self.run_number], dtype=np.int32),
                "run_momentum": np.array([self.run_momentum], dtype=np.int64),
                "n_eveto": np.array([self.n_eveto], dtype=np.float64),
                "n_tagger": np.array([self.n_tagger], dtype=np.float64),
                "there_is_ACT5":np.array([self.there_is_ACT5], dtype = np.int32),
                "collimator_jaw_pos1": np.array([self.momentum_slit_negative], dtype=np.float64),
                "collimator_jaw_pos2": np.array([self.momentum_slit_positive], dtype=np.float64),
                "acceptance_jaw_pos1": np.array([self.acceptance_slit_negative], dtype=np.float64),
                "acceptance_jaw_pos2": np.array([self.acceptance_slit_positive], dtype=np.float64),
                "is_kaon_run": np.array([int(self.is_kaon_run)], dtype=np.int32),

            }


            #save as a separate branch the 1d results of interest

            results = {
                "act_eveto_cut":np.array([self.eveto_cut], dtype=np.float64),
                "act_tagger_cut":np.array([self.act35_cut_pi_mu], dtype=np.float64),
                "proton_tof_cut":np.array([self.proton_tof_cut], dtype=np.float64),
                "deuteron_tof_cut":np.array([self.deuteron_tof_cut], dtype=np.float64),

                "helium3_tof_cut":np.array([self.helium3_tof_cut], dtype=np.float64),
                "mu_tag_cut": np.array([self.mu_tag_cut], dtype=np.int64),
                "using_mu_tag_cut": np.array([self.using_mu_tag_cut], dtype=np.float64),
                
                #output the number of triggers identified as each particle, for reference 
                #here we need to sum from the df dataframe and not the df_all which doesn't have the PID info
                "n_electrons": np.array([sum(self.df["is_electron"])], dtype=np.int32),
                "n_muons": np.array([sum(self.df["is_muon"])], dtype=np.int32),
                "n_pions": np.array([sum(self.df["is_pion"])], dtype=np.int32),
                "n_protons": np.array([sum(self.df["is_proton"])], dtype=np.int32),
                "n_deuterium": np.array([sum(self.df["is_deuteron"])], dtype=np.int32),
                "n_helium3": np.array([sum(self.df["is_helium3"])], dtype=np.int32),
                
#                 "pion_purity":np.array([self.pion_purity], dtype=np.float64),
#                 "pion_efficiency":np.array([self.pion_efficiency], dtype=np.float64),
#                 "muon_purity":np.array([self.muon_purity], dtype=np.float64),
#                 "muon_efficiency":np.array([self.muon_efficiency], dtype=np.float64),
            }
            
            if scalar_info:
                for prefix, d in [

                    ("tof_mean", self.particle_tof_mean),
                    ("tof_std", self.particle_tof_std),
                    ("tof_eom", self.particle_tof_eom),

                    ("momentum_at_CERN_pipe_mean", self.particle_mom_mean),

                    # ("momentum_at_CERN_pipe_err", self.particle_mom_mean_eom),

                    ("momentum_at_CERN_pipe_err_minus",
                    self.particle_mom_mean_err_minus),
                    ("momentum_at_CERN_pipe_err_plus",
                    self.particle_mom_mean_err_plus),
                    
                    ("momentum_at_WCTE_window_mean", self.particle_mom_final_mean),
                    ("momentum_at_WCTE_window_err_minus",
                    self.particle_mom_final_mean_err_minus),
                    ("momentum_at_WCTE_window_err_plus",
                    self.particle_mom_final_mean_err_plus),

                    ("tof_t0t4_mean", self.particle_tof_t0t4_mean),
                    ("tof_t0t4_std", self.particle_tof_t0t4_std),
                    ("tof_t0t4_eom", self.particle_tof_t0t4_eom),

                    ("momentum_t0t4_at_CERN_pipe_mean", self.particle_mom_mean_t0t4),

                    ("momentum_t0t4_at_CERN_pipe_err_minus",
                    self.particle_mom_mean_t0t4_err_minus),

                    ("momentum_t0t4_at_CERN_pipe_err_plus",
                    self.particle_mom_mean_t0t4_err_plus),

                    ("momentum_t0t4_at_WCTE_window_mean",
                    self.particle_mom_final_mean_t0t4),

                    ("momentum_t0t4_at_WCTE_window_err_minus",
                    self.particle_mom_final_mean_t0t4_err_minus),

                    ("momentum_t0t4_at_WCTE_window_err_plus",
                    self.particle_mom_final_mean_t0t4_err_plus)

                ]:

                    for key, value in d.items():
                        results[f"{prefix}_{key}"] = np.array([value], dtype=np.float64)

                results["n_triggers_total"] = nTriggers

                        

                    
#             results["n_electrons"] = np.array([sum(self.df_all["is_electron"]) ], dtype=np.float64) 
#             results["n_muons"] =  np.array([sum(self.df_all["is_muon"])], dtype=np.float64) 
#             results["n_pions"] =  np.array([sum(self.df_all["is_pion"])], dtype=np.float64)   
#             results["n_protons"] =  np.array([sum(self.df_all["is_proton"])], dtype=np.float64)   
#             results["n_deuterons"] =   np.array([sum(self.df_all["is_deuteron"])], dtype=np.float64)   
#             results["n_helium3"] =   np.array([sum(self.df_all["is_helium3"])], dtype=np.float64)   
#             results["n_lithium6"] =   np.array([sum(self.df_all["is_lithium6"])], dtype=np.float64)   
#             results["n_tritium"] =   np.array([sum(self.df_all["is_tritium"])], dtype=np.float64)   

                
                f["scalar_results"] = results
            
            print(f"Saved output file to {output_name}")
            
            
            
            
    def study_electrons(self, cut_line):
        '''This function is dedicated to understanding why some electrons are missed by the ACT02 tag and check why some muons and pions are tagged as electrons'''

        lbl = self._plabels
        #step 1: make a selection of the particles that are not tagged as electrons by ACT02 but deposit a lot of light in the ACT35
        mask = (self.df["is_electron"] == 0) & (self.df["act_tagger"] >= cut_line) & (self.df["tof"] < self.proton_tof_cut)
        df_e = self.df[mask]
        
        self.plot_ACT35_left_vs_right(cut_line, "non-tagged e-like triggers")
        
        bins = np.linspace(0, self.eveto_cut, 50)
        
        
        fig, axs = plt.subplots(3, 3, figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        bins = np.linspace(0, 25, 50)
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            i = i+3
            for j, ax in enumerate(ax_row):
                
                j = j+3
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        

        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_l"], bins = (bins, bins), norm=LogNorm())
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} left (PE)", fontsize = 12)
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.hist2d(df_e[f"act{i}_r"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                
                ax.set_xlabel(f"ACT{i} right (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
     ################################################
     ### Here check using the tof which ones are the muons and which are the other
   
        print(f"The difference between the muon TOF ({self.particle_tof_mean['electron']} "r"$\pm$" f" {self.particle_tof_std['electron']} ns) and the electron TOF ({self.particle_tof_mean['muon']} "r"$\pm$" f" {self.particle_tof_std['muon']} ns)")
    
        
        mid_tof_e_mu = self.particle_tof_mean["electron"]+ 3 * self.particle_tof_std["electron"]
        
        #(self.particle_tof_mean["muon"]-self.particle_tof_mean["electron"])/2
        df_e_true = df_e[df_e["tof"]<mid_tof_e_mu]
        df_mu_true = df_e[df_e["tof"]>mid_tof_e_mu]
    
        
        
        ### check against the tof
        tof_bins = np.linspace(12, self.proton_tof_cut, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        
        ax.hist(df_e_true["tof"], bins = tof_bins, color = "red", label = f"e-like triggers not tagged by ACT20 with ACT35 > {cut_line}")
        ax.hist(df_mu_true["tof"], bins = tof_bins, color = "black", label = f"mu-like triggers not tagged by ACT20 with ACT35 > {cut_line}")
        ax.hist(self.df["tof"][self.df["is_electron"]], bins = tof_bins, label = "Triggers tagged as electrons by ACT20", histtype = "step")
        ax.hist(self.df["tof"][self.df["is_muon"]], bins = tof_bins, label = lbl["mus"], histtype = "step")
        ax.hist(self.df["tof"][self.df["is_pion"]], bins = tof_bins, label = lbl["pis"], histtype = "step")
        ax.axvline(mid_tof_e_mu, linestyle = "--", color = "k", label = f"e/mu tof cut: {mid_tof_e_mu:.2f}")
        fig.suptitle(f"Study of triggers \n not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        ax.legend()
        ax.set_yscale("log")
        ax.grid()
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        ###### check the distribution of muons and electrons
        
        fig, axs = plt.subplots(3, 3, figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_r"], color = "black", label = f"{lbl['mu'].lower()}-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        #################
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            i = i+3
            for j, ax in enumerate(ax_row):
                
                j = j+3
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_r"], color = "black", label = f"{lbl['mu'].lower()}-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_l"], color = "black", label = f"{lbl['mu'].lower()}-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_l"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} left (PE)", fontsize = 12)
                ax.legend()
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_r"], df_mu_true[f"act{j}_r"], color = "black", label = f"{lbl['mu'].lower()}-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_r"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                ax.set_xlabel(f"ACT{i} right (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
    def plot_number_particles_per_POT(self):
        '''This function plots the number of particles of each type recorded per spill and then per POT, required for the beam flux paper and represent an example of how to read POT information from Arturo's readings of the nxcals CERN database'''

        lbl = self._plabels

#         #making a complete dataframe with all of the entries, including the rejected ones
#         df_comp = self.df_all.copy()
        
#         for col in self.df.columns:
#             if col not in df_comp.columns:
#                 df_comp[col] = np.nan  # create empty column first if you want aligned length
#                 df_comp.loc[df_comp["is_kept"], col] = self.df[col].values
        
        spill_index = [s for s in self.df["spill_number"].unique()]
        number_e_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_electron"]) for s in self.df["spill_number"].unique()])
        number_mu_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_muon"]) for s in self.df["spill_number"].unique()])
        number_pi_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_pion"]) for s in self.df["spill_number"].unique()])
        number_p_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_proton"]) for s in self.df["spill_number"].unique()])
        number_D_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_deuteron"]) for s in self.df["spill_number"].unique()])
        number_3He_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_helium3"]) for s in self.df["spill_number"].unique()])
        
        number_rejected_per_spill = np.array([len(self.df_all[(self.df_all["spill_number"]==s) & (self.df_all["is_kept"]==0)]) for s in self.df_all["spill_number"].unique()])
        spill_index_all = [s for s in self.df_all["spill_number"].unique()]
        
        
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(spill_index, number_e_per_spill, "x", label = f"{lbl['es']} ({sum(self.df['is_electron'])})")
        ax.plot(spill_index, number_mu_per_spill, "x", label = f"{lbl['mus']}  ({sum(self.df['is_muon'])})")
        ax.plot(spill_index, number_pi_per_spill, "x", label = f"{lbl['pis']}  ({sum(self.df['is_pion'])})")
        ax.plot(spill_index, number_p_per_spill, "x", label = f"{lbl['ps']}  ({sum(self.df['is_proton'])})")
        ax.plot(spill_index, number_D_per_spill, "x", label = f"{lbl['Ds']}  ({sum(self.df['is_deuteron'])})")
        ax.plot(spill_index, number_3He_per_spill, "x", label = f"{lbl['He3']}  ({sum(self.df['is_helium3'])})")
        n_rejected_triggers = len(self.df_all["tof"])-len(self.df["tof"])
        ax.plot(spill_index_all, number_rejected_per_spill, "x", label = f"Rejected triggers ({n_rejected_triggers})", color =  "darkgray")
        ax.set_ylabel("Number of particles", fontsize = 20)
        ax.set_xlabel("Spill index", fontsize = 20)
        ax.legend(fontsize = 16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\n ({np.array(spill_index).max()-np.array(spill_index).min()} spills)", fontsize = 20)
        self.pdf_global.savefig(fig)
        plt.close()
        
        ## Here read the number of POT per spill from the nxcals data 
        #The spills are always in order, we do not need to re-arrange them, in principle
        df_pot = None
#        if self.run_number == 1606:
#            df_pot = pd.read_csv(f"/eos/experiment/wcte/user_data/fiorenti/nxcals/pot/run_{self.run_number}_pot.csv", header = 0)
#            n_pot_per_trigger = np.array(df_pot["POT0"])
#            n_pot_per_trigger = n_pot_per_trigger[0:len(spill_index)]
#        else:
        try:
            df_pot = pd.read_csv(f"/eos/experiment/wcte/user_data/fiorenti/nxcals/pot/brb_run_{self.run_number}_pot.csv")
            #the first column is the time stamp, the second is the number of POT for that spill 
            df_pot = df_pot.rename(columns={df_pot.columns[0]: "timestamp", df_pot.columns[1]: "POT"})
            n_pot_per_trigger = np.array(df_pot["POT"])
            n_pot_per_trigger = n_pot_per_trigger[0:len(spill_index)]
        except:
            return 0


        #decide that there are a bin for each ten spills
#         n_bins = int(max(spill_index)/10)

        print("Mean number of protons per spill", np.array(number_p_per_spill).mean())

        min_number = max(max(number_e_per_spill), max(number_mu_per_spill), max(number_pi_per_spill), max(number_p_per_spill), max(number_D_per_spill), max(number_3He_per_spill))/min(n_pot_per_trigger)*1.5

        max_limit = abs(self.run_momentum)**2 * 25/770**2

        n_bins = np.linspace(0, max_limit, 100)
        n_bins_narrow = np.linspace(0, max_limit, 300)
        fig, ax = plt.subplots(figsize = (8, 6))
        
        
        bin_centers = (n_bins[1:]+n_bins[:-1])/2

        min_fraction = 0.05 

        if max(number_e_per_spill) > min_fraction:

        
            h_e, _, _ = ax.hist(number_e_per_spill/n_pot_per_trigger, bins = n_bins, label = lbl["es"], color = "blue", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_e, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "blue", label = f"{lbl['es']}: mean {popt[1]:.2f}, std {popt[2]:.2f}")
            except:
                popt  = [0, 0, 0]
        
        
        if max(number_mu_per_spill) > min_fraction:
        
            h_mu, _, _ = ax.hist(number_mu_per_spill/n_pot_per_trigger, bins = n_bins, label = lbl["mus"], color = "orange", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_mu, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "orange", label = f"{lbl['mus']}: mean {popt[1]:.2f}, std {popt[2]:.2f}")
            
            except:
                popt  = [0, 0, 0]


        if max(number_pi_per_spill) > min_fraction:
        
            h_pi, _, _ = ax.hist(number_pi_per_spill/n_pot_per_trigger, bins = n_bins, label = lbl["pis"], color = "green", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_pi, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "green", label = f"{lbl['pis']}: mean {popt[1]:.2f}, std {popt[2]:.2f}")
        
            except:
                popt  = [0, 0, 0]
        
        if max(number_p_per_spill) > min_fraction:
            h_p, _, _ = ax.hist(number_p_per_spill/n_pot_per_trigger, bins = n_bins, label = lbl["ps"], color = "red", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_p, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "red", label = f"{lbl['ps']}: mean {popt[1]:.2f}, std {popt[2]:.2f}")
            except:
                popt  = [0, 0, 0]
        
        if max(number_D_per_spill) > min_fraction:
            h_D, _, _ = ax.hist(number_D_per_spill/n_pot_per_trigger, bins = n_bins, label = "Deuterium", color = "black", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_D, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "black", label = f"Deuterium: mean {popt[1]:.2f}, std {popt[2]:.2f}")
            except:
                popt = [0, 0, 0]
        
        
        if max(number_3He_per_spill) > min_fraction:
            h_3He, _, _ = ax.hist(number_3He_per_spill/n_pot_per_trigger, bins = n_bins, label = lbl["He3"], color = "magenta", histtype = "step")
            try:
                popt, pcov = fit_gaussian(h_3He, bin_centers)
                plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "magenta", label = f"3He: mean {popt[1]:.2f}, std {popt[2]:.2f}")
            except:
                popt = [0, 0, 0]
       
        
        
        ax.set_xlabel("Number of particles per 10^10 POT", fontsize = 20)
        ax.set_ylabel("Number of spills", fontsize = 20)
        ax.legend(fontsize = 14)
        ax.grid()

        ax.set_xlim(-2, max_limit+2)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        self.pdf_global.savefig(fig)

        print(f"Mean number of electrons per 10^10 POT: {sum(number_e_per_spill)/sum(n_pot_per_trigger):.2f}")
        plt.close()

        #for each spill also record the number of POT in total so we can output it with the rest of the beam analysis data 
        self.df["n_pot_per_spill"] = 0.0
        for s in self.df["spill_number"].unique():
            if s < len(n_pot_per_trigger):
                # print(s)
                # print("Spill: ", s, " number of POT:", n_pot_per_trigger[int(s)])
                self.df.loc[self.df["spill_number"]==s, "n_pot_per_spill"] = n_pot_per_trigger[int(s)] 
        
        
    def study_beam_structure(self):
        """This function studies the timing difference between consecutive triggers to estimate the probability that a given bunch holds a particle and also study the temporal structure of the beam. It will be useful to correlate this with the scalar information (at some point)"""
        
        #step 1: make a histogram of the timings for a given spill
        #work with all the data so we are not biased by our selection (though we will have for sure the online electron veto impact to keep in mind)  
        
        spill_number = 6
        df_with_diffs = self.df.copy()
        #self.df_all[self.df_all["spill_number"]==spill_number].copy() #.copy()
        df_with_diffs["ref0_dt"] = df_with_diffs.groupby("spill_number")["ref0_time"].transform(lambda x: x - x.min())
        df_with_diffs["ref1_dt"] = df_with_diffs.groupby("spill_number")["ref1_time"].transform(lambda x: x - x.min())


        
        bins = np.linspace (0, 30, 30)
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(df_with_diffs["ref0_dt"], bins = bins, color = "blue", label = "ref0", histtype="step")
        ax.hist(df_with_diffs["ref1_dt"], bins = bins, color = "black", label = "ref1", histtype="step")
        
        ax.set_xlabel("Time since begining of spill", fontsize=20)
        ax.set_ylabel("Number of triggers", fontsize=20)
        
        ax.legend(fontsize=20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n Ref times distribution, total number of spills {max(df_with_diffs['spill_number'])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        df_with_diffs["t0_times"] = df_with_diffs.groupby("spill_number")["t0_time"].transform(lambda x: x - x.min())
        df_with_diffs["t1_times"] = df_with_diffs.groupby("spill_number")["t1_time"].transform(lambda x: x - x.min())
        df_with_diffs["t5_times"] = df_with_diffs.groupby("spill_number")["t5_time"].transform(lambda x: x - x.min())
        df_with_diffs["t4_times"] = df_with_diffs.groupby("spill_number")["t4_time"].transform(lambda x: x - x.min())


        
        bins = np.linspace (-210, -140, 60)
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(df_with_diffs["t0_time"], bins = bins, color = "blue", label = "T0 average times", histtype="step")
        ax.hist(df_with_diffs["t1_time"], bins = bins, color = "black", label = "T1 average times", histtype="step")
        ax.hist(df_with_diffs["t4_time"], bins = bins, color = "green", label = "T4 average times", histtype="step")
        ax.hist(df_with_diffs["t5_time"], bins = bins, color = "red", label = "T5 average times", histtype="step")
        
        ax.set_xlabel("Recorded time", fontsize=20)
        ax.set_ylabel("Number of triggers", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        

        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
        ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
        ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
        ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylim(-210, -120)
        
        
        ax.set_ylabel("Time of events in given spill", fontsize=20)
        ax.set_xlabel("Trigger index within spill", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, axs = plt.subplots(2, 2, figsize = (18, 16))
        y = np.arange(len(df_with_diffs))
        y_bins = np.linspace(0, len(df_with_diffs), 50)
        
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") 
        
        bins_2d = [y_bins, np.linspace(-220, -140, 80)]
        axs[0,0].hist2d(y, df_with_diffs["t0_time"], bins = bins_2d, label = "T0 average times")
        axs[0,0].set_title("T0", fontsize = 20)
        axs[0,1].hist2d(y, df_with_diffs["t1_time"], bins = bins_2d,  label = "T1 average times")
        axs[0,1].set_title("T1", fontsize = 20)
        
        axs[1,0].hist2d(y, df_with_diffs["t4_time"], bins = bins_2d,  label = "T4 average times")
        axs[1,0].set_title("T4", fontsize = 20)
        
        axs[1,1].hist2d(y, df_with_diffs["t5_time"], bins = bins_2d,  label = "T5 average times")
        axs[1,1].set_title("T5", fontsize = 20)
        
        
        for axes in axs:
            for ax in axes:
#                 ax.set_ylim(-210, -120)        
                ax.set_ylabel("Time of events in given spill", fontsize=10)
                ax.set_xlabel("Trigger index within spill", fontsize=10)
#                 ax.legend(fontsize=10)
                #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        
        df_p = self.df[self.df["is_proton"]==True].copy()#
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        for s in [5, 6,7]:
        #I want to plot the tof of protons as a function of the registered T4 hit 
            ax.plot(df_p[df_p["spill_number"]==s]["t4_time"], df_p[df_p["spill_number"]==s]["tof_t0t4"], marker = "x", label = f"T0-T4 TOF vs T4 time spill {s}", linestyle = "")
        

        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("T4 time of event", fontsize=20)
        ax.set_xlim(-200,-125)
        ax.set_ylim(25, 50)
        ax.legend(fontsize=16)
        
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        for s in [5, 6,7]:
        #I want to plot the tof of protons as a function of the registered T4 hit 
            ax.plot(df_p[df_p["spill_number"]==s]["ref0_time"], df_p[df_p["spill_number"]==s]["tof_t0t4"], marker = "x", label = f"T0-T4 TOF vs ref0 time spill {s}", linestyle = "")
        

        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("ref0 time of event", fontsize=20)
#         ax.set_xlim(-200,-125)
        ax.set_ylim(25, 50)
        ax.legend(fontsize=16)
        
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        
        df_p = self.df[self.df["is_proton"]==True].copy()
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t0_time"], df_p["tof_t0t4"], marker = "x", color = "red", label = "T0-T4 TOF vs T0 time", linestyle = "")
        ax.plot(df_p["t5_time"], df_p["tof_t0t5"], marker = "x", color = "black", label = "T0-T5 TOF vs T5 time", linestyle = "")
        ax.plot(df_p["t4_time"], df_p["tof_t4t5"], marker = "x", color = "green", label = "T4-T5 TOF vs T4 time", linestyle = "")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("TS time of event", fontsize=20)
#         ax.set_xlim(-200,-125)
#         ax.set_ylim(25, 50)
        ax.set_xlim(-250,-125)
        ax.set_ylim(20, 60)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        bins = np.linspace (0, 60, 60)
        
        fig, ax = plt.subplots(figsize = (8, 6))

        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.hist(df_p["tof_t0t4"],  bins = bins, color = "blue", label = "T0-T4 TOF (protons)", histtype = "step")
        ax.hist(df_p["tof"], bins = bins, color = "red", label = "T0-T1 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t4t5"]+25, bins = bins, color = "green", label = "T4-T5 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t0t5"]-30, bins = bins, color = "black", label = "T0-T5 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t4t1"]+20, bins = bins, color = "magenta", label = "T1-T4 TOF(protons)", histtype = "step")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_xlabel("TOF", fontsize=20)
        ax.set_ylabel("Number of events", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t0_time"], df_p["t1_time"], marker = "x", color = "red", label = "T1 vs T0 time", linestyle = "")
        ax.plot(df_p["t4_time"], df_p["t0_time"], marker = "x", color = "black", label = "T4 vs T0 time", linestyle = "")
        ax.plot(df_p["t1_time"], df_p["t4_time"], marker = "x", color = "green", label = "T1 vs T4 time", linestyle = "")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylabel("TS time", fontsize=20)
        ax.set_xlabel("TS time", fontsize=20)
        ax.set_xlim(-220,-150)
        ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        #How the T4 tof as a function of the difference between T4 l and T4 r
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t4_l"]-df_p["t4_r"], df_p["tof_t0t4"], marker = "x", color = "red", label = "T0-T4 TOF vs T4L-T4R time", linestyle = "")
        ax.plot(df_p["t4_l"]-df_p["t4_r"], df_p["tof_t4t1"]+20, marker = "x", color = "black", label = "T1-T4 TOF vs T4L-T4R time", linestyle = "")

        
        
        ax.set_ylabel("TOF", fontsize=20)
        ax.set_xlabel("T4L-T4R", fontsize=20)
        ax.set_xlim(-15,15)
        ax.set_ylim(0,50)
#         ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
         #How the T4 tof as a function of the difference between T4 l and T4 r
        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace(-15, 15, 30)
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.hist(df_p["t4_l"]-df_p["t4_r"], color = "red", bins = bins, label = "T0-T4 TOF vs T4L-T4R time", histtype = "step")              
        
        ax.set_ylabel("Number of events", fontsize=20)
        ax.set_xlabel("T4L-T4R", fontsize=20)
        ax.set_xlim(-15,15)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.scatter(df_p["t4_l"]+40, df_p["t4_l"]-df_p["t0_time"], marker = "x", color = "red", label = "T4L-T0 vs T4L+40ns")
        ax.scatter(df_p["t4_r"], df_p["t4_r"]-df_p["t0_time"], marker = "x", color = "blue", label = "T4R-T0 vs T4R")
        
        ax.scatter(df_p["t4_l"]+20, df_p["t4_r"]-df_p["t0_time"], marker = "x", color = "green", label = "T4R-T0 vs T4L+20ns")
        
        ax.set_xlabel("T4 L or R", fontsize=20)
        ax.set_ylabel("T4L-T0 or TR-T0", fontsize=20)
        ax.set_xlim(-200,-100)
        ax.set_ylim(0,50)
#         ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
    def plot_event_quality_bitmask(self):
        '''Plot a histogram of the event quality bitmask and the digitisation issues'''
        fig, axes = plt.subplots(2, 1, figsize = (8, 6))
        bins = np.arange(-0.5, 65.5, 1)
        small_bins = np.arange(-0.5, 8, 1)
        #I want to plot the tof of protons as a function of the registered T4 hit 
        axes[0].hist(self.df_all["evt_quality_bitmask"], bins = bins,  histtype = "step", label = "Event quality bitmask")
        axes[1].hist(self.df_all["digi_issues_bitmask"], bins = small_bins,  histtype = "step", label = "Digitization issue bitmask")
        
        for i in range(2):
            axes[i].set_ylabel("Number of events", fontsize=20)
            axes[i].grid()
            axes[i].legend(fontsize=16)
            axes[i].set_yscale("log")
        
        axes[0].set_xlabel("Event quality bitmask value", fontsize=16)
        axes[1].set_xlabel("Digitization issue bitmask value", fontsize=16)

        accepted_triggers = np.sum(self.df_all["is_kept"])
        rejected_triggers = np.sum(np.where(self.df_all["is_kept"], 0, 1))
        
        axes[0].set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {accepted_triggers} accepted triggers; {rejected_triggers} rejected") 
        fig.tight_layout()
        self.pdf_global.savefig(fig)
        plt.close()
        
   
        
        
        
        
        
