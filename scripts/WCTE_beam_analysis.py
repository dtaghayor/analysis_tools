'''## PID using the beam monitors
This code is the default way of identifying particles using the beam monitor. It is providing only a template event selection that is not optimised for any Physics analyses. It should serve as an exmple around which to build your own beam particle identification code. '''

#Step 0, import libraries
import numpy as np
import importlib
#this is the file with the necessary functions for performing PID 
import sys
#path to analysis tools - change with where your path is, though it might just work with mine on eos
sys.path.append("../") #neeed to acess the analysis_tools folded to acess
from analysis_tools import BeamAnalysis # as bm
# import cProfile

from analysis_tools import ReadBeamRunInfo 

import argparse
import os

# Number of events to read in debug mode
DEBUG_N_EVENTS = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Beam analysis configuration loader"
    )

    parser.add_argument(
        "-r", "--run_number", required=True, type=int,
        help="Run number to analyse"
    )
    parser.add_argument(
        "-i", "--input_files", required=True, nargs='+',
        help="Path(s) to WCTEReadoutWindows ROOT file(s)"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to write output ROOT file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=f"Enable debug mode: limits to {DEBUG_N_EVENTS} events"
    )
    
    parser.add_argument(
        "--no_acts", action="store_true",
        help=f"Enable minimum biais mode: downstream ACTs do not have the same refractive index"
    )

    return parser.parse_args()


#Step 0: read  from the json file which run you want and its properties

#set here if we require that having hits in T5 is required or not (typically set to TRUE for WCTE tank analyses)
require_t5 = True

args = parse_args()

if args.no_acts:
    run_info = ReadBeamRunInfo(no_acts = True)
else:
    run_info = ReadBeamRunInfo()

#The beam config holds information about the colimator slit status, in case it's needed
run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, beam_config = run_info.get_info_run_number(args.run_number)
run_info.print_run_summary(there_is_ACT5)

#choose the number of events to read in, set to -1 if you want to read all events
n_events = -1
if args.debug and n_events == -1:
    print(f"Debug mode: limiting to {DEBUG_N_EVENTS} events")
    n_events = DEBUG_N_EVENTS

#output_filename
os.makedirs(args.output_dir, exist_ok=True)

for input_file in args.input_files:

    if f"R{args.run_number}" not in os.path.basename(input_file):
        print(f"[ERROR] '{input_file}' does not match run number R{args.run_number}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = os.path.join(args.output_dir, f"{base}_beam_analysis.root")
    pdf_name = f"{base}_PID.pdf"

    print(f"\n{'#'*60}\n  {os.path.basename(input_file)}\n{'#'*60}")

    #Set up a beam analysis class
    ana = BeamAnalysis(run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, args.output_dir, pdf_name, is_beam_paper_analysis = False)

    #Store into memory the number of events desired,
    # set require_t5 to False if you do not require that the particle reaches T5
    ana.open_file(n_events, require_t5 = require_t5, input_file = input_file, output_file=output_filename)
    
    #In case no triggers are found then we do not need to run the analysis, otherwise we have already output the file in open file and closed the pdf properly 
    if ana.run_analysis:

        #Step 2: Adjust the 1pe calibration: need to check the accuracy on the plots
        ana.adjust_1pe_calibration()

        #Step 3: proton and heavier particle tagging with T0-T1 TOF
        #We need to tag protons before any other particles to avoid double-counting
        ana.tag_protons_TOF()
        #TODO: identify protons that produce knock-on electrons


        #Step 4: tag electrons using ACT0-2 finding the minimum in the cut line
        ana.tag_electrons_ACT02()

        #Step 5: check visually that the electron and proton removal makes sense in ACT35
        ana.plot_ACT35_left_vs_right()

        #Step 5: check visually that the electron and proton removal makes sense in ACT02
        ana.plot_ACT02_left_vs_right()

        #Step 6: make the muon/pion separation, using the muon tagger in case 
        #at least 0.5% of muons and pions are above the cut line (at hiogh momentum). This is necessary in case the 
        #Number of particles is too high to clearly see a minimum between the muons and pions
        ana.tag_muons_pions_ACT35()


        #This corrects any offset in the TOF (e.g. from cable length) that can cause the TOF 
        #of electrons to be different from L/c This has to be calibrated to give meaningful momentum 
        #estimates later on
        ana.measure_particle_TOF()

        #This function extimates both the mean momentum for each particle type and for each trigger
        #We take the the error on the tof for each trigger is the resolution of the TS0-TS1 measurement
        #Taken as the std of the gaussian fit to the electron TOF
        ana.estimate_particle_momentum()


        #estimate the number of events per POT
        ana.plot_number_particles_per_POT()


        #Check the number of triggers that are rejected and why
        ana.plot_event_quality_bitmask()

        #Step X: end_analysis, necessary to cleanly close files 
        ana.end_analysis()

        #Output to a root file
        ana.output_to_root(output_filename)
