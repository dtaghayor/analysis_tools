#!/usr/bin/env python3
"""
Full WCTE data production pipeline for a single run.

  Step 1 : Hardware or self-trigger processing (wf*, calibrate, dq)
  Step 2 : Beam monitor PID analysis              [optional, controlled by run info JSON]
  Step 3 : T5 offline analysis                    [optional, C++ executable in extern/]
  Step 4 : Merge                                  [placeholder — C++ merger not yet available]

  * wf (waveform processing) is hw-trigger only.

Trigger type and whether to run beam analysis are read from the run info JSON.

Use --steps to run a subset of the pipeline (default: all steps).
Earlier steps automatically include their downstream dependencies:
  --steps wf         -> wf + calibrate + dq
  --steps calibrate  -> calibrate + dq
  --steps dq         -> dq only
  --steps beam       -> beam only
  --steps t5         -> t5 only
  --steps dq beam t5 -> dq + beam + t5

Example usage:
  python run_pipeline.py -r 1602 \\
      -i /eos/.../WCTE_offline_R1602S0_VME_matched.root \\
      -o /eos/.../output \\
      --debug
"""

import os
import sys
import argparse
import subprocess

from run_hw_trigger_pipeline   import run_hw_pipeline
from run_self_trigger_pipeline import run_self_pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from analysis_tools.production_utils import get_run_info

ALL_STEPS = {"wf", "calibrate", "dq", "beam", "t5"}


def parse_args():
    parser = argparse.ArgumentParser(description="Full WCTE production pipeline for a single run")
    parser.add_argument("-r", "--run_number", required=True,
                        help="Run number to process")
    parser.add_argument("-i", "--input_files", required=True, nargs="+",
                        help="Raw WCTEReadoutWindows ROOT input files")
    parser.add_argument("-o", "--output_base", required=True,
                        help="Base output directory; <run_number>/ subdir created automatically")
    parser.add_argument("--steps", nargs="*", choices=sorted(ALL_STEPS),
                        default=None, metavar="STEP",
                        help="Steps to run (default: all). Choices: "
                             "wf (hw only), calibrate, dq, beam, t5. "
                             "Earlier steps automatically pull in downstream steps.")
    parser.add_argument("--not_official_const", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # ── Read run info from JSON ────────────────────────────────────────────────
    try:
        run_info = get_run_info(args.run_number)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    
    trigger_type       = run_info["trigger_type"]
    beam_analysis_type = run_info["beam_analysis_type"]

    kaon_runs_list = [2284, 2288, 2292, 2293, 2295, 2297, 2299, 2300, 2301, 2304, 2320, 2323, 2324, 2338, 2342]

    if args.run_number in kaon_runs_list:
        # Handle kaon runs differently
        beam_analysis_type = "kaon_run"


    # ── Resolve which steps to run ─────────────────────────────────────────────
    # Steps to pass to the hw_trigger or self_trigger pipeline
    if args.steps is None:
        # Default: run all pipeline steps (None tells the pipeline function to use its own default)
        trigger_type_pipeline_steps = None
        run_trigger_pipeline        = True
        run_vme_processing          = (beam_analysis_type in ["normal", "missing_act"])
        run_t5_analysis             = True
    else:
        #user has specified steps to run specific steps
        requested                   = set(args.steps)
        run_vme_processing          = (beam_analysis_type in ["normal", "missing_act", "kaon_run"]) and "beam" in requested
        run_t5_analysis             = "t5" in requested
        pipeline_requested          = requested - {"beam", "t5"}
        trigger_type_pipeline_steps = pipeline_requested or None
        run_trigger_pipeline        = bool(pipeline_requested)

    run_dir = os.path.join(args.output_base, str(args.run_number))
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Run {args.run_number}  |  trigger: {trigger_type}")
    if args.steps is None:
        print(f"  Trigger Specific Pipeline steps : all")
    else: 
        print(f"  Trigger Specific Pipeline steps : {', '.join(sorted(trigger_type_pipeline_steps)) if trigger_type_pipeline_steps else 'none'}")
    print(f"  Beam analysis  : {beam_analysis_type}")
    print(f"{'='*60}\n")

    # ── Step 1: Trigger-specific pipeline ─────────────────────────────────────
    if run_trigger_pipeline:
        if trigger_type == "hw":
            success = run_hw_pipeline(
                input_files=args.input_files,
                run_number=args.run_number,
                output_base=args.output_base,
                steps_to_run=trigger_type_pipeline_steps,
                debug=args.debug,
                not_official_const=args.not_official_const,
            )
        elif trigger_type == "self":
            success = run_self_pipeline(
                input_files=args.input_files,
                run_number=args.run_number,
                output_base=args.output_base,
                steps_to_run=trigger_type_pipeline_steps,
                debug=args.debug,
                not_official_const=args.not_official_const,
            )

        if not success:
            print(f"[ERROR] Pipeline step failed for run {args.run_number} — aborting.")

    # ── Step 2: Beam analysis ─────────────────────────────────────────────────
    if run_vme_processing:
        beam_dir = os.path.join(args.output_base, str(args.run_number), "beam_data")
        os.makedirs(beam_dir, exist_ok=True)
        beam_cmd = (
            [sys.executable, os.path.join(SCRIPT_DIR, "WCTE_beam_analysis.py"),
             "-r", str(args.run_number),
             "-o", beam_dir]
            + ["-i"] + args.input_files
            + (["--debug"] if args.debug else [])
        )
        if beam_analysis_type == "missing_act":
            beam_cmd.append("--no_acts")
        if beam_analysis_type == "kaon_run":
            beam_cmd.append("--is_kaon_run")
        result = subprocess.run(beam_cmd)
        if result.returncode != 0:
            print(f"[ERROR] Beam analysis failed for run {args.run_number}")

    # ── Step 3: T5 analysis ───────────────────────────────────────────────────
    if run_t5_analysis:
        t5_dir = os.path.join(args.output_base, str(args.run_number), "t5_analysis")
        os.makedirs(t5_dir, exist_ok=True)
        
        # Dynamically construct the path to the T5 executable relative to this script
        t5_executable = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "extern", "T5_analysis", "analyze_T5"))
        
            
        if not os.path.exists(t5_executable):
            print(f"[ERROR] T5 executable not found at: {t5_executable}")
            print("Please compile it using 'make' in the T5_analysis directory.")
            
            
        t5_cmd = [
            t5_executable,
            "-r", str(args.run_number)
        ]
        
        if args.input_files:
            t5_input_files = []
            if trigger_type == "hw":
                #hardeware trigger uses processed waveforms
                wf_dir = os.path.join(run_dir, "processed_waveforms")
                for infile in args.input_files:
                    base = os.path.splitext(os.path.basename(infile))[0]
                    t5_input_files.append(os.path.join(wf_dir, f"{base}_processed_waveforms.root"))
            else:
                #self trigger uses raw data
                t5_input_files = args.input_files

            for infile in t5_input_files:
                t5_cmd.extend(["-i", infile])
                
        t5_cmd.extend(["-o", t5_dir])
            
        if args.debug:
            t5_cmd.append("-d")
        print(f"\n[T5 ANALYSIS] Running T5 analysis for run {args.run_number}")
        result = subprocess.run(t5_cmd)
        if result.returncode != 0:
            print(f"[ERROR] T5 analysis failed for run {args.run_number}")


    print(f"\n*** Pipeline complete for run {args.run_number} ***\n")
