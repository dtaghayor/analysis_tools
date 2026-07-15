#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

// ROOT Headers
#include "TBranch.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << " -r <run_number> -p <production_dir> -i <input_files...> -t "
               "<trigger_type> [-b] [-d]\n"
            << "Options:\n"
            << "  -r, --run        Run number to merge (required)\n"
            << "  -p, --dir        Base production directory (required, e.g. "
               "/eos/.../20260316_production_test)\n"
            << "  -i, --input      Raw WCTE input files (required)\n"
            << "  -t, --trigger    Trigger type: 'hw' or 'self' (required)\n"
            << "  -b, --beam       Flag indicating that beam code should also "
               "be merged (optional)\n"
            << "  -d, --debug      Enable debug output\n"
            << "  -h, --help       Print this help message\n";
}

int main(int argc, char **argv) {
  // -------------------------------------------------------------
  // Parse Input Arguments
  // -------------------------------------------------------------
  int run_number = -1;
  std::string prod_dir = "";
  std::vector<std::string> raw_files;
  std::string trigger_type = "";
  bool merge_beam = false;
  bool debug = false;

  const char *const short_opts = "r:p:i:t:bhd";
  const option long_opts[] = {{"run", required_argument, nullptr, 'r'},
                              {"dir", required_argument, nullptr, 'p'},
                              {"input", required_argument, nullptr, 'i'},
                              {"trigger", required_argument, nullptr, 't'},
                              {"beam", no_argument, nullptr, 'b'},
                              {"debug", no_argument, nullptr, 'd'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, no_argument, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) !=
         -1) {
    switch (opt) {
    case 'r':
      run_number = std::stoi(optarg);
      break;
    case 'p':
      prod_dir = optarg;
      break;
    case 'i':
      raw_files.push_back(optarg);
      // Consume any other arguments before the next '-'
      while (optind < argc && argv[optind][0] != '-') {
        raw_files.push_back(argv[optind]);
        optind++;
      }
      break;
    case 't':
      trigger_type = optarg;
      break;
    case 'b':
      merge_beam = true;
      break;
    case 'd':
      debug = true;
      break;
    case 'h':
    case '?':
      print_usage(argv[0]);
      return 1;
    }
  }

  if (run_number == -1 || prod_dir.empty() || raw_files.empty() ||
      trigger_type.empty()) {
    std::cerr << "[ERROR] Missing required arguments.\n\n";
    print_usage(argv[0]);
    return 1;
  }

  if (trigger_type != "hw" && trigger_type != "self") {
    std::cerr << "[ERROR] Trigger type must be 'hw' or 'self'. Got: '"
              << trigger_type << "'\n";
    return 1;
  }

  // -------------------------------------------------------------
  // Display Configuration
  // -------------------------------------------------------------
  std::cout << "=========================================\n";
  std::cout << "  Merge Configuration\n";
  std::cout << "=========================================\n";
  std::cout << "Run Number      : " << run_number << "\n";
  std::cout << "Production Dir  : " << prod_dir << "\n";
  std::cout << "Input Files     : " << raw_files.size() << " provided\n";
  std::cout << "Trigger Type    : " << trigger_type << "\n";
  std::cout << "Merge Beam Code : " << (merge_beam ? "Yes" : "No") << "\n";
  std::cout << "Debug Mode      : " << (debug ? "Yes" : "No") << "\n";
  std::cout << "=========================================\n\n";

  // Build the specific run directory exactly like the python scripts do
  TString run_dir = Form("%s/%d", prod_dir.c_str(), run_number);

  if (gSystem->AccessPathName(run_dir.Data())) {
    std::cerr << "[ERROR] Run directory does not exist or is not readable:\n  "
              << run_dir << "\n";
    return 1;
  }

  // -------------------------------------------------------------
  // Main Merging Logic to be populated by user
  // -------------------------------------------------------------

  // -------------------------------------------------------------
  // Memory Variables for Trees
  // -------------------------------------------------------------
  // BEAM variables
  double t0_time, t1_time, t4_time;
  double t0_time_second_hit, t1_time_second_hit, t4_time_second_hit;
  double time_t0_0, time_t0_1, time_t0_2, time_t0_3;
  double time_t1_0, time_t1_1, time_t1_2, time_t1_3;
  double time_t4_0, time_t4_1;
  double t5_time, t4_l_time, t4_r_time;
  double t4_l_second_hit, t4_r_second_hit;
  double act0_l_charge, act1_l_charge, act2_l_charge, act3_l_charge,
      act4_l_charge, act5_l_charge;
  double act0_r_charge, act1_r_charge, act2_r_charge, act3_r_charge,
      act4_r_charge, act5_r_charge;
  double act0_l_time, act0_r_time;
  double act1_l_time, act1_r_time;
  double act2_l_time, act2_r_time;
  double act3_l_time, act3_r_time;
  double act4_l_time, act4_r_time;
  double act5_l_time, act5_r_time;
  double mu_tag_l_time, mu_tag_r_time;
  double tof_t0t1, tof_t0t4, tof_t4t1, tof_t0t5, tof_t1t5, tof_t4t5;
  double ref0_time, ref1_time;
  double act_eveto, act_tagger, tof_corr, tof_t0t4_corr;

  double charge_t0_0, charge_t0_1, charge_t0_2, charge_t0_3;
  double charge_t1_0, charge_t1_1, charge_t1_2, charge_t1_3;
  double charge_t4_0, charge_t4_1;
  int event_id;
  double mu_tag_l_charge, mu_tag_r_charge, mu_tag_total;
  int spill_number, evt_quality_bitmask, digi_issues_bitmask;

  // T5 variables
  int T5_event_nr;
  int t5_hit_bitmask;
  int t5_n_main_bunch_particles;
  double t5_main_hit_time;
  double t5_main_hit_charge;
  double t5_main_hit_pos_x;
  double t5_main_hit_pos_y;
  double t5_main_hit_pos_x_error;
  double t5_main_hit_pos_y_error;
  std::vector<double> *t5_all_hits_pos_x = nullptr;
  std::vector<double> *t5_all_hits_pos_y = nullptr;
  std::vector<double> *t5_all_hits_pos_x_error = nullptr;
  std::vector<double> *t5_all_hits_pos_y_error = nullptr;
  std::vector<double> *t5_all_hits_time = nullptr;
  std::vector<double> *t5_all_hits_charge = nullptr;
  std::vector<int> *t5_all_hits_is_in_time_window = nullptr;
  std::vector<int> *t5_all_hits_is_in_bounds = nullptr;

  // Production 1.0 - out of date
  // int T5_particle_nr;
  // bool T5_HasValidHit;
  // bool T5_HasMultipleScintillatorsHit;
  // bool T5_HasOutOfTimeWindow;
  // bool T5_HasInTimeWindow;
  // std::vector<int> *T5_hit_is_in_bounds = nullptr;
  // std::vector<double> *T5_hit_pos_x = nullptr;
  // std::vector<double> *T5_hit_pos_y = nullptr;
  // std::vector<double> *T5_hit_time = nullptr;
  // std::vector<bool> *T5_secondary_hit_is_in_bounds = nullptr;
  // std::vector<double> *T5_secondary_hit_pos_x = nullptr;
  // std::vector<double> *T5_secondary_hit_pos_y = nullptr;
  // std::vector<double> *T5_secondary_hit_time = nullptr;

  // CALIBRATED HITS variables (Input Arrays)
  const int MAX_HITS = 200000;
  int nhit_pmt_calibrated_times = 0;
  double *hit_pmt_calibrated_times_in = new double[MAX_HITS];
  int nhit_pmt_charges = 0;
  double *hit_pmt_charges_in = new double[MAX_HITS];
  int nhit_mpmt_slot = 0;
  int *hit_mpmt_slot_in = new int[MAX_HITS];
  int nhit_pmt_pos = 0;
  int *hit_pmt_pos_in = new int[MAX_HITS];

  // CALIBRATED HITS variables (Output Vectors)
  std::vector<double> *out_hit_pmt_calibrated_times = new std::vector<double>();
  std::vector<double> *out_hit_pmt_charges = new std::vector<double>();
  std::vector<int> *out_hit_mpmt_slot = new std::vector<int>();
  std::vector<int> *out_hit_pmt_pos = new std::vector<int>();

  // DQ FLAGS variables
  int nhit_pmt_readout_mask = 0;
  int *hit_pmt_readout_mask_in = new int[MAX_HITS];
  int window_data_quality_mask = 0;
  int dq_readout_number = 0;
  std::vector<int> *out_hit_pmt_readout_mask = new std::vector<int>();

  // CROSS-TREE variables
  int raw_readout_number = 0;
  int cal_readout_number = 0;

  // TRIGGER BOARD HITS AND WAVEFORMS (Input vectors)
  std::vector<int> *raw_hit_card_ids = nullptr;
  std::vector<int> *raw_hit_channel_ids = nullptr;
  std::vector<float> *raw_hit_charges = nullptr;
  std::vector<double> *raw_hit_times = nullptr;

  std::vector<int> *raw_waveform_card_ids = nullptr;
  std::vector<int> *raw_waveform_channel_ids = nullptr;
  std::vector<double> *raw_waveform_times = nullptr;
  std::vector<std::vector<double>> *raw_waveforms = nullptr;

  // TRIGGER BOARD HITS AND WAVEFORMS (Output vectors)
  std::vector<int> *tb_hit_card_ids = new std::vector<int>();
  std::vector<int> *tb_hit_channel_ids = new std::vector<int>();
  std::vector<float> *tb_hit_charges = new std::vector<float>();
  std::vector<double> *tb_hit_times = new std::vector<double>();

  std::vector<int> *tb_waveform_card_ids = new std::vector<int>();
  std::vector<int> *tb_waveform_channel_ids = new std::vector<int>();
  std::vector<double> *tb_waveform_times = new std::vector<double>();
  std::vector<std::vector<double>> *tb_waveforms =
      new std::vector<std::vector<double>>();

  // PROCESSED WAVEFORMS variables (Output Vectors for HW Trigger Board Hits)
  int nhit_card_wf = 0;
  int *hit_card_wf = new int[MAX_HITS];
  int nhit_chan_wf = 0;
  int *hit_chan_wf = new int[MAX_HITS];
  int nhit_charge_wf = 0;
  double *hit_charge_wf = new double[MAX_HITS];
  int nhit_time_wf = 0;
  double *hit_time_wf = new double[MAX_HITS];

  // Set up IO Paths
  TString cal_dir = Form("%s/calibrated_hits", run_dir.Data());
  TString dq_dir = Form("%s/dq_flags", run_dir.Data());
  TString t5_dir = Form("%s/t5_analysis", run_dir.Data());
  TString beam_dir = Form("%s/beam_data", run_dir.Data());
  TString wf_dir = Form("%s/processed_waveforms", run_dir.Data());

  for (const auto &raw_file_str : raw_files) {
    TString raw_file_path = raw_file_str.c_str();
    TString filename = gSystem->BaseName(raw_file_path);
    TString stem = filename.ReplaceAll(".root", "");

    std::cout << "\n=========================================\n"
              << "[INFO] Processing file: " << raw_file_path << "\n"
              << "=========================================\n";

    TString beam_file_path =
        Form("%s/%s_beam_analysis.root", beam_dir.Data(), stem.Data());
    TString t5_file_path = Form("%s/%s_T5.root", t5_dir.Data(), stem.Data());
    TString cal_file_path =
        Form("%s/%s_calibrated_hits.root", cal_dir.Data(), stem.Data());
    if (trigger_type == "hw") {
      t5_file_path =
          Form("%s/%s_processed_waveforms_T5.root", t5_dir.Data(), stem.Data());
      cal_file_path = Form("%s/%s_processed_waveforms_calibrated_hits.root",
                           cal_dir.Data(), stem.Data());
    }

    TFile *f_beam = merge_beam ? new TFile(beam_file_path, "READ") : nullptr;
    TFile *f_t5 = new TFile(t5_file_path, "READ");
    TFile *f_cal = new TFile(cal_file_path, "READ");
    TFile *f_raw = new TFile(raw_file_path, "READ");

    TString dq_file_path = Form("%s/%s_%s_trigger_dq_flags.root", dq_dir.Data(),
                                stem.Data(), trigger_type.c_str());
    TFile *f_dq = new TFile(dq_file_path, "READ");

    if (merge_beam && (!f_beam || f_beam->IsZombie())) {
      std::cerr << "[ERROR] Cannot open beam file: " << beam_file_path << "\n";
      return 1;
    }
    if (!f_t5 || f_t5->IsZombie()) {
      std::cerr << "[ERROR] Cannot open T5 file: " << t5_file_path << "\n";
      return 1;
    }
    if (!f_cal || f_cal->IsZombie()) {
      std::cerr << "[ERROR] Cannot open Calibrated hits file: " << cal_file_path
                << "\n";
      return 1;
    }
    if (!f_raw || f_raw->IsZombie()) {
      std::cerr << "[ERROR] Cannot open raw file: " << raw_file_path << "\n";
      return 1;
    }
    if (!f_dq || f_dq->IsZombie()) {
      std::cerr << "[ERROR] Cannot open DQ flags file: " << dq_file_path
                << "\n";
      return 1;
    }

    TString wf_file_path =
        Form("%s/%s_processed_waveforms.root", wf_dir.Data(), stem.Data());
    TFile *f_wf = nullptr;
    TTree *t_wf = nullptr;
    if (trigger_type == "hw") {
      f_wf = new TFile(wf_file_path, "READ");
      if (!f_wf || f_wf->IsZombie()) {
        std::cerr << "[ERROR] Cannot open Processed Waveforms file: "
                  << wf_file_path << "\n";
        throw std::runtime_error(std::string("Cannot open file: ") +
                                 wf_file_path.Data());
      }
      t_wf = (TTree *)f_wf->Get("ProcessedWaveforms");
      if (!t_wf) {
        std::cerr
            << "[ERROR] 'ProcessedWaveforms' tree not found in wf file.\n";
        throw std::runtime_error("Tree ProcessedWaveforms missing");
      }
    }

    TTree *t_beam =
        merge_beam ? (TTree *)f_beam->Get("beam_analysis") : nullptr;
    TTree *t_t5 = (TTree *)f_t5->Get("T5_Events");
    TTree *t_cal = (TTree *)f_cal->Get("CalibratedHits");
    TTree *t_raw = (TTree *)f_raw->Get("WCTEReadoutWindows");
    TTree *t_dq = (TTree *)f_dq->Get("DataQualityFlags");

    if (merge_beam && !t_beam) {
      std::cerr << "[ERROR] 'beam_analysis' tree not found in beam file.\n";
      return 1;
    }
    if (!t_t5) {
      std::cerr << "[ERROR] 'T5_Events' tree not found in T5 file.\n";
      return 1;
    }
    if (!t_cal) {
      std::cerr << "[ERROR] 'CalibratedHits' tree not found in calibrated hits "
                   "file.\n";
      return 1;
    }
    if (!t_raw) {
      std::cerr << "[ERROR] 'WCTEReadoutWindows' tree not found in raw file.\n";
      return 1;
    }
    if (!t_dq) {
      std::cerr
          << "[ERROR] 'DataQualityFlags' tree not found in DQ flags file.\n";
      return 1;
    }
    Long64_t n_entries = t_raw->GetEntries();
    if (!debug) {
      if (merge_beam && t_beam->GetEntries() != n_entries) {
        std::cerr << "[ERROR] Mismatch in entries! Raw: " << n_entries
                  << " Beam: " << t_beam->GetEntries() << "\n";
        return 1;
      }
      if (t_cal->GetEntries() != n_entries) {
        std::cerr << "[ERROR] Mismatch in entries! Raw: " << n_entries
                  << " CalibratedHits: " << t_cal->GetEntries() << "\n";
        return 1;
      }
      if (t_t5->GetEntries() != n_entries) {
        std::cerr << "[ERROR] Mismatch in entries! Raw: " << n_entries
                  << " T5: " << t_t5->GetEntries() << "\n";
        return 1;
      }
      if (t_dq->GetEntries() != n_entries) {
        std::cerr << "[ERROR] Mismatch in entries! Raw: " << n_entries
                  << " DQ: " << t_dq->GetEntries() << "\n";
        return 1;
      }
      if (trigger_type == "hw" && t_wf->GetEntries() != n_entries) {
        std::cerr << "[ERROR] Mismatch in entries! Raw: " << n_entries
                  << " WF: " << t_wf->GetEntries() << "\n";
        return 1;
      }
    } else {
      std::cout << "Debug mode entries in each file " << t_raw->GetEntries()
                << " " << t_cal->GetEntries() << " " << t_t5->GetEntries()
                << " " << t_dq->GetEntries() << "\n";
      n_entries = std::min({t_raw->GetEntries(), t_cal->GetEntries(),
                            t_t5->GetEntries(), t_dq->GetEntries()});

      if (merge_beam) {
        std::cout << "Debug mode entries in beam file " << t_beam->GetEntries()
                  << "\n";
        n_entries = std::min(n_entries, t_beam->GetEntries());
      }
    }

    // Set Branches - DQ FLAGS
    t_dq->SetBranchAddress("nhit_pmt_readout_mask", &nhit_pmt_readout_mask);
    t_dq->SetBranchAddress("hit_pmt_readout_mask", hit_pmt_readout_mask_in);
    t_dq->SetBranchAddress("window_data_quality_mask",
                           &window_data_quality_mask);
    t_dq->SetBranchAddress("readout_number", &dq_readout_number);

    // Set Branches - RAW
    t_raw->SetBranchAddress("readout_number", &raw_readout_number);

    t_raw->SetBranchAddress("hit_mpmt_card_ids", &raw_hit_card_ids);
    t_raw->SetBranchAddress("hit_pmt_channel_ids", &raw_hit_channel_ids);
    t_raw->SetBranchAddress("hit_pmt_charges", &raw_hit_charges);
    t_raw->SetBranchAddress("hit_pmt_times", &raw_hit_times);

    t_raw->SetBranchAddress("pmt_waveform_mpmt_card_ids",
                            &raw_waveform_card_ids);
    t_raw->SetBranchAddress("pmt_waveform_pmt_channel_ids",
                            &raw_waveform_channel_ids);
    t_raw->SetBranchAddress("pmt_waveform_times", &raw_waveform_times);
    t_raw->SetBranchAddress("pmt_waveforms", &raw_waveforms);

    // Set Branches - CALIBRATED HITS
    t_cal->SetBranchAddress("readout_number", &cal_readout_number);
    t_cal->SetBranchAddress("nhit_pmt_calibrated_times",
                            &nhit_pmt_calibrated_times);
    t_cal->SetBranchAddress("hit_pmt_calibrated_times",
                            hit_pmt_calibrated_times_in);
    t_cal->SetBranchAddress("nhit_pmt_charges", &nhit_pmt_charges);
    t_cal->SetBranchAddress("hit_pmt_charges", hit_pmt_charges_in);
    t_cal->SetBranchAddress("nhit_mpmt_slot", &nhit_mpmt_slot);
    t_cal->SetBranchAddress("hit_mpmt_slot", hit_mpmt_slot_in);
    t_cal->SetBranchAddress("nhit_pmt_pos", &nhit_pmt_pos);
    t_cal->SetBranchAddress("hit_pmt_pos", hit_pmt_pos_in);

    // Set Branches - PROCESSED WAVEFORMS
    if (trigger_type == "hw") {
      t_wf->SetBranchAddress("nhit_card", &nhit_card_wf);
      t_wf->SetBranchAddress("hit_card", hit_card_wf);
      t_wf->SetBranchAddress("nhit_chan", &nhit_chan_wf);
      t_wf->SetBranchAddress("hit_chan", hit_chan_wf);
      t_wf->SetBranchAddress("nhit_charge", &nhit_charge_wf);
      t_wf->SetBranchAddress("hit_charge", hit_charge_wf);
      t_wf->SetBranchAddress("nhit_time", &nhit_time_wf);
      t_wf->SetBranchAddress("hit_time", hit_time_wf);
    }

    // Set Branches - BEAM
    if (merge_beam) {
      t_beam->SetBranchAddress("t0_time", &t0_time);
      t_beam->SetBranchAddress("t1_time", &t1_time);
      t_beam->SetBranchAddress("t4_time", &t4_time);
      t_beam->SetBranchAddress("t0_time_second_hit", &t0_time_second_hit);
      t_beam->SetBranchAddress("t1_time_second_hit", &t1_time_second_hit);
      t_beam->SetBranchAddress("t4_time_second_hit", &t4_time_second_hit);
      t_beam->SetBranchAddress("time_t0_0", &time_t0_0);
      t_beam->SetBranchAddress("time_t0_1", &time_t0_1);
      t_beam->SetBranchAddress("time_t0_2", &time_t0_2);
      t_beam->SetBranchAddress("time_t0_3", &time_t0_3);
      t_beam->SetBranchAddress("time_t1_0", &time_t1_0);
      t_beam->SetBranchAddress("time_t1_1", &time_t1_1);
      t_beam->SetBranchAddress("time_t1_2", &time_t1_2);
      t_beam->SetBranchAddress("time_t1_3", &time_t1_3);
      t_beam->SetBranchAddress("time_t4_0", &time_t4_0);
      t_beam->SetBranchAddress("time_t4_1", &time_t4_1);
      t_beam->SetBranchAddress("t5_time", &t5_time);
      t_beam->SetBranchAddress("t4_l_time", &t4_l_time);
      t_beam->SetBranchAddress("t4_r_time", &t4_r_time);
      t_beam->SetBranchAddress("t4_l_second_hit", &t4_l_second_hit);
      t_beam->SetBranchAddress("t4_r_second_hit", &t4_r_second_hit);
      t_beam->SetBranchAddress("act0_l_charge", &act0_l_charge);
      t_beam->SetBranchAddress("act1_l_charge", &act1_l_charge);
      t_beam->SetBranchAddress("act2_l_charge", &act2_l_charge);
      t_beam->SetBranchAddress("act3_l_charge", &act3_l_charge);
      t_beam->SetBranchAddress("act4_l_charge", &act4_l_charge);
      t_beam->SetBranchAddress("act5_l_charge", &act5_l_charge);
      t_beam->SetBranchAddress("act0_r_charge", &act0_r_charge);
      t_beam->SetBranchAddress("act1_r_charge", &act1_r_charge);
      t_beam->SetBranchAddress("act2_r_charge", &act2_r_charge);
      t_beam->SetBranchAddress("act3_r_charge", &act3_r_charge);
      t_beam->SetBranchAddress("act4_r_charge", &act4_r_charge);
      t_beam->SetBranchAddress("act5_r_charge", &act5_r_charge);
      t_beam->SetBranchAddress("act0_l_time", &act0_l_time);
      t_beam->SetBranchAddress("act0_r_time", &act0_r_time);
      t_beam->SetBranchAddress("act1_l_time", &act1_l_time);
      t_beam->SetBranchAddress("act1_r_time", &act1_r_time);
      t_beam->SetBranchAddress("act2_l_time", &act2_l_time);
      t_beam->SetBranchAddress("act2_r_time", &act2_r_time);
      t_beam->SetBranchAddress("act3_l_time", &act3_l_time);
      t_beam->SetBranchAddress("act3_r_time", &act3_r_time);
      t_beam->SetBranchAddress("act4_l_time", &act4_l_time);
      t_beam->SetBranchAddress("act4_r_time", &act4_r_time);
      t_beam->SetBranchAddress("act5_l_time", &act5_l_time);
      t_beam->SetBranchAddress("act5_r_time", &act5_r_time);
      t_beam->SetBranchAddress("mu_tag_l_time", &mu_tag_l_time);
      t_beam->SetBranchAddress("mu_tag_r_time", &mu_tag_r_time);
      t_beam->SetBranchAddress("tof_t0t1", &tof_t0t1);
      t_beam->SetBranchAddress("tof_t0t4", &tof_t0t4);
      t_beam->SetBranchAddress("tof_t4t1", &tof_t4t1);
      t_beam->SetBranchAddress("tof_t0t5", &tof_t0t5);
      t_beam->SetBranchAddress("tof_t1t5", &tof_t1t5);
      t_beam->SetBranchAddress("tof_t4t5", &tof_t4t5);
      t_beam->SetBranchAddress("ref0_time", &ref0_time);
      t_beam->SetBranchAddress("ref1_time", &ref1_time);
      t_beam->SetBranchAddress("act_eveto", &act_eveto);
      t_beam->SetBranchAddress("act_tagger", &act_tagger);
      t_beam->SetBranchAddress("tof_corr", &tof_corr);
      t_beam->SetBranchAddress("tof_t0t4_corr", &tof_t0t4_corr);
      t_beam->SetBranchAddress("charge_t0_0", &charge_t0_0);
      t_beam->SetBranchAddress("charge_t0_1", &charge_t0_1);
      t_beam->SetBranchAddress("charge_t0_2", &charge_t0_2);
      t_beam->SetBranchAddress("charge_t0_3", &charge_t0_3);
      t_beam->SetBranchAddress("charge_t1_0", &charge_t1_0);
      t_beam->SetBranchAddress("charge_t1_1", &charge_t1_1);
      t_beam->SetBranchAddress("charge_t1_2", &charge_t1_2);
      t_beam->SetBranchAddress("charge_t1_3", &charge_t1_3);
      t_beam->SetBranchAddress("charge_t4_0", &charge_t4_0);
      t_beam->SetBranchAddress("charge_t4_1", &charge_t4_1);
      t_beam->SetBranchAddress("event_id", &event_id);
      t_beam->SetBranchAddress("mu_tag_l_charge", &mu_tag_l_charge);
      t_beam->SetBranchAddress("mu_tag_r_charge", &mu_tag_r_charge);
      t_beam->SetBranchAddress("mu_tag_total", &mu_tag_total);
      t_beam->SetBranchAddress("spill_number", &spill_number);
      t_beam->SetBranchAddress("evt_quality_bitmask", &evt_quality_bitmask);
      t_beam->SetBranchAddress("digi_issues_bitmask", &digi_issues_bitmask);
    }

    // Set Branches - T5
    t_t5->SetBranchAddress("event_nr", &T5_event_nr);
    t_t5->SetBranchAddress("t5_hit_bitmask", &t5_hit_bitmask);
    t_t5->SetBranchAddress("t5_n_main_bunch_particles",
                           &t5_n_main_bunch_particles);
    t_t5->SetBranchAddress("t5_main_hit_time", &t5_main_hit_time);
    t_t5->SetBranchAddress("t5_main_hit_charge", &t5_main_hit_charge);
    t_t5->SetBranchAddress("t5_main_hit_pos_x", &t5_main_hit_pos_x);
    t_t5->SetBranchAddress("t5_main_hit_pos_y", &t5_main_hit_pos_y);
    t_t5->SetBranchAddress("t5_main_hit_pos_x_error", &t5_main_hit_pos_x_error);
    t_t5->SetBranchAddress("t5_main_hit_pos_y_error", &t5_main_hit_pos_y_error);
    t_t5->SetBranchAddress("t5_all_hits_pos_x", &t5_all_hits_pos_x);
    t_t5->SetBranchAddress("t5_all_hits_pos_y", &t5_all_hits_pos_y);
    t_t5->SetBranchAddress("t5_all_hits_pos_x_error", &t5_all_hits_pos_x_error);
    t_t5->SetBranchAddress("t5_all_hits_pos_y_error", &t5_all_hits_pos_y_error);
    t_t5->SetBranchAddress("t5_all_hits_time", &t5_all_hits_time);
    t_t5->SetBranchAddress("t5_all_hits_charge", &t5_all_hits_charge);
    t_t5->SetBranchAddress("t5_all_hits_is_in_time_window", &t5_all_hits_is_in_time_window);
    t_t5->SetBranchAddress("t5_all_hits_is_in_bounds", &t5_all_hits_is_in_bounds);


    // t_t5->SetBranchAddress("T5_particle_nr", &T5_particle_nr);
    // t_t5->SetBranchAddress("T5_HasValidHit", &T5_HasValidHit);
    // t_t5->SetBranchAddress("T5_HasMultipleScintillatorsHit",
    //                        &T5_HasMultipleScintillatorsHit);
    // t_t5->SetBranchAddress("T5_HasOutOfTimeWindow", &T5_HasOutOfTimeWindow);
    // t_t5->SetBranchAddress("T5_HasInTimeWindow", &T5_HasInTimeWindow);
    // t_t5->SetBranchAddress("T5_hit_is_in_bounds", &T5_hit_is_in_bounds);
    // t_t5->SetBranchAddress("T5_hit_pos_x", &T5_hit_pos_x);
    // t_t5->SetBranchAddress("T5_hit_pos_y", &T5_hit_pos_y);
    // t_t5->SetBranchAddress("T5_hit_time", &T5_hit_time);
    // t_t5->SetBranchAddress("T5_secondary_hit_is_in_bounds",
    //                        &T5_secondary_hit_is_in_bounds);
    // t_t5->SetBranchAddress("T5_secondary_hit_pos_x", &T5_secondary_hit_pos_x);
    // t_t5->SetBranchAddress("T5_secondary_hit_pos_y", &T5_secondary_hit_pos_y);
    // t_t5->SetBranchAddress("T5_secondary_hit_time", &T5_secondary_hit_time);

    // Prepare Output File & Tree
    TString suffix = stem;
    int suffix_idx = suffix.Index("VME_matched");
    if (suffix_idx != kNPOS) {
      suffix.Remove(0, suffix_idx + 11);
    } else {
      suffix = "";
    }
    TString out_file_path = Form("%s/WCTE_merged_production_R%d%s.root",
                                 run_dir.Data(), run_number, suffix.Data());
    TFile *f_out = new TFile(out_file_path, "RECREATE");

    // Clone raw tree structure
    t_raw->SetBranchStatus("*", 0);
    t_raw->SetBranchStatus("window_time", 1);
    t_raw->SetBranchStatus("start_counter", 1);
    t_raw->SetBranchStatus("run_id", 1);
    t_raw->SetBranchStatus("sub_run_id", 1);
    t_raw->SetBranchStatus("spill_counter", 1);
    t_raw->SetBranchStatus("event_number", 1);
    t_raw->SetBranchStatus("readout_number", 1);

    TTree *t_out = t_raw->CloneTree(0);
    t_out->SetName("WCTEReadoutWindows");
    t_out->SetTitle("Merged Data Production Pipeline Output");

    // Re-enable all branches if needed later (though we just write these
    // specified ones)
    t_raw->SetBranchStatus("*", 1);

    // Create Output Branches for CALIBRATED HITS (Vectors)
    t_out->Branch("hit_pmt_calibrated_times", &out_hit_pmt_calibrated_times);
    t_out->Branch("hit_pmt_charges", &out_hit_pmt_charges);
    t_out->Branch("hit_mpmt_slot_ids", &out_hit_mpmt_slot);
    t_out->Branch("hit_pmt_position_ids", &out_hit_pmt_pos);

    // Set Output Branches for DQ FLAGS
    t_out->Branch("hit_pmt_readout_mask", &out_hit_pmt_readout_mask);
    t_out->Branch("window_data_quality_mask", &window_data_quality_mask);

    // Create Trigger Board Branches
    t_out->Branch("trigger_board_hit_card_ids", &tb_hit_card_ids);
    t_out->Branch("trigger_board_hit_channel_ids", &tb_hit_channel_ids);
    t_out->Branch("trigger_board_hit_charges", &tb_hit_charges);
    t_out->Branch("trigger_board_hit_times", &tb_hit_times);

    t_out->Branch("trigger_board_waveform_card_ids", &tb_waveform_card_ids);
    t_out->Branch("trigger_board_waveform_channel_ids",
                  &tb_waveform_channel_ids);
    t_out->Branch("trigger_board_waveform_times", &tb_waveform_times);
    t_out->Branch("trigger_board_waveforms", &tb_waveforms);

    // Create vme_ prefixed Output Branches for Beam
    if (merge_beam) {
      t_out->Branch("vme_t0_time", &t0_time);
      t_out->Branch("vme_t1_time", &t1_time);
      t_out->Branch("vme_t4_time", &t4_time);
      t_out->Branch("vme_t0_time_second_hit", &t0_time_second_hit);
      t_out->Branch("vme_t1_time_second_hit", &t1_time_second_hit);
      t_out->Branch("vme_t4_time_second_hit", &t4_time_second_hit);
      t_out->Branch("vme_time_t0_0", &time_t0_0);
      t_out->Branch("vme_time_t0_1", &time_t0_1);
      t_out->Branch("vme_time_t0_2", &time_t0_2);
      t_out->Branch("vme_time_t0_3", &time_t0_3);
      t_out->Branch("vme_time_t1_0", &time_t1_0);
      t_out->Branch("vme_time_t1_1", &time_t1_1);
      t_out->Branch("vme_time_t1_2", &time_t1_2);
      t_out->Branch("vme_time_t1_3", &time_t1_3);
      t_out->Branch("vme_time_t4_0", &time_t4_0);
      t_out->Branch("vme_time_t4_1", &time_t4_1);
      t_out->Branch("vme_t5_time", &t5_time);
      t_out->Branch("vme_t4_l_time", &t4_l_time);
      t_out->Branch("vme_t4_r_time", &t4_r_time);
      t_out->Branch("vme_t4_l_second_hit", &t4_l_second_hit);
      t_out->Branch("vme_t4_r_second_hit", &t4_r_second_hit);
      t_out->Branch("vme_act0_l_charge", &act0_l_charge);
      t_out->Branch("vme_act1_l_charge", &act1_l_charge);
      t_out->Branch("vme_act2_l_charge", &act2_l_charge);
      t_out->Branch("vme_act3_l_charge", &act3_l_charge);
      t_out->Branch("vme_act4_l_charge", &act4_l_charge);
      t_out->Branch("vme_act5_l_charge", &act5_l_charge);
      t_out->Branch("vme_act0_r_charge", &act0_r_charge);
      t_out->Branch("vme_act1_r_charge", &act1_r_charge);
      t_out->Branch("vme_act2_r_charge", &act2_r_charge);
      t_out->Branch("vme_act3_r_charge", &act3_r_charge);
      t_out->Branch("vme_act4_r_charge", &act4_r_charge);
      t_out->Branch("vme_act5_r_charge", &act5_r_charge);
      t_out->Branch("vme_act0_l_time", &act0_l_time);
      t_out->Branch("vme_act0_r_time", &act0_r_time);
      t_out->Branch("vme_act1_l_time", &act1_l_time);
      t_out->Branch("vme_act1_r_time", &act1_r_time);
      t_out->Branch("vme_act2_l_time", &act2_l_time);
      t_out->Branch("vme_act2_r_time", &act2_r_time);
      t_out->Branch("vme_act3_l_time", &act3_l_time);
      t_out->Branch("vme_act3_r_time", &act3_r_time);
      t_out->Branch("vme_act4_l_time", &act4_l_time);
      t_out->Branch("vme_act4_r_time", &act4_r_time);
      t_out->Branch("vme_act5_l_time", &act5_l_time);
      t_out->Branch("vme_act5_r_time", &act5_r_time);
      t_out->Branch("vme_mu_tag_l_time", &mu_tag_l_time);
      t_out->Branch("vme_mu_tag_r_time", &mu_tag_r_time);
      t_out->Branch("vme_tof_t0t1", &tof_t0t1);
      t_out->Branch("vme_tof_t0t4", &tof_t0t4);
      t_out->Branch("vme_tof_t4t1", &tof_t4t1);
      t_out->Branch("vme_tof_t0t5", &tof_t0t5);
      t_out->Branch("vme_tof_t1t5", &tof_t1t5);
      t_out->Branch("vme_tof_t4t5", &tof_t4t5);
      t_out->Branch("vme_ref0_time", &ref0_time);
      t_out->Branch("vme_ref1_time", &ref1_time);
      t_out->Branch("vme_act_eveto", &act_eveto);
      t_out->Branch("vme_act_tagger", &act_tagger);
      t_out->Branch("vme_tof_corr", &tof_corr);
      t_out->Branch("vme_tof_t0t4_corr", &tof_t0t4_corr);
      t_out->Branch("vme_charge_t0_0", &charge_t0_0);
      t_out->Branch("vme_charge_t0_1", &charge_t0_1);
      t_out->Branch("vme_charge_t0_2", &charge_t0_2);
      t_out->Branch("vme_charge_t0_3", &charge_t0_3);
      t_out->Branch("vme_charge_t1_0", &charge_t1_0);
      t_out->Branch("vme_charge_t1_1", &charge_t1_1);
      t_out->Branch("vme_charge_t1_2", &charge_t1_2);
      t_out->Branch("vme_charge_t1_3", &charge_t1_3);
      t_out->Branch("vme_charge_t4_0", &charge_t4_0);
      t_out->Branch("vme_charge_t4_1", &charge_t4_1);
      t_out->Branch("vme_event_id", &event_id);
      t_out->Branch("vme_mu_tag_l_charge", &mu_tag_l_charge);
      t_out->Branch("vme_mu_tag_r_charge", &mu_tag_r_charge);
      t_out->Branch("vme_mu_tag_total", &mu_tag_total);
      t_out->Branch("vme_spill_number", &spill_number);
      t_out->Branch("vme_evt_quality_bitmask", &evt_quality_bitmask);
      t_out->Branch("vme_digi_issues_bitmask", &digi_issues_bitmask);
    }

    // Create vme_ prefixed Output Branches for T5
    t_out->Branch("T5_event_nr", &T5_event_nr);
    t_out->Branch("T5_hit_bitmask", &t5_hit_bitmask);
    t_out->Branch("T5_n_main_bunch_particles", &t5_n_main_bunch_particles);
    t_out->Branch("T5_main_hit_time", &t5_main_hit_time);
    t_out->Branch("T5_main_hit_charge", &t5_main_hit_charge);
    t_out->Branch("T5_main_hit_pos_x", &t5_main_hit_pos_x);
    t_out->Branch("T5_main_hit_pos_y", &t5_main_hit_pos_y);
    t_out->Branch("T5_main_hit_pos_x_error", &t5_main_hit_pos_x_error);
    t_out->Branch("T5_main_hit_pos_y_error", &t5_main_hit_pos_y_error);
    t_out->Branch("T5_all_hits_pos_x", &t5_all_hits_pos_x);
    t_out->Branch("T5_all_hits_pos_y", &t5_all_hits_pos_y);
    t_out->Branch("T5_all_hits_pos_x_error", &t5_all_hits_pos_x_error);
    t_out->Branch("T5_all_hits_pos_y_error", &t5_all_hits_pos_y_error);
    t_out->Branch("T5_all_hits_time", &t5_all_hits_time);
    t_out->Branch("T5_all_hits_charge", &t5_all_hits_charge);
    t_out->Branch("T5_all_hits_is_in_time_window", &t5_all_hits_is_in_time_window);
    t_out->Branch("T5_all_hits_is_in_bounds", &t5_all_hits_is_in_bounds);


    // t_out->Branch("T5_particle_nr", &T5_particle_nr);
    // t_out->Branch("T5_HasValidHit", &T5_HasValidHit);
    // t_out->Branch("T5_HasMultipleScintillatorsHit",
    //               &T5_HasMultipleScintillatorsHit);
    // t_out->Branch("T5_HasOutOfTimeWindow", &T5_HasOutOfTimeWindow);
    // t_out->Branch("T5_HasInTimeWindow", &T5_HasInTimeWindow);
    // t_out->Branch("T5_hit_is_in_bounds", &T5_hit_is_in_bounds);
    // t_out->Branch("T5_hit_pos_x", &T5_hit_pos_x);
    // t_out->Branch("T5_hit_pos_y", &T5_hit_pos_y);
    // t_out->Branch("T5_hit_time", &T5_hit_time);
    // t_out->Branch("T5_secondary_hit_is_in_bounds",
    //               &T5_secondary_hit_is_in_bounds);
    // t_out->Branch("T5_secondary_hit_pos_x", &T5_secondary_hit_pos_x);
    // t_out->Branch("T5_secondary_hit_pos_y", &T5_secondary_hit_pos_y);
    // t_out->Branch("T5_secondary_hit_time", &T5_secondary_hit_time);

    // -------------------------------------------------------------
    // Main Merging Loop
    // -------------------------------------------------------------
    std::cout << "[INFO] Commencing file merge for " << n_entries
              << " events...\n";
    for (Long64_t i = 0; i < n_entries; ++i) {
      // Load event data into memory variables
      t_raw->GetEntry(i);
      t_t5->GetEntry(i);
      if (merge_beam) {
        t_beam->GetEntry(i);
      }
      t_cal->GetEntry(i);
      t_dq->GetEntry(i);

      // --- Execute minimal custom processing here if needed ---
      if (raw_readout_number != cal_readout_number ||
          raw_readout_number != dq_readout_number) {
        std::cerr << "[ERROR] Readout number mismatch in event " << i
                  << "! Raw: " << raw_readout_number
                  << " | Cal: " << cal_readout_number
                  << " | DQ: " << dq_readout_number << "\n";
        throw std::runtime_error("Readout number mismatch in event " +
                                 std::to_string(i));
      }

      // Ensure array lengths are exactly the same
      if (nhit_pmt_calibrated_times != nhit_pmt_charges ||
          nhit_pmt_calibrated_times != nhit_mpmt_slot ||
          nhit_pmt_calibrated_times != nhit_pmt_pos ||
          nhit_pmt_calibrated_times != nhit_pmt_readout_mask) {
        std::cerr << "[WARNING] Array length mismatch in event " << i << "!\n";
        std::cerr << "  times: " << nhit_pmt_calibrated_times
                  << ", charges: " << nhit_pmt_charges
                  << ", slots: " << nhit_mpmt_slot << ", pos: " << nhit_pmt_pos
                  << ", dq_mask: " << nhit_pmt_readout_mask << "\n";
        throw std::runtime_error("Array length mismatch in event " +
                                 std::to_string(i));
      }

      // Convert the static arrays into standard vectors for writing out to file
      out_hit_pmt_calibrated_times->clear();
      out_hit_pmt_charges->clear();
      out_hit_mpmt_slot->clear();
      out_hit_pmt_pos->clear();
      out_hit_pmt_readout_mask->clear();

      tb_hit_card_ids->clear();
      tb_hit_channel_ids->clear();
      tb_hit_charges->clear();
      tb_hit_times->clear();

      tb_waveform_card_ids->clear();
      tb_waveform_channel_ids->clear();
      tb_waveform_times->clear();
      tb_waveforms->clear();

      // Check that the hit vectors are all the same size
      if (raw_hit_card_ids->size() != raw_hit_channel_ids->size() ||
          raw_hit_card_ids->size() != raw_hit_charges->size() ||
          raw_hit_card_ids->size() != raw_hit_times->size()) {
        throw std::runtime_error("Hit vector length mismatch in event " +
                                 std::to_string(i));
      }

      // add trigger mainboard hits to separate array
      if (trigger_type == "hw") {
        t_wf->GetEntry(i);
        for (int k = 0; k < nhit_card_wf; ++k) {
          if (hit_card_wf[k] > 120) {
            tb_hit_card_ids->push_back(hit_card_wf[k]);
            tb_hit_channel_ids->push_back(hit_chan_wf[k]);
            tb_hit_charges->push_back(static_cast<float>(hit_charge_wf[k]));
            tb_hit_times->push_back(hit_time_wf[k]);
          }
        }
      } else {
        for (size_t k = 0; k < raw_hit_card_ids->size(); ++k) {
          if ((*raw_hit_card_ids)[k] > 120) {
            tb_hit_card_ids->push_back((*raw_hit_card_ids)[k]);
            tb_hit_channel_ids->push_back((*raw_hit_channel_ids)[k]);
            tb_hit_charges->push_back((*raw_hit_charges)[k]);
            tb_hit_times->push_back((*raw_hit_times)[k]);
          }
        }
      }

      // check that the waveform vectors are all the same size
      if (raw_waveform_card_ids->size() != raw_waveform_channel_ids->size() ||
          raw_waveform_card_ids->size() != raw_waveform_times->size() ||
          raw_waveform_card_ids->size() != raw_waveforms->size()) {
        throw std::runtime_error("Waveform vector length mismatch in event " +
                                 std::to_string(i));
      }
      // add trigger mainboard waveforms to separate array
      for (size_t k = 0; k < raw_waveform_card_ids->size(); ++k) {
        if ((*raw_waveform_card_ids)[k] > 120) {
          tb_waveform_card_ids->push_back((*raw_waveform_card_ids)[k]);
          tb_waveform_channel_ids->push_back((*raw_waveform_channel_ids)[k]);
          tb_waveform_times->push_back((*raw_waveform_times)[k]);
          tb_waveforms->push_back((*raw_waveforms)[k]);
        }
      }

      for (int j = 0; j < nhit_pmt_calibrated_times; ++j) {
        out_hit_pmt_calibrated_times->push_back(hit_pmt_calibrated_times_in[j]);
        out_hit_pmt_charges->push_back(hit_pmt_charges_in[j]);
        out_hit_mpmt_slot->push_back(hit_mpmt_slot_in[j]);
        out_hit_pmt_pos->push_back(hit_pmt_pos_in[j]);
      }

      for (int j = 0; j < nhit_pmt_readout_mask; ++j) {
        out_hit_pmt_readout_mask->push_back(hit_pmt_readout_mask_in[j]);
      }

      // Fill output tree with exact copy of the data
      t_out->Fill();

      if (i % 100 == 0)
        std::cout << "  Processed " << i << " events\n";
    }

    // Wrap up
    f_out->cd();
    t_out->Write();

    // -------------------------------------------------------------
    // Auxiliary TTree Copies
    // -------------------------------------------------------------
    f_out->cd();

    // 1. BEAM Trees
    if (merge_beam && f_beam) {
      TTree *in_scalar = (TTree *)f_beam->Get("scalar_results");
      if (in_scalar) {
        TTree *out_scalar = in_scalar->CloneTree(-1, "fast");
        out_scalar->SetName("vme_analysis_scalar_results");
        out_scalar->Write();
      }
      TTree *in_run_info = (TTree *)f_beam->Get("run_info");
      if (in_run_info) {
        TTree *out_run_info = in_run_info->CloneTree(-1, "fast");
        out_run_info->SetName("vme_analysis_run_info");
        out_run_info->Write();
      }
    }

    // 2. Metrics Tree
    if (f_dq) {
      TTree *in_metrics = (TTree *)f_dq->Get("Metrics");
      if (in_metrics) {
        TTree *out_metrics = in_metrics->CloneTree(-1, "fast");
        out_metrics->SetName("DataQualityMetrics");
        out_metrics->Write();
      }

      // 3. Configuration Tree from the data quality
      TTree *in_config = (TTree *)f_dq->Get("Configuration");
      if (in_config && in_config->GetEntries() > 0) {
        char run_configuration_in[2048] = {0};
        int n_good_wcte_pmts = 0, n_wcte_pmts_with_timing_constant = 0,
            n_wcte_pmts_slow_control_stable = 0, n_manually_masked_pmts = 0;
        int good_wcte_pmts_in[2500], wcte_pmts_with_timing_constant_in[2500],
            wcte_pmts_slow_control_stable_in[2500],
            manually_masked_pmts_in[2500];

        int n_bad_current = 0, n_bad_pmt_status = 0, n_coarse = 0,
            n_missing = 0, n_no_data = 0, n_sporadic = 0, n_pmt_trip = 0;
        int bad_current_in[2500], bad_pmt_status_in[2500], coarse_in[2500],
            missing_in[2500], no_data_in[2500], sporadic_in[2500],
            pmt_trip_in[2500];

        in_config->SetBranchStatus("*", 0);
        in_config->SetBranchStatus("run_configuration", 1);
        in_config->SetBranchAddress("run_configuration", run_configuration_in);

        in_config->SetBranchStatus("ngood_wcte_pmts", 1);
        in_config->SetBranchAddress("ngood_wcte_pmts", &n_good_wcte_pmts);
        in_config->SetBranchStatus("good_wcte_pmts", 1);
        in_config->SetBranchAddress("good_wcte_pmts", good_wcte_pmts_in);

        in_config->SetBranchStatus("nwcte_pmts_with_timing_constant", 1);
        in_config->SetBranchAddress("nwcte_pmts_with_timing_constant",
                                    &n_wcte_pmts_with_timing_constant);
        in_config->SetBranchStatus("wcte_pmts_with_timing_constant", 1);
        in_config->SetBranchAddress("wcte_pmts_with_timing_constant",
                                    wcte_pmts_with_timing_constant_in);

        in_config->SetBranchStatus("nwcte_pmts_slow_control_stable", 1);
        in_config->SetBranchAddress("nwcte_pmts_slow_control_stable",
                                    &n_wcte_pmts_slow_control_stable);
        in_config->SetBranchStatus("wcte_pmts_slow_control_stable", 1);
        in_config->SetBranchAddress("wcte_pmts_slow_control_stable",
                                    wcte_pmts_slow_control_stable_in);

        in_config->SetBranchStatus("nmanually_masked_pmts", 1);
        in_config->SetBranchAddress("nmanually_masked_pmts",
                                    &n_manually_masked_pmts);
        in_config->SetBranchStatus("manually_masked_pmts", 1);
        in_config->SetBranchAddress("manually_masked_pmts",
                                    manually_masked_pmts_in);

        in_config->SetBranchStatus("nslow_control_mask_bad_current", 1);
        in_config->SetBranchAddress("nslow_control_mask_bad_current",
                                    &n_bad_current);
        in_config->SetBranchStatus("slow_control_mask_bad_current", 1);
        in_config->SetBranchAddress("slow_control_mask_bad_current",
                                    bad_current_in);

        in_config->SetBranchStatus("nslow_control_mask_bad_pmt_status", 1);
        in_config->SetBranchAddress("nslow_control_mask_bad_pmt_status",
                                    &n_bad_pmt_status);
        in_config->SetBranchStatus("slow_control_mask_bad_pmt_status", 1);
        in_config->SetBranchAddress("slow_control_mask_bad_pmt_status",
                                    bad_pmt_status_in);

        in_config->SetBranchStatus(
            "nslow_control_mask_coarse_counter_reset_failed", 1);
        in_config->SetBranchAddress(
            "nslow_control_mask_coarse_counter_reset_failed", &n_coarse);
        in_config->SetBranchStatus(
            "slow_control_mask_coarse_counter_reset_failed", 1);
        in_config->SetBranchAddress(
            "slow_control_mask_coarse_counter_reset_failed", coarse_in);

        in_config->SetBranchStatus("nslow_control_mask_missing_monitoring_data",
                                   1);
        in_config->SetBranchAddress(
            "nslow_control_mask_missing_monitoring_data", &n_missing);
        in_config->SetBranchStatus("slow_control_mask_missing_monitoring_data",
                                   1);
        in_config->SetBranchAddress("slow_control_mask_missing_monitoring_data",
                                    missing_in);

        in_config->SetBranchStatus("nslow_control_mask_no_data", 1);
        in_config->SetBranchAddress("nslow_control_mask_no_data", &n_no_data);
        in_config->SetBranchStatus("slow_control_mask_no_data", 1);
        in_config->SetBranchAddress("slow_control_mask_no_data", no_data_in);

        in_config->SetBranchStatus(
            "nslow_control_mask_sporadic_monitoring_packets", 1);
        in_config->SetBranchAddress(
            "nslow_control_mask_sporadic_monitoring_packets", &n_sporadic);
        in_config->SetBranchStatus(
            "slow_control_mask_sporadic_monitoring_packets", 1);
        in_config->SetBranchAddress(
            "slow_control_mask_sporadic_monitoring_packets", sporadic_in);

        in_config->SetBranchStatus("nslow_control_mask_pmt_trip", 1);
        in_config->SetBranchAddress("nslow_control_mask_pmt_trip", &n_pmt_trip);
        in_config->SetBranchStatus("slow_control_mask_pmt_trip", 1);
        in_config->SetBranchAddress("slow_control_mask_pmt_trip", pmt_trip_in);

        in_config->GetEntry(0);

        f_out->cd();
        TTree *out_config = new TTree("Configuration", "Configuration Data");

        std::string *run_configuration_out =
            new std::string(run_configuration_in);
        std::vector<int> *out_good_wcte_pmts = new std::vector<int>(
            good_wcte_pmts_in, good_wcte_pmts_in + n_good_wcte_pmts);
        std::vector<int> *out_wcte_pmts_with_timing_constant =
            new std::vector<int>(wcte_pmts_with_timing_constant_in,
                                 wcte_pmts_with_timing_constant_in +
                                     n_wcte_pmts_with_timing_constant);
        std::vector<int> *out_wcte_pmts_slow_control_stable =
            new std::vector<int>(wcte_pmts_slow_control_stable_in,
                                 wcte_pmts_slow_control_stable_in +
                                     n_wcte_pmts_slow_control_stable);
        std::vector<int> *out_manually_masked_pmts = new std::vector<int>(
            manually_masked_pmts_in,
            manually_masked_pmts_in + n_manually_masked_pmts);

        std::vector<int> *out_bad_current = new std::vector<int>(
            bad_current_in, bad_current_in + n_bad_current);
        std::vector<int> *out_bad_pmt_status = new std::vector<int>(
            bad_pmt_status_in, bad_pmt_status_in + n_bad_pmt_status);
        std::vector<int> *out_coarse =
            new std::vector<int>(coarse_in, coarse_in + n_coarse);
        std::vector<int> *out_missing =
            new std::vector<int>(missing_in, missing_in + n_missing);
        std::vector<int> *out_no_data =
            new std::vector<int>(no_data_in, no_data_in + n_no_data);
        std::vector<int> *out_sporadic =
            new std::vector<int>(sporadic_in, sporadic_in + n_sporadic);
        std::vector<int> *out_pmt_trip =
            new std::vector<int>(pmt_trip_in, pmt_trip_in + n_pmt_trip);

        out_config->Branch("run_configuration", &run_configuration_out);
        out_config->Branch("good_wcte_pmts", &out_good_wcte_pmts);
        out_config->Branch("wcte_pmts_with_timing_constant",
                           &out_wcte_pmts_with_timing_constant);
        out_config->Branch("wcte_pmts_slow_control_stable",
                           &out_wcte_pmts_slow_control_stable);
        out_config->Branch("manually_masked_pmts", &out_manually_masked_pmts);
        out_config->Branch("slow_control_mask_bad_current", &out_bad_current);
        out_config->Branch("slow_control_mask_bad_pmt_status",
                           &out_bad_pmt_status);
        out_config->Branch("slow_control_mask_coarse_counter_reset_failed",
                           &out_coarse);
        out_config->Branch("slow_control_mask_missing_monitoring_data",
                           &out_missing);
        out_config->Branch("slow_control_mask_no_data", &out_no_data);
        out_config->Branch("slow_control_mask_sporadic_monitoring_packets",
                           &out_sporadic);
        out_config->Branch("slow_control_mask_pmt_trip", &out_pmt_trip);

        out_config->Fill();
        out_config->Write();

        delete run_configuration_out;
        delete out_good_wcte_pmts;
        delete out_wcte_pmts_with_timing_constant;
        delete out_wcte_pmts_slow_control_stable;
        delete out_manually_masked_pmts;
        delete out_bad_current;
        delete out_bad_pmt_status;
        delete out_coarse;
        delete out_missing;
        delete out_no_data;
        delete out_sporadic;
        delete out_pmt_trip;
      }
    }

    f_out->Close();

    std::cout << "\n[INFO] Merge successfully completed: " << out_file_path
              << "\n";

    // Clean up
    f_raw->Close();
    f_cal->Close();
    f_t5->Close();
    f_dq->Close();
    if (f_beam)
      f_beam->Close();
    if (f_wf)
      f_wf->Close();

  } // End of file processing loop

  return 0;
}
