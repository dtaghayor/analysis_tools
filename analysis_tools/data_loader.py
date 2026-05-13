import uproot
import awkward as ak

class DataLoader:
    """
    A class for loading data the processed WCTE data and apply data quality cuts
    """

    def __init__(self, file_name, branches_to_load=None):
        #if branches to load is none then load all branches are loaded
        self.file_name = file_name
        self.branches_to_load = branches_to_load
        
        try:
            self.file = uproot.open(self.file_name)
        except Exception as e:
            raise RuntimeError(f"Failed to open ROOT file: {self.file_name}\n{e}")
        
        available_branches = self.file["WCTEReadoutWindows"].keys()

        #must load the mask branches if they are not already loaded for data quality cuts
        if self.branches_to_load is not None:            
            if "window_data_quality_mask" not in self.branches_to_load and "window_data_quality_mask" in available_branches:
                self.branches_to_load.append("window_data_quality_mask")
            if "hit_pmt_readout_mask" not in self.branches_to_load and "hit_pmt_readout_mask" in available_branches:
                self.branches_to_load.append("hit_pmt_readout_mask")
            if "vme_digi_issues_bitmask" not in self.branches_to_load and "vme_digi_issues_bitmask" in available_branches:
                self.branches_to_load.append("vme_digi_issues_bitmask")
            if "vme_evt_quality_bitmask" not in self.branches_to_load and "vme_evt_quality_bitmask" in available_branches:
                self.branches_to_load.append("vme_evt_quality_bitmask")
            if "T5_HasValidHit" not in self.branches_to_load and "T5_HasValidHit" in available_branches:
                self.branches_to_load.append("T5_HasValidHit")
            if "T5_HasMultipleScintillatorsHit" not in self.branches_to_load and "T5_HasMultipleScintillatorsHit" in available_branches:
                self.branches_to_load.append("T5_HasMultipleScintillatorsHit")
            if "T5_HasInTimeWindow" not in self.branches_to_load and "T5_HasInTimeWindow" in available_branches:
                self.branches_to_load.append("T5_HasInTimeWindow")

        self.mPMT_data_quality_cuts = False
        self.vme_event_quality_cuts = False
        self.t5_event_quality_cuts = False


    def iterate(self, verbose=False,**kwargs):
        """Iterate over the tree in batches using uproot.iterate"""
        defaults = {
        "step_size": "100 MB",
        "library": "ak",
        }
        defaults.update(kwargs)
        yield from (
            self._apply_all_data_quality_cuts(batch,verbose)
            for batch in self.file["WCTEReadoutWindows"].iterate(
                expressions=self.branches_to_load,
                **defaults,
            )
        )
    
    def _apply_all_data_quality_cuts(self, batch, verbose=False):
        if verbose:
            print(f"\nBatch loaded with {len(batch)} events")
        if self.mPMT_data_quality_cuts:            
            batch = batch[batch["window_data_quality_mask"]==0]
            if verbose:
                print(f"After window_data_quality_mask cut: {len(batch)} events")

            batch["hit_pmt_calibrated_times"] =  batch["hit_pmt_calibrated_times"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_charges"] =  batch["hit_pmt_charges"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_mpmt_slot_ids"] =  batch["hit_mpmt_slot_ids"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_position_ids"] =  batch["hit_pmt_position_ids"][batch["hit_pmt_readout_mask"]==0]
            batch["hit_pmt_readout_mask"] =  batch["hit_pmt_readout_mask"][batch["hit_pmt_readout_mask"]==0]

        if self.vme_event_quality_cuts:
                          
            batch = batch[(batch["vme_digi_issues_bitmask"]==0) & (batch["vme_evt_quality_bitmask"]==0)]
            if verbose:
                print(f"After vme_event_quality_cuts cut: {len(batch)} events")  
    
        if self.t5_event_quality_cuts:
            #a valid hit, only one hit in the main beam bunch, within time window
            batch = batch[(batch["T5_HasValidHit"]==True)&(batch["T5_HasMultipleScintillatorsHit"]==False)&(batch["T5_HasInTimeWindow"]==True)]
            if verbose:
                print(f"After t5_event_quality_cuts cut: {len(batch)} events")  

        return batch

    def apply_mPMT_data_quality_cuts(self):
        self.mPMT_data_quality_cuts = True
    
    def apply_vme_event_quality_cuts(self):
        self.vme_event_quality_cuts = True
    
    def apply_t5_event_quality_cuts(self):
        self.t5_event_quality_cuts = True

    def get_good_wcte_pmts(self):
        config = self.get_configuration()
        good_wcte_pmts = config["good_wcte_pmts"]
        good_wcte_pmts_slots = good_wcte_pmts//100
        good_wcte_pmts_positions = good_wcte_pmts%100
        return good_wcte_pmts_slots, good_wcte_pmts_positions
            
    def get_configuration(self):
        tree = self.file['Configuration']
        config = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return config[0]
    
    def get_data_quality_metrics(self):
        tree = self.file['DataQualityMetrics']
        data_quality_metrics = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return data_quality_metrics[0]
    
    def get_vme_analysis_scalar_results(self):
        tree = self.file['vme_analysis_scalar_results']
        vme_analysis_scalar_results = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return vme_analysis_scalar_results[0]
    
    def get_vme_analysis_run_info(self):
        tree = self.file['vme_analysis_run_info']
        vme_analysis_run_info = tree.arrays(library="ak", entry_start=0, entry_stop=1)
        return vme_analysis_run_info[0]
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
