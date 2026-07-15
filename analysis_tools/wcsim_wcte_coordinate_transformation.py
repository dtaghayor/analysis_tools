import numpy as np
from importlib.resources import files

class WCSimCoordinateTransform:
    def __init__(self):
        """
        Define the offset
        """
        
        self.vertical_offset = 424.7625 #mm

        self.units={
            "mm": 1,
            "cm": 10**(-1),
            "m": 10**(-3)
        }


    # ───────────────────────────── #
    # 1. WCSim → WCTE               #
    # ───────────────────────────── #
    
    def wcsim_to_wcte(self, wcsim_coords, unit="mm"):
        """
        Transform a set of WCSim coordinates into a set of WCTE coordinates
        WCSim coordinates MUST be given following the convention: 
        wcsim_coords must be an nx3 array or numpy array of length 3
        wcsim_coords[:,0] is x
        wcsim_coords[:,1] is y
        wcsim_coords[:,2] is z      
                  
                                                                        
        Use a list or an Array for the WCSim coords                        
        """

        if unit not in self.units:
            raise ValueError(f"Invalid unit '{unit}'. Choose from: {list(self.units.keys())}")
        
        offset = self.vertical_offset
            
        wcte_coords = np.atleast_2d(wcsim_coords).copy().astype(float)

        wcte_coords[:,1] += offset*self.units[unit]

        return wcte_coords.squeeze() 
    
    # ───────────────────────────── #
    # 2. WCTE → WCSim               #
    # ───────────────────────────── #

    def wcte_to_wcsim(self, wcte_coords, unit="mm"):
        """
        Transform a set of WCTE coordinates into a set of WCSim coordinates
        WCTE coordinates MUST be given following the convention: 
        wcte_coords must be an nx3 array or numpy array of length 3
        wcte_coords[:,0] is x
        wcte_coords[:,1] is y
        wcte_coords[:,2] is z      
                  
                                                                        
        Use a list or an Array for the WCSim coords                        
        """

        offset = self.vertical_offset
            
        wcsim_coords = np.atleast_2d(wcte_coords).copy().astype(float)

        wcsim_coords[:,1] -= offset

        return wcsim_coords.squeeze() 


class WCSimPMTMapping:
    def __init__(self, geo_path=None):
        """
        geo_path: path al archivo de geometría (WCSim)
        """
        self._slotpos_to_tube = {}
        self._tube_to_slotpos = {}
        self.default_geopath = files('analysis_tools.data').joinpath('wcsim_v1_12_29_mapping_geofile.txt')
        if geo_path is None:
            print("Loading default WCSim mapping for WCSim 1.12.29")
            geo_path = self.default_geopath
        self._slotpos_to_tube, self._tube_to_slotpos = self._load_mapping(geo_path)
        self.build_fast_array_lookup()

    # ───────────────────────────── #
    # LOAD MAPPING                  #
    # ───────────────────────────── #
    
    def _load_mapping(self, geo_path):
        data = np.loadtxt(geo_path, skiprows=5, usecols=(0, 1, 2), dtype=int)
        slotpos_to_tube = {}
        tube_to_slotpos = {}
        for tube_no, slot, pos in data:
            pos0 = pos - 1  # convert to 0-index
            
            tube_to_slotpos[tube_no] = (slot, pos0)
            slotpos_to_tube[(slot, pos0)] = tube_no
        return slotpos_to_tube, tube_to_slotpos
    
    def build_fast_array_lookup(self):
        """
        Build fast arrays from the dictionaries for vectorised lookup
        """
        if not self._slotpos_to_tube or not self._tube_to_slotpos:
            raise ValueError("No mapping data available to build fast array lookup")
        
        #build some fast arrays from the dictionaries for vectorised lookup
        max_slot = 105
        max_pos = 18
        max_tube = max(self._tube_to_slotpos.keys())
        self._lookup_wcte_to_tube = np.full((max_slot+1, max_pos+1), -1)
        self._valid_slotpos = np.full((max_slot+1, max_pos+1), False)
        
        for (slot, pos), tube_no in self._slotpos_to_tube.items():
            self._lookup_wcte_to_tube[slot, pos] = tube_no
            self._valid_slotpos[slot, pos] = True
        
        self._lookup_tube_to_wcte = np.full((max_tube + 1, 2), -1)
        self._valid_tubes = np.full((max_tube + 1, ), False)
        for tube_no in self._tube_to_slotpos.keys():
            slot, pos = self._tube_to_slotpos[tube_no]
            self._lookup_tube_to_wcte[tube_no] = [slot, pos]
            self._valid_tubes[tube_no] = True

    
    def check_mapping_consistency_with_default(self, user_geo_path):
        """ 
        Check if a geofile loaded by the user is consistent with the default geofile
        """

        user_slotpos_to_tube, user_tube_to_slotpos = self._load_mapping(user_geo_path)
        default_slotpos_to_tube, default_tube_to_slotpos = self._load_mapping(self.default_geopath)
        #Check if they're the same
        if user_slotpos_to_tube == default_slotpos_to_tube and user_tube_to_slotpos == default_tube_to_slotpos:
            print("The specified geofile is consistent with the default geofile")
        else:
            print("The specified geofile is NOT consistent with the default geofile")
        
    
    def map_wcsim_tube_no_to_wcte_slot_pos(self, tube_no, use_watchmal_npz = False):
        """
        Map WCSim tube number to WCTE slot and position.

        Returns
        -------
        slots : np.ndarray
            WCTE slot number(s) corresponding to each tube number.
        positions : np.ndarray
            WCTE position number(s) corresponding to each tube number.

        Usage
        -----
        slots, positions = mapper.map_wcsim_tube_no_to_wcte_slot_pos(tube_nos)
        """
        tube_no = np.atleast_1d(np.asarray(tube_no))
        if use_watchmal_npz:
            tube_no += 1 #first convert back to tube number starting at 1
        out_of_bounds = (tube_no < 0) | (tube_no >= len(self._valid_tubes))
        if np.any(out_of_bounds):
            raise ValueError("Tube number(s) not in the mapping and out of range: \n" + str(tube_no[out_of_bounds]))
        if np.any(self._valid_tubes[tube_no]==False):
            raise ValueError("Invalid tube number(s) looked up not in mapping: \n" + str(tube_no[self._valid_tubes[tube_no]==False]))

        result = self._lookup_tube_to_wcte[tube_no]
        slots, positions = result.T
        return slots.squeeze(), positions.squeeze()
    
    def map_wcte_slot_pos_to_wcsim_tube_no(self, slot, pos, use_watchmal_npz = False):
        slot = np.atleast_1d(np.asarray(slot))
        pos = np.atleast_1d(np.asarray(pos))
        max_slot, max_pos = self._valid_slotpos.shape
        out_of_bounds = (slot < 0) | (slot >= max_slot) | (pos < 0) | (pos >= max_pos)
        if np.any(out_of_bounds):
            raise ValueError("Slot/pos are not in the mapping and out of range: \n" + str(slot[out_of_bounds]) + " " + str(pos[out_of_bounds]))
        if np.any(self._valid_slotpos[slot, pos]==False):
            raise ValueError("Invalid slot/pos looked up not in mapping: \n" + str(slot[self._valid_slotpos[slot, pos]==False]) + " " + str(pos[self._valid_slotpos[slot, pos]==False]))
        tube_no = self._lookup_wcte_to_tube[slot, pos]
        if use_watchmal_npz:
            tube_no -= 1 #convert to tube numbers starting at 0
        return tube_no
