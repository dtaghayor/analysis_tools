import numpy as np

class coordinateTransform:
    def __init__(self):
        # Define the offset
        self.vertical_offset = 424.7625 #mm

    def wcsim_to_wcte(self, wcsim_coords, offset=None):
        # Transform a set of WCSim coordinates into a set of WCTE coordinates
        # WCSim coordinates MUST be given following the convention:
        # X is the horizontal axis perpendicular to the beam direction
        # Y is the vertical axis, where +Y means upwards and -Y downwards
        # Z is the horizontal axis parallel to the beam direction

        # Use a list or an Array for the WCSim coords
        
        if offset is None:
            offset = self.vertical_offset
            
        wcte_coords = np.zeros(3)
        
        wcte_x = wcsim_coords[0]
        wcte_y = wcsim_coords[1]
        wcte_z = wcsim_coords[2]

        wcte_coords[0] = wcte_x
        wcte_coords[1] = wcte_y + offset
        wcte_coords[2] = wcte_z

        return wcte_coords
    
    def wcte_to_wcsim(self, wcte_coords, offset=None):
        # Transform a set of WCTE coordinates into a set of WCSim coordinates
        # WCTE coordinates MUST be given following the convention:
        # X is the horizontal axis perpendicular to the beam direction
        # Y is the vertical axis, where +Y means upwards and -Y downwards
        # Z is the horizontal axis parallel to the beam direction

        # Use a list or an Array for the WCTE coords

        if offset is None:
            offset = self.vertical_offset
            
        wcsim_coords = np.zeros(3)
        
        wcte_x = wcte_coords[0]
        wcte_y = wcte_coords[1]
        wcte_z = wcte_coords[2]

        wcsim_coords[0] = wcte_x
        wcsim_coords[1] = wcte_y - offset
        wcsim_coords[2] = wcte_z

        return wcsim_coords


