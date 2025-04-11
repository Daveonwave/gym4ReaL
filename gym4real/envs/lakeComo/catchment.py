from utils import Utils

class Catchment:
    def __init__(self, pCM=None):
        if pCM is None:
            # Default constructor, does nothing
            self.cModel = None
            self.inflow = None
        else:
            self.cModel = pCM["CM"]
            if self.cModel == 1:
                file_info = pCM["inflow_file"]
                self.inflow = Utils.load_matrix(
                    file_info["filename"],
                    file_info["row"],
                    file_info["col"]
                )
            else:
                # Placeholder for other catchment models (e.g., HBV)
                self.inflow = None

    def get_inflow(self, pt, ps):
        """
        Retrieve inflow for simulation point ps and time step pt.
        """
        return self.inflow[ps][pt]
