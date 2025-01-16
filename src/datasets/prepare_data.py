import os
import shutil
from abc import ABC, abstractmethod

# TODO: 
# update & integrate with the rest of the codebase. Right now, this
# is a standalone class that does not have much functionality beyond the 
# first stage.
class PrepareDataset(ABC):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.output_root = '../../data/outputs/' + '_'.join(map(str, self.data_folder.split('/')))
        self.prepare_output_dir()
        
    def prepare_output_dir(self):
        if os.path.exists(self.output_root): 
            delete = input("Output path already exists. Please grant permission to remove current folder. (Y/N) ")
            if delete.lower() == 'y':
                shutil.rmtree(self.output_root)
            else:
                print(f"Error: Permission denied. Unable to proceed because the folder '{self.output_root}' already exists.")
                sys.exit(1)  # Terminate the program with an error status

        os.makedirs(self.output_root, exist_ok=True)
        print(f"-------\nOutput folder created at {self.output_root}\n-------")
        self._img_out = os.path.join(self.output_root, "images")
        self._mask_out = os.path.join(self.output_root, "masks")
        os.makedirs(self._img_out, exist_ok=True)
        os.makedirs(self._mask_out, exist_ok=True)
    
    @abstractmethod
    def load_cam_params(self):
        """
        Writes cameras.json file, containing information about the cameras, to output_root.
        """
        pass
    
    @abstractmethod
    def populate_directory(self):
        """
        Read the images, labels, and masks and writes them to the output directory within respective folders. 
        """
        pass