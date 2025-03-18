import os
import socket
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()

# use these if you need to check which machine/user is running the code
if 'borong-System-Product-Name' in hostname:
    DEFAULTS['user'] = os.getlogin()
    DEFAULTS['user_id'] = os.getuid()

    DEFAULTS['output_root'] = '/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/outputs'
    DEFAULTS['data_root'] = '/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/input'
    DEFAULTS['aux_root'] = '/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/input'
    
    DEFAULTS['server'] = 'local'
    DEFAULTS['hostname'] = hostname

elif 'ait-server' in hostname:
    DEFAULTS['data_root'] = f'/data/agrigorev/02_Projects/opengaga/Inputs/'
    DEFAULTS['output_root'] = f'/data/agrigorev/02_Projects/opengaga/Outputs/'
    DEFAULTS['aux_root'] = f'/data/agrigorev/02_Projects/opengaga/aux_data/'
    DEFAULTS['server'] = 'ait'
    DEFAULTS['hostname'] = hostname

    DEFAULTS['temp_folder'] = "/data/agrigorev/temp/s2_debug"

elif 'ohws68' in hostname:
    DEFAULTS['data_root'] = f'/media/sdb/Data/opengaga/Inputs/'
    DEFAULTS['output_root'] = f'/media/sdb/Data/opengaga/Outputs/'
    DEFAULTS['aux_root'] = f'/media/sdb/Data/opengaga/aux_data/'
    DEFAULTS['server'] = 'agrigorev'
    DEFAULTS['hostname'] = hostname
    DEFAULTS['temp_folder'] = '/local/home/agrigorev/Data/temp/s2_debug'


DEFAULTS['stage1'] = 'stage1'
DEFAULTS['stage2'] = 'stage2'
DEFAULTS['stage3'] = 'stage3'


DEFAULTS['rgb_images'] = 'rgb_images'
DEFAULTS['garment_masks'] = 'garment_masks'
DEFAULTS['foreground_masks'] = 'foreground_masks'

# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)