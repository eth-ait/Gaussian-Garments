import os
import socket
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()



DEFAULTS['output_root'] = '/path/to/output/dir'
DEFAULTS['data_root'] = '/path/to/input/dir'
DEFAULTS['aux_root'] = '/path/to/auxiliary/dir'


DEFAULTS['stage1'] = 'stage1'
DEFAULTS['stage2'] = 'stage2'
DEFAULTS['stage3'] = 'stage3'


DEFAULTS['rgb_images'] = 'rgb_images'
DEFAULTS['garment_masks'] = 'garment_masks'
DEFAULTS['foreground_masks'] = 'foreground_masks'

# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)

for d in ['data_root', 'aux_root']:
    if not os.path.exists(DEFAULTS[d]):
        raise FileNotFoundError(f"DEFAULTS.{d} ({DEFAULTS[d]}) does not exist! Follow instructions in DataPreparation.md to set it up.")