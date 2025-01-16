import os
import socket
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()

# use these if you need to check which machine/user is running the code
if 'borong-System-Product-Name' in hostname:
    DEFAULTS['user'] = os.getlogin()
    DEFAULTS['user_id'] = os.getuid()

    prefix = f'/run/user/' + str(DEFAULTS['user_id'])
    DEFAULTS['host_root'] = prefix + '/gvfs/smb-share:server=mocap-stor-02.inf.ethz.ch,share=ssd/data1/HOODs/Datasets/'
    # DEFAULTS['host_root'] = prefix + '/gvfs/smb-share:server=hilliges.scratch.inf.ethz.ch,share=scratch-hilliges/HOODs/Datasets/'
    
    DEFAULTS['server'] = 'local'
    DEFAULTS['hostname'] = hostname

elif 'ait-server' in hostname:
    DEFAULTS['host_root'] = f'/mnt/scratch/HOODs/Datasets/'
    DEFAULTS['server'] = 'ait'
    DEFAULTS['hostname'] = hostname

# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)