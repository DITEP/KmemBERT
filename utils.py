import os
import platform

def get_root():
    # Ruche
    if platform.platform() == 'Linux-3.10.0-957.el7.x86_64-x86_64-with-centos-7.6.1810-Core':
        return '/gpfs/workdir/piatc'
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("GR") + 1])

def pretty_time(t):
    # Tranforms time t in seconds into a pretty string
    return f"{int(t//60)}m{int(t%60)}s"