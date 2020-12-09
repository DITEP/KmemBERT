import os
import platform

def get_root():
    # Ruche
    if platform.platform() == 'Linux-3.10.0-957.el7.x86_64-x86_64-with-centos-7.6.1810-Core':
        return '/gpfs/workdir/piatc'
    return os.sep.join(os.getcwd().split(os.sep)[0 : os.getcwd().split(os.sep).index("French-Transformers") + 1])