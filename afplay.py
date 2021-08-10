# import modules

import subprocess
import io


def afplay(filepath):

    params = io.getInfo(filepath)
    time = params[3] / params[2]
    cmd = 'afplay -q 1 %s'%(filepath)
    subprocess.Popen(cmd, shell=True)
    time.sleep()

    return
