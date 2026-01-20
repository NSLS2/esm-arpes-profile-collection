#esm_ui_config_file = "/home2/xf21id1/constructor/esm_usr1.ui"
sys.path.insert(0,'/home2/xf21id1/PEAK-notebooks-alpha13/PythonNotebooks/')
esm_ui_config_file = "/home2/xf21id1/git/bsstudio-xf/21id/esm_usr2.ui"
esm_ui_old_config_file = "/home2/xf21id1/git/bsstudio-xf_old/21id/esm_usr2.ui"

def update_gui():
    import os
    os.system("git -C /opt/bsstudio/bsstudio-xf pull")

def esm_gui1(filename=esm_ui_config_file, detached=False, verbose=False):
    # Note: install it in the developer mode with the following command before
    # it's packaged with conda
    #   $ cd /home2/xf21id1/git/bsstudio/
    #   $ pip install -e .
    import os
    os.environ["BSSTUDIO_LOG_FILE_NAME"] = "/var/log/bsstudio/bsstudio.log"
    import sys
    import bsstudio
    import warnings
    #warnings.simplefilter("ignore")
    bsstudio.load(filename, noexec=detached, verbose=verbose)

def esm_gui(filename=esm_ui_config_file, detached=False, verbose=False):
    #import cProfile
    #cProfile.run("esm_gui1('"+filename+"')", "profile_results")
    esm_gui1(filename, detached=detached, verbose=verbose)

def old_gui():
    esm_gui(esm_ui_old_config_file)

def gui_designer():
    import os
    os.system("~/opt_bsstudio/gui_designer.sh")

def bsstudio_profile_results():
    import pstats
    p = pstats.Stats('profile_results')
    p.sort_stats('cumulative').print_stats(10)
