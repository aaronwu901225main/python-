import numpy as np
import pip

try:
    np_version = np.version.version
    print("NumPy version:", np_version)
    print("NumPy is installed.")
    
    # Check if the installed version is outdated
    latest_version = pip.get_installed_distributions()["numpy"].version
    if np_version < latest_version:
        print("Updating NumPy to the latest version...")
        pip.main(["install", "--upgrade", "numpy"])
        print("NumPy updated to version", latest_version)
    else:
        print("NumPy is already up to date.")
        
except ImportError:
    print("NumPy is not installed.")