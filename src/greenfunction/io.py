import json
import numpy as np

def save_data(filename, **kwargs):
    '''
    Saves the result of compuation to filename.npz
    '''
    if not filename.endswith('.npz'):
        filename += '.npz'
    np.savez_compressed(filename, **kwargs)

def load_data(filename):
    '''
    Tries to load data from filename (should be a .npz file)
    '''
    try:
        return np.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def save_settings(filename, settings):
    '''
    Saves the current settings to a .json file. Run with --help or see cli.py for available settings
    '''
    if not filename.endswith('.json'):
        filename += '.json'

    safe_settings = settings.copy()
    if 'corner' in safe_settings:
        safe_settings['corner'] = str(safe_settings['corner']).strip("()")

    with open(filename, 'w') as f:
        json.dump(settings, f, indent=4)