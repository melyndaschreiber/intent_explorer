"""
A Module including a series of functions that create a tmp and data folder to extract datasets from the original sources. 
"""

import mne
from mne import create_info
from mne.io import RawArray
import pandas as pd
import numpy as np

from mne.preprocessing import create_eog_epochs, ICA
from mne import Epochs, pick_types

def check_directory():
    # Check if there is directory called data
    import os
    wd = os.getcwd()
    print(wd)
    if not os.path.isdir('data'):
        # Create the folder
        os.makedirs('data')

    if not os.path.isdir('tmp'):
        # Create the folder
        os.makedirs('tmp')

def ulm_execution():
    import os
    from os import path
    import requests
    wd = os.getcwd()
    check_directory()
    print('Upper Limb Movement Execution')
    # Download the file
    from urllib.request import urlopen
    from zipfile import ZipFile
    from progress.bar import ChargingBar

    print('Download dataset paper reference.')
    paper_reference_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable'
    paper = requests.get(paper_reference_url)
    with open('data/Upper_limb_movements_can_be_decoded_from_the_time_domain_of_low_frequency_EEG.pdf', 'wb') as f:
        f.write(paper.content)

    print('Download dataset information')
    dataset_description_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable'
    dataset_description = requests.get(dataset_description_url)
    with open('data/Upper_limb_movements_dataset_description.pdf', 'wb') as f:
        f.write(dataset_description.content)

    bar = ChargingBar('Downloading subject data', max=20)
    subjects = range(1, 16)
    for subject in subjects:
        if os.path.exists('tmp\S%d.zip' % subject):
            print('Already downloaded subject', subject, 'in the dataset.')
            pass
        else:
            ulm_zip_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2017/S%02d_ME.zip' % subject # Create URL name
            print('Downloading subject', subject, 'from ' + ulm_zip_url)
            ulm_dwl = urlopen(ulm_zip_url) # Download zip file
            print('Saving data into the tmp folder.')
            tempzip = open('tmp\S%d.zip' % subject, "wb")
            tempzip.write(ulm_dwl.read())
            tempzip.close()

            print('Extracting files into data folder.')
            zf = ZipFile('tmp\S%d.zip' % subject)
            zf.extractall(path = wd + '\data')
            zf.close()

        bar.next()
    bar.finish()

def ulm_imagination():
    import os
    from os import path
    import requests
    wd = os.getcwd()
    check_directory()
    print('Upper Limb Movement Imagination')

    print('Download dataset paper reference.')
    paper_reference_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable'
    paper = requests.get(paper_reference_url)
    with open('data/Upper_limb_movements_can_be_decoded_from_the_time_domain_of_low_frequency_EEG.pdf', 'wb') as f:
        f.write(paper.content)

    print('Download dataset information')
    dataset_description_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable'
    dataset_description = requests.get(dataset_description_url)
    with open('data/Upper_limb_movements_dataset_description.pdf', 'wb') as f:
        f.write(dataset_description.content)

    # Download the file
    from urllib.request import urlopen
    from zipfile import ZipFile
    from progress.bar import ChargingBar

    bar = ChargingBar('Downloading subject data', max=20)
    subjects = range(1, 16)
    for subject in subjects:
        if os.path.exists('tmp\S%d.zip' % subject):
            print('Already downloaded subject', subject, 'in the dataset.')
            pass
        else:
            ulm_zip_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2017/S%02d_MI.zip' % subject # Create URL name
            print('Downloading subject', subject, 'from ' + ulm_zip_url)
            ulm_dwl = urlopen(ulm_zip_url) # Download zip file
            print('Saving data into the tmp folder.')
            tempzip = open('tmp\S%d.zip' % subject, "wb")
            tempzip.write(ulm_dwl.read())
            tempzip.close()

            print('Extracting files into data folder.')
            zf = ZipFile('tmp\S%d.zip' % subject)
            zf.extractall(path = wd + '\data')
            zf.close()
        bar.next()
    bar.finish()

def sci_offline():
    import os
    import requests
    check_directory()
    from os import path
    wd = os.getcwd()
    print('Spinal Cord Injury Offline')

    print('Download dataset paper reference.')
    paper_reference_url = 'https://www.nature.com/articles/s41598-019-43594-9.pdf'
    paper = requests.get(paper_reference_url)
    with open('data/Attempted_Arm_and_Hand_Movements_can_be_Decoded_from_Low_Frequency_EEG_from_Persons_with_Spinal_Cord_Injury.pdf', 'wb') as f:
        f.write(paper.content)

    print('Download dataset information')
    dataset_description_url = 'https://lampx.tugraz.at/~bci/database/001-2019/dataset_description_v1-1.pdf'
    dataset_description = requests.get(dataset_description_url)
    with open('data/Spinal_Cord_Injury_dataset_description.pdf', 'wb') as f:
        f.write(dataset_description.content)

    # Download the file
    from urllib.request import urlopen
    from zipfile import ZipFile
    from progress.bar import ChargingBar

    bar = ChargingBar('Downloading subject data', max=20)
    subjects = range(1, 10)
    for subject in subjects:
        if os.path.exists('tmp\P%2d.zip' % subject):
            print('Already downloaded subject', subject, 'in the dataset.')
            pass
        else:
            sci_offline_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2019/P0%d.zip' % subject # Create URL name
            print('Downloading subject', subject, 'from ' + sci_offline_url)
            sci_offline_dwl = urlopen(sci_offline_url) # Download zip file
            print('Saving data into the tmp folder.')
            tempzip = open('tmp\P0%d.zip' % subject, "wb")
            tempzip.write(sci_offline_dwl.read())
            tempzip.close()

            print('Extracting files into data folder.')
            zf = ZipFile('tmp\P0%d.zip' % subject)
            zf.extractall(path = wd + '\data')
            zf.close()
        bar.next()
    bar.finish()

def sci_online():
    import os
    import requests
    check_directory()
    from os import path
    wd = os.getcwd()
    print('Spinal Cord Injury Online')
    # Download the file
    from urllib.request import urlopen
    from zipfile import ZipFile
    from progress.bar import ChargingBar

    bar = ChargingBar('Downloading subject data', max=20)
    sessions = [1, 2]
    session_type = ['Train', 'Test']
    for session in sessions:
        for trial_type in session_type:
            sci_online_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2019/P09%%20Online%%20Session%%20%d%%20%s.zip' % (session, trial_type) # Create URL name
            print('Downloading subject data from ' + sci_online_url)
            sci_online_dwl = urlopen(sci_online_url) # Download zip file
            print('Saving data into the tmp folder.')
            tempzip = open('tmp\session%d_%s.zip' % (session, trial_type), "wb")
            tempzip.write(sci_online_dwl.read())
            tempzip.close()

            print('Extracting files into data folder.')
            zf = ZipFile('tmp\session%d_%s.zip' % (session, trial_type))
            zf.extractall(path = wd + '\data')
            zf.close()
            bar.next()
        bar.finish()
    
def way_eeg_gal():

    import os
    import requests
    check_directory()
    from os import path
    wd = os.getcwd()
    print('WAY EEG GAL')

    print('Download dataset paper reference.')
    paper_reference_url = 'https://www.nature.com/articles/sdata201447.pdf'
    paper = requests.get(paper_reference_url)
    with open('data/Multichannel_EEG_recordings_during_3,936_grasp_and_lift_trials_with_varying_weight_and_friction.pdf', 'wb') as f:
        f.write(paper.content)

    print('Download dataset information')
    dataset_description_url = 'https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2014.47/MediaObjects/41597_2014_BFsdata201447_MOESM67_ESM.pdf'
    dataset_description = requests.get(dataset_description_url)
    with open('data/WAY_EEG_GAL_dataset_description.pdf', 'wb') as f:
        f.write(dataset_description.content)

    # Download the file
    from urllib.request import urlopen
    from zipfile import ZipFile
    from progress.bar import ChargingBar

    bar = ChargingBar('Downloading subject data', max=20)
    dwl_links = {'Participant1': 'https://ndownloader.figshare.com/files/3229301', 
                 'Participant2': 'https://ndownloader.figshare.com/files/3229304', 
                 'Participant3': 'https://ndownloader.figshare.com/files/3229307',
                 'Participant4': 'https://ndownloader.figshare.com/files/3229310',
                 'Participant5': 'https://ndownloader.figshare.com/files/3229313',
                 'Participant6': 'https://ndownloader.figshare.com/files/3209486',
                 'Participant7': 'https://ndownloader.figshare.com/files/3209501',
                 'Participant8': 'https://ndownloader.figshare.com/files/3209504',
                 'Participant9': 'https://ndownloader.figshare.com/files/3209495',
                 'Participant10': 'https://ndownloader.figshare.com/files/3209492',
                 'Participant11': 'https://ndownloader.figshare.com/files/3209498',
                 'Participant12': 'https://ndownloader.figshare.com/files/3209489'}

    for key, value in list(dwl_links.items()):
        print(key)
        print(value)
        if os.path.exists('tmp\%s.zip' % key):
            print('Already downloaded ', key, 'in the dataset.')
            pass
        else:
            print('Downloading subject ', key, 'from ' + value)
            way_dwl = urlopen(value) # Download zip file
            print('Saving data into the tmp folder.')
            tempzip = open('tmp\%s.zip' % key, "wb")
            tempzip.write(way_dwl.read())
            tempzip.close()

            print('Extracting files into data folder.')
            zf = ZipFile('tmp\%s.zip' % key)
            zf.extractall(path = wd + '\data')
            zf.close()
        bar.next()
    bar.finish()

def read_execution(subject, trial):
    """
    Create the Raw Data Array for data coming from the Upper Limb Movement Dataset.
    Args:
        file_location: A string value representing individual datasets.
        paradigm: A string value representing Healthy or Spinal Cord Injury
            Health: Subjects from the upper limb dataset
            SCI_Offline: Subjects from the Spinal Cord Injury offline dataset
            SCI_Online: Subjects from the Spinal Cord Injury online dataset
    Returns:
        raw: A Raw Array for further manipulation.
    """
    import os
    wd = os.getcwd() + '\data'
    file_location = wd + '/motorexecution_subject%d_run%d.gdf' % (subject, trial)
    
    eog_names = ["eog-r", "eog-m", "eog-l"]
    glove_sensors = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle",
                    "middle_near", "middle_far", "middle_ring", "ring_near", "ring_far", "ring_little", "litte_near",
                        "litte_far", "thumb_palm", "wrist_bend", "roll", "pitch", "gesture"]
    stim_sensors = ['STIM']

    print('Loading a dataset with a healthy population.')
    raw = mne.io.read_raw_gdf(file_location, preload=True)
    raw.set_eeg_reference()
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)

    # Create data labels
    eeg_names = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h",
                "FFC6h", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h",
                "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "TTP7h", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
                "TTP8h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h",
                "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", "P1", "Pz", "P2", "P4", "PPO1h",
                "PPO2h"]
    exo_sensors = ["handPosX", "handPosY", "handPosZ", "elbowPosX", "elbowPosY", "elbowPosZ", "ShoulderAdd",
                            "ShoulderFlex", "ShoulderRot", "Elbow", "ProSupination", "Wrist", "GripPressure"]

    channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(glove_sensors) + ['misc']*len(exo_sensors) + ['stim']*len(stim_sensors)
    montage = mne.channels.make_standard_montage('standard_1005')
    channel_names = eeg_names + eog_names + glove_sensors + exo_sensors + stim_sensors
    info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)

    raw_data = raw.get_data()
    trial_data = raw_data[0:97, :]
    new_channel = np.zeros((1, len(raw.times)))
    new_channel[0, events[:, 0]] = events[:, 2]

    all_data = np.concatenate([trial_data, new_channel], axis = 0)
    raw_data_pd = pd.DataFrame(all_data.T)
    raw_data_pd.columns = channel_names
    raw = RawArray(raw_data_pd.T, info, verbose=False).load_data()
    return raw

def read_imagination(subject, trial):
    """
    Create the Raw Data Array for data coming from the Upper Limb Movement Dataset.
    Args:
        file_location: A string value representing individual datasets.
        paradigm: A string value representing Healthy or Spinal Cord Injury
            Health: Subjects from the upper limb dataset
            SCI_Offline: Subjects from the Spinal Cord Injury offline dataset
            SCI_Online: Subjects from the Spinal Cord Injury online dataset
    Returns:
        raw: A Raw Array for further manipulation.
    """
    import os
    wd = os.getcwd() + '\data'
    file_location = wd + '/motorimagination_subject%d_run%d.gdf' % (subject, trial)
    
    eog_names = ["eog-r", "eog-m", "eog-l"]
    glove_sensors = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle",
                    "middle_near", "middle_far", "middle_ring", "ring_near", "ring_far", "ring_little", "litte_near",
                        "litte_far", "thumb_palm", "wrist_bend", "roll", "pitch", "gesture"]
    stim_sensors = ['STIM']

    print('Loading a dataset with a healthy population.')
    raw = mne.io.read_raw_gdf(file_location, preload=True)
    raw.set_eeg_reference()
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)

    # Create data labels
    eeg_names = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h",
                "FFC6h", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h",
                "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "TTP7h", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
                "TTP8h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h",
                "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", "P1", "Pz", "P2", "P4", "PPO1h",
                "PPO2h"]
    exo_sensors = ["handPosX", "handPosY", "handPosZ", "elbowPosX", "elbowPosY", "elbowPosZ", "ShoulderAdd",
                            "ShoulderFlex", "ShoulderRot", "Elbow", "ProSupination", "Wrist", "GripPressure"]

    channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(glove_sensors) + ['misc']*len(exo_sensors) + ['stim']*len(stim_sensors)
    montage = mne.channels.make_standard_montage('standard_1005')
    channel_names = eeg_names + eog_names + glove_sensors + exo_sensors + stim_sensors
    info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)

    raw_data = raw.get_data()
    trial_data = raw_data[0:97, :]
    new_channel = np.zeros((1, len(raw.times)))
    new_channel[0, events[:, 0]] = events[:, 2]

    all_data = np.concatenate([trial_data, new_channel], axis = 0)
    raw_data_pd = pd.DataFrame(all_data.T)
    raw_data_pd.columns = channel_names
    raw = RawArray(raw_data_pd.T, info, verbose=False).load_data()
    return raw
        
def read_sci_offline(subject, trial):
    
    import os
    wd = os.getcwd() + '\data'
    file_location = wd + '/P%02d Run %d.gdf' % (subject, trial)
    raw = mne.io.read_raw_gdf(file_location)
    events, event_dict = mne.events_from_annotations(raw)
    eeg_names = ["AFz", "F3", "F1", "Fz", "F2", "F4",
                "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
                "FC5", "FCC6h", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h",
                "CCP2h", "CCP4h", "CPP2h", "CPP4h", "CPP6h","P5",
                "P3", "P1", "Pz", "P2", "P4", "P6",
                "PPO1h", "PPO2h", "POz", "FC3", "FC1", "FCz",
                "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FCC1h",
                "FCC2h", "FCC4h", "CCP6h", "CP5", "CP3", "CP1",
                "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h","CPP1h"]
    eog_names = ["eog-r", "eog-m", "eog-l"]
    stim_sensors = ['STIM']

    channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['stim']*len(stim_sensors)
    montage = mne.channels.make_standard_montage('standard_1005')
    channel_names = eeg_names + eog_names + stim_sensors
    info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)

    raw_data = raw.get_data()
    trial_data = raw_data[0:64, :]
    new_channel = np.zeros((1, len(raw.times)))
    new_channel[0, events[:, 0]] = events[:, 2]

    all_data = np.concatenate([trial_data, new_channel], axis = 0)
    raw_data_pd = pd.DataFrame(all_data.T)
    raw_data_pd.columns = channel_names
    raw = RawArray(raw_data_pd.T, info, verbose=False).load_data()
    return raw

def read_sci_online(session, type_of_trial, trial):

    import os
    wd = os.getcwd() + '\data'
    file_location = wd + '/P09 Run %d.gdf' % (trial)
    raw = mne.io.read_raw_gdf(file_location)
    events, event_dict = mne.events_from_annotations(raw)
    eeg_names = ["AFz", "F3", "F1", "Fz", "F2", "F4",
                "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
                "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
                "CCP5h", "CCP3h", "CCP1h","CCP2h", "CCP4h", "CCP6h", 
                "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
                "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h",
                "P5","P3", "P1", "Pz", "P2", "P4", "P6",
                "PPO1h", "PPO2h", "POz"]

    eog_names = ["eog-l", "eog-m", "eog-r"]

    if type_of_trial == "Train":

        sci_sensors = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle",
                        "middle_near", "middle_far", "middle_ring", "ring_near", "ring_far", "ring_little", "litte_near",
                            "litte_far", "thumb_palm", "wrtist_bend", "roll", "pitch", "gesture", "button_press"]

        stim_sensors = ['stim']

        channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(sci_sensors) +['stim']*len(stim_sensors) 
        montage = mne.channels.make_standard_montage('standard_1005')
        channel_names = eeg_names + eog_names + sci_sensors + stim_sensors
        info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)
    else:
        sci_sensors = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle",
                        "middle_near", "middle_far", "middle_ring", "ring_near", "ring_far", "ring_little", "litte_near",
                            "litte_far", "thumb_palm", "wrtist_bend", "roll", "pitch", "gesture"]

        classifier_sensors = ['predicted_class', 'prob_zc1', 'prob_zc2', 'prob_778', 'prob_779', 'prob_780', 'detected_class', 'button_press']

        stim_sensors = ['stim']

        channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(sci_sensors) + ['misc']*len(classifier_sensors) +['stim']*len(stim_sensors) 
        montage = mne.channels.make_standard_montage('standard_1005')
        channel_names = eeg_names + eog_names + sci_sensors + classifier_sensors + stim_sensors
        info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)

    raw_data = raw.get_data()
    trial_data = raw_data[0:len(raw_data), :]
    new_channel = np.zeros((1, len(raw.times)))
    new_channel[0, events[:, 0]] = events[:, 2]

    all_data = np.concatenate([trial_data, new_channel], axis = 0)
    raw_data_pd = pd.DataFrame(all_data.T)
    raw_data_pd.columns = channel_names 
    raw = RawArray(raw_data_pd.T, info, verbose=False).load_data()
    return raw

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
