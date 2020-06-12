import numpy as np
import matplotlib.pyplot as plt
from random import sample

import mne
from mne import Epochs, pick_types, concatenate_epochs
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, cross_val_score, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from mne.time_frequency import psd_multitaper

class Prep:

    def __init__(self, dataset, subject, bands, low_pass = 2, high_pass  = 25):
        self.dataset = dataset
        self.subject = subject
        self.bands = bands
        self.low_pass = low_pass
        self.high_pass = high_pass

    def create_raw_data(self, trial):

        """
        Create a Raw Data Array for one subjecs and one trial for all datasets.
        Args:
            dataset: A string value reperesenting each dataset
                    1: WAY-EEG-GAl
                    2: Motor Imagery
                    3: Motor Execution
                    4: Spinal Cord Injury
            subject: An integer value for each subject.
            trial: An integer value for each trial.
            wd: By default the working directory is set. The working directory can be changes manually
        Returns:
            raw: A raw array of eeg, sensors and stim.
        """

        import os
        import datasets
        wd = os.getcwd() + '\data'

        print(self.dataset + ' Selected')
        print('Attempting to load data from subject ' + str(self.subject) + ' and trial ' + str(trial))

        subject = self.subject
        if self.dataset == 'Motor Imagination':
            raw = datasets.read_imagination(subject, trial)
            
        elif self.dataset == 'Motor Execution':
            raw = datasets.read_execution(subject, trial)

        elif self.dataset == 'SCI Offline':
            raw = datasets.read_sci_offline(subject, trial)

        elif self.dataset == 'SCI Online':
            #file_location = r'C:/Users/ergo/Datasets/SCI/data/P09 Online Session %d %s/P09 Run %d.gdf' % (self.session, self.type_of_trial, trial)
            #file_location = self.wd + r'P09 Online Session %d Train/P09 Run %d.gdf' % (self.session, trial)
            #raw, event_id = read_execution_imagination(file_location, 'SCI Online', self.type_of_trial)
            raw, event_id = datasets.read_sci_online(subject, session, type_of_trial)
            print('The ' + self.dataset + ' dataset has been loaded. Observing the session ' + str(session) + ' for ' + type_of_trial)

        else:
            print('Not a known dataset. Check the dataset string.')
        
        print('The ' + self.dataset + ' dataset has been loaded.')
        self.raw = raw
        return raw

    def create_event_array_for_movement_onset(self, trial, param5=False):
        """Create a function to label event of motor activation

        Args:
            subject: Integer representing thr subject
            trial: Integer representing the trial
            param5: show plot: True or False

        Returns:
            unique_events: An event array to be used for future imports
            raw_original: The original raw array
        """

        import mne
        import numpy as np
        from mne import Epochs, pick_types

        if self.dataset == 'Motor Execution':
            chosen_events = dict(elbow_Flex=1, elbow_Extend=2, supination=3, pronation=4, hand_Close=5,
                        hand_Open=6, rest=7)

            self.added_event_id = dict(flex=5000, extend=5001, sup=5002, pro=5003, close=5004, hopen=5005, rest=14)

            raw = Prep.create_raw_data(self, trial)
            raw_filtered = raw.copy()
            picks = pick_types(raw.info, eeg=False, stim=False, include=["Elbow", "ProSupination", "GripPressure"])
            raw_filtered.filter(None, 10, fir_design='firwin', picks=picks)

            # Pick the channels to investigate
            pick_elbow = pick_types(raw_filtered.info, eeg=False, stim=False, include=["Elbow"])
            pick_wrist = pick_types(raw_filtered.info, eeg=False, stim=False, include=["ProSupination"])
            pick_grip = pick_types(raw_filtered.info, eeg=False, stim=False, include=["GripPressure"])
            pick_hand_position_x = pick_types(raw_filtered.info, eeg=False, stim=False, include=["handPosX"])

            # Get the events
            tmin, tmax = 0, 3
            events = mne.find_events(raw_filtered.copy().pick_channels(["STIM"]))

            # Get epoch data
            elbow_f_epochs = Epochs(raw_filtered.copy(), events, dict(elbow_Flex=1), tmin, tmax, proj=True, picks=pick_elbow,
                                    baseline=None, preload=True)
            elbow_e_epochs = Epochs(raw_filtered.copy(), events, dict(elbow_Extend=2), tmin, tmax, proj=True,
                                    picks=pick_elbow, baseline=None, preload=True)
            wrist_p_epochs = Epochs(raw_filtered.copy(), events, dict(pronation=3), tmin, tmax, proj=True, picks=pick_wrist,
                                    baseline=None, preload=True)
            wrist_s_epochs = Epochs(raw_filtered.copy(), events, dict(supination=4), tmin, tmax, proj=True, picks=pick_wrist,
                                    baseline=None, preload=True)
            hand_c_epochs = Epochs(raw_filtered.copy(), events, dict(hand_Close=5), tmin, tmax, proj=True, picks=pick_grip,
                                baseline=None, preload=True)
            hand_o_epochs = Epochs(raw_filtered.copy(), events, dict(hand_Open=6), tmin, tmax, proj=True, picks=pick_hand_position_x,
                                baseline=None, preload=True)

            elbow_f = elbow_f_epochs.get_data()
            elbow_e = elbow_e_epochs.get_data()
            wrist_p = wrist_p_epochs.get_data()
            wrist_s = wrist_s_epochs.get_data()
            hand_c = hand_c_epochs.get_data()
            hand_o = hand_o_epochs.get_data()

            all_data = [elbow_f, elbow_e, wrist_s, wrist_p, hand_c, hand_o]

            # Event time, 0, EventLabelNumber
            set_thresh = 0.20
            for i in range(len(all_data)):  # For all event types elbow flex etc.
                time_index = []
                start_event_index = events[np.where(events[:, 2] == list(chosen_events.values())[i]), 0]
                start_event_index = start_event_index[0].astype(int)

                for j in range(len(all_data[i])):  # For each instance of elbow flex
                    first_point = all_data[i][j][0][0]
                    last_point = all_data[i][j][0][-1]
                    if first_point > last_point:  # first larger than last?
                        this_range = first_point - last_point
                        data_thresh = first_point - set_thresh * this_range
                        for idx, h in enumerate(all_data[i][j][0]):
                            if h < data_thresh:
                                time_index.append(idx)  # Get time index for each instance of elbow flex
                                break
                    else:
                        this_range = last_point - first_point
                        data_thresh = first_point + set_thresh * this_range
                        for idx, h in enumerate(all_data[i][j][0]):
                            if h > data_thresh:
                                time_index.append(idx)  # Get time index for each instance of elbow flex
                                break

                stim_times = start_event_index + time_index
                zero_array = np.zeros(len(stim_times))
                stim_label_array = zero_array + 5000 + i
                # Create an array of the three arrays
                new_events = np.vstack([stim_times, zero_array, stim_label_array]).T
                events = np.vstack([events, new_events])

            # Combine arrays by odering the time index
            sorted_events_index = np.argsort(events[:, 0], axis=0)
            reorder_events = events[sorted_events_index]
            reorder_events = reorder_events.astype(int)
            unique_events = np.unique(reorder_events, axis=0)
            revised_events = dict(elbow_Flex=1536, flex=5000, elbow_Extend=1537, extend=5001,
                                supination=1538, sup=5002, pronation=1539, pro=5003,
                                hand_Close=1540, close=5004, hand_Open=1541, hopen=5005, rest=1542)
            if param5 is True:
                raw.plot(block=True, scalings='auto', events=events, event_id=revised_events)
            #raw.add_events(events=unique_events, stim_channel="STIM")
            raw.events = new_events
            self.unique_events = unique_events
    
            return unique_events, raw

    def filter_with_ica(self, thresh=3):
        """Use ICA to filter data for artifacts.

        Args:
            raw_data: Raw array of data
            events: Event Array
            event_id: A Dictionary of events
            thresh: A float that indicates a threshold

        Resturns:
            raw: ICA and bandpass filtered data
        """

        from mne.preprocessing import create_eog_epochs, ICA
        from mne import Epochs, pick_types

        try:
            raw_ica = self.raw.copy().crop(2, self.raw.times.max()-1)
            raw_ica.filter(self.low_pass, self.high_pass, n_jobs=-1, fir_design='firwin')
            # Run ICA
            method = 'fastica'
            # Choose other parameters
            n_components, random_state = 0.95, 42  # if float, select n_components by explained variance of PCA
            ica = ICA(n_components=n_components, method=method, random_state=random_state)
            ica.fit(raw_ica, picks=pick_types(raw_ica.info, eeg=True, misc=False, stim=False, eog=False))
            eog_epochs = create_eog_epochs(raw_ica)  # get single EOG trials
            eog_inds, scores = ica.find_bads_eog(eog_epochs, threshold=thresh)  # find via correlation
            ica.exclude.extend(eog_inds)
            ica.apply(raw_ica)

            self.raw_ica = raw_ica
            # self.raw = raw_copy
        except:
            raw_ica = self.raw.copy().crop(2, self.raw.times.max()-1)
            raw_ica.filter(self.low_pass, self.high_pass, n_jobs=-1, fir_design='firwin')
            self.raw_ica = self.raw
        return raw_ica
    
    def create_training_testing_unfiltered_raw(self, num_trials = 11, testing_percentage=20, thresh = 3.0):

        """Create training and testing unfiltered raw where a specified number of trials are left out for the testing set.
        Parameters: 
        -----------
            dataset: str
                The dataset of choice. 
            subject: int 
                The subject to investigate.
                testing_percentage: int, default = 20%
                    The percentage of data used for testing only.
                thresh: double, default = 3.0
                    The amount of ICA filter into the signal.
        Returns:
        --------
                training_raws:
                    The subject's concatenated raws meant for the training phase.
                testing_raws:
                    The subject's concatenated raws meant for the testing phase.
        """
        import random
        import numpy as np
        import mne

        if testing_percentage % 10 != 0: 
            print('Choose a percentage that is a multiple of 10.')        
        else:
            # Split the testing and training trials
            list_of_trials = np.arange(1,num_trials,1)
            
            # Load all training data as a concatenated raw.
            list_of_training_raws = []
            train_percent = 10-(testing_percentage/10)
            training_trials = list_of_trials[:int(train_percent)]
            print('Training trials are ', training_trials)
            testing_trials = list_of_trials[int(-(testing_percentage/10)):]
            print('Testing trials are ', testing_trials)
            for trial in training_trials:
                if self.dataset == 'Motor Execution':
                    events, raw = Prep.create_event_array_for_movement_onset(self, trial)
                raw_ica = Prep.filter_with_ica(self)
                list_of_training_raws.append(raw_ica)
            training_raws = mne.concatenate_raws(list_of_training_raws)
            
            # Load all testing data as a concatenated raw.
            list_of_testing_raws = []
            for trial in testing_trials:
                if self.dataset == 'Motor Execution':
                    Prep.create_event_array_for_movement_onset(self, trial)
                raw_ica = Prep.filter_with_ica(self)
                list_of_testing_raws.append(raw_ica)
            testing_raws = mne.concatenate_raws(list_of_testing_raws)

            self.training_raws = training_raws
            self.testing_raws = testing_raws

            return training_raws, testing_raws

    def create_filtered_epochs(self, event_id, event1 = None, event2 = None, tmin=-1.25, tmax=0.25, filter_design='firwin'):
        """
        Takes concatenated raws and creates filtered concatenated raws based off of a series of conditions.
        
        Parameters:
        -----------
            raws: concatenated_raw
                Raw files but preferrably a concatenated_raw
            condition: str
                wide_band_pass: Filtered data from 8-32 Hz.
                alpha_beta: Filtered data for alpha (8-12 Hz) and beta (16-24 Hz).
            event_id: dict
                A dictionary of events and corresponding event codes.
            filter_design: str
                Filter design based off of the preset 
        
        Returns:
        --------
            filtered_raw: concatenated_raw
                Filtered raw file based off of concatenated raw.
        """
        import mne
        import numpy as np
        from mne import Epochs, pick_types

        self.tmin = tmin
        self.tmax = tmax

        # Create filtered epochs for testing and training           
        epochs_test, epochs_train, events_test, events_train = {}, {}, {}, {} 

        # For all the bands in the list, extract the epochs
        for key in list(self.bands.keys()):

            fmin, fmax = self.bands[key][0], self.bands[key][1]
            temp_train = self.training_raws.filter(fmin, fmax, fir_design=filter_design)
            temp_events = mne.find_events(temp_train, stim_channel='STIM', shortest_event=1)
            temp_picks = pick_types(temp_train.info, eeg=True, stim=False, eog=False)
            value_train = Epochs(temp_train, temp_events, event_id, tmin, tmax, proj=True, picks=temp_picks, baseline=(tmin, tmin+0.1), preload=True)
            events_train[key] = temp_events
            epochs_train[key] = value_train
            
            
            temp_test = self.testing_raws.filter(fmin, fmax, fir_design=filter_design)
            temp_events = mne.find_events(temp_test, stim_channel='STIM', shortest_event=1)
            temp_picks = pick_types(temp_test.info, eeg=True, stim=False, eog=False)
            value_test = Epochs(temp_test, temp_events, event_id, tmin, tmax, proj=True, picks=temp_picks, baseline=(tmin, tmin+0.1), preload=True)
            events_test[key] = temp_events
            epochs_test[key] = value_test

        self.events_test = events_test
        self.epochs_test = epochs_test
        self.events_train = events_train
        self.epochs_train = epochs_train
           
    def flatten_psd(self, elec_psd, tested_freqs, freq_range = [2, 25]):
        """Flatten the power spectral density using the FOOOF algorithm.

        Args:
            tested_freqs: An array containing the frequencies of interest.
            elec_psd: An array containing the original electrode power spectral density.

        Returns:
            results: A dictionary containing the flat spectrum, offset, slope, peak parameters, guassian parameters,
            FOOOFED spectra, R2 and the error.
        """
        from fooof import FOOOF
        import numpy as np

        # Update settings to fit a more constrained model, to reduce overfitting
        e = FOOOF(peak_width_limits=[2, 8], max_n_peaks=6)

        # Add data to FOOOF object
        #freq_range = [8, 33]
        #fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.4)
        # freq_range = 2*len Need to change the frequency resultion
        e.add_data(tested_freqs, elec_psd, freq_range)
        e.fit(tested_freqs, elec_psd, freq_range)

        results = dict()
        results['PSD'] = elec_psd
        results['tested_Frequencies'] = tested_freqs
        results['flat_Spectrum'] = e._peak_fit
        results['Offset'] = e.aperiodic_params_[0]
        results['Slope'] = e.aperiodic_params_[1]
        results['PeakParams'] = e.peak_params_
        #results['GaussianParams'] = e._gaussian_params
        results['Foofed'] = e.fooofed_spectrum_
        results['R2'] = e.r_squared_
        results['OError'] = e.error_
        
        self.fooof_results = results
        
        return results

    def create_fooof_epochs(self, event_id, tmin=-1.25, tmax=0.25, fmin = 2, fmax = 25):

        fooof_results = {}
        fmin, fmax = 2, 25

        for this_event in list(event_id.keys()):
            print(this_event)
            fooof_results[this_event] = []
            train_event_data = self.epochs_train['wide'][this_event]
            train_psds, train_freqs = psd_multitaper(train_event_data, low_bias=True, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=False, picks='eeg', n_jobs=-1)
         
            for instance in train_psds:
                current_event_data = []
                for electrode in instance:
                    temp_result = self.flatten_psd(electrode, train_freqs, freq_range = [fmin, fmax])
                    current_event_data.append(temp_result)
                fooof_results[this_event].append(current_event_data)
                training_fooof = fooof_results
        self.training_fooof = fooof_results

        fooof_results = {}
        for this_event in list(event_id.keys()):
            print(this_event)
            fooof_results[this_event] = []
            test_event_data = self.epochs_test['wide'][this_event]  
            test_psds, test_freqs = psd_multitaper(test_event_data, low_bias=True, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=False, picks='eeg', n_jobs=-1)

            for instance in test_psds:
                current_event_data = []
                for electrode in instance:
                    temp_result = self.flatten_psd(electrode, test_freqs, freq_range = [fmin, fmax])
                    current_event_data.append(temp_result)
                fooof_results[this_event].append(current_event_data)
                testing_fooof = fooof_results
        self.testing_fooof = fooof_results

    def multiband_binary_OVR_class_blah(self, event1, csp = CSP(n_components=4, reg=None, log=True, norm_trace=False), lda = LinearDiscriminantAnalysis()):

        # Setup parameters for machine learning
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        g = globals()
        csp_array = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for i in self.bands.keys()]
        lda_array = [LinearDiscriminantAnalysis() for i in self.bands.keys()]

        # For each band in the self
        total_predicted_train, total_predicted_val, total_predicted_test = [], [], []
        total_actual_train, total_actual_val, total_actual_test = [], [], []
        event1_train, event1_test = {}, {}
        for k, key in enumerate(list(self.bands.keys())):
            
            print(key)
            # Define the CSP and LDA parameters
            g["csp_" + str(key)] = csp_array[k]
            g["lda_" + str(key)] = lda_array[k]

            # Create a the train self for a second event as a series of other equally represented events
            new_events_list = list(self.epochs_train[key].event_id)
            new_events_list.remove(event1)
            print(new_events_list)

            num_epochs_train_count = []
            all_events = self.epochs_train['wide'].event_id
            for i in all_events:
                num_epochs_train_count.append(len(self.epochs_train['wide'][i]))
            train_count = min(num_epochs_train_count)
            print(train_count)

            floor_train_count = (len(all_events)-1)*int(np.floor(train_count/(len(all_events)-1)))
            num_epochs_train = np.arange(0, floor_train_count, 1)
            chosen_train = sample(list(num_epochs_train), int(len(self.epochs_train[key][new_events_list[1]])/6))

            sev1_train = self.epochs_train[key][new_events_list[0]][chosen_train]
            sev2_train = self.epochs_train[key][new_events_list[1]][chosen_train]
            sev3_train = self.epochs_train[key][new_events_list[2]][chosen_train]
            sev4_train = self.epochs_train[key][new_events_list[3]][chosen_train]
            sev5_train = self.epochs_train[key][new_events_list[4]][chosen_train]
            sev6_train = self.epochs_train[key][new_events_list[5]][chosen_train]

            event2_epochs_train = mne.concatenate_epochs([sev1_train, sev2_train, sev3_train, sev4_train, sev5_train, sev6_train])
            event2_train_data = event2_epochs_train.get_data()
            event2_train_labels = np.array([0] * len(num_epochs_train))

            # Create a the test data for a second event as a series of other equally represented events
            num_epochs_test_count = []
            all_events = self.epochs_test['wide'].event_id
            for i in all_events:
                num_epochs_test_count.append(len(self.epochs_test['wide'][i]))
            test_count = min(num_epochs_test_count)
            print(test_count)

            floor_test_count = (len(all_events)-1)*int(np.floor(test_count/(len(all_events)-1)))
            num_epochs_test = np.arange(0, floor_test_count, 1)
            chosen_test = sample(list(num_epochs_test), int(len(self.epochs_test[key][new_events_list[1]])/6))

            sev1_test = self.epochs_test[key][new_events_list[0]][chosen_test]
            sev2_test = self.epochs_test[key][new_events_list[1]][chosen_test]
            sev3_test = self.epochs_test[key][new_events_list[2]][chosen_test]
            sev4_test = self.epochs_test[key][new_events_list[3]][chosen_test]
            sev5_test = self.epochs_test[key][new_events_list[4]][chosen_test]
            sev6_test = self.epochs_test[key][new_events_list[5]][chosen_test]

            event2_epochs_test = mne.concatenate_epochs([sev1_test, sev2_test, sev3_test, sev4_test, sev5_test, sev6_test])
            event2_test_data = event2_epochs_test.get_data()
            event2_test_labels = np.array([0] * len(num_epochs_test))

            # Create training and testing dataset for both events
            event1_train[key] = self.epochs_train[key][event1][:floor_train_count]
            event1_test[key] = self.epochs_test[key][event1][:floor_test_count]

            # Extract training data from epochs and events
            event1_train_labels = event1_train[key].events[:,2]
            event1_train_data = event1_train[key].get_data()

            # Extract testing data from epochs and events
            event1_test_labels = event1_test[key].events[:,2]
            event1_test_data = event1_test[key].get_data()

            # Setup empty lists to store data
            scores_window = []
            predicted_train, predicted_val, predicted_test = [], [], []
            actual_train, actual_val, actual_test = [], [], []

            # Statify shuffle split the training and validation data
            for train_index, val_index in sss.split(event1_train_data, event1_train_labels):

                # Create event1 train and validation sets
                event1_x_train, event1_x_val = event1_train_data[train_index], event1_train_data[val_index]
                event1_y_train, event1_y_val = event1_train_labels[train_index], event1_train_labels[val_index]

                # Create event2 train and validation sets
                event2_x_train, event2_x_val = event2_train_data[train_index], event2_train_data[val_index]
                event2_y_train, event2_y_val = event2_train_labels[train_index], event2_train_labels[val_index]

                # Concatenate epochs of different events
                x_train = np.concatenate([event1_x_train, event2_x_train], axis = 0)
                x_val = np.concatenate([event1_x_val, event2_x_val], axis = 0)
                y_train = np.concatenate([event1_y_train, event2_y_train])
                y_val = np.concatenate([event1_y_val, event2_y_val], axis = 0)

                # Transform train and validation sets using CSP
                x_train_csp = g["csp_" + str(key)].fit_transform(x_train, y_train)
                x_val_csp = g["csp_" + str(key)].transform(x_val)

                # Store original classification 
                actual_train.extend(y_train)
                actual_val.extend(y_val)
                total_actual_train.extend(y_train)
                total_actual_val.extend(y_val)

                # Fit classifier 
                g["lda_" + str(key)].fit(x_train_csp, y_train)

                # Make predictions
                current_train = g["lda_" + str(key)].predict(x_train_csp)
                current_val = g["lda_" + str(key)].predict(x_val_csp)

                predicted_train.extend(current_train)
                predicted_val.extend(current_val)
                total_predicted_train.extend(current_train)
                total_predicted_val.extend(current_val)
                scores_window.append(g["lda_" + str(key)].score(x_val_csp, y_val))

            x_test = np.concatenate([event1_test_data, event2_test_data], axis = 0)
            y_test = np.concatenate([event1_test_labels, event2_test_labels], axis = 0)
            x_test_csp = g["csp_" + str(key)].transform(x_test)

            # Fit classifier
            validation_score = np.mean(scores_window)
            print('The validation mean score is ', np.mean(scores_window), 'with a standard deviation of ', np.std(scores_window))

            predicted_test = g["lda_" + str(key)].predict(x_test_csp)
            actual_test = y_test
            total_actual_test.extend(actual_test)
            total_predicted_test.extend(predicted_test)
            test_score = g["lda_" + str(key)].score(x_test_csp, y_test)
            print('The test score is ', test_score)

            # Save the actual and predicted values
            self.actual_train = actual_train
            self.predicted_train = predicted_train
            self.actual_val = actual_val
            self.predicted_val = predicted_val
            self.actual_test = actual_test
            self.predicted_test = predicted_test

            self.total_actual_train = total_actual_train
            self.total_predicted_train = total_predicted_train
            self.total_actual_val = total_actual_val
            self.total_predicted_val = total_predicted_val
            self.total_actual_test = total_actual_test
            self.total_predicted_test = total_predicted_test

            # Save the confusion matrix
            self.train_cm = confusion_matrix(total_actual_train, total_predicted_train)
            self.val_cm = confusion_matrix(total_actual_val, total_predicted_val)
            self.test_cm = confusion_matrix(total_actual_test, total_predicted_test)

            #data.train_score = np.mean(train_scores)
            self.validation_score = validation_score
            self.test_score = test_score

            return validation_score, test_score

    def OVR(self, event1, csp = CSP(n_components=12, reg=None, log=True, norm_trace=False), lda = LinearDiscriminantAnalysis()):
        
        # Setup parameters for machine learning
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        g = globals()
        csp_array = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for i in self.bands.keys()]
        lda_array = [LinearDiscriminantAnalysis() for i in self.bands.keys()]

        # For each band in the self
        total_predicted_train, total_predicted_val, total_predicted_test = [], [], []
        total_actual_train, total_actual_val, total_actual_test = [], [], []
        event1_train, event1_test = {}, {}
        for k, key in enumerate(list(self.bands.keys())):
            
            print(key)
            # Define the CSP and LDA parameters
            g["csp_" + str(key)] = csp_array[k]
            g["lda_" + str(key)] = lda_array[k]

            # Create a the train self for a second event as a series of other equally represented events
            new_events_list = list(self.epochs_train[key].event_id)
            new_events_list.remove(event1)
            print(new_events_list)

            num_epochs_train_count = []
            all_events = self.epochs_train['wide'].event_id
            for i in all_events:
                num_epochs_train_count.append(len(self.epochs_train['wide'][i]))
            train_count = min(num_epochs_train_count)
            print(train_count)

            # floor_train_count = (len(all_events)-1)*int(np.floor(train_count/(len(all_events)-1)))
            # num_epochs_train = np.arange(0, floor_train_count, 1)
            # chosen_train = sample(list(num_epochs_train), int(len(self.epochs_train[key][new_events_list[1]])/6))

            num_epochs_train = np.arange(0,len(self.epochs_train[key][event1]),1)
            chosen_train = sample(list(num_epochs_train), int(len(self.epochs_train[key][new_events_list[1]])/6))

            sev1_train = self.epochs_train[key][new_events_list[0]][chosen_train]
            sev2_train = self.epochs_train[key][new_events_list[1]][chosen_train]
            sev3_train = self.epochs_train[key][new_events_list[2]][chosen_train]
            sev4_train = self.epochs_train[key][new_events_list[3]][chosen_train]
            sev5_train = self.epochs_train[key][new_events_list[4]][chosen_train]
            sev6_train = self.epochs_train[key][new_events_list[5]][chosen_train]

            event2_epochs_train = mne.concatenate_epochs([sev1_train, sev2_train, sev3_train, sev4_train, sev5_train, sev6_train])
            event2_train_data = event2_epochs_train.get_data()
            event2_train_labels = np.array([0] * len(num_epochs_train))

            # Create a the test data for a second event as a series of other equally represented events
            num_epochs_test_count = []
            all_events = self.epochs_test['wide'].event_id
            for i in all_events:
                num_epochs_test_count.append(len(self.epochs_test['wide'][i]))
            test_count = min(num_epochs_test_count)
            print(test_count)

            # floor_test_count = (len(all_events)-1)*int(np.floor(test_count/(len(all_events)-1)))
            # num_epochs_test = np.arange(0, floor_test_count, 1)
            # chosen_test = sample(list(num_epochs_test), int(len(self.epochs_test[key][new_events_list[1]])/6))

            num_epochs_test = np.arange(0,len(self.epochs_test[key][event1]),1)
            chosen_test = sample(list(num_epochs_test), int(len(self.epochs_test[key][new_events_list[1]])/6))

            sev1_test = self.epochs_test[key][new_events_list[0]][chosen_test]
            sev2_test = self.epochs_test[key][new_events_list[1]][chosen_test]
            sev3_test = self.epochs_test[key][new_events_list[2]][chosen_test]
            sev4_test = self.epochs_test[key][new_events_list[3]][chosen_test]
            sev5_test = self.epochs_test[key][new_events_list[4]][chosen_test]
            sev6_test = self.epochs_test[key][new_events_list[5]][chosen_test]

            event2_epochs_test = mne.concatenate_epochs([sev1_test, sev2_test, sev3_test, sev4_test, sev5_test, sev6_test])
            event2_test_data = event2_epochs_test.get_data()
            event2_test_labels = np.array([0] * len(num_epochs_test))

            # Create training and testing dataset for both events
            event1_train[key] = self.epochs_train[key][event1]
            event1_test[key] = self.epochs_test[key][event1]

            # Extract training data from epochs and events
            event1_train_labels = event1_train[key].events[:,2]
            event1_train_data = event1_train[key].get_data()

            # Extract testing data from epochs and events
            event1_test_labels = event1_test[key].events[:,2]
            event1_test_data = event1_test[key].get_data()

            # Setup empty lists to store data
            scores_window = []
            predicted_train, predicted_val, predicted_test = [], [], []
            actual_train, actual_val, actual_test = [], [], []

            # Statify shuffle split the training and validation data
            for train_index, val_index in sss.split(event1_train_data, event1_train_labels):

                # Create event1 train and validation sets
                event1_x_train, event1_x_val = event1_train_data[train_index], event1_train_data[val_index]
                event1_y_train, event1_y_val = event1_train_labels[train_index], event1_train_labels[val_index]

                # Create event2 train and validation sets
                event2_x_train, event2_x_val = event2_train_data[train_index], event2_train_data[val_index]
                event2_y_train, event2_y_val = event2_train_labels[train_index], event2_train_labels[val_index]

                # Concatenate epochs of different events
                x_train = np.concatenate([event1_x_train, event2_x_train], axis = 0)
                x_val = np.concatenate([event1_x_val, event2_x_val], axis = 0)
                y_train = np.concatenate([event1_y_train, event2_y_train])
                y_val = np.concatenate([event1_y_val, event2_y_val], axis = 0)

                # Transform train and validation sets using CSP
                x_train_csp = g["csp_" + str(key)].fit_transform(x_train, y_train)
                x_val_csp = g["csp_" + str(key)].transform(x_val)

                # Store original classification 
                actual_train.extend(y_train)
                actual_val.extend(y_val)
                total_actual_train.extend(y_train)
                total_actual_val.extend(y_val)

                # Fit classifier 
                g["lda_" + str(key)].fit(x_train_csp, y_train)

                # Make predictions
                current_train = g["lda_" + str(key)].predict(x_train_csp)
                current_val = g["lda_" + str(key)].predict(x_val_csp)

                predicted_train.extend(current_train)
                predicted_val.extend(current_val)
                total_predicted_train.extend(current_train)
                total_predicted_val.extend(current_val)
                scores_window.append(g["lda_" + str(key)].score(x_val_csp, y_val))

            x_test = np.concatenate([event1_test_data, event2_test_data], axis = 0)
            y_test = np.concatenate([event1_test_labels, event2_test_labels], axis = 0)
            x_test_csp = g["csp_" + str(key)].transform(x_test)

            # Fit classifier
            validation_score = np.mean(scores_window)
            print('The validation mean score is ', np.mean(scores_window), 'with a standard deviation of ', np.std(scores_window))

            predicted_test = g["lda_" + str(key)].predict(x_test_csp)
            actual_test = y_test
            total_actual_test.extend(actual_test)
            total_predicted_test.extend(predicted_test)
            test_score = g["lda_" + str(key)].score(x_test_csp, y_test)
            print('The test score is ', test_score)

            # Save the actual and predicted values
            self.actual_train = actual_train
            self.predicted_train = predicted_train
            self.actual_val = actual_val
            self.predicted_val = predicted_val
            self.actual_test = actual_test
            self.predicted_test = predicted_test

            self.total_actual_train = actual_train
            self.total_predicted_train = predicted_train
            self.total_actual_val = actual_val
            self.total_predicted_val = predicted_val
            self.total_actual_test = actual_test
            self.total_predicted_test = predicted_test

            # Save the confusion matrix
            self.train_cm = confusion_matrix(actual_train, predicted_train)
            self.val_cm = confusion_matrix(actual_val, predicted_val)
            self.test_cm = confusion_matrix(actual_test, predicted_test)

            self.validation_score = validation_score
            self.test_score = test_score

            return validation_score, test_score

    def OVR_best_params(self, event1):
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_validate
        import random
        random.seed(a=42)
        # csp = CSP(n_components=12, reg=None, log=True, norm_trace=False)
        # lda = LinearDiscriminantAnalysis()
        # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        # For each band in the self
        total_predicted_train, total_predicted_val, total_predicted_test = [], [], []
        total_actual_train, total_actual_val, total_actual_test = [], [], []
        event1_train, event1_test = {}, {}

        for k, key in enumerate(list(self.bands.keys())):
            
            print(key)

            # Create a the train self for a second event as a series of other equally represented events
            new_events_list = list(self.epochs_train[key].event_id)
            new_events_list.remove(event1)
            print(new_events_list)

            num_epochs_train = np.arange(0,len(self.epochs_train[key][event1]),1)
            chosen_train = sample(list(num_epochs_train), int(len(self.epochs_train[key][new_events_list[1]])/6))

            sev1_train = self.epochs_train[key][new_events_list[0]][chosen_train]
            sev2_train = self.epochs_train[key][new_events_list[1]][chosen_train]
            sev3_train = self.epochs_train[key][new_events_list[2]][chosen_train]
            sev4_train = self.epochs_train[key][new_events_list[3]][chosen_train]
            sev5_train = self.epochs_train[key][new_events_list[4]][chosen_train]
            sev6_train = self.epochs_train[key][new_events_list[5]][chosen_train]

            event2_epochs_train = mne.concatenate_epochs([sev1_train, sev2_train, sev3_train, sev4_train, sev5_train, sev6_train])
            event2_train_data = event2_epochs_train.get_data()
            event2_train_labels = np.array([0] * len(num_epochs_train))

            num_epochs_test = np.arange(0,len(self.epochs_test[key][event1]),1)
            chosen_test = sample(list(num_epochs_test), int(len(self.epochs_test[key][new_events_list[1]])/6))

            sev1_test = self.epochs_test[key][new_events_list[0]][chosen_test]
            sev2_test = self.epochs_test[key][new_events_list[1]][chosen_test]
            sev3_test = self.epochs_test[key][new_events_list[2]][chosen_test]
            sev4_test = self.epochs_test[key][new_events_list[3]][chosen_test]
            sev5_test = self.epochs_test[key][new_events_list[4]][chosen_test]
            sev6_test = self.epochs_test[key][new_events_list[5]][chosen_test]

            event2_epochs_test = mne.concatenate_epochs([sev1_test, sev2_test, sev3_test, sev4_test, sev5_test, sev6_test])
            event2_test_data = event2_epochs_test.get_data()
            event2_test_labels = np.array([0] * len(num_epochs_test))

            # Create training and testing dataset for both events
            event1_train[key] = self.epochs_train[key][event1]
            event1_test[key] = self.epochs_test[key][event1]

            # Extract training data from epochs and events
            event1_train_labels = event1_train[key].events[:,2]
            event1_train_data = event1_train[key].get_data()

            # Extract testing data from epochs and events
            event1_test_labels = event1_test[key].events[:,2]
            event1_test_data = event1_test[key].get_data()

            # Concatenate epochs of different events
            X = np.concatenate([event1_train_data, event2_train_data], axis = 0)
            x_test = np.concatenate([event1_test_data, event2_test_data], axis = 0)
            y = np.concatenate([event1_train_labels, event2_train_labels])
            y_test = np.concatenate([event1_test_labels, event2_test_labels], axis = 0)

        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([('CSP', csp), ('LDA', lda)])

        param_grid = [
            {'CSP__n_components': [4, 6, 8, 10, 12, 14], 
            'CSP__log': [True, False],
            'LDA__solver': ['lsqr', 'eigen'], 
            'LDA__shrinkage': ['auto', 0, 0.5, 1]}
            ]

        gs = GridSearchCV(estimator=clf, 
                            param_grid=param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1, 
                            cv=10)

        # run gridearch
        gs = gs.fit(X, y)
        for i in range(len(gs.cv_results_['params'])):
            print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])

        clf = gs.best_estimator_
        scores = cross_validate(clf, X, y, cv = cv, return_train_score = True)

        train_score = np.mean(scores['train_score'])
        val_score = np.mean(scores['test_score'])

        csp = gs.best_estimator_[0]
        X_csp_train = csp.fit_transform(X,y)
        X_csp_test = csp.transform(x_test)

        lda = gs.best_estimator_[1]
        lda.fit(X_csp_train, y)
        test_score = lda.score(X_csp_test, y_test)

        actual_test = y_test
        predicted_test = lda.predict(X_csp_test)

        print("Train accuracy: %f" % train_score)
        print("Validation accuracy: %f" % val_score)
        print("Test accuracy: %f" % test_score)

        return train_score, val_score, test_score, clf, actual_test, predicted_test

    def extract_metrics(self, dataset_type, actual, predicted): # This is a crappy function. Fix this.
        
        import itertools
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
        from IPython.core.display import display

        print(dataset_type, ' classification report')
        report = classification_report(actual, predicted, output_dict=True) 
        report_df = pd.DataFrame(report).transpose()
        display(report_df)  
        display(report_df)  

        return report_df


    def OVO_best_params(self, event1, event2):
        # Import libraries for machine learning
        from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, cross_val_predict
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
        from sklearn.pipeline import Pipeline
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

        import pandas as pd
        import numpy as np
        from random import sample
        import mne
        from mne.decoding import CSP
        import matplotlib.pyplot as plt

        mne.set_log_level(30) 
        
        # Import libraries for machine learning
        total_predicted_test,total_actual_test = [], []
        event1_train, event2_train, event1_test, event2_test = {}, {}, {}, {}
        train_percentage, val_percentage, test_percentage = [], [], []
        for key in enumerate(list(self.bands.keys())):
            key = key[1]

            # Create training and testing dataset for both events
            event1_train[key] = self.epochs_train[key][event1]
            event2_train[key] = self.epochs_train[key][event2]
            event1_test[key] = self.epochs_test[key][event1]
            event2_test[key] = self.epochs_test[key][event2]

            # Extract training data from epochs and events
            event1_train_labels = event1_train[key].events[:,2]
            event1_train_data = event1_train[key].get_data()
            event2_train_labels = event2_train[key].events[:,2]
            event2_train_data = event2_train[key].get_data()

            # Extract testing data from epochs and events
            event1_test_labels = event1_test[key].events[:,2]
            event1_test_data = event1_test[key].get_data()
            event2_test_labels = event2_test[key].events[:,2]
            event2_test_data = event2_test[key].get_data()

            X = np.concatenate([event1_train_data, event2_train_data], axis = 0)
            y = np.concatenate([event1_train_labels, event2_train_labels])
            x_test = np.concatenate([event1_test_data, event2_test_data], axis = 0)
            y_test = np.concatenate([event1_test_labels, event2_test_labels])

        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([('CSP', csp), ('LDA', lda)])

        param_grid = [
            {'CSP__n_components': [4, 6, 8, 10, 12, 14], 
            'CSP__log': [True, False],
            'LDA__solver': ['lsqr', 'eigen'], 
            'LDA__shrinkage': ['auto', 0, 0.5, 1]}
            ]

        gs = GridSearchCV(estimator=clf, 
                            param_grid=param_grid, 
                            scoring='accuracy', 
                            n_jobs=-1, 
                            cv=10)

        # run gridearch
        gs = gs.fit(X, y)
        for i in range(len(gs.cv_results_['params'])):
            print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])

        clf = gs.best_estimator_
        scores = cross_validate(clf, X, y, cv = cv, return_train_score = True)

        train_score = np.mean(scores['train_score'])
        val_score = np.mean(scores['test_score'])

        csp = gs.best_estimator_[0]
        X_csp_train = csp.fit_transform(X,y)
        X_csp_test = csp.transform(x_test)

        lda = gs.best_estimator_[1]
        lda.fit(X_csp_train, y)
        test_score = lda.score(X_csp_test, y_test)

        actual_test = y_test
        predicted_test = lda.predict(X_csp_test)

        print("Train accuracy: %f" % train_score)
        print("Validation accuracy: %f" % val_score)
        print("Test accuracy: %f" % test_score)

        return train_score, val_score, test_score, clf, actual_test, predicted_test