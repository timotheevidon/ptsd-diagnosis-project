from wav_transformer import WavTransformer
import os
import librosa
from time import time
import random
import numpy as np
import scipy.signal as sg

class DataAugmentor(WavTransformer):
    """Creates a handful of files which are similar to the original ones.
    Useful to provide more training data
    The original file will be saved as <original_file_name>_0.wav
    The k^th new file will be named <original_file_name>_<k+1>.wav
    """
    def __init__(self, input_folder, output_folder = "data_augmented", name = "Data Augmentation", processes = 4, samples_per_sample=3, manipulations_per_sample = 2):
        super().__init__(input_folder, output_folder, name=name, processes=processes)
        self.samples_per_sample = samples_per_sample
        self.manipulations_per_sample = manipulations_per_sample
        self.augmenting_functions = [self.add_noise, self.update_pitch, self.stretch_time, self.frequency_filter]
        
    def add_noise(self, data, min_ratio=0, max_ratio=0.05):
        """Adds white noise to the data
        min_ratio and max_ratio control the desired average noise_to_signal ratio
        """
        ratio = random.uniform(min_ratio, max_ratio)
        mean_value = np.mean(abs(data))
        noise = np.random.randn(len(data))
        augmented_data = data + ((ratio * mean_value) * noise)
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data
                
    def update_pitch(self, data, min_ratio=-2, max_ratio=2):
        """Changes the pitch of the sample by a certain number of harmonic half-steps
        ratio decides how many harmonic half-steps to go down or up"""
        ratio = random.randint(min_ratio, max_ratio)
        return librosa.effects.pitch_shift(data, self.sr, ratio)
    
    def stretch_time(self, data, min_ratio = 0.9, max_ratio = 1.1):
        """Changes the speed of the sound extract
        ratio decides how much slower of faster it should be"""
        ratio = random.uniform(min_ratio, max_ratio)
        return librosa.effects.time_stretch(data, ratio)
    
    def frequency_filter(self, data, min_n=1, max_n=2, min_ratio = 0.75, max_ratio = 1):
        """Applies a low_pass or a band_pass filter on the sample, mimicking a different recording setup
        n controls the order of the filter : how harsh the transformation is
        ratio controls the characteristic frequency of the filter"""
        btype = random.choice(['lowpass', 'highpass'])
        ratio = random.uniform(min_ratio, max_ratio)
        n = random.randint(min_n, max_n)
        if btype == "lowpass":
            freq = 0.9*ratio
            sos = sg.butter(n, freq, btype=btype, output='sos')
            return sg.sosfiltfilt(sos, data)
        elif btype == "highpass":
            freq = 0.1*ratio
            sos = sg.butter(n, 1-ratio, btype=btype, output='sos')
            return sg.sosfiltfilt(sos, data)
                
    def run_manipulation_sample(self, data):
        """Applies manipulations_per_sample transformations on average"""
        x = data.copy()
        n_manipulations = np.random.poisson(self.manipulations_per_sample) # On average, we perform 2 manipulations before saving
        for _ in range(max(1, n_manipulations)):
            manipulator = random.choice(self.augmenting_functions)
            x = manipulator(x)
        return x

    def transform(self, filename, data):
        """Creates samples_per_samples samples from the data"""
        samples = [data]
        n_samples = np.random.poisson(self.samples_per_sample) # On average, we turn a sample into 3
        for sample in range(n_samples):
            try:
                sample = self.run_manipulation_sample(data)
                samples.append(sample)
            except MemoryError as e:
                print(f"Failed manipulation {sample}/{n_samples} for {filename}")
        return samples

    def run_file(self, filename):
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)
        data, sr = librosa.load(input_path)
        self.sr = sr
        samples = self.transform(filename, data)
        k=0
        for sample in samples:
            path = output_path.replace(".wav", f"_aug_{k}.wav")
            k+=1
            self.save_file(path, sample)

if __name__ == "__main__":
    input = os.path.join("datasets", "audio_test", "test")
    output = os.path.join("datasets", "audio_test", "converted")
    da = DataAugmentor(input, output, processes=4)
    da.run()