from abc import abstractmethod
import os
import soundfile as sf
from multiprocessing import Pool
import librosa
from time import time
from tqdm import tqdm

class WavTransformer():
    """Transforms a folder full of .wav into transformed .wav files
    Allows for parallel computation
    To transform data, implement this class and add it to the WavPipeline

    processes: controls the number of threads
    """
    def __init__(self, input_folder, output_folder, name="Transformer", processes = 4):
        self.name = name
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processes = processes
        os.makedirs(output_folder, exist_ok=True)

    def save_file(self, output_path, data):
        """Saves data"""
        sf.write(output_path, data, self.sr)

    @abstractmethod
    def transform(self, filename, data):
        """Implement this function when you inherit the class"""
        pass

    def run_file(self, filename):
        """Load data, run the transformer on it, then save the new data"""
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)
        data, sr = librosa.load(input_path)
        self.sr = sr
        data = self.transform(filename, data)
        self.save_file(output_path, data)

    def run(self):
        """Runs the wave transformer on the input dataset in parallel.
        processes controls the number of threads"""
        print(f"{self.name}")
        self.filenames = os.listdir(self.input_folder)
        with Pool(processes = self.processes) as pool:
            with tqdm(total=len(self.filenames), smoothing=0.1) as pbar:
                for _ in pool.imap(self.run_file, self.filenames):
                    pbar.update()

if __name__ == "__main__":
    input = os.path.join("datasets", "audio_test", "test")
    output = os.path.join("datasets", "audio_test", "converted")
    wt = WavTransformer(input, output, processes=4)
    wt.run()