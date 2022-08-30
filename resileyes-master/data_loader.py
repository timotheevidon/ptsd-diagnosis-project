from abc import ABC, abstractmethod
import warnings
from sklearn.model_selection import GroupKFold

class DataLoader(ABC):

    def __init__(self, **kwargs):
        """An object specifying how to load and preprocess data from a folder of wav file to be ready to use as model input"""

    @abstractmethod
    def run(self):
        """returns two arrays or dataframes X,y with the data and the labels associated"""
        pass

    def generate_split(self):
        """splits X,y"""
        pass

class OpenSmileDataLoader(DataLoader):

    def __init__(self, input_folder, feature_extractor, labeller, n_splits=5):
        """An object specifying how to load and split data from a folder of wav file to be ready to use as model input
        input_folder: folder containing the audio files to process
        feature_extractor: instance of FeatureExtractor that will load and transform audio files into features
        labeller: instance of Labeller that will get the labels for the data
        n_splits: number of splits to perform with KFold for cross-validation
        """
        super(OpenSmileDataLoader, self).__init__()
        self.input_folder = input_folder
        self.feature_extractor = feature_extractor
        self.labeller = labeller
        self.n_splits = n_splits
        self.k_fold = GroupKFold(n_splits=n_splits)
        self.load()

    def load(self):
        """loads transformed data"""
        self.data = self.feature_extractor.run()

    def run(self):
        """loads data, labels it and returns it"""
        try:
            data = self.data
        except AttributeError: # if data isn't loaded
            warnings.warn("DataLoader field 'data' is not initialized, you should use self.load before self.run." 
            "This should have been called during initialization.\n"
            "Calling self.load from method 'run'...")
            self.load()
        X, y, groups = self.labeller.run(data) # get labels for data
        return X,y,groups

    def generate_split(self):
        """loads data, labels it and returns the splits"""
        X, y, groups = self.run()
        indexes = self.k_fold.split(X, y, groups=groups)
        for (train, test) in indexes:
            yield (X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test])

if __name__ == '__main__':    
    import argparse
    from feature_extractor import get_feature_extractor
    from labeller import CsvLabeller
    parser = argparse.ArgumentParser(description='Process an audio file (.wav)')

    parser.add_argument('-audio_folder', metavar='f', type=str,
                        help='path to the folder containing the audio extracts (must be .wav)')
    parser.add_argument('-config_folder', metavar='f', type=str,
                        help='path to the folder containing the config files')
    parser.add_argument('-target_file', type=str,
                        help='path to the target file')
    
    args = parser.parse_args()

    # TARGET_FILE = args.target_file
    # AUDIO_FOLDER = args.audio_folder
    # CONFIG_FOLDER = args.config_folder

    TARGET_FILE = "c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/TARGET_FILES/labels.csv"
    AUDIO_FOLDER = "e:/DAIC-WOZ-dataset/dataset épuré/audios"
    CONFIG_FOLDER = 'c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/resileyes_config/working_config'

    extractor = get_feature_extractor(input_folder=AUDIO_FOLDER, config_folder=CONFIG_FOLDER)
    labeller = CsvLabeller(TARGET_FILE)

    loader = OpenSmileDataLoader(
        input_folder=AUDIO_FOLDER, feature_extractor=extractor, labeller=labeller, n_splits=3)

    X,y,groups  = loader.run()
    folds = list(loader.generate_split())
    X_tr, X_te, y_tr, y_te = folds[0]
    n_folds = len(folds)
    print('X : {} | y : {} | n_folds : {} | X_tr : {} | y_tr : {} | X_te : {} | y_te : {}'.format(
        X.shape, y.shape, n_folds, X_tr.shape, y_tr.shape, X_te.shape, y_te.shape))
    print(X,y)
    
