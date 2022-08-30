from abc import ABC, abstractmethod
import pandas as pd
import logging

class Labeller(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self, data):
        pass

class CsvLabeller(Labeller):
    """"labels data given a data file and a target file
    target_file: csv file with a column named file containing the full name of the files to be labelled
    label_name: name of the column containing the labels"""
    def __init__(self, target_file, label_name='label'):
        super(CsvLabeller, self).__init__()
        self.target_file = target_file
        self.targets = pd.read_csv(target_file, index_col='file')
        self.label_name = label_name
        if label_name not in self.targets.columns:
            raise AttributeError("label file has no column {} ! columns available : {}".format(label_name, self.targets.columns))

    def run(self, data):
        """labels the data
        data: dataframe which index are the file names. columns can be whatever (usually features)
        returns:   
            X: (DataFrame) same as data
            y: (Series) labels s.t X.iloc[i] = y.iloc[i]
            groups: (Series) groups.iloc[i] = name of the original file before augmentation for row i (cf get_source)
        """
        source = self.group_source(data)
        df = data.copy()
        df['source'] = source
        df = pd.merge(df, self.targets, how='inner', left_on='source', right_index=True, validate='m:1') 
        if len(df) < len(data): # if size of merge inferior to size of intial data, warn
            logging.warn("Some data went missing during labelling. Returning {}/{} labelled files. "
            "This probably means that the files from your feature extraction don't have the same name "
            "as those in the 'file' column of your label file".format(len(df), len(data)))
        X = df.drop([self.label_name,'source'],axis=1)
        y = df[self.label_name]
        groups = df['source']
        return X,y,groups

    def group_source(self, df):
        """apply get_source on all rows of the dataframe"""
        files = pd.Series(df.index.map(self.get_source), index=df.index)
        return files

    def get_source(self, file, extension='.wav'):
        """gets the name of the source file for a file (useful when using file augmentation to avoid having 
        two versions of the same file in the train and test set)"""
        splits = file.split('_aug_')
        if len(splits) == 1: # if no _AUG_ in filename (eg original file)
            return file
        return ''.join(splits[:-1]) + extension

if __name__ == '__main__':    
    import argparse
    parser = argparse.ArgumentParser(description='Labels data')

    parser.add_argument('-data_file', metavar='f', type=str,
                        help='path to the csv file containing the data')
    parser.add_argument('-target_file', type=str,
                        help='path to the target file')
    parser.add_argument('-output_x', metavar='out', type=str,
                        help='file where to save the result for data X')
    parser.add_argument('-output_y', metavar='out', type=str,
                        help='file where to save the result for labels y')
    args = parser.parse_args()
    DATA_FILE = args.data_file
    TARGET_FILE = args.target_file
    OUTPUT_X= args.output_x
    OUTPUT_Y = args.output_y

    
    labeller = CsvLabeller(TARGET_FILE)
    df = pd.read_csv(DATA_FILE, index_col='file')
    X, y  = labeller.run(df)
    X.to_csv(OUTPUT_X) # write them in the output file
    y.to_csv(OUTPUT_Y)
