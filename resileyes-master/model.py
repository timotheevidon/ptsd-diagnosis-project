#Models and pre-processors
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#Utils
from joblib import dump, load
import numpy as np
import pandas as pd
#Internal
from feature_extractor import FeatureExtractor, OpenSmileFeatureExtractor, get_feature_extractor
from data_loader import DataLoader, OpenSmileDataLoader
from labeller import CsvLabeller

class Model():
    ''' A model class for our classification task '''
    def __init__(self, type_, dataloader, params={}, transformers=['Imputer','StandardScaler','PCA'], PCA_components=25):
        ''' Here, params must be a dictionnary of parameters allowed by the classifier'''
        self.type_ = type_
        self.params = params
        self.dataloader = dataloader
        self.score = -1
        self.PCA_components= PCA_components
        self.transformers = []
        clf={}
        '''Allowed Classifiers, you can add more as long as they have a fit and a predict method'''
        if self.type_ == 'SVC':
            clf = SVC()
        elif self.type_ == 'RandomForest':
            clf = RandomForestClassifier()
        elif self.type_ == 'SGD':
            clf = SGDClassifier()
        elif self.type_ == 'DT':
            clf = DecisionTreeClassifier()
        elif self.type_ == 'AdaBoost':
            clf = AdaBoostClassifier()
        else:
            raise NameError('Allowed models are SVC - RandomForest - SGD - KNN - DT - AdaBoost')
        try:
            clf.set_params(**params)
        except: 
            raise NameError('ERROR IN MODEL PARAMETERS', params)
        self.clf = clf
        '''The imputer is used to replace Nan and np.inf in the data by zeros'''
        if 'Imputer' in transformers:
            self.transformers.append(('SimpleImputer',SimpleImputer(missing_values=np.nan, strategy='mean')))
        if 'StandardScaler' in transformers:
            self.transformers.append(('StandardScaler',StandardScaler()))
        if 'PCA' in transformers:
            self.transformers.append(('PCA',PCA(n_components=PCA_components)))
        self.transformers.append(('clf',clf))
        self.clf = Pipeline(steps = self.transformers)

    def train(self, X_train, y_train):
        self.clf.fit(np.array(X_train), np.array(y_train).ravel())

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def evaluate(self, metric='f1'):
        ''' F1-score is the most appropriate metric for this task'''
        scores = []
        for X_train, y_train, X_test, y_test in list(self.dataloader.generate_split()):
            self.train(X_train, y_train)
            y_pred = self.predict(X_test)
            if metric == 'f1':
                scores.append(f1_score(y_test, y_pred))
            elif metric == 'accuracy':
                scores.append(accuracy_score(y_test, y_pred))
            elif metric == 'recall':
                scores.append(recall_score(y_test, y_pred))
            else:
                raise NameError('Allowed metrics are f1 - accuracy - recall')
        self.score = sum(scores)/len(scores)
        return self.score

    def save(self, directory):
        dump(self.clf, directory) 
    
    def load(self, directory):
        self.clf = load(directory) 

if __name__ == "__main__":
    TARGET_FILE = "c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/TARGET_FILES/labels.csv"
    AUDIO_FOLDER = "e:/DAIC-WOZ-dataset/dataset épuré/audios"
    CONFIG_FILE = 'eGeMAPSv02'

    labeller = CsvLabeller(TARGET_FILE)
    extractor = get_feature_extractor(AUDIO_FOLDER, CONFIG_FILE)
    loader = OpenSmileDataLoader(input_folder=AUDIO_FOLDER, feature_extractor=extractor, labeller=labeller, n_splits=4)

    # params = {'kernel':"linear", 'gamma':10000, 'C':1, 'coef0':0}
    # model_test = Model('SVC', loader, params, PCA_components=50)
    # results = model_test.evaluate()
    # print(results)

    params = {'loss': 'squared_hinge', 'penalty': 'l1', 'alpha': 0.0025}
    model_test = Model('SGD', loader, params, PCA_components=50)
    results = model_test.evaluate()
    print(results)

