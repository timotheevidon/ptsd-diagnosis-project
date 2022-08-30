from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from model import Model
import numpy as np
from hyper_optimizer import Parameter
from data_loader import DataLoader, OpenSmileDataLoader
from feature_extractor import FeatureExtractor, OpenSmileFeatureExtractor, get_feature_extractor
from labeller import CsvLabeller

class Voter(Model):
    ''' A class able to combine the results of multiple classifiers'''
    def __init__(self, models, dataloader):
        self.models = models
        self.dataloader = dataloader

    def train(self, X_train, y_train):
        '''Every model is trained on the same data'''
        for model in self.models:
            model.train(np.array(X_train), np.array(y_train).ravel())

    def predict(self, X_test):
        '''The final prediction will be the average of all predictions'''
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X_test))
        average_prediction = sum(predictions) / len(predictions)
        for i in range (len(average_prediction)):
            average_prediction[i] = round(average_prediction[i])
        return average_prediction
    
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

if __name__ == "__main__":

    TARGET_FILE = "c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/TARGET_FILES/labels.csv"
    AUDIO_FOLDER = "e:/DAIC-WOZ-dataset/dataset épuré/audios"
    CONFIG_FOLDER = 'c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/resileyes_config/working_config'

    labeller = CsvLabeller(TARGET_FILE)
    extractor = get_feature_extractor(AUDIO_FOLDER, CONFIG_FOLDER)
    loader = OpenSmileDataLoader(input_folder=AUDIO_FOLDER, feature_extractor=extractor, labeller=labeller, n_splits=3)

    svm = Model('SVC', loader, {'kernel': 'linear'}, PCA_components=40)
    rf = Model('RandomForest', loader, {'criterion': 'entropy', 'max_depth': 70, 'n_estimators': 40}, PCA_components=40)
    ada = Model('AdaBoost', loader,  {'algorithm': 'SAMME', 'learning_rate': 0.889, 'n_estimators': 200}, PCA_components=40)
    sgd = Model('SGD', loader, {'loss': 'squared_hinge', 'penalty': 'l1', 'alpha': 0.0025}, PCA_components=40)
    
    models = Voter([svm, ada, sgd], loader)
    results = models.evaluate()
    print(results)