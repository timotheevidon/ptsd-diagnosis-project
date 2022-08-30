import numpy as np 
import pandas as pd
from model import Model
from tqdm import tqdm
import itertools
from data_loader import DataLoader, OpenSmileDataLoader
from feature_extractor import FeatureExtractor, OpenSmileFeatureExtractor, get_feature_extractor
from labeller import CsvLabeller

class Parameter():
    ''' Class allowing to define hyper-parameters and their range'''
    def __init__(self, name, min_=0, max_=np.Infinity, count=1, list_=[], constraint=None):
        '''If the hyper-parameter is categorical (strings), use the list'''
        '''If the hyper-parameter is a float value, use max_, min_ and count'''
        self.name = name
        self.min_ = min_
        self.max_ = max_
        self.count = count
        self.list_ = list_
        self.constraint = constraint
        if self.list_ != [] and self.max_ != np.Infinity:
            raise NameError('CANNOT DECLARE PARAMETER WITH list_ AND max_)')

    def get_elements(self):
        ''' Allows to gets all the possible instances of a hyper-parameter'''
        if self.max_ != np.Infinity:
            self.list_ = np.linspace(self.min_, self.max_, self.count)
        elements = []
        for element in self.list_:
            if self.constraint == 'int':
                elements.append( {self.name : int(element)} )
            else:
                elements.append( {self.name : element} )
        return elements

class Optimizer:
    ''' Optimizers will find the best hyper-parameters for a model'''
    def __init__(self, type_, dataloader, params, PCA_components=[25]):
        ''' params must be a list of instances of the Parameter class'''
        self.type_ = type_
        self.params = params
        self.dataloader = dataloader
        self.PCA_components= PCA_components
        self.get_models()

    def get_params(self):
        for param in self.params:
            yield param.get_elements()

    def get_models(self):
        ''' Instanciates our models with the different hyper-parameter combinaisons'''
        self.models = []
        params = [x.get_elements() for x in self.params]
        for PCA_component in self.PCA_components:
            for param_combinaison_tuple in itertools.product(*params):
                param_combinaison = {}
                for elt in param_combinaison_tuple:
                    param_combinaison.update(elt)
                self.models.append(Model(self.type_, self.dataloader, param_combinaison, PCA_components = PCA_component))

    def get_report(self, report_name='test',metric='f1'):
        ''' Returns best hyper-parameter combinaison and score'''
        grid_search_report = [['score', 'PCA components','params']]
        self.best_model = 0
        best_score = -1
        best_pca = 0
        best_params = {}
        for model in tqdm(self.models) :
            model.evaluate(metric)
            if model.score > best_score:
                best_score = model.score
                best_params = model.params
                self.best_model = model
                best_pca = model.PCA_components
            grid_search_report.append([model.score, model.PCA_components, model.params])
            report_df = pd.DataFrame(grid_search_report)
            report_df.to_csv('./{}.csv'.format(report_name), index=False)
            del model
        return best_score, best_params, best_pca

if __name__ == "__main__":

    TARGET_FILE = "c:/Users/pc/Desktop/Vidon/Centrale/resileyes/new_pull/resileyes/TARGET_FILES/labels.csv"
    AUDIO_FOLDER = "e:/DAIC-WOZ-dataset/dataset épuré/audios"
    CONFIG_FILE = 'eGeMAPSv02'

    labeller = CsvLabeller(TARGET_FILE)
    extractor = get_feature_extractor(AUDIO_FOLDER, CONFIG_FILE, type='DefaultOpenSmileFeatureExtractor')
    loader = OpenSmileDataLoader(input_folder=AUDIO_FOLDER, feature_extractor=extractor, labeller=labeller, n_splits=4)

    ''' study to execute to search for optimized SVC'''
    # kernel = Parameter('kernel', list_ = ['linear'])
    # gamma = Parameter('gamma', min_=1,max_=10000,count=10)
    # tol = Parameter('tol', min_=1e-5,max_=1e-2,count=10)
    # c = Parameter('C', min_=1,max_=1e4,count=10)
    # coef0 = Parameter('coef0', min_=0,max_=1e4,count=10, constraint='int')
    # SVC_parameters = [kernel, c, gamma, tol, coef0]
    # type_ = 'SVC'
    # PCA_components = [40]#,[i for i in range(5,80,1)]
    # optim = Optimizer(type_, loader, SVC_parameters, PCA_components=PCA_components)
        # Ce qui est bizarre c est que c et gamma n'ont pas d'influence sur le score F1
        # => les données sont presque lineairement séparable ?
        # = 84% de score F1 obtenu pour 40 PCA components

    ''' study to execute to search for optimized  Random Forest'''
    # criterion = Parameter('criterion', list_=['gini', 'entropy'])
    # max_depth = Parameter('max_depth', min_=10, max_=200, count=10, constraint='int')
    # n_estimators = Parameter('n_estimators', min_=10, max_=200, count=10, constraint='int')
    # RF_parameters = [criterion, max_depth, n_estimators]
    # PCA_components = [i for i in range(30,80,10)]
    # type_ = 'RandomForest'
    # optim = Optimizer(type_, loader, RF_parameters, PCA_components=PCA_components)
        # BEST RESULT :  56.97191697191698  %
        # WITH PARAMS :  {'criterion': 'entropy', 'max_depth': 70, 'n_estimators': 40}
        # WITH CLASSIFIER : RandomForest
        # WITH PCA components :  40
        # Les resultats sont pas oufs avec les RF 

    ''' study to execute to search for optimized AdaBoost'''
    # algorithm = Parameter('algorithm', list_=['SAMME', 'SAMME.R'])
    # learning_rate = Parameter('learning_rate', min_=1e-3, max_=1, count=10)
    # n_estimators = Parameter('n_estimators', min_=1000, max_=300, count=10, constraint='int')
    # AdaBoost_parameters = [algorithm, learning_rate, n_estimators]
    # PCA_components = [i for i in range(30,80,10)]
    # type_ = 'AdaBoost'
    # optim = Optimizer(type_, loader, AdaBoost_parameters, PCA_components=PCA_components)
        # BEST RESULT :  76.84222684222685  %
        # WITH PARAMS :  {'algorithm': 'SAMME', 'learning_rate': 0.889, 'n_estimators': 200}
        # WITH PCA components :  30
        # WITH CLASSIFIER : AdaBoost

    ''' study to execute to search for optimized DT'''
    # criterion = Parameter('criterion', list_=['gini', 'entropy'])
    # max_depth = Parameter('max_depth', min_=10, max_=200, count=100, constraint='int')
    # DT_parameters = [criterion, max_depth]
    # PCA_components = [i for i in range(30,60,100)]
    # type_ = 'DT'
    # optim = Optimizer(type_, loader, DT_parameters, PCA_components=PCA_components)
    #     BEST RESULT :  55.85784313725492  %
    #     WITH PARAMS :  {'criterion': 'gini', 'max_depth': 178}
    #     WITH PCA components :  30
    #     WITH CLASSIFIER : DT

    ''' study to execute to search for optimized SGD'''
    loss = Parameter('loss', list_=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
    penalty = Parameter('penalty', list_ = ['l1','l2','elasticnet'])
    alpha = Parameter('alpha', min_=1e-4, max_=1e-2, count=100)
    SGD_parameters = [loss, penalty, alpha]
    PCA_components = [i for i in range(50,60,10)]
    type_ = 'SGD'
    optim = Optimizer(type_, loader, SGD_parameters, PCA_components=PCA_components)
        # BEST RESULT :  88.15422565422566  %
        # WITH PARAMS :  {'loss': 'squared_hinge', 'penalty': 'l1', 'alpha': 0.0025}
        # WITH PCA components :  50
        # WITH CLASSIFIER : SGD
        # A prendre avec des pincettes car le SGD n'est pas deterministe, ce résultat est difficilement reproductible

    report = optim.get_report()
    print('BEST RESULT : ', report[0]*100, ' %')
    print('WITH PARAMS : ', report[1])
    print('WITH PCA components : ', report[2])
    print('WITH CLASSIFIER :', type_)
