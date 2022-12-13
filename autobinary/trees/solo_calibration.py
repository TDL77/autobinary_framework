import pandas as pd
import pickle

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..custom_metrics.balance_cover import BalanceCover
from ..utils.folders import create_folder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

import matplotlib.pyplot as plt

# Функция калибровки
def plot_calibration_curve(y_true: pd.DataFrame, y_prob: pd.DataFrame, n_bins: int, ax = None, normalize = False):
    
    prob_true, prob_pred = calibration_curve(y_true,
                                            y_prob,
                                            n_bins=n_bins,
                                            normalize=normalize)
    if ax is None:
        ax = plt.gca()
        
    # Идеальная калибровка
    ax.plot([0,1],[0,1],':',c='k')
    
    curve = ax.plot(prob_pred, prob_true, marker='o')
    
    # Заголовки осей
    ax.set_xlabel('Спрогнозированная вероятность в бине (предсказани)')
    ax.set_ylabel('Доля наблюдений положительного класса в бине (на самом деле)')
    
    #Задаем укладку
    ax.set(aspect='equal')
    
    return curve

class FinalModel():
    
    def __init__(self,
                 prep_pipe_final: object=None,
                 model_final: object=None,
                 base_pipe: object=None,
                 num_columns: list=None,
                 cat_columns: list=None,
                 model_type: str=None, 
                 model_params: dict=None, 
                 task_type: str='classification'):
        
        
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.features = self.num_columns+self.cat_columns
        self.prep_pipe = prep_pipe_final
        self.model = model_final
        self.base_pipe = base_pipe
        self.model_type = model_type
        self.model_params = model_params
        self.task_type = task_type
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        
        if self.prep_pipe is not None and self.model is not None:
            print('Модель обучена! Доступны только метрики и калибровка!')
            
        else:
            
            if len(self.num_columns)>0 and len(self.cat_columns)>0:
                prep_pipe = self.base_pipe(
                    num_columns=self.num_columns,
                    cat_columns=self.cat_columns,
                    kind='all')

            elif len(self.num_columns)==0 and len(self.cat_columns)>0:
                prep_pipe = self.base_pipe(
                    cat_columns=self.cat_columns,
                    kind='cat')

            elif len(self.num_columns)>0 and len(self.cat_columns)==0:
                prep_pipe = self.base_pipe(
                    num_columns=self.num_columns,
                    kind='num')

            self.prep_pipe = prep_pipe.fit(X_train[self.features],y_train)

            if self.task_type=='classification' or self.task_type=='multiclassification':

                if self.model_type=='xgboost':
                    self.model = XGBClassifier(**self.model_params)
                elif self.model_type=='catboost':
                    self.model = CatBoostClassifier(**self.model_params) 
                elif self.model_type=='lightboost':
                    self.model = LGBMClassifier(**self.model_params)
                elif self.model_type=='decisiontree':
                    self.model = DecisionTreeClassifier(**self.model_params)            
                elif self.model_type=='randomforest':
                    self.model = RandomForestClassifier(**self.model_params) 

            elif self.task_type=='regression':

                if self.model_type=='xgboost':
                    self.model = XGBRegressor(**self.model_params)
                elif self.model_type=='catboost':
                    self.model = CatBoostRegressor(**self.model_params)
                elif self.model_type=='lightboost':
                    self.model = LGBMRegressor(**self.model_params)
                elif self.model_type=='decisiontree':
                    self.model = DecisionTreeRegressor(**self.model_params)            
                elif self.model_type=='randomforest':
                    self.model = RandomForestRegressor(**self.model_params)

            self.model.fit(self.prep_pipe.transform(X_train[self.features]),y_train)

            return 'Пайплайн и модель обучены!'
    
    def predict(self, X: pd.DataFrame):
        
        if self.task_type=='classification':       
            predict = self.model.predict_proba(self.prep_pipe.transform(X))[:, 1]
        else:
            predict = self.model.predict(self.prep_pipe.transform(X))
            
        return predict
    
    def balance_cover(self, X: pd.DataFrame, y: pd.DataFrame, min_bin: int, n_obs: int):
        
        if self.task_type!='classification':
            print('Метрика только для классификации')
            
        else:
            
            res = pd.DataFrame()
            res['target'] = y
            res['proba'] = self.model.predict_proba(self.prep_pipe.transform(X))[:, 1]
            res = res.sort_values('proba', ascending=False).reset_index(drop=True)
            
            metr = BalanceCover(res, target='target')
            metr.calc_scores(min_bin, n_obs)
            metr.sample_describe()
            metr.plot_scores()
            
            return metr.output
        
    def calibration(self,X_calib: pd.DataFrame, y_calib: pd.DataFrame, n_bins: int=10):
    
        scores_base = self.model.predict_proba(self.prep_pipe.transform(X_calib))[:,1]
        
        self.model_sigm = CalibratedClassifierCV(
            base_estimator=self.model,
            cv='prefit',
            method='sigmoid')

        self.model_sigm.fit(self.prep_pipe.transform(X_calib),y_calib)
        scores_sigm = self.model_sigm.predict_proba(self.prep_pipe.transform(X_calib[self.features]))[:,1]

        self.model_iso = CalibratedClassifierCV(
            base_estimator=self.model,
            cv='prefit',
            method='isotonic')

        self.model_iso.fit(self.prep_pipe.transform(X_calib),y_calib)
        scores_iso = self.model_iso.predict_proba(self.prep_pipe.transform(X_calib[self.features]))[:,1]
        
        
        
        string = "{}: \n Оценка Брайера {:.3f} \n Logloss {:.3f} \n AUC {:.3F}"

        fig, axes = plt.subplots(1,3, figsize=(15,15))

        for name, s, ax in zip(['Без калибровки', 'Сигмоидная', 'Изотоническая'], 
                               [scores_base, scores_sigm, scores_iso], axes):
            plot_calibration_curve(y_true=y_calib,
                                  y_prob=s,
                                  n_bins=n_bins,
                                  ax=ax)

            ax.set_title(string.format(name,
                                      brier_score_loss(y_calib,s),
                                      log_loss(y_calib,s),
                                      roc_auc_score(y_calib,s)))
            

        plt.tight_layout()
        
    def calibration_compare(self, X_calib: pd.DataFrame, y_calib: pd.DataFrame, calib_compare:str='scores_iso'):
    
        calib_df = pd.DataFrame()
        calib_df['scores_base'] = self.model.predict_proba(self.prep_pipe.transform(X_calib))[:,1]
        calib_df['scores_sigm'] = self.model_sigm.predict_proba(self.prep_pipe.transform(X_calib[self.features]))[:,1]
        calib_df['scores_iso'] = self.model_iso.predict_proba(self.prep_pipe.transform(X_calib[self.features]))[:,1]
        calib_df['target'] = y_calib.reset_index(drop=True)

        calib_df = calib_df.sort_values('scores_base', ascending=False).reset_index(drop=True)
        
        plt.figure(figsize=(16,8))
        plt.plot(calib_df.index.tolist(), calib_df['scores_base'], label='scores_base')
        plt.plot(calib_df.index.tolist(), calib_df[calib_compare], label=calib_compare)
        plt.ylabel('Вероятность')
        plt.xlabel('Номер клиента (отсортированы по вероятности)')

        plt.grid()
        plt.legend()
        plt.show()
        
        return calib_df
    
    def get_pickles(self,
                    columns: bool=True,
                    prep_pipe: bool=True,
                    model: bool=True,
                    calibration: bool=False,
                    type_calibration: str='iso',
                    path: str='final_results'):
        
        
        create_folder(path)
        if columns:
            pickle.dump(self.num_columns, open('./{}/num_columns_final.sav'.format(path), 'wb'))
            pickle.dump(self.cat_columns, open('./{}/cat_columns_final.sav'.format(path), 'wb'))
        if prep_pipe:
            pickle.dump(self.prep_pipe, open('./{}/prep_pipe_final.sav'.format(path), 'wb'))
        if model:
            pickle.dump(self.model, open('./{}/model_final.sav'.format(path), 'wb'))
        if calibration:
            if type_calibration=='iso':
                pickle.dump(self.model_iso, open('./{}/calib_final.sav'.format(path), 'wb'))
            elif type_calibration=='sigm':
                pickle.dump(self.model_sigm, open('./{}/calib_final.sav'.format(path), 'wb'))
        
        return 'Все атрибуты сохранены!'