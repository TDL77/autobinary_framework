import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.utils import shuffle
from ..utils.default_parameters import *

from sklearn.inspection import permutation_importance
from eli5.permutation_importance import get_score_importances
from copy import deepcopy

from tqdm import tqdm, tqdm_notebook
from joblib import Parallel, delayed
from collections import ChainMap


class NansAnalysis():
    
    def __init__(self, df: pd.DataFrame=None):
        
        """
        Описание: Класс NansAnalysis предназначен для анализа пропусков.
        
            df - выборка для проверки.
        
        """
        
        self.df = df
        
    def to_types(self, num_columns: list = None):

        """
        Описание: Функция to_types предназначена для замены всех бесконечных значений на пропуски, а также приведения количественных переменных к типу float.

            num_columns - список количественных признаков.
        """
        
        self.df.replace([-np.inf, np.inf], np.nan, inplace=True)
        self.df.replace(['-infinity', 'infinity', '+infinity'], np.nan, inplace=True)
        self.df.replace(['-Infinity', 'Infinity', '+Infinity'], np.nan, inplace=True)

        if num_columns is not None:
            for i in num_columns:
                self.df[i] = self.df[i].astype(float)

        print('Все бесконечные значения заменены на пропуски, количественные переменные приведены к типу float!')
        
        return self.df

    def fit(self, percent: float=0.95, top: int=5):

        """
        Описание: Функция fit предназначена дла анализа выборки на количество пропусков.

            percent - процент при котором столбец исключается из дальнейшего анализа;
            top - топ признаков для табличного представления результата по количеству и доли пропусков в столбце.
        """    

        null = pd.DataFrame(self.df.isna().sum().reset_index()).rename(columns={'index':'feature',0:'cnt_null'})
        null['share_nans'] = (null.cnt_null/len(self.df)).round(4)
#        display(null.sort_values('cnt_null', ascending=False).head(top))
        
        self.nans_df = null.sort_values('cnt_null', ascending=False)

        feature_not_nan = null[null.cnt_null<=len(self.df)*percent].feature.tolist()

        print(f'Количество столбцов до: {self.df.shape[1]}')
        print('==================================================')
        print(f'Удалены столбцы, имеющие долю пропусков > {percent*100} %, количество оставшихся : {len(feature_not_nan)} ')

        return feature_not_nan    


class PrimarySelection():

    def __init__(self, df_train: pd.DataFrame=None, base_pipe: object=None, num_columns: list=None, cat_columns: list=None, target: str=None, 
                 model_type: str='xgboost', model_params: dict=None, task_type: str='classification', random_state: int=42):

        """
        Описание: Класс PrimarySelection предназначен для первичного отбора признаков. Включает в себя:
            1) corr_analysis - корреляционный анализ;
            2) depth_analysis - анализ относительно глубины;
            3) permutation_analysis - анализ случайного перемешивания фактора.
        
        df_train - обучающее множество;
        prep_pipe - конвейер предобработки признаков;
        num_columns / cat_columns - количественные и категориальные признаки соответственно;
        target - название переменной таргета;
        model_type - тип обучаемого алгоритма 'xgboost' / 'catboost' / 'lightboost' / 'decisiontree' / 'randomforest';
        model_params - параметры обучаемого алгоритма;
        task_type - тип задачи 'classification' / 'regression' / 'multiclassification';
        random_state - параметр воспроизводимости результата (контроль).
        """
        
        self.df_train = df_train
        self.base_pipe = base_pipe
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.target = target
        self.model_type = model_type
        self.task_type = task_type
        self.random_state = random_state
        
        if model_params is None:
            
            if self.task_type == 'classification':
            
                if model_type == 'xgboost':
                    self.model_params = clf_params_xgb
                elif model_type == 'catboost':
                    self.model_params = clf_params_ctb
                elif model_type == 'lightboost':
                    self.model_params = clf_params_lgb
                elif model_type == 'decisiontree':
                    self.model_params = clf_params_dt
                elif model_type == 'randomforest':
                    self.model_params = clf_params_rf
                    
            elif self.task_type == 'multiclassification':
            
                if model_type == 'xgboost':
                    self.model_params = mclf_params_xgb
                elif model_type == 'catboost':
                    self.model_params = mclf_params_ctb
                elif model_type == 'lightboost':
                    self.model_params = mclf_params_lgb
                elif model_type == 'decisiontree':
                    self.model_params = mclf_params_dt
                elif model_type == 'randomforest':
                    self.model_params = mclf_params_rf
                    
            elif self.task_type == 'regression':
            
                if model_type == 'xgboost':
                    self.model_params = reg_params_xgb
                elif model_type == 'catboost':
                    self.model_params = reg_params_ctb
                elif model_type == 'lightboost':
                    self.model_params = reg_params_lgb
                elif model_type == 'decisiontree':
                    self.model_params = reg_params_dt
                elif model_type == 'randomforest':
                    self.model_params = reg_params_rf
            
        else:
            self.model_params = model_params
        
        print('Класс первичного отбора факторов инициализирован!')
        
        self.random_feature = np.random.randn(len(self.df_train))
        
    def _preprocessing(self, X_tr: pd.DataFrame = None, y_tr: pd.DataFrame=None, X_te: pd.DataFrame = None,
                       num_columns: list=None, cat_columns: list=None):
        
        """
        Описание: Локальная функция _preprocessing предназначена для трансформации количественных и категориальных факторов.
            
            X_tr - обучающее множество;
            y_tr - целевая переменная обучающего множества;
            X_te - тестовое множество;
            num_columns / cat_columns - количественные и категориальные признаки соответственно.
        """
        
        X_tr.reset_index(drop=True,inplace=True)
        y_tr.reset_index(drop=True,inplace=True)
        
        prep_pipe = self.base_pipe(num_columns = num_columns,
                                   cat_columns = cat_columns)
        
        prep_pipe.fit(X_tr, y_tr)

        X_tr = prep_pipe.transform(X_tr)
        
        if X_te is None:
            return X_tr
        else:
            
            X_te.reset_index(drop=True,inplace=True)
            X_te = prep_pipe.transform(X_te)
            
            return X_tr, X_te
        
    def _m_gini(self, y_true,y_pred):

        n_class = y_true.nunique()
        ginis = []
        target_dummy = pd.get_dummies(y_true, drop_first=False).values

        for i in range(n_class):
            ginis.append((2*roc_auc_score(target_dummy[:, i], y_pred)-1)*100)

        return np.mean(ginis)

    def _get_f_metric(self, df, task_type):

        """
        Описание: Функция get_f_metric предназначена для расчета метрики по признаку.

            df - выборка.
        """       

        try:
            if task_type=='classification':
                
                return (2*roc_auc_score(df.iloc[:,0],df.iloc[:,1])-1)*100

            elif task_type=='multiclassification': 
                
                return self._m_gini(df.iloc[:,0],df.iloc[:,1])

            elif task_type=='regression':
                
                return r2_score(df.iloc[:,0],df.iloc[:,1])

        except:
            return 0 
        
    def _get_features_metric(self, df, task_type):

        """
        Описание: Функция get_features_metric предназначена для расчета метрики по всем признакам и представляет словарь 'Признак - значение метрики'.

            df - выборка.
        """       

        return {f: self._get_f_metric(df[[df.columns[0],f]], task_type) for f in list(df.columns[1:])}
    
    def _permute(self, col, model: object, X: pd.DataFrame, y: pd.DataFrame, n_iter: int, metric=None, higher_is_better: bool=True, task_type: str='classification', random_state: int=None):

        """
        Описание: Функция permute предназначена для перемешивания признака и пересчет скорра модели.

            model - объект модели;
            X - признаковое пространство;
            y - целевая переменная;
            n_iter - количество итераций для перемешиваний;
            metric - метрика, для перерасчета качества модели;
            higher_is_better - направленность метрики auc~True / mse~False;
            task_type - тип задачи 'classification' / 'regression' / 'multiclassification';
            random_state - параметр воспроизводимости результата (контроль).

        """

        d = {col: []}
        if task_type=='classification':       
            base_score = metric(y, model.predict_proba(X)[:, 1])
        else:
            base_score = metric(y, model.predict(X))

        for _ in range(n_iter):
            X[col] = shuffle(X[col].values, random_state=random_state)
            if task_type=='classification':
                temp_prediction = model.predict_proba(X)[:, 1]
            else:
                temp_prediction = model.predict(X)
            score = metric(y.values, temp_prediction)

            if higher_is_better:
                d[col].append(base_score-score)
            else:
                d[col].append(base_score+score)

        return d

    def _kib_permute(self, model: object, X: pd.DataFrame, y: pd.DataFrame,
                     metric=None, n_iter: int=5, n_jobs: int=-1, higher_is_better: bool=True, 
                     task_type: str='classification', random_state: int=None):

        """
        Описание: Функция kib_permute предназначена для формирования словаря 'Признак - Cреднее значение метрики после перемешивания'.

            model - объект модели;
            X - признаковое пространство;
            y - целевая переменная;
            metric - метрика, для перерасчета качества модели;
            n_iter - количество итераций для перемешиваний;
            n_jobs - количество ядер;
            higher_is_better - направленность метрики auc~True / mse~False;
            task_type - тип задачи 'classification' / 'regression';
            random_state - параметр воспроизводимости результата (контроль).

        """    


        result = Parallel(n_jobs=n_jobs,max_nbytes='50M')(delayed(self._permute)(col, model, X, y, 
                                                          n_iter, metric, higher_is_better, task_type, random_state) for col in tqdm(X.columns.tolist()))


        dict_imp = dict(ChainMap(*result))

        for i in dict_imp.keys(): dict_imp[i] = np.mean(dict_imp[i])

        return dict_imp

    def corr_analysis(self, method='spearman', threshold: float=0.8, drop_with_most_correlations: bool=True):
        
        """
        Описание: Функция corr_analysis предназначена для проведения корреляционного анализа и отборе факторов на основе корреляций и Джини значения.
        
            method - метод расчета корреляций (spearman / pearson);
            threshold - порог при котором фактор является коррелирующим;
            drop_with_most_correlations:
                True - исключается фактор с наибольшим количеством коррелирующих с ним факторов с корреляцией выше порогового значения;
                False - исключается фактор с наименьшим Джини из списка коррелирующих факторов.
            top - топ признаков для табличного представления результата.
        """
            
        df_train = deepcopy(self.df_train)

        features = self.num_columns + self.cat_columns

        df_train[features] = self._preprocessing(X_tr = df_train, y_tr = df_train[self.target], 
                                                 num_columns=self.num_columns, cat_columns=self.cat_columns)

        samples = {name: sample for name, sample in {'metric': df_train}.items() if not sample.empty} if df_train is not None else None

        metrics = {name: self._get_features_metric(df_train[[self.target] + [f for f in features if f in df_train.columns]], self.task_type) for name, df_train in samples.items()}

        self.metric_res = pd.DataFrame(metrics).round(4).abs().sort_values('metric', ascending=False)

        correlations = df_train[features].corr(method=method).abs()
        to_check_correlation=True
        features_to_drop = {}


        while to_check_correlation:
            to_check_correlation=False
            corr_number = {}
            significantly_correlated = {}

            for var in correlations:
                var_corr = correlations[var]
                var_corr = var_corr[(var_corr.index != var) & (var_corr > threshold)].sort_values(ascending=False).copy()
                corr_number[var] = var_corr.shape[0]
                significantly_correlated[var] = str(var_corr.index.tolist())

            if drop_with_most_correlations:
                with_correlation = {x: self.metric_res['metric'][x] for x in corr_number
                                    if corr_number[x] == max([corr_number[x] for x in corr_number])
                                    and corr_number[x] > 0}
            else:
                with_correlation = {x: self.metric_res['metric'][x] for x in corr_number if corr_number[x] > 0}

            if len(with_correlation)>0:
                feature_to_drop = min(with_correlation, key=with_correlation.get)
                features_to_drop[feature_to_drop] = significantly_correlated[feature_to_drop]
                correlations = correlations.drop(feature_to_drop, axis=1).drop(feature_to_drop, axis=0).copy()

                to_check_correlation = True

            self.feat_after_corr = list(set(features) - set(features_to_drop))

        print(f'Количество факторов до: {len(features)}')
        self.corr_df = pd.DataFrame(features_to_drop.values(), features_to_drop.keys()).rename(columns={0: f'Корреляция более {threshold*100}%'})
        print('==================================================')
        print(f'Количество факторов после корреляционного анализа: {len(self.feat_after_corr)}')

        return self.feat_after_corr

    def depth_analysis(self, features:list=None, max_depth:int=5, top: int=5):

        """
        Описание: Функция depth_analysis предназначена для анализа важности признаков относительно глубины алгоритма. Просиходит обучение алгоритма с изменением глубины дерева от 1 до заданного значения. На каждом значении глубины определяется значимость факторов, далее значение по каждому фактору усредняется. Итоговым набором факторов выступают те, среднее значение которых > 0.

            features - список факторов для расчета важностей с изменением глубины дерева;
            max_depth - максимальное значение глубины дерева;
            top - топ признаков для табличного представления результата.

        """
        df_train = deepcopy(self.df_train)
        
        max_depth_grid = list(range(1,max_depth+1))
        fi = list()
        
        num_columns = list(filter(lambda x: x in features, self.num_columns))
        cat_columns = list(filter(lambda x: x in features, self.cat_columns))
        
        X_train = self._preprocessing(X_tr = df_train, y_tr = df_train[self.target],
                                      num_columns=num_columns, cat_columns=cat_columns)
        
        rank_df = pd.DataFrame(X_train.columns,columns=['index']).set_index(['index'])

        for max_depth in tqdm_notebook(max_depth_grid):

            fi_feat = []
            new_params = self.model_params.copy()

            if self.model_type=='catboost':
                new_params['depth'] = max_depth
            else:
                new_params['max_depth'] = max_depth

            if self.task_type=='classification' or self.task_type=='multiclassification':

                if self.model_type=='xgboost':
                    model = XGBClassifier(**new_params)
                elif self.model_type=='catboost':
                    model = CatBoostClassifier(**new_params)
                elif self.model_type=='lightboost':
                    model = LGBMClassifier(**new_params)
                elif self.model_type=='decisiontree':
                    model = DecisionTreeClassifier(**new_params)
                elif self.model_type=='randomforest':
                    model = RandomForestClassifier(**new_params)

            elif self.task_type=='regression':

                if self.model_type=='xgboost':
                    model = XGBRegressor(**new_params)
                elif self.model_type=='catboost':
                    model = CatBoostRegressor(**new_params)
                elif self.model_type=='lightboost':
                    model = LGBMRegressor(**new_params)
                elif self.model_type=='decisiontree':
                    model = DecisionTreeRegressor(**new_params)
                elif self.model_type=='randomforest':
                    model = RandomForestRegressor(**new_params)

            model.fit(X_train, df_train[self.target])

            if self.model_type=='xgboost':
                xgbimp = list(model.get_booster().get_score(importance_type='gain').values())
                fi.append(xgbimp+[i*0 for i in range(len(X_train.columns)-len(xgbimp))])
                fi_feat.append(xgbimp+[i*0 for i in range(len(X_train.columns)-len(xgbimp))])

            elif self.model_type=='catboost':
                fi.append(model.get_feature_importance())
                fi_feat.append(model.get_feature_importance())

            elif self.model_type=='lightboost':
                fi.append(model.booster_.feature_importance(importance_type='gain'))
                fi_feat.append(model.booster_.feature_importance(importance_type='gain'))

            elif self.model_type=='decisiontree' or self.model_type=='randomforest':
                fi.append(model.feature_importances_)
                fi_feat.append(model.feature_importances_)

            rank = pd.DataFrame(np.array(fi_feat).T,
                              columns=['importance'],
                              index=X_train.columns).sort_values('importance', ascending=True)

            len_list = len(rank[rank.importance>0].index)
            rank[f'rank_depth_{max_depth}'] = [0 * i for i in range(len(rank)-len_list)]+[i/sum(range(1,len_list+1)) for i in range(1,len_list+1)]

            rank_df[f'rank_depth_{max_depth}'] = rank[f'rank_depth_{max_depth}']

        fi = pd.DataFrame(np.array(fi).T,
                  columns=['importance_depth_' + str(idx) for idx in range(1,len(fi)+1)],
                  index=X_train.columns)

        # вычисляем усредненные важности и добавляем столбец с ними
        fi['mean_importance'] = fi.mean(axis=1)
        rank_df['mean_rank'] = rank_df.mean(axis=1)

        fi['mean_rank'] = rank_df['mean_rank']
        self.deth_features_importance = fi[fi.mean_importance>0].index.tolist()
        self.deth_features_rank = fi[fi.mean_rank>0].index.tolist()
        self.fi = fi.sort_values('mean_importance',ascending=False)

        print(f'Количество признаков до отбора: {len(features)}')
        print('==================================================')
        print(f'Количество признаков после mean importance относительно глубины: {len(self.deth_features_importance)}')
        print(f'Количество признаков после mean rank относительно глубины: {len(self.deth_features_rank)}')
        
#        print(self.fi.head(top))

        return self.deth_features_importance, self.deth_features_rank
    
    def permutation_analysis(self, features: list=None, strat: object=None, group: str=None,
                             n_iter: int=5, permute_type: str='sklearn', 
                             n_jobs: int=-1, metric=None, higher_is_better: bool=True):

        """
        Описание: функция permutation_analysis предназначена для расчета permutation importance.

            features - список факторов для расчета важностей с изменением глубины дерева;
            n_iter - количество итераций для перемешиваний;
            permute_type - используемая библиотека для расчета permutation importance 'sklearn' / 'eli5' / 'kib';
            n_jobs - количество ядер (используется только для permutation importance от 'sklearn' и 'kib');
            metric - метрика, для перерасчета качества модели (используется только для permutation importance от 'kib');
            higher_is_better - направленность метрики auc~True / mse~False (используется только для permutation importance от 'kib').
        
        Последовательность действий выполняемых алгоритмом:

            1) Происходит обучение алгоритма;
            2) Происходит расчет метрики;
            3) Происходит перемешивание одного из факторов, остальные остаются неизменными;
            4) Происходит пересчет метрики с одним из перемешанных факторов;
            5) Происходит расчет разницы метрики 2) и метрики 4);
            6) Происходит повтор 5) пункта n_iter раз;
            7) Происходит усреднение пунка 6)
            8) Происходит отбор признаков либо по факторам выше значения random_feature на тесте, либо permutation importance значение на тесте > 0.

        """
        
        df_train = deepcopy(self.df_train)
        
        self.n_iter_permute = n_iter
        self.permute_type = permute_type
        
        df_train.reset_index(drop=True,inplace=True)
        if group:
            groups = df_train[group]
            folds_perm = list(strat.split(df_train, df_train[self.target], groups=groups))
        else:
            folds_perm = list(strat.split(df_train, df_train[self.target]))

        df_train_perm = df_train.iloc[folds_perm[1][0]].reset_index(drop=True)
        print(f'Размер обучающего подмножества для Permutation importance: {df_train_perm.shape} ; Среднее значение таргета: {df_train_perm[self.target].mean()}')
        print()
        df_test_perm = df_train.iloc[folds_perm[1][1]].reset_index(drop=True)
        print(f'Размер тестового подмножества для Permutation importance: {df_test_perm.shape} ; Среднее значение таргета: {df_test_perm[self.target].mean()}')  
        print('==================================================')

        if self.task_type=='classification' or self.task_type=='multiclassification':

            if self.model_type=='xgboost':
                model = XGBClassifier(**self.model_params)
            elif self.model_type=='catboost':
                model = CatBoostClassifier(**self.model_params)
            elif self.model_type=='lightboost':
                model = LGBMClassifier(**self.model_params)
            elif self.model_type=='decisiontree':
                model = DecisionTreeClassifier(**self.model_params)
            elif self.model_type=='randomforest':
                model = RandomForestClassifier(**self.model_params)

        elif self.task_type=='regression':

            if self.model_type=='xgboost':
                model = XGBRegressor(**self.model_params)
            elif self.model_type=='catboost':
                model = CatBoostRegressor(**self.model_params)
            elif self.model_type=='lightboost':
                model = LGBMRegressor(**self.model_params)
            elif self.model_type=='decisiontree':
                model = DecisionTreeRegressor(**self.model_params)
            elif self.model_type=='randomforest':
                model = RandomForestRegressor(**self.model_params)

        num_columns = list(filter(lambda x: x in features, self.num_columns)) 
        cat_columns = list(filter(lambda x: x in features, self.cat_columns))
        
        X_train, X_test = self._preprocessing(X_tr = df_train_perm, y_tr = df_train_perm[self.target],
                                              X_te = df_test_perm,
                                              num_columns=num_columns, cat_columns=cat_columns)
        
        X_train['random_feature'] = self.random_feature[folds_perm[1][0]]
        X_test['random_feature'] = self.random_feature[folds_perm[1][1]]
        
        
        model.fit(X_train[features+['random_feature']], df_train_perm[self.target])
        self.permute_feature_names = X_train.columns.tolist()
        
        # Обучение Permutation importance из разных библиотек
        if permute_type=='sklearn':
            result_tr = permutation_importance(model, X_train, df_train_perm[self.target], n_repeats=n_iter, random_state=self.random_state, n_jobs=n_jobs)
            result_te = permutation_importance(model, X_test, df_test_perm[self.target], n_repeats=n_iter, random_state=self.random_state, n_jobs=n_jobs)

            # Создание важности и словаря факторов
            sorted_idx = result_tr.importances_mean.argsort()
            feature_names = np.array(self.permute_feature_names)[sorted_idx]

            data_tr = {'Feature':feature_names,
                       'Perm_Importance_Tr':result_tr.importances_mean[sorted_idx]}
            data_te = {'Feature':feature_names,
                       'Perm_Importance_Te':result_te.importances_mean[sorted_idx]}
                
        elif permute_type=='eli5':
            _, result_tr = get_score_importances(model.score, X_train.values, df_train_perm[self.target], n_iter=n_iter, random_state=self.random_state)
            _, result_te = get_score_importances(model.score, X_test.values, df_test_perm[self.target], n_iter=n_iter, random_state=self.random_state)

            # Создание важности и словаря факторов
            data_tr = {'Feature':self.permute_feature_names,
                       'Perm_Importance_Tr':np.mean(result_tr, axis=0)}
            data_te = {'Feature':self.permute_feature_names,
                       'Perm_Importance_Te':np.mean(result_te, axis=0)}

        elif permute_type=='kib':
            print('Расчет Permutation Importance на Train')
            result_tr = self._kib_permute(model, X_train, df_train_perm[self.target], metric=metric, n_iter=n_iter, n_jobs=n_jobs, higher_is_better=higher_is_better, task_type=self.task_type, random_state=self.random_state)
            print('Расчет Permutation Importance на Test')
            result_te = self._kib_permute(model, X_test, df_test_perm[self.target], metric=metric, n_iter=n_iter, n_jobs=n_jobs, higher_is_better=higher_is_better, task_type=self.task_type, random_state=self.random_state)
            
            # Создание важности и словаря факторов
            data_tr = {'Feature':result_tr.keys(),
                       'Perm_Importance_Tr':result_tr.values()}
            data_te = {'Feature':result_te.keys(),
                       'Perm_Importance_Te':result_te.values()}

        # Создание датасета и сортировка PI на тесте по убыванию
        self.pi_df = (pd.DataFrame(data_tr).merge(pd.DataFrame(data_te),how='left',on='Feature')).set_index('Feature').sort_values(by=['Perm_Importance_Te'], ascending=False)
        
        return self.pi_df

    def permutation_plot(self, top: int=None, figsize=(10,6)):

        """
        Описание: функция permutation_plot предназначена для отрисовки бар плота по признакам на тестовом признаковом пространстве.

            top - количество признаков для отрисовки бар плота. Если не указано значение, будут отрисованы все признаки, участвующие при обучении алгоритма.

        """
        if top is None:
            x = self.pi_df['Perm_Importance_Te']
            y = y=self.pi_df.index
        else:
            x = self.pi_df['Perm_Importance_Te'][:top]
            y = y=self.pi_df.index[:top]

        # Параметры для рисунка
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y=y,color='dodgerblue')
        plt.title(self.permute_type + ' Feature Importance on Test')
        plt.xlabel('Permutation Importance')
        plt.ylabel('Feature Names')

    def select_features(self):

        """
        Описание: функция select_features предназначена для отбора признаков по результатам Permutation Importance.

            Отбор происходит по значению permutation importance > random_feature значения на тестовом множестве / значение permutation importance >= 0 на обучающем множестве / значение permutation importance >=0 на тестовом множестве.

        """

        random_score = self.pi_df.loc['random_feature'].Perm_Importance_Te

        if random_score>0:
            self.selected_features = self.pi_df[self.pi_df.Perm_Importance_Te>random_score].index.tolist()
        elif random_score<=0:
            self.selected_features = self.pi_df[self.pi_df.Perm_Importance_Te>=0].index.tolist()

        print(len(self.permute_feature_names), 'признаков было до Permutation Importance', '\n')
        print(len(self.selected_features), 'признаков после Permutation Importance от', self.permute_type)

        return self.selected_features