import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class UpliftCalibration:
    
    def __init__(self, df:pd.DataFrame, type_score: str='probability', type_calib: str='bins', 
                 strategy: str='all', woe: object=None, bins: int=5):
        
        self.df = df ## Датафрейм с таргетом, флагом коммуникации и скорром
        self.type_score = type_score ## Тип скорра для калибровки ('probability' / 'uplift')
        self.type_calib = type_calib ## Тип калибровки, через перцентиль или с помощью woe ('bins' / 'woe')
        self.strategy = strategy ## Вся выборка ('all') / С коммуникацией ('trt') / Без коммуникации ('crtl')
        self.woe = woe ## Объект обучения для WOE
        self.bins = bins ## Количество бинов для перцентиля
        
    def fit(self,target:str='target',treatment:str='treatment',score:str='proba',ascending:bool=False):
        
        ## target - название столбца таргета
        ## treatment - название столбца флага коммуникации
        ## score - название столбца со скорром
        ## ascending - направление калибровки
        
        if self.strategy == 'all':
            df1 = self.df
        elif self.strategy == 'trt':
            df1 = self.df[self.df[treatment] == 1]
        elif self.strategy == 'ctrl':
            df1 = self.df[self.df[treatment] == 0]

        # учимся на части выборки    
            
        if self.type_calib=='bins':
            df1 = df1.sort_values(score, ascending=False).reset_index(drop=True)

            percentiles1 = [round(p * 100 / self.bins) for p in range(1, self.bins + 1)]

            percentiles = [f"0-{percentiles1[0]}"] + \
                [f"{percentiles1[i]}-{percentiles1[i + 1]}" for i in range(len(percentiles1) - 1)]

            # возвращается кортеж:
            _, self.list_bounders = pd.qcut(
                df1[score],
                q=self.bins,
                precision=10,
                retbins=True)

            if self.type_score == 'uplift':
                self.list_bounders[0] = -100
                self.list_bounders[len(self.list_bounders)-1] = 100
            else:
                self.list_bounders[0] = 0
                self.list_bounders[len(self.list_bounders)-1] = 1            
        
        else:
            
            self.woe.fit(df1[[score]], df1[[target]])

            # преобразовываем всю изначальную выборку
            new_df1 = self.woe.transform(self.df[[score]])

            self.list_bounders = self.woe.optimal_edges.tolist()
            if self.type_score == 'uplift':
                self.list_bounders[0] = -100
                self.list_bounders[len(self.list_bounders)-1] = 100
            else:
                self.list_bounders[0] = 0
                self.list_bounders[len(self.list_bounders)-1] = 1  

            percentiles1 = [round(p * 100 / len(self.list_bounders)-1) for p in range(1, len(self.list_bounders))]

            percentiles = [f"0-{percentiles1[0]}"] + \
                [f"{percentiles1[i]}-{percentiles1[i + 1]}" for i in range(len(percentiles1)-1)]

        percentiles.reverse()


        df1['interval'] = pd.cut(df1[score], bins=self.list_bounders, precision=10)
        df1['name_interval'] = pd.cut(df1[score], bins=self.list_bounders, labels=percentiles)

        df1['left_b'] = df1['interval'].apply(lambda x: x.left)
        df1['right_b'] = df1['interval'].apply(lambda x: x.right)
        
        df1['interval'] = df1['interval'].astype(str)
        df_trt = df1[df1[treatment]==1]
        df_ctrl = df1[df1[treatment]==0]
        
        # Группировка и расчет количества наблюдений с коммуникацией/без коммуникации/в общем относительно бинов
        trt = df_trt.reset_index().\
        groupby(['interval', 'name_interval', 'left_b', 'right_b']).\
        agg({'index':np.size, target:np.sum, score: np.mean}).\
        rename(columns={'index':'n_trt',target:'tar1_trt',score:'mean_pred_trt'}).\
        reset_index().dropna()
        trt['tar0_trt'] = trt.n_trt-trt.tar1_trt

        ctrl = df_ctrl.reset_index().\
        groupby(['interval', 'name_interval', 'left_b', 'right_b']).\
        agg({'index':np.size, target:np.sum, score: np.mean}).\
        rename(columns={'index':'n_ctrl',target:'tar1_ctrl',score:'mean_pred_ctrl'}).\
        reset_index().dropna()
        ctrl['tar0_ctrl'] = ctrl.n_ctrl-ctrl.tar1_ctrl

        all_t = pd.DataFrame({'interval':'total',
                                'n_trt':len(df_trt),
                                'tar1_trt':df_trt[target].sum(),
                                'tar0_trt':len(df_trt)-df_trt[target].sum()}, index=[0]).\
        merge(pd.DataFrame({'interval':'total',
                                'n_ctrl':len(df_ctrl),
                                'tar1_ctrl':df_ctrl[target].sum(),
                                'tar0_ctrl':len(df_ctrl)-df_ctrl[target].sum()}, index=[0]),how='left', on='interval')

        final = trt.merge(ctrl.drop(columns=['left_b','right_b']), how='left', on=['interval','name_interval']).\
        append(all_t)[['interval', 'name_interval', 'left_b', 'right_b', 'n_trt', 'tar1_trt',
               'tar0_trt', 'n_ctrl', 'tar1_ctrl', 'tar0_ctrl']]
        
        
        # Расчитываем аплифт
        final['resp_rate_trt'] = final['tar1_trt']/final['n_trt']
        final['resp_rate_ctrl'] = final['tar1_ctrl']/final['n_ctrl']
        final['real_uplift'] = final['resp_rate_trt'] - final['resp_rate_ctrl']

        sort = final[final['interval'] != 'total'].sort_values(['left_b'], ascending=ascending).reset_index(drop=True)
        total = final[final['interval'] == 'total'].reset_index(drop=True)

        final = pd.concat([sort, total], axis=0).reset_index(drop=True)

        self.df = None
        self.final = final.to_dict()
        
        return final
        
    def apply(self, score: pd.DataFrame, precision: int=10):
        
        df = pd.DataFrame({'score': score, 'interval': pd.cut(score, bins=self.list_bounders, precision=precision).astype(str)})
        df = df.merge(
            pd.DataFrame(self.final)[['interval', 'name_interval', 'real_uplift']],
            on = 'interval',
            how='left'
        )
        
        self.applied = df.to_dict()  
        
        return df
        
    def plot_table(self, ascending: bool=False):
        
        df = pd.DataFrame(self.final)
        df = df[df['interval'] != 'total']
        df = df.sort_values(['interval'], ascending=ascending).reset_index(drop=True)
        df = df.reset_index()
        
        percentiles = df['name_interval']
        response_rate_trmnt = df['resp_rate_trt']
        response_rate_ctrl = df['resp_rate_ctrl']
        uplift_score = df['real_uplift']

        _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(
            percentiles, 
            response_rate_trmnt, 
            linewidth=2, 
            color='forestgreen', 
            label='treatment\nresponse rate')

        axes.errorbar(
            percentiles, 
            response_rate_ctrl,
            linewidth=2, 
            color='orange', 
            label='control\nresponse rate')

        axes.errorbar(
            percentiles, 
            uplift_score, 
            linewidth=2, 
            color='red', 
            label='uplift')

        axes.fill_between(percentiles, response_rate_trmnt,
                          response_rate_ctrl, alpha=0.1, color='red')

        axes.legend(loc='upper right')
        axes.set_title(
            f'Uplift by percentile')
        axes.set_xlabel('Percentile')
        axes.set_ylabel(
            'Uplift = treatment response rate - control response rate')
        axes.grid()