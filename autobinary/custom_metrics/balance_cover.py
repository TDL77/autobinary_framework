import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class BalanceCover:
    def __init__(self, entr_df, target='target', proba='proba', save=False, path='./', name=''):
        self.entr_df = entr_df
        self.target=target
        self.proba = proba
        self.save = save
        self.path = path
        self.name = name

    def sample_describe(self, n=2):
        df = self.entr_df.copy()
        print('Всего записей в выборке: ', df.shape[0])
        print('Всего таргетов в выборке: ', df[df[self.target] == 1].shape[0])
        print()
        print('Баланс классов: ', round(df[df[self.target] == 1].shape[0]*100/df.shape[0], n), ' %')
    
    def _calc_balance(self, counter):
    
        df = self.entr_df.copy()
        
        # базовый баланс классов
        bal1 = df[df[self.target] == 1].shape[0]/df.shape[0]
        # баланс классов в бакете
        bal2 = df[0:counter][df[self.target] == 1].shape[0]/counter
        # количество таргетов в бакете
        count_t = df[0:counter][df[self.target] == 1].shape[0]
        
        if df[0:counter][self.target].nunique() == 1:
            gini = 100
        else:
            gini = (roc_auc_score(df[0:counter][self.target], df[0:counter][self.proba]))*100

        return 100*bal1, 100*bal2, bal2/bal1, count_t, 100*df[0:counter][df[self.target] == 1].shape[0]/df[df[self.target] == 1].shape[0], gini

    def calc_scores(self, step, end):

        l_balans = []
        l_cover = []
        l_count = []
        l_base_bal = []
        l_bucket_bal = []
        l_turget_bucket = []
        l_gini = []

        for value in range(step, end+step, step):
            base_bal, bucket_bal, bal, turget_bucket, cov, gini = self._calc_balance(counter=value)
            l_balans.append(bal)
            l_cover.append(cov)
            l_count.append(value)
            l_base_bal.append(base_bal)
            l_bucket_bal.append(bucket_bal)
            l_turget_bucket.append(turget_bucket)
            l_gini.append(gini)
    
        df_output = pd.DataFrame()
        
        df_output['start_bucket'] = l_count
        df_output['start_bucket'] = df_output['start_bucket'] - df_output['start_bucket']
        df_output['end_bucket'] = l_count
        df_output['turget_in_bucket'] = l_turget_bucket
        df_output['bucket_bal (%)'] = l_bucket_bal
        df_output['coverage (%)'] = l_cover
        df_output['base_bal (%)'] = l_base_bal
        df_output['bucket_bal/base_bal'] = l_balans
        df_output['auc'] = l_gini
        
        self.output = df_output

    def _plot_scores_cov(self):
        plt.plot(self.output['end_bucket'], self.output['bucket_bal/base_bal'], label = 'Отношение балансов')
        plt.plot(self.output['end_bucket'], self.output['coverage (%)'], label = 'Покрытие')
        plt.grid()
        plt.legend()
        plt.title('Отношение балансов и покрытие')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('Процент(%)/выигрыш(раз)')
        plt.show()
        if self.save:
            plt.savefig('{}/{}_sc_cov.png'.format(self.path, self.name), dpi=75, bbox_inches='tight')
            plt.close()
        
    def _plot_scores_gini(self):
        plt.plot(self.output['end_bucket'], self.output['auc'], label = 'AUC')
        plt.grid()
        plt.legend()
        plt.title('AUC по срезам')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('AUC, %')
        plt.show()
        if self.save:
            plt.savefig('{}/{}_sc_gini.png'.format(self.path, self.name), dpi=75, bbox_inches='tight')
            plt.close()

    def plot_scores(self):
        self._plot_scores_cov()          
        self._plot_scores_gini()
            
    def _calc_balance_2(self, counter, step):
    
        df = self.entr_df.copy()
        
        # базовый баланс классов
        bal1 = df[df[self.target] == 1].shape[0]/df.shape[0]
        # баланс классов в бакете
        bal2 = df[counter:counter+step][df[self.target] == 1].shape[0]/(step)
        # количество таргетов в бакете
        count_t = df[counter:counter+step][df[self.target] == 1].shape[0]
        
        if df[counter:counter+step][self.target].nunique() == 1:
            gini = 100
        else:
            gini = (roc_auc_score(df[counter:counter+step][self.target], df[counter:counter+step][self.proba]))*100

        return 100*bal1, 100*bal2, bal2/bal1, count_t, 100*df[counter:counter+step][df[self.target] == 1].shape[0]/df[df[self.target] == 1].shape[0], gini
    
    def calc_scores_2(self, step, end):

        l_balans = []
        l_cover = []
        l_count = []
        l_base_bal = []
        l_bucket_bal = []
        l_turget_bucket = []
        l_gini = []

        for value in range(0, end, step):
            base_bal, bucket_bal, bal, turget_bucket, cov, gini= self._calc_balance_2(counter=value, step=step)
        
            l_balans.append(bal)
            l_cover.append(cov)
            l_count.append(value)
            l_base_bal.append(base_bal)
            l_bucket_bal.append(bucket_bal)
            l_turget_bucket.append(turget_bucket)
            l_gini.append(gini)
    
        df_output2 = pd.DataFrame()
        
        df_output2['start_bucket'] = l_count
        df_output2['end_bucket'] = df_output2['start_bucket']+step
        df_output2['target_in_bucket'] = l_turget_bucket
        df_output2['bucket_bal (%)'] = l_bucket_bal
        df_output2['coverage (%)'] = l_cover
        df_output2['base_bal (%)'] = l_base_bal
        df_output2['bucket_bal/base_bal'] = l_balans
        df_output2['auc'] = l_gini
        
        self.output2 = df_output2

    def _plot_scores_cov_2(self):
        plt.plot(self.output2['start_bucket'], self.output2['bucket_bal (%)'], label = '% таргетов')
        plt.grid()
        plt.legend()
        plt.title('Процент таргетов в бакете')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('Процент(%)')
        plt.show()
        if self.save:
            plt.savefig('{}/{}_sc_cov_2.png'.format(self.path,self.name), dpi=75, bbox_inches='tight')
            plt.close()
        
    def _plot_scores_gini_2(self):
        plt.plot(self.output2['end_bucket'], self.output2['auc'], label = 'AUC')
        plt.grid()
        plt.legend()
        plt.title('AUC по срезам')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('AUC, %')
        plt.show()
        if self.save:
            plt.savefig('{}/{}_sc_gini_2.png'.format(self.path,self.name), dpi=75, bbox_inches='tight')
            plt.close()

    def plot_scores_2(self):
        self._plot_scores_cov_2()
        self._plot_scores_gini_2()
