import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, skew
from sklearn.metrics import r2_score

def target_dist(df: pd.DataFrame, target: str, save=False, path='./', name=''):
    
    skewness = np.round(skew((df[target])),2)

    _, ax = plt.subplots(figsize=(16,8))
    plt.title('Распределение таргета')
    plt.text(x=0.9,y=0.9, ha='center', va='center', s='Ассиметрия = {}'.format(skewness), transform=ax.transAxes)
    sns.distplot((df[target]))
    plt.xlabel('Таргет')
    plt.grid(True)
    plt.show()
    if save:
        plt.savefig('{}/{}_plot_dist.png'.format(path, name), dpi=75, bbox_inches='tight')
        plt.close()

class RegressionMetrics:
    def __init__(self, entr_df, target='target', predict='prediction', save=False, path='./', name=''):
        self.entr_df = entr_df
        self.target = target
        self.predict = predict
        self.save = save
        self.path = path
        self.name = name

    def target_describe(self):
        df = self.entr_df.copy()
        min_t, max_t = '{0:,}'.format(min(df["target"])).replace(',', ' '), '{0:,}'.format(max(df["target"])).replace(',', ' ')
        
        print('Всего записей в выборке: ', df.shape[0])
        print('----------')
        print(f'Минимальное значение таргета: {min_t} \nМаксимальное значение таргета: {max_t}')
    
    def plot_prediction(self, size=(12,8)):
        
        target = self.entr_df[self.target]
        prediction = self.entr_df[self.predict]
        eps=0.1
        corr = round(spearmanr(target, prediction)[0],2)
#        r2 = round(r2_score(target, prediction),2)
        _, ax = plt.subplots(figsize=size)
        ax.scatter(prediction, target, alpha=0.3)
        props = {'xlim': (np.min(prediction) - eps, np.max(prediction) + eps),
                'ylim': (np.min(target) - eps, np.max(target) + eps),
                'title': 'Предсказание против Таргета',
                'xlabel': 'Предсказание',
                'ylabel': 'Таргет'}
        ax.set(**props)
        diag, = ax.plot(props['xlim'], props['ylim'], ls='--', c='blue')
#        plt.text(x=0.87,y=0.9,ha='center',va='center',s='R2 = {}'.format(r2), transform=ax.transAxes)
        plt.text(x=0.87,y=0.87,ha='center',va='center',s='Корреляция Спирмена = {}'.format(corr), transform=ax.transAxes)
        plt.show()
        if self.save:
            plt.savefig('{}/{}_plot_prediction.png'.format(self.path, self.name), dpi=75, bbox_inches='tight')
            plt.close()

    def m_bin_table(self, n_bins: int=5):
        
        df_output = self.entr_df.copy()
        
        df_output['mape'] = -100 * (df_output['target'] - df_output['prediction']) / df_output['target']
        df_output['smape'] = -200 * (df_output['target'] - df_output['prediction'])/(df_output['target'] + df_output['prediction'])        
        
        _, target_range = pd.qcut(df_output[self.target], q=n_bins, precision=1, retbins=True)
        len_r = len(target_range)
        target_range = [int(i) for i in target_range]
        target_range_l = ['{0:,}'.format(i).replace(',', ' ') for i in target_range]
        target_labels = [f"< {target_range_l[1]}"] + \
            [f"{target_range_l[i]} - {target_range_l[i + 1]}" for i in range(1,len(target_range_l)-2)] + \
            [f"> {target_range_l[-2]}"]

        min_t, max_t = '{0:,}'.format(min(df_output[self.target])).replace(',', ' '), '{0:,}'.format(max(df_output[self.target])).replace(',', ' ')
        
        bins_m = [-np.inf, -500, -100, -50, -10, 10, 50, 100, 500, np.inf]
        names_m = ['<-500', '-500 до -100', '-100 до -50', '-50 до -10', '-10 до 10', '10 до 50',  '50 до 100','100 до 500', '>+500']
        df_output['target_range'] = pd.cut(df_output['target'], bins=target_range, labels=target_labels)
        df_output['mape_range'] = pd.cut(df_output['mape'], bins=bins_m, labels=names_m)
        df_output['smape_range'] = pd.cut(df_output['smape'], bins=bins_m, labels=names_m)
        
        self.output = df_output
        return self.output
        
    def m_bin_plot(self, metric: str='mape', size=(12,8)):
        
        t_train = pd.pivot_table(self.output, values=metric, index='target_range', columns=f'{metric}_range', aggfunc='count')
        ax = t_train.plot.bar(stacked=True, figsize=(12,6))
        ax.legend(loc=1, bbox_to_anchor=(1.3, 1), title=f'{metric.upper()}, %')
        ax.set_ylabel('Количество клиентов')
        ax.set_xlabel('Бакеты поля (таргет)')
        ax.set_title(f'Распределение ошибки {metric.upper()}')
        plt.xticks(rotation=45)
        plt.show()
        if self.save:
            plt.savefig('{}/{}_plot_{}_bin.png'.format(self.path, self.name,metric.upper()), dpi=75, bbox_inches='tight')
            plt.close()
