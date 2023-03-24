import os
# Определение стандартных гиперпараметров моделей и список моделей

clf_params_xgb = {'eta':0.01,
              'n_estimators':500,
              'subsample':0.9,
              'max_depth':6,
              'objective':'binary:logistic',
              'n_jobs':-1,
              'random_state':42,
              'eval_metric':'logloss'}

clf_params_ctb = {'learning_rate':0.01,
              'iterations':500,
              'subsample':0.9,
              'depth':6,
              'loss_function':'Logloss',
              'thread_count':-1,
              'random_state':42,
              'verbose':0}

clf_params_lgb = {'learning_rate':0.01,
              'n_estimators':500,
              'subsample':0.9,
              'max_depth':6,
              'objective':'binary',
              'metric':'binary_logloss',
              'n_jobs':-1,
              'random_state':42,
              'verbose':-1}

clf_params_dt = {'criterion':'gini',
             'max_depth':6,
             'random_state':42}

clf_params_rf = {'criterion':'gini',
             'max_depth':6,
             'random_state':42,
             'n_estimators':500}

mclf_params_xgb = {
    'eta':0.01,
    'n_estimators':500,
    'subsample':0.9,
    'max_depth':6,
    'objective':'multi:softmax',
    'num_class':3,
    'n_jobs':-1,
    'random_state':42,
    'eval_metric':'auc'}

mclf_params_ctb = {
    'learning_rate':0.01,
    'iterations':500,
    'subsample':0.9,
    'depth':6,
    'loss_function':'Logloss',
    'objective':'MultiClass',
    'thread_count':-1,
    'random_state':42,
    'eval_metric':'MultiClass',
    'bootstrap_type':'MVS',
    'verbose':0}

mclf_params_lgb = {
    'learning_rate':0.01,
    'n_estimators':500,
    'subsample':0.9,
    'max_depth':6,
    'objective':'multiclass',
    'num_class': 3,
    'n_jobs':-1,
    'random_state':42,
    'metric':'auc_mu'}

mclf_params_dt = {
             'max_depth':6,
             'random_state':42}

mclf_params_rf = {
             'max_depth':6,
             'random_state':42,
             'n_estimators':500}

reg_params_xgb = {
    'eta':0.01,
    'n_estimators':500,
    'subsample':0.9,
    'max_depth':6,
    'objective':'reg:squarederror',
    'n_jobs':-1,
    'random_state':42,
    'eval_metric':'rmse'}

reg_params_ctb = {
    'learning_rate':0.01,
    'iterations':500,
    'subsample':0.9,
    'depth':6,
    'loss_function':'RMSE',
    'thread_count':-1,
    'random_state':42,
    'custom_metric':'RMSE',
    'verbose':0}

reg_params_lgb = {
    'learning_rate':0.01,
    'n_estimators':500,
    'subsample':0.9,
    'max_depth':6,
    'objective':'regression',
    'n_jobs':-1,
    'random_state':42,
    'metric':'rmse'}

reg_params_dt = {
             'max_depth':6,
             'random_state':42}

reg_params_rf = {
             'max_depth':6,
             'random_state':42,
             'n_estimators':500}