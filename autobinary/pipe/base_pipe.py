from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .sent_columns import SentColumns
from category_encoders import CatBoostEncoder

def base_pipe(num_columns:list=None, cat_columns:list=None, fill_value:float=-1e24):

    if (num_columns is None or len(num_columns)==0) and (cat_columns is None or len(cat_columns)==0):
        print('Ни один из списков переменных не определены!')  

    elif num_columns is not None and (cat_columns is None or len(cat_columns)==0):
        print('Определены только количественные переменные!')
     
    elif (num_columns is None or len(num_columns)==0) and cat_columns is not None:
        print('Определены только категориальные переменные!')
        
    elif num_columns is not None and cat_columns is not None:
        print('Определены количественные и категориальные переменные!')

    # создаем конвейер для количественных переменных
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value = fill_value))
    ])

    # создаем конвейер для категориальных переменных
    cat_pipe = Pipeline([
        ('catenc', CatBoostEncoder(cols=cat_columns))
    ])

    transformers = [('num', num_pipe, num_columns),
                    ('cat', cat_pipe, cat_columns)]

    # передаем список трансформеров в ColumnTransformer
    transformer = ColumnTransformer(transformers=transformers)

    # задаем итоговый конвейер
    prep_pipe = Pipeline([
        ('transform', transformer),
        ('sent_columns', SentColumns(columns=num_columns+cat_columns))
    ])  

    return prep_pipe
