# Autobinary Framework version 1.0.11

Библиотека Autobinary является набором инструментов, которые позволяют автоматизировать процесс построения модели для решения определенных бизнес задач.

## Что позволяет:

  1. Провести первичный разведочный анализ и обработать факторы;
  2. Провести первичный отбор факторов из всех доступных;
  3. Провести первичное обучение по необходимой кросс - валидационной схеме;
  4. Провести поиск оптимального набора гиперапатмеров;
  5. Провести глубокий отбор факторов для финализации модели;
  6. Провести калибровку финальной модели при необходимости;
  7. Провести визуализацию оптимизационных и бизнес метрик;
  8. Провести интерпретационный анализ факторов.

## Как использовать:

#### Сценарий с установкой:
  1. Переместить установочный файл autobinary-1.0.10.tar.gz в необходимую папку
  2. Выполнить установку при помощи: !pip install autobinary-1.0.10.tar.gz
  3. Выполнить импорт библиотеки при помощи: import autobinary


#### Сценарий с ручной корректировкой библиотеки:
  1. Переместить папку autobinary на локальное пространство;
  2. Прописать путь к папке autobinary;
  3. Импортировать необходимые инструменты из библиотеки autobinary.

## Требования:

  * pandas >= 1.3.1
  * numpy >= 1.21.5 
  * catboost >= 0.25.1
  * matplotlib >= 3.1.0
  * sklearn >= 0.24.2 and <1.2.0
  * pdpbox == 0.2.0

## В папках репозитория приведены детальные примеры использования библиотеки:

  1. 01_Feature_vs_Target:

    * Примеры анализа целевой переменной относительно фактора для задач классификации;

    * Примеры анализа целевой переменной относительно фактора для задач регрессии.

  2. 02_CV_importances_for_trees:

    * Примеры обучения различных алгоритмов для решения задач классификации по кросс-валидационной схеме;
    
    * Примеры обучения различных алгоритмов для решения задач регрессии по кросс-валидационной схеме;
    
    * Примеры обучения различных алгоритмов для решения задач мультиклассификации по кросс-валидационной схеме;

    * Расчет важностей факторов после обучения алгоритма;

  3. 03_Tuning_parameters_Optuna:

    * Примеры поиска оптимального набора гиперпараметров с помощью библиотеки Optuna.
    
  4. 04_Explaining_output:
  
    * Примеры интерпретации влияния факторов на целевую переменную с помощью библиотеки Shap;
    
    * Примеры интерпретации влияния факторов на целевую переменную с помощью библиотеки PDPbox.
    
  5. 05_Uplift_models:
  
    * Примеры Solo model решения задач uplift с необходимой кросс - валидационной схемой;
    
    * Примеры Two models (Vanilla) решения задач uplift с необходимой кросс - валидационной схемой;
    
    * Примеры Two models (DDR control) решения задач uplift с необходимой кросс - валидационной схемой;
    
    * Примеры Two models (Treatment control) решения задач uplift с необходимой кросс - валидационной схемой.
    
  6. 06_Base_uplift_calibration:
  
    * Примеры калибровки для response задач;
    
    * Примеры калибровки для uplift задач;
    
    * Примеры калибровки для других видов задач;

  7. 07_Feature_selection:

    * Примеры первичного отбора факторов из всех доступных с помощью анализа пропусков, корреляционного анализа, анализа глубины деревьев, а также метода Permutation Importance (для бинарной классификации, регрессии и мультиклассовой классификации);
    
    * Примеры глубокого отбора факторов с помощью методов Forward и Backward selection;

    * Примеры отбора факторов с помощью метода Target Permutation.

  8. 08_Custom_metrics:

    * Примеры визуализации известных и кастомных метрик для детального понимания качества алгоритма в задачах бинарной классификации и uplift;

    * Примеры визуализации известных и кастомных метрик для детального понимания качества алгоритма в задачах регрессии.

  9. 09_Finalization_calibration:

    * Привер финализации и калибровки модели при существующей модели и с обучением при заданных параметрах для задачи бинарной классификации;
    
    * Привер финализации и калибровки модели при существующей модели и с обучением при заданных параметрах для задачи регрессии;

  10. 10_Full_Fitting_model:

    * Пример всего процесса построения и финализации модели в ноутбуке для задачи бинарной классификации (вероятность выживаемости при крушении титаника).


#### Авторы:
* Василий Сизов - https://github.com/Vasily-Sizov
* Дмитрий Тимохин - https://github.com/dmitrytimokhin
* Павел Зеленский - https://github.com/vselenskiy777
* Руслан Попов - https://github.com/RuslanPopov98
