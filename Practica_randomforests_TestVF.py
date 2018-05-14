# -*- coding: utf-8 -*-

# 0. load data
import pandas as pd

#**********VALORES DE ENTRENAMIENTO**************: 
dengue= pd.read_csv('DengueTraining_Data_Features_Totales.csv',sep=';')
dengue.head()
#Características seleccionadas:
features= ('weekofyear','precipitation_amt_mm','reanalysis_air_temp_k', 'reanalysis_avg_temp_k','reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k', 'station_avg_temp_c','station_diur_temp_rng_c', 'station_max_temp_c','station_min_temp_c', 'station_precip_mm') 
X_train= dengue[['weekofyear','precipitation_amt_mm','reanalysis_air_temp_k', 'reanalysis_avg_temp_k','reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k', 'station_avg_temp_c','station_diur_temp_rng_c', 'station_max_temp_c','station_min_temp_c', 'station_precip_mm']]
y_train= dengue['total_cases']

#**********VALORES PRUEBA*************:
dengueTest= pd.read_csv('Dengue_Test_Data_FeaturesTotales.csv',sep=';')
dengueTest.head()
#Características Seleccionadas
features= ('weekofyear','precipitation_amt_mm','reanalysis_air_temp_k', 'reanalysis_avg_temp_k','reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k', 'station_avg_temp_c','station_diur_temp_rng_c', 'station_max_temp_c','station_min_temp_c', 'station_precip_mm') 
X_test = dengueTest[['weekofyear','precipitation_amt_mm','reanalysis_air_temp_k', 'reanalysis_avg_temp_k','reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k','reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_sat_precip_amt_mm', 'reanalysis_tdtr_k', 'station_avg_temp_c','station_diur_temp_rng_c', 'station_max_temp_c','station_min_temp_c', 'station_precip_mm']]


# Definición del Conjunto de entrenamiento para cálculo del fit y posterior cálculo de predicción
from sklearn.ensemble import RandomForestRegressor
# Parametrización del modelo. Se define el número de árboles (6) y máximo de profundidad o niveles de éstos (3). 
regressor = RandomForestRegressor(n_estimators=6, max_depth = 3, criterion='mae', random_state=0)
# Construcción del Modelo
regressor.fit(X_train, y_train)
# Prueba. Cálculo de predicción
y_pred = regressor.predict(X_test).astype(int)

#Para Generar Archivo CSV de predicción
submission = pd.read_csv('Dengue_Submission_Format.csv',index_col=[0, 1, 2])
#y_pred.size
#submission.total_cases.size
submission.total_cases=y_pred
#submission
submission.to_csv('C:/Users/Gustavo Balmaceda/Desktop/MASTER 2018/DSI/BLOQUE 2/Trabajo de Prediccion/Iquitos/dengue_final_test.csv')
print ('Verifica el archivo dengue_final_test.csv en el directorio definido')
# FEATURE RELEVANCIES
#print ('Feature Relevances')
pd.DataFrame({'Attributes': features,'Random Forests':regressor.feature_importances_})
    
    

