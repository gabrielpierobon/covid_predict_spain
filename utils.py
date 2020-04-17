from pytrends.request import TrendReq
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

def trending_topic_search(topic):
    '''
    Función que permite extraer una tendencia de los ultimos 3 meses de Google Trends usando la api pytrends
    '''
    
    data = pd.DataFrame()

    pytrend = TrendReq(retries=3)

    for i in [topic]:
        pytrend.build_payload(kw_list=[i], geo="ES", timeframe='today 3-m')
        df2 = pytrend.interest_over_time().transpose()
        data = data.append(df2[:-1])
        data.transpose().plot()
        
def trending_topic_search_long(topic):
    '''
    Función que permite extraer una tendencia de los ultimos 5 años de Google Trends usando la api pytrends
    '''  

    data = pd.DataFrame()

    pytrend = TrendReq(retries=3)

    for i in [topic]:
        pytrend.build_payload(kw_list=[i], geo="ES", timeframe='today 5-y')
        df2 = pytrend.interest_over_time().transpose()
        data = data.append(df2[:-1])
        data.transpose().plot()

# Draw Plot
def plot_sectores_raw(df,Rubro, xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(df[df['Rubro']==Rubro].index, df[df['Rubro']==Rubro].value, color='tab:red')
    plt.gca().set(title=Rubro, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def draw_seasonal_plot(df, Rubro, years):
    # Prep Colors
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)
    # Draw Plot
    plt.figure(figsize=(12,7), dpi= 80)
    for i, y in enumerate(years):   
        plt.plot('month', 'value', data=df.loc[(df.year==y) & (df.Rubro==Rubro), :], color=mycolors[i], label=y)
        plt.text(df.loc[(df.year==y) & (df.Rubro==Rubro), :].shape[0]-.9, 
                 df.loc[(df.year==y) & (df.Rubro==Rubro), 'value'][-1:].values[0], 
                 y, fontsize=12, color=mycolors[i])
    # Decoration
    # plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(("Seasonal Plot de {}").format(Rubro), fontsize=10)
    plt.show()

def draw_boxplots(df, Rubro):
    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x='year', y='value', data=df, ax=axes[0])
    sns.boxplot(x='month', y='value', data=df.loc[(~df.year.isin([1991, 2008])) & (df.Rubro==Rubro), :])

    # Set Title
    axes[0].set_title(('Year-wise Box Plot\n(The Trend)\n{}').format(Rubro), fontsize=18); 
    axes[1].set_title(('Month-wise Box Plot\n(The Seasonality)\n{}').format(Rubro), fontsize=18)
    plt.show()

def decomposition_plot(df, Rubro):
    # Multiplicative Decomposition 
    result_mul = seasonal_decompose(df[df.Rubro==Rubro]['value'], model='multiplicative', extrapolate_trend='freq')

    # Additive Decomposition
    result_add = seasonal_decompose(df[df.Rubro==Rubro]['value'], model='additive', extrapolate_trend='freq')
    
    df_mult = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
    df_mult.columns = ['seas', 'trend', 'resid', 'actual_values']
    
    df_add = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
    df_add.columns = ['seas', 'trend', 'resid', 'actual_values']

    # Plot
    plt.rcParams.update({'figure.figsize': (6,4)})
    result_mul.plot().suptitle(('Multiplicative Decompose: {}').format(Rubro), fontsize=7)
    result_add.plot().suptitle(('Additive Decompose: {}').format(Rubro), fontsize=7)
    plt.show()
    
    return df_mult, df_add


def extract_google_trends(dict_palabras, Rubro, extract):
    if extract==True:

        data = pd.DataFrame()

        pytrend = TrendReq(retries=3)
        
        #creación de la lista de palabras a partir del diccionario
        lista_palabras = []
        for i in list(dict_palabras[Rubro].keys()):
            lista_palabras = lista_palabras + dict_palabras[Rubro][i]    

        for i in lista_palabras:
            pytrend.build_payload(kw_list=[i], geo="ES", timeframe='today 5-y')
            df2 = pytrend.interest_over_time().transpose()
            data = data.append(df2[:-1])

        for i in lista_palabras:
            data[data.index==i].transpose().plot()
            
        data.to_csv(str('./data/trends_')+Rubro+str('.csv'), index=True)

    else:
        data = pd.read_csv(str('./data/trends_')+Rubro+str('.csv'),index_col=0)
        data.columns.name = 'date'
        
    return data

def prepare_trend_data(df, Rubro, dict_palabras, use_external, qtr_test = 2, delay = 0):
    if use_external == True:
        df = df.astype('int')
        data_trends_transposed = df.transpose()
        data_trends_transposed['date'] = data_trends_transposed.index
        data_trends_transposed['date'] = pd.to_datetime(data_trends_transposed['date'], format='%Y-%m-%d')
        data_trends_transposed['quarter'] = data_trends_transposed['date'].dt.quarter
        data_trends_transposed['year'] = data_trends_transposed['date'].dt.year
        data_trends_transposed = data_trends_transposed[data_trends_transposed['date']>='2016-01-01']
        data_trends_transposed = data_trends_transposed.drop(['date'], axis=1)
        data_trends_transposed = data_trends_transposed.groupby(['year','quarter']).mean().reset_index()
        data_trends_transposed = data_trends_transposed.drop(['quarter','year'], axis=1)
        
        for i in list(dict_palabras[Rubro].keys()):
            data_trends_transposed['externa_'+str(i)]=data_trends_transposed[list(dict_palabras[Rubro][i])].mean(axis=1)
            
        data_trends_transposed = data_trends_transposed[list(data_trends_transposed.filter(like='externa').columns)]
            
        for i in list(dict_palabras[Rubro].keys()):
            data_trends_transposed['externa_'+str(i)+str(delay)] = data_trends_transposed[['externa_'+str(i)]].shift(delay, axis = 0).fillna(data_trends_transposed[['externa_'+str(i)]].tail(1)['externa_'+str(i)].values[0])
            data_trends_transposed = data_trends_transposed.drop('externa_'+str(i),axis=1)

        print('Trends Plot')
        data_trends_transposed.plot()
        
        data_trends_train = data_trends_transposed[:-(qtr_test)]
        data_trends_test = data_trends_transposed.tail(qtr_test)

    else:
        data_trends_transposed = pd.read_csv('./data/PIB_adj.csv',index_col=0)
        data_trends_train = data_trends_transposed[:-(qtr_test)]
        data_trends_test = data_trends_transposed.tail(qtr_test)
    
    return data_trends_transposed,data_trends_train,data_trends_test

def predict_sarimax(df, Rubro, external_train, external_test, tipo, use_external, trend):
    if use_external==True:
        model = pm.auto_arima(df[df.Rubro==Rubro].value,
                              # start_p=1, start_q=1,
                              exogenous=external_train,
                              test='adf',
                              seasonal_test = 'ch',
                              max_p=6, max_q=6, max_d=3,
                              max_P=6, max_Q=6, max_D=3,
                              m=4,
                              alpha = 0.05,
                              trend = trend,
                              # start_P=0, 
                              seasonal=True,
                              d=None, D=None, trace=True,
                              error_action='ignore',  
                              suppress_warnings=True,
                              random_state=123,
                              random=True,
                              maxiter = 400,
                              n_fits=400,
                              stepwise=False)
        
        print(model.summary())

        # Forecast
        n_periods = 2
        fc, confint = model.predict(n_periods=n_periods, 
                                    exogenous=external_test, 
                                    return_conf_int=True)
    else:
        model = pm.auto_arima(df[df.Rubro==Rubro].value,
                              # start_p=1, start_q=1,
                              exogenous=external_train,
                              test='adf',
                              seasonal_test = 'ch',
                              max_p=6, max_q=6, max_d=3,
                              max_P=6, max_Q=6, max_D=3,
                              m=4,
                              alpha = 0.05,
                              trend = trend,
                              # start_P=0, 
                              seasonal=True,
                              d=None, D=None, trace=True,
                              error_action='ignore',  
                              suppress_warnings=True,
                              random_state=123,
                              random=True,
                              maxiter = 400,
                              n_fits=400,
                              stepwise=False)
        
        print(model.summary())

        # Forecast
        n_periods = 2
        fc, confint = model.predict(n_periods=n_periods, 
                                    exogenous=external_test, 
                                    return_conf_int=True)
    
    
    index_of_fc = np.arange(len(df[df.Rubro==Rubro].value), len(df[df.Rubro==Rubro].value)+n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    
    # Original Series
    original_series = df[df.Rubro==Rubro].value.reset_index().value
    
    prediction_df = pd.DataFrame(original_series).append(pd.DataFrame(fc_series, columns=['value'])).astype('int')
    prediction_df.columns = [Rubro+("_{}").format(tipo)]
    
    return prediction_df

def predict_rubro(df, Rubro, dict_palabras, use_external=True, delay=0, trend='ct', extract=True):
    #Obtener los datos de Google Trends
    print("Obteniendo los datos de Google Trends...")
    data_trends = extract_google_trends(dict_palabras, Rubro = Rubro, extract=extract)
    
    #Preparar los datos para entrenamiento y test
    print("Preparando los datos para entrenamiento y test...")
    data_prepared, data_train, data_test = prepare_trend_data(data_trends, Rubro, dict_palabras, use_external=use_external, qtr_test = 2, delay=delay)
    
    #Predecir con ajustes
    print("Prediciendo con ajustes...")
    predictions_ajustada = predict_sarimax(df, Rubro, data_train, data_test,'adj', use_external=use_external, trend=trend)
    
    #Predecir sin ajustes
    print("Prediciendo sin ajustes")
    predictions_original = predict_sarimax(df, Rubro, None, None, 'orig', use_external=use_external, trend=trend)
    
    #Unir las predicciones en un único dataframe
    print("Uniendo las predicciones...")
    predictions_comparar = predictions_original.join(predictions_ajustada)
    
    #Graficar
    predictions_comparar.plot()
    
    return predictions_comparar,data_prepared