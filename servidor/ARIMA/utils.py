import numpy as np

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def rd(n):
  return round(n,4)

def summary(model, series):
    Numberofobservations = len(series)
    aic = model.aic()
    aicc = model.aicc()
    bic = model.bic()
    hqic = model.hqic()
    ModelARIMA = model.to_dict()['order']

    AutoArimaResults1 = []
    row1=[]
    row1.append('Model ARIMA: '+ str(ModelARIMA))
    row1.append('Nr. Observations: '+str(Numberofobservations))
    row2=[]
    row2.append('AIC: '+ str(rd(aic)))
    row2.append('BIC: '+str(rd(bic)))
    row3=[]
    row3.append('AICc: '+ str(rd(aicc)))
    row3.append('HQIC: '+str(rd(hqic)))
    AutoArimaResults1.append(row1)
    AutoArimaResults1.append(row2)
    AutoArimaResults1.append(row3)

    numberpar = len(model.pvalues())
    numberapar = ModelARIMA[0]
    AutoArimaResults2= []
    for i in range(numberpar):
      row =[]
      if i == numberpar -1:
        row.append("Sigma2")

      elif numberpar==  ModelARIMA[0]+ ModelARIMA[2] +2:
        if i==0:
           row.append('Intercept')

        elif i <= numberapar:
           row.append('AR'+ str(i))
        else:
           row.append('MA'+ str(i -numberapar))
      else:
        if i < numberapar:
           row.append('AR'+ str(i+1))
        else:
           row.append('MA'+ str(i+1 -numberapar))

      row.append(rd(model.params()[i]))
      row.append(rd(model.bse()[i]))
      row.append(rd(model.pvalues()[i]))

      conf ='['+str(rd(model.conf_int()[i][0]))+', '+str(rd(model.conf_int()[i][1]))+']'
      row.append(conf)
      AutoArimaResults2.append(row)


    JB=sm.stats.jarque_bera(model.resid())
    JBtest = rd(JB[0])
    JBpvalue =  rd(JB[1])
    Skewness =  rd(JB[2])
    Kurtosis = rd(JB[3])

    AutoArimaResults3= []
    row1=[]
    row1.append('Mean of residuals: '+ str(rd(model.resid().mean())))
    row1.append(f'Skewness: {Skewness}')
    AutoArimaResults3.append(row1)
    row2=[]
    row2.append('SD of residuals: '+ str(rd(model.resid().std())))
    row2.append(f'Kurtosis: {Kurtosis}')
    AutoArimaResults3.append(row2)

    AutoArimaResults4= []
    Numberofobservations = len(series)
    for i in [1,10,20,2]:
        row = []
        if i ==1:
          row.append('Ljung-Box log(Nr.Obs) lags')
          lblog = acorr_ljungbox(model.resid(), lags=[int(np.log(Numberofobservations))], model_df=model.df_model(), return_df=False)
          row.append('Q-test: ' + str(rd(lblog[0][0])))
          row.append('p-value: ' + str(rd(lblog[1][0])))
        elif i ==2:
          row.append('Jarque-Bera')
          row.append('JB-test: '+ str(rd(JB[0])))
          row.append('p-value: '+ str(rd(JB[1])))
        else:
          row.append(f'Ljung-Box {i} lags')
          lb = acorr_ljungbox(model.resid(), lags=[i], model_df=model.df_model(), return_df=False)
          row.append('Q-test: ' + str(rd(lb[0][0])))
          row.append('p-value: ' + str(rd(lb[1][0])))
        AutoArimaResults4.append(row)
    return [AutoArimaResults1,AutoArimaResults2,AutoArimaResults3,AutoArimaResults4]







def get_graph():
    buffer =BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plotstock(series, stock):
    plt.figure(figsize=(10,6))
    plt.xticks(rotation=45)
    plt.plot(series)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    plt.title(f'Plot of {stock} stock')
    graph = get_graph()
    plt.clf()
    return graph


def get_plotdif(logseries, df, dff):
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(1, 3)
    fig.autofmt_xdate(rotation=45)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    axes[0].plot(logseries); axes[0].set_title('Log Original Series')
    axes[1].plot(df); axes[1].set_title('Log 1st Order Differencing')
    axes[2].plot(dff); axes[2].set_title('Log 2nd Order Differencing')
    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    fig.tight_layout()
    PlotDif = get_graph()
    plt.clf()
    return PlotDif

def get_plotacf(logseries, df, dff):
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(1, 3)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    plot_acf(logseries, ax=axes[0]); axes[0].set_title('ACF of Log Series')
    plot_acf(df, ax=axes[1]); axes[1].set_title('ACF of Log 1st DIFF')
    plot_acf(dff, ax=axes[2]); axes[2].set_title('ACF Log 2nd DIFF')
    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    fig.tight_layout()
    plotacf = get_graph()
    plt.clf()
    return plotacf

def get_plotpacf(logseries, df, dff):
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(1, 3)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    plot_pacf(logseries, ax=axes[0]); axes[0].set_title('PACF of Log Series')
    plot_pacf(df, ax=axes[1]); axes[1].set_title('PACF of Log 1st DIFF')
    plot_pacf(dff, ax=axes[2]); axes[2].set_title('PACF Log 2nd DIFF')
    figure = plt.gcf()
    figure.set_size_inches(12, 6)
    fig.tight_layout()
    plotpacf = get_graph()
    plt.clf()
    return plotpacf


def get_plotdiagnostics(model):
    plt.figure(figsize=(12, 6))
    model.plot_diagnostics(figsize=(10,10))
    plotdiagnostics = get_graph()
    plt.clf()
    return plotdiagnostics

def get_plotforecast(logseries, fc_series, lower_series, upper_series):
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.plot(logseries)
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                lower_series,
                upper_series,
                color='k', alpha=.15)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    plotforecast = get_graph()
    return plotforecast
