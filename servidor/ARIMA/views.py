### http packages ###

from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse
from django.http import HttpResponseRedirect
from django import forms



###  numerical packages   ###
from .utils import summary, get_graph, get_plotstock, get_plotdif, get_plotacf, get_plotpacf, get_plotdiagnostics, get_plotforecast
import numpy as np, pandas as pd
import yfinance as yf
import datetime
import pmdarima as pm
import re
import datetime
import matplotlib.pyplot as plt
###  Form to introduce stock and time information ###
class StockForm(forms.Form):
       newstock = forms.CharField(label="Stock")
       newstart = forms.DateField(initial='2016-10-1', label="Start Date")
       newend = forms.DateField(initial=datetime.date.today, label="End Date")

# Create your views here.
success= True
stock="test"
start=datetime.datetime(2016, 10, 1)
end=datetime.date.today()
plotstock="test"
plotdif="test"
plotacf="test"
plotpacf="test"
plotdiagnostics="test"
plotforecast="test"
AutoArimaResults1="test"
AutoArimaResults2="test"
AutoArimaResults3="test"
AutoArimaResults4="test"
AutoArimaResults="test"

def index(request):
    global success
    global stock
    global start
    global end
    global plotstock
    global plotdif
    global plotacf
    global plotpacf
    global plotdiagnostics
    global plotforecast
    global AutoArimaResults1
    global AutoArimaResults2
    global AutoArimaResults3
    global AutoArimaResults4
    global AutoArimaResults
    if request.method == "GET" and stock =="test":
        return render(request, "ARIMA/index.html",{
                "stock": stock,
                "form": StockForm()
        })

    elif request.method == "POST":

        # Take in the data the user submitted and save it as for
        form = StockForm(request.POST)

        # Check if form data is valid (server-side)
        if form.is_valid():

            # Isolate the stock from the 'cleaned' version of form data

             #check if date inputed is valid for stock dowload
                if form.cleaned_data['newstart'] < datetime.date(1970, 1, 2):
                    newstart=datetime.date(1970,1,2)
                else:
                    newstart=form.cleaned_data["newstart"]

                if form.cleaned_data["newend"] >datetime.date.today():
                    newend=datetime.date.today()
                else:
                    newend=form.cleaned_data["newend"]

                if newstart >= newend:
                     success= False
                     return render(request, "ARIMA/index.html", {
                                            "form": StockForm(),
                                            "datewrong": True})

		#get the stock in standard form
                newstock = re.sub(r'[\W_]+', '', form.cleaned_data["newstock"]).upper()

              #check if data exists
                test_download= yf.download(newstock, start=datetime.date(1985,1,1), end=datetime.date.today(), interval="5d")
                if len(test_download['Adj Close'])==0:
                     sucess=False
                     return render(request, "ARIMA/index.html", {
                                            "form": StockForm(),
                                            "stockwrong": True})



               #check if information submitted is equal to previous data submitted
                if stock!=newstock or  start!=newstart or end!=newend or success == False:
                     stock = form.cleaned_data["newstock"]
                     start =newstart
                     end =newend
                else:
                    return render(request, "ARIMA/index.html", {
                                            "stock": stock,
                                            "form": StockForm(),
                                            "plotstock": plotstock,
                                            "plotdif": plotdif,
                                            "plotacf": plotacf,
                                            "plotpacf": plotpacf,
                                            "plotdiagnostics": plotdiagnostics,
                                            "plotforecast": plotforecast,
                                            "AutoArimaResults1": AutoArimaResults[0],
                                            "AutoArimaResults2": AutoArimaResults[1],
                                            "AutoArimaResults3": AutoArimaResults[2],
                                            "AutoArimaResults4": AutoArimaResults[3]
                    })


              #dowload stock information
                df = yf.download(stock, start=start, end=end, interval="5d")

           #filter only the adjusted close data
                series=df['Adj Close']
           #check if there is enough data
                if len(series)<50:
                     success = False
                     return render(request, "ARIMA/index.html", {
                                            "form": StockForm(),
                                            "notenough": True})

          #check if there is non empyt data
                if len(series)==0:
                    success= False
                    return render(request, "ARIMA/index.html", {
                                            "form": StockForm(),
                                            "nodata": True})

           #chart the schoosen stock(todo)
                plotstock = get_plotstock(series, stock)


           #interpolate series due to missing values
                series=series.interpolate()

           #take the log of the series
                logseries=np.log(series)


           #take the first and second order differences
                df=logseries.diff()[1:len(logseries.diff())]
                dff=df.diff()[1:len(df.diff())]

           #plot the  log of the series and the first and second differences(to do)
                plotdif = get_plotdif(logseries, df, dff)

           #plot the acf of the log of the original series and first and second differences(to do)
                plotacf = get_plotacf(logseries, df, dff)

           #plot the pacf of the log of the original series and first and second differences(to do)
                plotpacf = get_plotpacf(logseries, df, dff)


           #find the appropriate Arima model for the series
                model = pm.auto_arima(logseries, trace= True, seasonal=False)

              #write the summary of fitting of ARIMA model
                summary_string = str(model.summary())
                print(summary_string)
              #plot diagnostics of how well the model fist the data (to do)
                plotdiagnostics = get_plotdiagnostics(model)

                #make forecast for n periods
                n_periods=24
                fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)

                #creating timeline of forecast from end date
                start_date = end
                number_of_days = n_periods
                date_list = [end + datetime.timedelta(weeks=x) for x in range(number_of_days)]

                # indexing the forecast for future dates
                fc_series = pd.Series(fc, index=date_list)
                lower_series = pd.Series(confint[:, 0] , index=date_list)
                upper_series = pd.Series(confint[:, 1] , index=date_list)

                #plot forecast
                plt.figure(figsize=(10, 6))
                plt.xticks(rotation=45)
                plt.plot(series)
                plt.plot(np.exp(fc_series), color='darkgreen')
                plt.fill_between(lower_series.index,
                                       np.exp(lower_series),
                                       np.exp(upper_series),
                                       color='k', alpha=.15)
                plt.title(f'Plot of final forecast of {stock} stock')
                figure = plt.gcf()
                figure.set_size_inches(10, 6)
                plotforecast = get_graph()

            #Auto-arima results to writen for tables
                AutoArimaResults=summary(model, series)
    success = True
    return render(request, "ARIMA/index.html", {
            "stock": stock,
            "form": StockForm(),
            "plotstock": plotstock,
            "plotdif": plotdif,
            "plotacf": plotacf,
            "plotpacf": plotpacf,
            "plotdiagnostics": plotdiagnostics,
            "plotforecast": plotforecast,
            "AutoArimaResults1": AutoArimaResults[0],
            "AutoArimaResults2": AutoArimaResults[1],
            "AutoArimaResults3": AutoArimaResults[2],
            "AutoArimaResults4": AutoArimaResults[3]
    })



