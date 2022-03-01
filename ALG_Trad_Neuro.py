#utils file which contains all the auxiliary functions
#for the views.py file of the NEURAL app
#in the django framework

##################  Check the main function at the bottom for extra customization ##################


##################################################################
########################  Get the Data  ##########################
##################################################################

#Import Data Related Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf


#function that creates dataframe with chosen stock
#and chosen features to be used for predictive analysis
def get_data(stock,market=False, lags=0,
            rolling_mean=False,
            volatility=False,
            rolling_market=False,
            distance=False):

    #start and end dates of which financial information will be used
    start = datetime.date(2012,1,1)
    end =datetime.date.today()
    # datetime.date(2018,1,1)
    #datetime.date.today()
    Date = pd.date_range(start,end)

    #dowload all the stock data from the top ten stock in the sp500
    symbols=['AAPL', 'MSFT', 'AMZN','GOOGL','NVDA', 'JPM','TSLA', 'FB', 'GLD']

    #create new empty data frame for reoganized stock data
    Adjclose_prices = pd.DataFrame(index=Date)

    #dowload stock historical data
    data = yf.download(stock, start=start, end=end,interval="1d")
    df_tmp = pd.DataFrame(data=data['Adj Close'].to_numpy(), index=data.index, columns=[stock])
    Adjclose_prices = Adjclose_prices.join(df_tmp, how="outer")

    for symbol in symbols:
        #dowload the stock data in the top 10 of the sp500 index   
        data = yf.download(symbol, start=start, end=end,interval="1d")
        
        #check if the data is not empy
        if not data['Adj Close'].isnull().all():

            df_tmp = pd.DataFrame(data=data['Adj Close'].to_numpy(), index=data.index, columns=[symbol+'0'])
            Adjclose_prices = Adjclose_prices.join(df_tmp, how="outer")  #left join by default

    #drop rows with all nan
    #most likely correspond to weekends, holidays (non-trading days)
    Adjclose_prices.dropna(axis=0, how='all', inplace=True)

    #fill missing data for stocks without initial value
    Adjclose_prices.fillna(method='ffill', inplace=True)

    #fill missing data for stocks without forward value
    Adjclose_prices.fillna(method='bfill', inplace=True)


    #create empty dataframe for the returns of the stocks
    df_returns = pd.DataFrame()

    #create empty dataframe for the returns of the stocks
    df_returns = pd.DataFrame()

    df_returns[stock] = np.log(Adjclose_prices[stock]).diff()

    #add the returns of each stock in the adjusted price dataframe
    if market:
        for name in Adjclose_prices.columns:
            if name !=stock:
                df_returns[name] = np.log(Adjclose_prices[name]).diff()


    #if choosen, addd the rolling means 5 and 20 of all the stocks
    #of the top 10 stocks in the sp500
    if rolling_market:
        for name in Adjclose_prices.columns:
            #only add the rolling means of the top 10 sp500 stocks
            if name !=stock:
                df_returns[f"{name}_mean5"] = np.log(Adjclose_prices[name]).diff().rolling(5).mean()
                df_returns[f"{name}_mean20"] = np.log(Adjclose_prices[name]).diff().rolling(20).mean()


    #add the chosen number of lags of the stock if lag chosen
    #is different than zero
    if lags != 0:
        
        for lag in range(0, lags):
            col = f'lag_{lag+1}'
            df_returns[col] = df_returns[stock].shift(lag)
            symbols.append(col)

    #add the rolling mean of 5 and 20 of the stock if choosen
    if rolling_mean :

        df_returns['momentum5'] = df_returns[stock].rolling(5).mean()
        symbols.append('momentum5')

        df_returns['momentum20'] = df_returns[stock].rolling(20).mean()
        symbols.append('momentum20')

    #add the volatility of the stock if choosen
    if volatility:

        df_returns['volatility'] = df_returns[stock].rolling(20).std()

        symbols.append( 'volatility')

    #add the distance of the last 20th price if choosen
    if distance:
        df_returns['distance'] = (Adjclose_prices[stock] - Adjclose_prices[stock].rolling(50).mean())

    #shift the colum of the s&p500 index by one time-step
    df_returns[stock] = df_returns[stock].shift(-1)    


    #remove any missing values
    df_returns.dropna( inplace=True)


    return df_returns



###################################################################
################  Get the Train and Test Datasets  ################
###################################################################

#function that normalizes and then divides the data into train and test data
def get_train_test(data,stock,Ntest=750):

    #defining the cutoff date for training and test data
    cutoff = datetime.date.today() - datetime.timedelta(days = Ntest)
    cutoff='{dt.year}-{dt.month}-{dt.day}'.format(dt = cutoff)
    
    #defining the train and test set data
    training_data = data[data.index < cutoff].copy()
    test_data = data[data.index >= cutoff].copy()

    #normalization of feature data by gaussian normalization
    mu, std = training_data.mean(), training_data.std()
    training_data_ = (training_data - mu) / std
    test_data_ = (test_data - mu) / std

    #isolate the features and signals of the training data
    Xtrain=training_data.drop(stock, 1)
    Ytrain= (training_data[stock]>0)

    #isolate the features and signals of the test data
    Xtest=test_data.drop(stock, 1)
    Ytest= (test_data[stock]>0)

    #get the returns of the stock in the train data
    return_train=training_data[stock]

    #get the returns of the stock in the test data
    return_test=test_data[stock]

    return (Xtrain, Ytrain,
            Xtest,Ytest,
            return_train,return_test)


###################################################################
#######################   Get the Model   #########################
###################################################################


#Import the Tensor Flow and Keras packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.optimizers import Adam, RMSprop
import random

#Function to fix the Random Seed
def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)

#Function that renders the neural network into memory with two layers
#whose sizes is given by the user's choice
def get_model(col_size,first_layer=32,second_layer=32):
    
    #setting the learning rate for the training process
    optimizer = Adam(learning_rate=0.0001)

    #install a sequential model
    model = Sequential()
    #define the first and hidden layers and the output layer
    model.add(Dense(first_layer, activation='relu',
    input_shape=(col_size,)))
    model.add(Dense(second_layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #compiles the Sequential model object for classification
    model.compile(  optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model


###################################################################
######################   Train the Model   ########################
###################################################################

#Train the Model with the Train Dataset
def train_model(model,Xtrain,Ytrain):

    
    #fit the data with the neural network model
    model.fit(Xtrain,
        Ytrain,
        epochs=100, verbose=False,
        validation_split=0.2, shuffle=False)

    return model

###################################################################
###################   Evaluation of the Model   ###################
###################################################################

#Function to evaluate the Model
def evaluate_model(model,stock,
                        Xtrain,
                        Ytrain,
                        Xtest,
                        Ytest,
                        return_train,
                        return_test,
                        risk_free=0.05):

    
    #####  Evaluation of the Model by its Accuracy in the Validation Sets  #####
    
    
    #progress of the accuracy of the model in the train set and validation sets
    res = pd.DataFrame(model.history.history)

    res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')

    plt.savefig("validation.png")
    plt.close()

    
    ############# Evaluation of the model in the train data  #############

    print("""
######################################################################
###########  Evaluation of the Model in the Train Dataset  ###########
######################################################################
""")


    #make prediction in the test set
    pred = np.where(model.predict(Xtrain) > 0.5, 1, 0)

    #insert the predictions of whether the model predicts 
    # the price of the following day is going to rise (i.e create long positions) 
    Xtrain['prediction'] = np.where(pred > 0, 1, 0)


    #convert returns of the stock into a pandas dataframe
    return_train=return_train.to_frame()
  
    #compute the returns on the algorithmic strategy
    return_train['strategy'] = (Xtrain['prediction'] *
    return_train[stock])



    #Create graph with the cummulative returns of both strategies
    return_train[[stock, 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig("returns_train.png")
    plt.close()

    #Compute Accuracy, Mean, Volatility and Sharpe Ratio

    #compute the total return of strategy and buy and hold
    total_train_alg=return_train['strategy'].sum()
    total_train_hold=return_train[stock].sum()


    print(f"""Total Log Return of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(total_train_alg*100,3)}%
    Buy-and-Hold: {round(total_train_hold*100,3)}%
    """)

    #Accuracy of the Model in the Train Dataset
    accuracy_train_alg=(Xtrain['prediction']==Ytrain).mean()
    accuracy_train_hold=(Ytrain>0).mean()

    print(f"""Accuracy of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(accuracy_train_alg*100,3)}%
    Buy-and-Hold: {round(accuracy_train_hold*100,3)}%
    """)


    #mean returns
    mean_train_alg=return_train['strategy'].mean()
    mean_train_hold=return_train[stock].mean()
    
    print(f"""Mean Daily Log Return of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(mean_train_alg*100,3)}%
    Buy-and-Hold: {round(mean_train_hold*100,3)}%
    """)

    #mean standard deviations
    std_train_alg=return_train['strategy'].std()
    std_train_hold=return_train[stock].std()
    
    print(f"""Standard Deviation of Daily Log Returns of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(std_train_alg*100,3)}%
    Buy-and-Hold: {round(std_train_hold*100,3)}%
    """)

    #Sharpe Ratio
    SR_train_alg=(mean_train_alg-risk_free/256)/std_train_alg
    SR_train_hold= (mean_train_hold-risk_free/256)/std_train_hold
    
    print(f"""Sharpe Ratio of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(SR_train_alg,3)}
    Buy-and-Hold: {round(SR_train_hold,3)}
    """)


    ####################################################
    #### Evaluation of the model in the test data  ####
    ####################################################

    print("""######################################################################
###########  Evaluation of the Model on the Test Dataset  ############
######################################################################
""")


    #make prediction in the test set
    pred = np.where(model.predict(Xtest) > 0.5, 1, 0)

    #insert the predictions of whether the model predicts 
    # the price of the following day is going to rise (i.e create long positions) 
    Xtest['prediction'] = np.where(pred > 0, 1, 0)



    #convert returns of the stock into a pandas dataframe
    return_test=return_test.to_frame()
  
    #compute the returns on the algorithmic strategy
    return_test['strategy'] = (Xtest['prediction'] *
    return_test[stock])


    #Create graph with the cummulative returns of both strategies
    return_test[[stock, 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig("returns_test.png")
    plt.close()


    #compute mean, volatility and Sharpe ratio

    #compute the total return of strategy and buy and hold
    total_test_alg=return_test['strategy'].sum()
    total_test_hold=return_test[stock].sum()


    print(f"""Total Log Return of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(total_test_alg*100,3)}%
    Buy-and-Hold: {round(total_test_hold*100,3)}%
    """)

    #Accuracy of the Model in the Train Dataset
    accuracy_test_alg=(Xtest['prediction']==Ytest).mean()
    accuracy_test_hold=(Ytest>0).mean()

    print(f"""Accuracy of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(accuracy_test_alg*100,3)}%
    Buy-and-Hold: {round(accuracy_test_hold*100,3)}%
    """)


    #mean returns
    mean_test_alg=return_test['strategy'].mean()
    mean_test_hold=return_test[stock].mean()
    
    print(f"""Mean Daily Log Return of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(mean_test_alg*100,3)}%
    Buy-and-Hold: {round(mean_test_hold*100,3)}%
    """)

    #mean standard deviations
    std_test_alg=return_test['strategy'].std()
    std_test_hold=return_test[stock].std()
    
    print(f"""Standard Deviation of Daily Log Returns of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(std_test_alg*100,3)}%
    Buy-and-Hold: {round(std_test_hold*100,3)}%
    """)

    #Sharpe Ratio
    SR_test_alg=(mean_test_alg-risk_free/256)/std_test_alg
    SR_test_hold= (mean_test_hold-risk_free/256)/std_test_hold
    
    print(f"""Sharpe Ratio of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(SR_test_alg,3)}
    Buy-and-Hold: {round(SR_test_hold,3)}
    """)    


###################################################################
#######################    Main Program    ########################
###################################################################

#Funtion to run the main bulk of the Program
def main():
    
    ### Ask the user which data and features to apply the model (optional) ###
    
    #stock=input("Choose Stock tick: ")
    #lags=int(input("Number of Lags?: "))
    #volatility=bool(input("Volatility? (empty for no): "))
    #rolling_mean=bool(input("Rolling Mean? (empty for no): "))
    #market=bool(input('Market (empty for no)?: '))
    #rolling_market=bool(input("Rolling Market? (empty for no): "))
    #distance=bool(input("Distance? (empty for no): "))
    #first_layer=int(input("Size of the first layer?: "))
    #second_layer=int(input("Size of the second layer?: "))
    #Ntest=int(input("Size of test data?: "))
    #risk_free=float(input("Risk-Free Rate?: "))

    ################## Defaut Settings ##################
    stock= "SPY"            #stock symbol
    lags=5                  #use a chosen number of lags of the chosen stock
    volatility=True         #use the rolling volatility (20) of the chosen stock
    rolling_mean=True       #use the rolling means (5,20) of the chosen stock
    market=True             #use the previous stock returns of the top S&P500 stocks and the gold index
    rolling_market=True     #use the rolling means (5,20) of the top S&P500 stocks and the gold index
    distance=False          #use the distance of the previous 50th price
    first_layer=32          #size of the first layer in the neural network
    second_layer=32         #size of the second layer in the neural network
    Ntest=750               #Size of the test dataset
    risk_free=0.05          #yearly risk free rate

    #get the data with the features choosen by the user
    data=get_data(stock, lags=lags,
                volatility=volatility, 
                rolling_market=rolling_market,
                rolling_mean=rolling_mean,
                distance=distance,
                market=market)


    #Get the Train and Test Datasets
    (Xtrain, Ytrain,
    Xtest,Ytest,
    return_train,return_test)=get_train_test(data,stock,Ntest=Ntest)

    #Get the number/dimension of the features
    number_features=len(data.columns)-1

    #Render the neural network
    model=get_model(number_features,
                    first_layer=first_layer,
                    second_layer=second_layer)

    #Train the model using the train data
    model=train_model(model=model,Xtrain=Xtrain,Ytrain=Ytrain)

    #Print the features used in predicting the direction of the stock
    print(f'''
    Features used in for Prediciton of {stock} stock
    {data.columns}
''')

    #evaluate the model
    evaluate_model(model,stock,
                        Xtrain,Ytrain,
                        Xtest,Ytest,
                        return_train,return_test,
                        risk_free=risk_free)


###################################################################
#######################   Run the Program   #######################
###################################################################

#Run the Script
if __name__ == '__main__':            
    main()



####################################################################
##########################   The End   #############################
#################   Thanks for Using the Program   #################
####################################################################
