#utils file which contains all the auxiliary functions
#for the the views.py file of the REGRESSION app
#in the django framework



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
            rolling_market=False):

    #start and end dates of which financial information will be used
    start = datetime.date(2012,1,1)
    end =datetime.date.today()

    #create data range index
    Date = pd.date_range(start,end)

    #download all the stock data from the top ten stock in the sp500
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
            col = f'lag_{lag}'
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

    
    #shift the colum of the s&p500 index by one time-step
    df_returns[stock] = df_returns[stock].shift(-1)

    #remove any missing values
    df_returns.dropna( inplace=True)
    return df_returns


###################################################################
#######################   Get the Model   #########################
###################################################################

#import the linear, logistic,support vector machine
#and random forest models classifiers
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


##function that provided the type of predictor, provides the model
def get_model(number,C=0):
    

    if number =='1':
        #linear regression model was chosen
        model = Ridge(alpha=C)
    elif number =='2':
        #logistic regression model was chosen
        model = LogisticRegression(C=C)
    elif number =='3':
        #support vector machine was chosen
        model = svm.SVC(kernel='linear',C=C)
    else:
        model = RandomForestClassifier()
    
    return model


###################################################################
################  Get the Train and Test Datasets  ################
###################################################################

#function that creates the train and test data sets using the df_returs data set.
def get_train_test(df_returns,stock,number,Ntest=1000):

    #create train and test sets from the returns dataframes
    
    train = df_returns.iloc[1:-Ntest]
    test = df_returns.iloc[-Ntest:-1]

    #create the feature and signal test train dataframes
    Xtrain = train.drop(stock, 1)
    Ytrain = train[stock]
    
    #create the feature and signal test dataframes
    Xtest = test.drop(stock, 1)
    Ytest = test[stock]

    #if the type of predictor is not linear regression
    #then the signals are binary (i.e {-1,1})    
    if number !='1':
        Ytrain = (Ytrain > 0)
        Ytest = (Ytest > 0)

    #creating the train and test indexes
    train_idx = df_returns.index <= train.index[-1]
    test_idx = df_returns.index > train.index[-1]

    #first element of the train index and the last of the test index
    #are removed
    train_idx[0] = False
    test_idx[-1] = False

    #create returns with the train index for the purposes of cross valiation
    returns_train= df_returns.loc[train_idx,stock]    

    #return the Xtrain (i.e the train data of features)
    #return the Ytrain (i.e the train data of signals)
    #return the returns_train (i.e the returns of the train data)
    #return the Xtest (i.e the test data of features)
    #return the Xtest (i.e the test data of features)
    #return the train and test indexes of the data.
    return  Xtrain, Ytrain, returns_train, Xtest, Ytest, train_idx, test_idx


###################################################################
########################  Score the Model  ########################
###################################################################

#import the KFold function to peform cross validation on predictive models
from sklearn.model_selection import KFold

#function that applies cross validation of a given model (non random forest)
#using the training data with a 5 non random fold slips.
def score(Xtrain,Ytrain,returns_train,number,C=0):

    #create the 5 fold split indexes   
    kf5 = KFold(n_splits=5, shuffle=False)

    #create the initial score variable
    mean_return=0
    
    #for each split index, fit the data and evaluate the algorithmic strategy
    for train_index, test_index in kf5.split(Xtrain):
        #get the model
        model=get_model(number,C)
        
        #fit the data
        model.fit(Xtrain.iloc[train_index], Ytrain.iloc[train_index])    
        #make predictions of using the fitted model
        Ptrain = model.predict(Xtrain.iloc[test_index])
        #calculate the return using the algoritmic strategy
        mean_return+=(returns_train.iloc[test_index]*(Ptrain>0)).sum()

    #return the mean return of the algorithmic strategy using the model
    return mean_return/5


###################################################################
##########  Find the Best Hyperparameters for the Model  ##########
###################################################################

#function that finds the best hyperparameters of the chosen model
#using cross validation (if the choosen model is not random forest)
def best_model(Xtrain,Ytrain,returns_train,number):

    #In the case the type of classifier is the random forest
    if number not in ['1', '2', '3']:
        
        return get_model(number)

    #initialize the best hyperparameter C variable
    #and the current best score of the model
    best_C=0
    best_score=0
    
    #list of range of the hyperparameter C
    c_list= [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    for c in c_list:
        #get the score of the current c value
        new_score = score(Xtrain,Ytrain,returns_train,number,C=c)
        #print the new score
        print(new_score)
        
        #replace the best c value if the new score is greater
        if new_score >best_score:
            best_score = new_score
            best_C=c
    #print the best c 
    print(best_C)
    
    #get the model with the highest scoring hyperparameter C
    best_model=get_model(number,best_C)
    
    #return the model for testing
    return best_model


###################################################################
##################### Evaluation of the Model #####################
###################################################################

#Evaluate the chosen model with the best hyperparameters
#using the features also chosen by the user
def eval_model(stock,df_returns,
                Xtrain, Ytrain,
                returns_train,
                Xtest, Ytest,
                train_idx, test_idx,
                model,
                Ntest,
                risk_free):
 
    #fit the model with all the train data
    model.fit(Xtrain, Ytrain)

    #Data prediction using the model
    Ptrain = (model.predict(Xtrain)>0)
    Ptest = (model.predict(Xtest)>0)
    
    
    ########## Analytics of Algorithmic Strategy ###########
    
    #compute the accuracy of the classfier
    #on the train and test data sets
    accuracy_alg_train=(Ptrain==(Ytrain>0)).mean()
    accuracy_alg_test= (Ptest==(Ytest>0)).mean()


    #computation of the postion using the algorithmic strategy
    #of the selected model
    df_returns.loc[train_idx,'Position'] = Ptrain
    df_returns.loc[test_idx,'Position'] = Ptest

    #computation of the returns of the algorithmic strategy (long position only)
    df_returns['AlgoReturn'] = df_returns['Position'] * df_returns[stock]

    #Computation of the total returns of the algorithmic strategy
    #on the train and test data sets
    total_alg_train=df_returns.iloc[1:-Ntest]['AlgoReturn'].sum()
    total_alg_test=df_returns.iloc[-Ntest:-1]['AlgoReturn'].sum()
    
    #Compute the mean return of the algorithmic strategy
    #on the train and test data sets
    mean_alg_train=df_returns.iloc[1:-Ntest]['AlgoReturn'].mean()
    mean_alg_test=df_returns.iloc[-Ntest:-1]['AlgoReturn'].mean()

    #Compute the standard deviation return of the algorithmic
    #strategy on the train and test data sets
    std_alg_train=df_returns.iloc[1:-Ntest]['AlgoReturn'].std()
    std_alg_test=df_returns.iloc[-Ntest:-1]['AlgoReturn'].std()

    #Compute the Sharpe Ratio of the algorithmic
    #strategy on the train and test data sets
    sharpe_alg_train=(mean_alg_train-risk_free/256)/std_alg_train
    sharpe_alg_test=(mean_alg_test-risk_free/256)/std_alg_test

    
    ########## Analytics of Buy-and-Hold Strategy ##########

    #recover the returns of the stock in the train and test data sets
    train = df_returns.iloc[1:-Ntest]
    test = df_returns.iloc[-Ntest:-1]
    Ytrain = train[stock]
    Ytest = test[stock]
    
    
    #compute the accuracy of the buy and hold strategy
    accuracy_hold_train=(Ytrain>0).mean()
    accuracy_hold_test=(Ytest>0).mean()

    
    #Compute the total return of the buy and hold strategy
    #on the train and test data sets
    total_hold_train=Ytrain.sum()
    total_hold_test=Ytest.sum()
    
    #Compute the mean return of the buy and hold strategy
    #on the train and test data sets
    mean_hold_train=Ytrain.mean()
    mean_hold_test=Ytest.mean()

    #Compute the standard deviation return of the buy and hold
    #strategy on the train and test data sets
    std_hold_train=Ytrain.std()
    std_hold_test=Ytest.std()

    #Compute the Sharpe Ratio of the buy and hold
    #strategy on the train and test data sets
    sharpe_hold_train=(mean_hold_train-risk_free/256)/std_hold_train
    sharpe_hold_test=(mean_hold_test-risk_free/256)/std_hold_test

    ########## Plot both Strategies ##########

    #creating dataframe with returns of algorithmic and
    #buy and hold strategies
    returns_train=Ytrain.to_frame()
    returns_test=Ytest.to_frame()
    returns_train['strategy']=df_returns.iloc[1:-Ntest]['AlgoReturn']
    returns_test['strategy']=df_returns.iloc[-Ntest:-1]['AlgoReturn']

    #create data frame with all the returns of both strategies
    returns_train[stock]=pd.to_numeric(returns_train[stock])
    returns_train['strategy']=pd.to_numeric(returns_train['strategy'])
    returns_test[stock]=pd.to_numeric(returns_test[stock])
    returns_test['strategy']=pd.to_numeric(returns_test['strategy'])

    #Create graph with the cummulative returns of both strategies in the train data
    returns_train[[stock, 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig("returns_train.png")
    plt.close()

    #Create graph with the cummulative returns of both strategies in the test data
    returns_test[[stock, 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.savefig("returns_test.png")
    plt.close()


    ############# Evaluation of the model in the train data  #############
    
    print("""
######################################################################
##########  Evaluation of the Model in the Train Dataset  ############
######################################################################
""")

    #Total Log Returns
    print(f"""Total Daily Log Return of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(total_alg_train*100,3)}%
    Buy-and-Hold: {round(total_hold_train*100,3)}%
    """)

    #Accuracy of both strategies
    print(f"""Accuracy of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(accuracy_alg_train*100,3)}%
    Buy-and-Hold: {round(accuracy_hold_train*100,3)}%
    """)
    
    #Mean Returns
    print(f"""Mean Daily Log Return of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(mean_alg_train*100,3)}%
    Buy-and-Hold: {round(mean_hold_train*100,3)}%
    """)
    
    print(f"""Standard Deviation of Daily Log Returns of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(std_alg_train*100,3)}%
    Buy-and-Hold: {round(std_hold_train*100,3)}%
    """)

    print(f"""Sharpe Ratio of Strategy and Buy-and-Hold on the Train Dataset:
    Algorithm   : {round(sharpe_alg_train,3)}
    Buy-and-Hold: {round(sharpe_hold_train,3)}
    """)


    ############# Evaluation of the Model in the Test Data  #############
    
    print("""######################################################################
############  Evaluation of the Model on the Test Dataset  ###########
######################################################################
""")

    #Total Log Returns
    print(f"""Total Daily Log Return of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(total_alg_test*100,3)}%
    Buy-and-Hold: {round(total_hold_test*100,3)}%
    """)

    #Accuracy of both strategies
    print(f"""Accuracy of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(accuracy_alg_test*100,3)}%
    Buy-and-Hold: {round(accuracy_hold_test*100,3)}%
    """)
    
    #Mean Returns
    print(f"""Mean Daily Log Return of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(mean_alg_test*100,3)}%
    Buy-and-Hold: {round(mean_hold_test*100,3)}%
    """)
    
    print(f"""Standard Deviation of Daily Log Returns of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(std_alg_test*100,3)}%
    Buy-and-Hold: {round(std_hold_test*100,3)}%
    """)

    print(f"""Sharpe Ratio of Strategy and Buy-and-Hold on the Test Dataset:
    Algorithm   : {round(sharpe_alg_test,3)}
    Buy-and-Hold: {round(sharpe_hold_test,3)}
    """)


###################################################################
#######################    Main Program    ########################
###################################################################

#Function to run the main bulk of the Program
def main():
#request the user to choose the type of predictor
    number=input('''Please select the type of predictor
1) Linear Regression
2) Logistic Regression
3) Support Vector Machine
4) Random Forest
    ''')

    ###Ask the user which data and features to apply the model (optional) ###
    
    #stock=input("Choose Stock tick: ")
    #lags=int(input("Number of Lags?: "))
    #volatility=bool(input("Volatility? (empty for no): "))
    #rolling_mean=bool(input("Rolling Mean? (empty for no): "))
    #market=bool(input('Market (empty for no)?: '))
    #rolling_market=bool(input("Rolling Market? (empty for no): "))
    #Ntest=int(input("Size of test data?: "))
    #risk_free=float(input("Risk-Free Rate?: "))
    
    #### Defaut Settings ####
    stock= "SPY"            #stock symbol (default S&P500 index)
    lags=5                  #use a chosen number of lags of the chosen stock
    volatility=True         #use the rolling volatility (20) of the chosen stock
    rolling_mean=True       #use the rolling means (5,20) of the chosen stock
    market=True             #use the previous stock returns of the top S&P500 stocks and the gold index
    rolling_market=True     #use the rolling means (5,20) of the top S&P500 stocks and the gold index
    Ntest=550               #Size of the test dataset
    risk_free=0.05          #yearly risk free rate
    
    #Get the Data
    df_returns=get_data(stock,lags=lags,
                        market=market,
                	    rolling_mean=rolling_mean,
                        volatility=volatility,
                        rolling_market=rolling_market)

    #Create the train and test dataframes with respective indexes
    (Xtrain, Ytrain, returns_train,
     Xtest, Ytest,
     train_idx, test_idx)=get_train_test(df_returns,stock,number,Ntest)

    #Get the model with the best Hyperparameters
    #(if model chosen was not the Random Forest)
    model=best_model(Xtrain,Ytrain,returns_train,number)

    
    #Evaluate the performance of the model
    eval_model(stock,df_returns,
                Xtrain, Ytrain,
                returns_train,
                Xtest, Ytest,
                train_idx, test_idx,
                model,
                Ntest,
                risk_free)


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
