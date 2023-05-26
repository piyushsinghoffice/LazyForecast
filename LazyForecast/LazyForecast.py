#Importing libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import pmdarima as pm
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

#Disable tensorflow cuDNN warnings
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Import tensorflow libraries for deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense

#Import GUI libraries
from tqdm.notebook import tqdm_notebook
from tqdm.keras import TqdmCallback
from IPython.display import Markdown, display

#Disable any other warnings
import warnings
warnings.filterwarnings('ignore')

class Deep_Formatter:
    def __init__(self, n_periods, n_steps, n_members, verbose):
        self.n_periods = n_periods
        self.n_steps = n_steps
        self.n_members = n_members
        self.verbose = verbose
        self.n_features = 1

    # split a univariate sequence into samples
    def split_sequence(self, sequence):
        X, y = list(), list()

        for i in range(len(sequence)):

            # find the end of this pattern
            end_ix = i + self.n_steps

            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break

            # gather input and output parts of the pattern
            seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    # make predictions with the ensemble and calculate a prediction interval
    def predict_with_confidence(self, ensemble, X):

        # make predictions
        yhat = [model.predict(X, verbose=0) for model in ensemble]
        yhat = np.asarray(yhat)

        # calculate 95% gaussian prediction interval
        interval = 1.96 * yhat.std()
        lower, upper = yhat.mean() - interval, yhat.mean() + interval
        return yhat.mean(), lower, upper

    # fit an ensemble of models
    def fit_ensemble(func):
        def inner(self, X_train, y_train, type_of_network):
            ensemble = list()
            for i in tqdm_notebook(range(self.n_members), desc=f"Training Ensemble {type_of_network} Models\t"):
                # define and fit the model on the training set
                model = func(self, i+1, X_train, y_train)
                # store the model
                ensemble.append(model)

            return ensemble
        return inner

    # format data for running deep learning algorithms
    def Deep_Forcasting(func):
        def inner(self, data, type_of_network):
            # create numpy matrix filled with zeroes
            predictions  = np.zeros(shape = (self.n_periods, 1))
            confidence  = np.zeros(shape = (self.n_periods, 2))
            train_data_size = len(data) - self.n_periods

            train = data[:train_data_size]
            test = data[train_data_size - self.n_steps:]

            X_train, y_train = self.split_sequence(train)
            X_test, y_test = self.split_sequence(test)

            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
            ensemble = func(self, X_train, y_train, type_of_network)

            for period in range(self.n_periods):
                x_input = X_test[period].reshape((1, self.n_steps, self.n_features))
                fc, lower, upper = self.predict_with_confidence(ensemble, x_input)

                #change the respective period records
                predictions[period] = fc
                confidence[period] = lower, upper

            predictions = predictions.flatten()
            # mean_prediction = np.mean(predictions, axis=0)  
            return predictions, confidence

        return inner
    

class LazyForecast(Deep_Formatter):

    """Fit Time-Series algorithms to data, predict and score on data.
    Parameters
    ----------
    n_periods : int, optional (default = 5)\
        Number of days to predict
    n_steps : int, optional (default = 5)
        Number of previous observations
    n_members: int, optional (default = 20)
        Total number of neural network models for ensemble learning
    verbose: bool, optional (default=0)
        To display verbose set its value to 1

    Returns
    -------
    Object : LazyForecast Object
        Returns object for lazy time series forecasting.
    """

    def __init__(self, n_periods=5, n_steps=5, n_members=10, verbose=0):
        self.n_periods = n_periods
        self.n_steps = n_steps
        self.n_members = n_members
        self.verbose = verbose
        self.n_features = 1

        super().__init__(n_periods, n_steps, n_members, verbose)

        # All forecasting methods
        self.FORECASTERS = {
            "ARIMA": self.AUTO_ARIMA,                       # Auto Arima
            "MLP": self.MLP,                               # Multilayer Perceptron
            "VANILLA LSTM": self.VANILLA_LSTM,              # Single LSTM Layer
            "STACKED LSTM": self.STACKED_LSTM,              # 2 Stacked LSTM Layers
            "BIDIRECTIONAL LSTM": self.BIDIRECTIONAL_LSTM,   # Bidirectional LSTM Layer
            "RNN": self.RNN,                               # Simple RNN Layer
            "GRU": self.GRU                                # Gated Recurrent Units Layer
        }

    def printmd(self, string):
        display(Markdown(string))
    
    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def MLP(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def VANILLA_LSTM(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def STACKED_LSTM(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        model.add(LSTM(32, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def BIDIRECTIONAL_LSTM(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def RNN(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(SimpleRNN(32, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    @Deep_Formatter.Deep_Forcasting
    @Deep_Formatter.fit_ensemble
    def GRU(self, id_no, X_train, y_train):
        model = Sequential()
        model.add(GRU(32, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=200, verbose=0, callbacks=[TqdmCallback(epochs=200, verbose=0, desc=f"Model: {id_no}\t ")])

        return model

    def AUTO_ARIMA(self, data, model_name):
        # create numpy matrix filled with zeroes
        forecast  = np.zeros(shape = (self.n_periods, 1))
        confidence = np.zeros(shape = (self.n_periods, 2))

        for period in tqdm_notebook(range(self.n_periods), desc=f"Training {model_name} Models"):
            train_data_size = len(data) - self.n_periods + period

            train = data[:train_data_size]
            model = pm.auto_arima(train,
                                    start_p=1,
                                    start_q=0,
                                    test='pp',       # use adftest to find optimal 'd'
                                    max_p=3, max_q=2, # maximum p and q
                                    m=1,              # frequency of series
                                    d=None,           # let model determine 'd'
                                    seasonal=False,   # No Seasonality
                                    start_P=0, 
                                    D=0, 
                                    trace=False,
                                    error_action='ignore',  
                                    suppress_warnings=True, 
                                    stepwise=True)

            fc, confint = model.predict(n_periods=1, return_conf_int=True)
            #change the respective period records
            forecast[period] = fc
            confidence[period] = confint

        return forecast.flatten(), confidence

    def mean_directional_accuracy(self, actual, predicted):
        """
        Calculates the Mean Directional Accuracy (MDA) for two time series.

        Parameters:
        actual (array-like): The actual values for the time series.
        predicted (array-like): The predicted values for the time series.

        Returns:
        float: The MDA value.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        # calculate the signs of the differences between consecutive values
        actual_diff = np.diff(actual)
        actual_signs = np.sign(actual_diff)
        predicted_diff = np.diff(predicted)
        predicted_signs = np.sign(predicted_diff)

        # count the number of times the signs are the same
        num_correct = np.sum(actual_signs == predicted_signs)

        # calculate the MDA value
        mda = num_correct / (len(actual) - 1)

        return mda

    def forecast_accuracy(self, actual, predicted, model_name):
        mda = self.mean_directional_accuracy(actual, predicted)
        r_squared = r2_score(actual, predicted)
        rmse = rms = mean_squared_error(actual, predicted, squared=False)
        mape = mean_absolute_percentage_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        corr = np.corrcoef(predicted, actual)[0,1]

        return({'model': model_name, 'mda': mda, 'rmse':rmse, 'mape':mape, 
                'R2': r_squared, 'mae': mae, 'corr':corr})

    def Title(self, title):
        self.printmd('<hr style="height:2px;border-width:0;color:gray;background-color:gray">')
        self.printmd(f'**{title}**')

    def find_periods_with_lowest_confidence_intervals(self, forecasts, lower_bounds, upper_bounds):
        lowest_interval_width = float('inf')
        lowest_interval_periods = []

        # Iterate over the confidence interval arrays and find the periods with the lowest interval widths
        for i in range(len(lower_bounds)):
            interval_width = upper_bounds[i] - lower_bounds[i]

            if interval_width < lowest_interval_width:
                lowest_interval_width = interval_width
                lowest_interval_periods = [forecasts[i]]
            elif interval_width == lowest_interval_width:
                lowest_interval_periods.append(forecasts[i])

        return lowest_interval_periods

    def Plot(self, data, fc, confint, model_name, x_values, x_axis, y_axis):
        x_values = np.arange(len(data)) if x_values is None else x_values

        index_of_fc = np.arange(len(data)-self.n_periods, len(data))#100-124

        # make series for plotting purpose
        fc_series = pd.Series(fc.flatten(), index=index_of_fc)

        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        # Find periods with the lowest confidence intervals
        lowest_intervals = np.where((confint[:, 1] - confint[:, 0]) == (confint[:, 1] - confint[:, 0]).min())[0]

        # Plot
        plt.figure(figsize=(15,8))
        a, = plt.plot(x_values, data, label='Actual')#line plot for original data
        fc, = plt.plot(x_values[index_of_fc], fc_series, color='red', linestyle='--', label='Forecasted')#line plotfor forecasted series

        #Plot confidence interval
        cf = plt.fill_between(x_values[lower_series.index], 
                         lower_series, 
                         upper_series,
                         color=(1,0,0,0.15), edgecolor=(1,0,0,1), label="Confidence Interval")

        # Add markers for periods with the lowest confidence intervals
        yerr = 50
        low_color = "black"
        X_low, Y_low = index_of_fc[lowest_intervals], fc_series.iloc[lowest_intervals]

        # For plotting the lowest confidence interval
        e, _, _ = plt.errorbar(x_values[X_low], Y_low, yerr=yerr, ls='-', color=low_color)
        plt.plot(x_values[X_low], Y_low + yerr, marker="v", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)
        plt.plot(x_values[X_low], Y_low - yerr, marker="^", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)

        # For adding the legends in correct order
        em1, = plt.plot([],[], marker=">", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)
        em2, = plt.plot([],[], marker="<", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)

        plt.title(f"{model_name} Forecast")
        if x_axis is not None:
            plt.xlabel(x_axis)
        if y_axis is not None:
            plt.ylabel(y_axis)

        # Create legend handler for markers and error bar
        marker_error_handler = HandlerTuple(ndivide=None, pad=-0.2)
            
        lg = plt.legend(
                    [a, fc, cf, (em1,e,em2)],
                    ['Actual', 'Forecasted', "Confidence Interval", 'Lowest Confidence Interval'],
                    prop = {'size' : 18},
                    handler_map={tuple: marker_error_handler},
                    facecolor = (1,1,1,0.5))

        plt.show()

    def fit(self, data, x_axis=None, y_axis=None):
        """Fit Time-Series algorithms to data, predict and score on data.
        Parameters
        ----------
        data : array-like
            The time-series on which you want to fit the models.
            This may either be a Pandas ``Series`` object, or a numpy array.
        x_axis : string, optional (default = None)
            Name of the X-axis
        y_axis : string, optional (default = None)
            Name of the Y-axis

        Returns
        -------
        Scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        Predictions : Numpy Array
            Returns predictions of the best performing model in a numpy array.
        Confidence Interval : Numpy Array
            Returns confidence interval of the best performing model in a numpy array.
        """

        x_values = data[x_axis] if x_axis != None else None
        
        if isinstance(data, np.ndarray):
            y_values = pd.Series(data)
        
        elif isinstance(data, pd.DataFrame):
            y_values = pd.Series(data[y_axis])

        else:
            y_values = data

        if y_values.isnull().values.any():
            y_values.interpolate(inplace=True)

        actual_values = y_values.iloc[-self.n_periods:]

        eval_data = list()

        # create an empty dictionary to store thre predictions and confidence interval for all the models
        prediction, confint = dict(), dict()

        for model_name, model in self.FORECASTERS.items():
            # set title for each model
            self.Title(model_name)

            # predict using the models in the FORECASTER dictionary
            prediction[model_name], confint[model_name] = model(y_values, model_name)

            # plot the predictions and confidence interval
            self.Plot(y_values, prediction[model_name], confint[model_name], model_name, x_values=x_values, x_axis=x_axis, y_axis=y_axis)

            # find the forecast accuracy and store accuracy result in the evaluation list
            result = self.forecast_accuracy(prediction[model_name], actual_values, model_name)
            eval_data.append(result)

        evaluation_table = pd.DataFrame(eval_data)
        evaluation_table.set_index('model', inplace=True)

        # Higher MDA, Lower RMSE, Lower MAPE, Higher R2_Score, Lower MAE means good fit
        evaluation_table.sort_values(by=['mda','rmse','mape','R2','mae'],
                                    ascending=[False, True, True, False, True],
                                    inplace=True)
        
        top = evaluation_table.head(1).index[0]

        fc = prediction[top]
        conf_interval = confint[top]

        return evaluation_table, fc, conf_interval