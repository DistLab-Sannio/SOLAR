import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from data_util_common import canopy_dataset, show_result

if __name__ == '__main__':
    data, label = canopy_dataset()
    print(data.columns)
    features = ['Irradiance_1_Wm2']
    data = data[[label] + features]
    label_idx = list(data.columns).index(label)

    T = 24
    past = T
    future = T
    print(f"RESOLUTION: {T} - LOOKBACK WINDOW SIZE: {past} - PREDICTION WINDOW SIZE: {future}")

    # LOAD MODEL
    with open("sarima/sarima.pkl", 'rb') as pickle_file:
        sarima = pickle.load(pickle_file)

    test_timestamp = data.index[past:]
    test = data[label][past:].values
    exogenous_test = data[features][past:].values
    test_size = len(test)
    print(f"TEST SIZE {test_size}")
    predictions = sarima.predict(exog=exogenous_test, start=0, end=test_size-1)


    print(
        f"SARIMA\n MAE: {mean_absolute_error(test, predictions)} \n RMSE: {mean_squared_error(test, predictions, squared=True)}\n")

    results = pd.DataFrame().from_dict({"y_true": test, "y_pred": predictions})
    results['y_pred'] = [x if x > 0 else 0 for x in results['y_pred']]
    results.index = data.index[past:]
    results.to_csv("sarima/results_eval.csv", sep=",")
    show_result(results, "SARIMAX forecasting (Test)", "Time [h]", "Energy [Wh]")




