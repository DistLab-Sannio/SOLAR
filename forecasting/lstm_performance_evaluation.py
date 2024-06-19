from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

from data_util_common import canopy_dataset, show_result, train_val_test_split
from forecasting.lstm_util import prepare_data, evaluate

data, label = canopy_dataset()
numeric_features = ['Irradiance_1_Wm2']
data = data[[label] + numeric_features]

print(data.columns)
label_idx = list(data.columns).index(label)
print(label_idx)

T = 24
past = T * 7
future = T
step = 1
batch_size = 512
sequence_length = int(past / step)

print(f"RESOLUTION: {T} - LOOKBACK WINDOW SIZE: {past} - PREDICTION WINDOW SIZE: {future}")

# SPLIT DATA IN TRAIN AND TEST
split_idx = train_val_test_split(pd.to_datetime(data.index), [80, 20])
train_val_size = split_idx[0]
train = data.iloc[:train_val_size, :]
test = data.iloc[train_val_size:, :]

# FIT SCALER ON TRAIN DATA
pipeline = ColumnTransformer([
    ('label', MinMaxScaler(), [label]),
    ('num', MinMaxScaler(), numeric_features),
], remainder='passthrough')

train = pipeline.fit_transform(train)

# SPLIT TRAIN IN SUB-TRAIN AND VALIDATION
train_val_split = train_val_test_split(data.index[:train_val_size], [70, 30])
train_size = train_val_split[0]
val_size = train_val_size - train_size

val = train[train_size:, :]
train = train[:train_size, :]

# TRANSFORM TEST DATA
test = pipeline.transform(test)
test_size = len(test)
test_timestamp = data.index[train_val_size:]
print(f"TRAIN {train_size} VAL {val_size} TEST {test_size} - TOTAL {len(data)}")

#####################################################################################################################
# PREPARE DATA
# TRAIN
print(f"\nTRAIN - LEN {train_size}")
dataset_train = prepare_data(train, 0, train_size - past, past, train_size, sequence_length, 1, batch_size,
                             label_idx)

# VAL
print(f"\nVAL - LEN {val_size}")
dataset_val = prepare_data(val, 0, val_size - past, past, val_size, sequence_length, 1, batch_size, label_idx)

# TEST
print(f"\nTEST - LEN {test_size}")
dataset_test = prepare_data(test, 0, test_size, past, test_size, sequence_length, 1, batch_size, label_idx)

print("*** TRAINING *** --------------------------------------------------------------------------------------")
for batch in dataset_train.take(1):
    inputs, targets = batch

print(f"Input shape:  {inputs.numpy().shape}")
print(f"Target shape: {targets.numpy().shape}")

model = load_model('lstm.keras')

print("*** TESTING *** ------------------------------------------------------------------------------------------")
for batch in dataset_test.take(1):
    inputs, targets = batch

print(f"Input shape:  {inputs.numpy().shape}")
print(f"Target shape: {targets.numpy().shape}")

days = int((test_size - past) / T)
print(f"DAYS {days}")

results = evaluate(dataset_test, model, days, pipeline.named_transformers_['label'])
results.to_csv("lstm/results.csv", sep=",", index=False)

print(len(results))
results.index = pd.to_datetime(test_timestamp[past:test_size], format='%Y-%m-%d %H:%M %S')
print(results.head())

results.to_csv("lstm/results.csv", sep=",")
show_result(results, "LSTM Energy prediction", "Time [h]", "Energy [kWh]", 'lstm/lstm_results.html')

mae = mean_absolute_error(results['y_true'], results['y_pred'])
model.save(f'lstm/lstm_final.keras')

print(f"MAE: {mae}")