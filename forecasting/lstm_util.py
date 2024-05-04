import numpy as np
from plotly import graph_objs as go
import pandas as pd
from tensorflow.keras.utils import timeseries_dataset_from_array

def show_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = [x for x in range(1, len(loss))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=loss, name="Training loss", mode="lines+markers", line_color='#2a9d8f'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation loss", mode="lines+markers", line_color='#f4a261'))
    fig.update_layout(title_text='Training Loss', showlegend=True)
    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text="Loss")
    fig.write_html('lstm/loss.html')
    fig.show()


def prepare_data(data, x_start, x_end, y_start, y_end,
                 sequence_length, step, batch_size, label_idx):
    print(f"X = {x_start} - {x_end}")
    print(f"Y = {y_start} - {y_end}")

    x = data[x_start:x_end, :]
    y = data[y_start:y_end, label_idx]
    dataset = timeseries_dataset_from_array(x, y,
                                            sequence_length=sequence_length,
                                            sampling_rate=step,
                                            batch_size=batch_size,
                                            )
    return dataset


#MODEL EVALUATION
def evaluate(dataset_test, model, days, scaler):
    y_true = []
    y_pred = []
    for inputs, targets in dataset_test.take(days):
        predictions = model.predict(inputs, verbose=0)
        y_true += [x for x in targets.numpy()]
        y_pred += [x[0] if x[0] > 0 else 0 for x in predictions]

    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1))
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    results = pd.DataFrame(np.concatenate([y_true, y_pred], axis=1))
    results.rename(columns={0: "y_true", 1: "y_pred"}, inplace=True)
    return results

