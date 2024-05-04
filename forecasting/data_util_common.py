from plotly import graph_objs as go
import pandas as pd
import numpy as np

def show_result(df, title, x_title, y_title, filename):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=col, opacity=0.7))
    fig.update_layout(barmode='overlay')
    fig.update_layout(title=title)
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.write_html(filename)
    fig.show()

def show_box(df, label, title, x_title, y_title):
    fig = go.Figure()
    for row in range(len(df)):
        fig.add_trace(go.Box(y=np.array(df[label][row]), name=row, boxpoints='outliers'))
    fig.update_layout(title=title,
                      yaxis=dict(
                          autorange=True,
                          showgrid=True,
                          zeroline=True,
                          gridcolor='rgb(255, 255, 255)',
                          gridwidth=1,
                          zerolinecolor='rgb(255, 255, 255)',
                          zerolinewidth=2,
                      ),
                      paper_bgcolor='rgb(243, 243, 243)',
                      plot_bgcolor='rgb(243, 243, 243)',
                      showlegend=False
                      )

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.show()

def canopy_dataset():
    filename = 'data/NIST_Canopy_2015_2018_interpolated.csv'
    label = 'ACP_kW'
    data = pd.read_csv(filename, sep=',')
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
    data.set_index(data['TIMESTAMP'], drop=True, inplace=True)
    # data = filter_data(data, "2015-01-01", "2015-06-01")
    data = data.drop(columns=['TIMESTAMP', 'season'])
    data[label] = data[label].astype(float)
    time = [x.hour for x in pd.to_datetime(data.index).time]
    data ['Time'] = time
    print(data.columns)

    print("Correlation ----------------------------------------------------------------------------------------------")
    corr_data = data.drop(columns=[label]).corrwith(data[label])
    print(corr_data)

    pv_per_hour = data[[label]].copy()
    pv_per_hour['Time'] = time
    pv_per_hour = pv_per_hour.groupby("Time").agg({label: list}).reset_index()
    pv_per_hour = pv_per_hour.drop(columns=["Time"])
    show_box(pv_per_hour, label, "Energy distribution per hour", "Time [h]", "Energy [kWh]")
    del pv_per_hour
    return data, label

def filter_data(data, start_date, end_date):
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    data = data[(data.index >= start_date) & (data.index < end_date)]
    return data

def train_val_test_split(data, percentage):
    assert sum(percentage) == 100
    print(f"DATA LEN {len(data)}")
    print("*" * 40)
    idx = []
    for p in range(len(percentage)):
        i = round(len(data) * percentage[p ] /100)
        if p:
            i += idx[ p -1]
        if i < len(data):
            print(f"{data[i]} - {int(data[i].hour)}")
            i -= int(data[i].hour)
        else:
            print(f"{i}")
        idx.append(i)
        print(f"{percentage[p]}% - End idx: {i} - LEN {len(data[idx[ p -1] if p> 0 else p: i])}")
        if i < len(data):
            print(data[i].strftime('%Y-%m-%d %H:%M:%S'))
        else:
            print((data[i - 1] + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'))
        print("*" * 40)
    return idx





