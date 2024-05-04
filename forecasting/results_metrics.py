import os
import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import qqplot
import scipy
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import datetime
import plotly.graph_objects as go
import plotly.io as pio
from data_util_common import canopy_dataset, filter_data

pio.kaleido.scope.mathjax = None

def show_box(df, label, title, x_title, y_title):
    fig = go.Figure()
    for row in range(len(df)):
        fig.add_trace(go.Box(y=np.array(df[label][row]), name=row, boxpoints='outliers'))
    fig.update_layout(title=title,
                      yaxis=dict(
                          autorange=True,
                          showgrid=True,
                          zeroline=True,
                          # gridcolor='rgb(255, 255, 255)',
                          gridwidth=1,
                          # zerolinecolor='rgb(255, 255, 255)',
                          zerolinewidth=2,
                      ),
                      # paper_bgcolor='rgb(243, 243, 243)',
                      # plot_bgcolor='rgb(243, 243, 243)',
                      showlegend=False
                      )

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.write_html("horly_distribution.html")
    fig.show()


def show_line(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=pd.to_datetime(df['TIMESTAMP']), y=df['AC Power [kW]'], name='AC Power', mode='lines+markers', marker=dict(size=1), line_color='#2a9d8f'),
                  secondary_y=False,
                  )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['Irradiance [Wm2]'], name='Irradiance', mode='lines+markers', marker=dict(size=1), line_color='#f4a261'),
        secondary_y=True
        )

    fig.update_traces(line={'width': 1})
    fig.update_layout(title_text="")

    # Set x-axis title
    fig.update_xaxes(title_text="Time [h]")

    # Set y-axes titles
    fig.update_yaxes(title_text="AC Power [kWh]", secondary_y=False)
    fig.update_yaxes(title_text="Irradiance [Wm2]", title_font_color="#f4a261", secondary_y=True)
    fig['layout']['yaxis2']['showgrid'] = False
    fig.show()


def boxplot(df, title, x_title, y_title):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Box(y=df[col], name=col, boxpoints='outliers', marker=dict(
                     color='teal')))
    fig.update_layout(title=title,
                      showlegend=False
                      )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.show()

def show_line(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(df['TIMESTAMP']), y=df['y_true'], name='Actual values', mode='lines+markers', marker=dict(size=3), line_color='#2a9d8f'),)
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(df['TIMESTAMP']), y=df['y_pred'], name='Predicted values', mode='lines+markers', marker=dict(size=3), line_color='#f4a261'))
    fig.update_traces(line={'width': 1})
    fig.update_layout(title_text=title)
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=15,
            # color="RebeccaPurple"
        ),
        autosize=False,
        width=700,
        height=500,
        title={
            'text': f'{title}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            #'font': {'size': 15},
        },
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='#edede9',
            gridwidth=1,
            zerolinecolor='#edede9',
            zerolinewidth=2,
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Time [h]")
    # Set y-axes titles
    fig.update_yaxes(title_text="AC Power [kW]")
    fig.write_image(f"{title}.pdf", format="pdf")
    fig.show()


def mean_absolute_scaled_error(y_true, y_pred, Naive_data, period=1):
    naive_actual = Naive_data[period:]
    naive_pred = Naive_data[:-period]
    naive_mae = mean_absolute_error(naive_actual, naive_pred)
    return mean_absolute_error(y_true, y_pred) / naive_mae


def compute_metrics(res, name, verbose=True):
    T = 24
    original = pd.read_csv("data/NIST_Canopy_2015_2018_interpolated.csv")
    original.set_index(pd.to_datetime(original["TIMESTAMP"], format='mixed'), drop=True, inplace=True)
    start_date = pd.to_datetime(res.iloc[0, 0]) - datetime.timedelta(days=1)
    end_date = pd.to_datetime(res.iloc[-1, 0]) + datetime.timedelta(hours=1)
    Naive_data = filter_data(original, str(start_date).split(" ")[0], str(end_date).split(" ")[0])

    y_actual = res.iloc[:, 1]
    y_pred = res.iloc[:, 2]

    nmae_range = round(mean_absolute_error(y_actual, y_pred) / ((max(y_actual) - min(y_actual))), 3)
    nmae_mean = round(mean_absolute_error(y_actual, y_pred) / np.mean(y_actual), 3)
    nmae_iqr = round(mean_absolute_error(y_actual, y_pred) / scipy.stats.iqr(y_actual), 3)

    mase = mean_absolute_scaled_error(y_actual, y_pred, Naive_data['ACP_kW'].values, period=T)
    mae = mean_absolute_error(y_actual, y_pred)
    me = np.mean(y_actual-y_pred)

    if verbose:
        # print(f"{name} {len(res.index)}")
        print("-" * 5 + "Compute METRICS" + "-" * 5)
        print(f"{name}\n"
              f"   ME: {round(me, 3)} [kWh]\n"
              f"   MAE: {round(mae, 3)} [kWh]\n"
              f"   NMAE (MEAN): {nmae_mean}\n"
              f"   NMAE (RANGE): {nmae_range}\n"
              f"   NMAE (IQR): {nmae_iqr}\n"
              f"   MASE: {round(mase, 3)}\n"
              "\n"
              )

    return me, mae, nmae_mean, nmae_range, nmae_iqr, mase


def cross_validation_results(dirs):
    for model, source_dir in dirs.items():
        metrics = {"me": [], "mae": [], "nmae (range)": [], "nmae (mean)": [], "nmae (iqr)": []}
        for file in os.listdir(source_dir):
            cval_fold = pd.read_csv(os.path.join(source_dir, file))
            me, mae, nmae_mean, nmae_range, nmae_iqr, mase = compute_metrics(cval_fold, f"LSTM {file}", False)
            metrics["me"].append(me)
            metrics["mae"].append(mae)
            metrics["nmae (range)"].append(nmae_range)
            metrics["nmae (mean)"].append(nmae_mean)
            metrics["nmae (iqr)"].append(nmae_iqr)
        print(f"\n{model} VAL")
        for metric, val in metrics.items():
            print(f"    {metric}: {np.mean(val)}")

    # boxplot(pd.DataFrame(np.column_stack((s_metrics["nmae"], l_metrics['nmae'], lopt_metrics['nmae']))).rename(columns={0: "SARIMAX", 1: "LSTM", 2: "LSTM OPT"}),
    #         title="Blocked Cross Validation Results", y_title='NMAE - Energy [kWh]', x_title="Models")

def handout_results():
    print("-" * 10 + "TEST RESULTS" + "-" * 10)
    sarimax = pd.read_csv("sarima/results.csv", sep=",")
    sarimax['y_pred'] = [x if x > 0 else 0 for x in sarimax['y_pred']]
    show_line(sarimax, "SARIMAX forecasting")
    compute_metrics(sarimax, "SARIMAX")

    lstm = pd.read_csv("lstm/optimized/results.csv", sep=",")

    #error_analysis(lstm)
    #error_compare(sarimax, lstm)

    show_line(lstm, "Stacked Bi-LSTM forecasting")
    compute_metrics(lstm, "LSTM")


def quantile_plot(data):
    qqplot_data = qqplot(data, line='s').gca().lines
    fig = go.Figure()
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#f4a261'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    fig.show()

def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   normal = df[((df>(q1-1.5*IQR)) & (df<(q3+1.5*IQR)))]
   return outliers, normal


def error_compare(sarimax, lstm):
    error_sarimax = sarimax['y_true'] - sarimax['y_pred']
    mean_error = np.mean(error_sarimax)
    std_error = np.std(error_sarimax)
    print(
        f"MEAN ERROR {mean_error} (STD {std_error}) The model {'underestimates' if mean_error > 0 else 'overestimates'}.")

    error_lstm = lstm['y_true'] - lstm['y_pred']
    mean_error = np.mean(error_lstm)
    std_error = np.std(error_lstm)
    print(
        f"MEAN ERROR {mean_error} (STD {std_error}) The model {'underestimates' if mean_error > 0 else 'overestimates'}.")

    fig = go.Figure()
    fig.add_trace(go.Box(y=error_sarimax, name='sarimax'))
    fig.add_trace(go.Box(y=error_lstm, name='lstm'))
    fig.show()

def error_analysis(data):
    error = data['y_true'] - data['y_pred']
    mean_error = np.mean(error)
    std_error = np.std(error)
    print(f"MEAN ERROR {mean_error} (STD {std_error}) The model {'underestimates' if mean_error > 0 else 'overestimates'}.")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=error, histnorm='percent',
                               marker={'color': 'lightcoral', 'opacity': 0.75}))
    fig.update_xaxes(title='Error')
    fig.update_layout(title='Error distribution')
    fig.show()

    quantile_plot(error)

    outliers, normal = find_outliers_IQR(error)
    # print(outliers)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['y_true'], y=data['y_pred'].mask(outliers > 0), name='Overestimated error (5%)',
    #                          mode='markers', line_color='#f4a261'))
    # fig.add_trace(go.Scatter(x=data['y_true'], y=data['y_pred'].mask(outliers < 0), name='Underestimation error (5%)',
    #                          mode='markers', line_color='#d5bdaf'))
    # fig.add_trace(go.Scatter(x=data['y_true'], y=data['y_pred'].iloc[~outliers.index], name='Average error',
    #                          mode='markers', line_color='#2a9d8f'))
    # fig.update_xaxes(title='Actual values')
    # fig.update_yaxes(title='Predicted values')
    # fig.show()


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['y_true'], y=data['y_pred'].iloc[outliers[outliers<0].index],
                             name='Overestimated error',
                             mode='markers', line_color='#f4a261'))
    fig.add_trace(go.Scatter(x=data['y_true'], y=data['y_pred'].iloc[outliers[outliers>0].index],
                             name='Underestimation error',
                             mode='markers', line_color='#2a9d8f'))
    fig.update_xaxes(title='Actual values')
    fig.update_yaxes(title='Predicted values')
    fig.show()

    canopy_data = canopy_dataset()
    irradiance = filter_data(canopy_data, data['TIMESTAMP'].values[0], data['TIMESTAMP'].values[-1])['Irradiance [Wm2]']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=irradiance.values, y=data['y_pred'].iloc[outliers[outliers<0].index],
                             name='Overestimated error',
                             mode='markers', line_color='#f4a261'))
    fig.add_trace(go.Scatter(x=irradiance.values, y=data['y_pred'].iloc[outliers[outliers>0].index],
                             name='Underestimation error',
                             mode='markers', line_color='#2a9d8f'))
    fig.update_xaxes(title='Actual values')
    fig.update_yaxes(title='Predicted values')
    fig.show()

def plot_lstm_sarimax_results():
    sarimax = pd.read_csv("sarima/results.csv", sep=",")
    sarimax['y_pred'] = [x if x > 0 else 0 for x in sarimax['y_pred']]
    lstm = pd.read_csv("lstm/optimized/results.csv", sep=",")
    sarimax = sarimax[len(sarimax)-len(lstm):]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(sarimax['TIMESTAMP']), y=sarimax['y_true'], name='Actual values', mode='lines+markers',
                   marker=dict(size=3), line_color='#2a9d8f'), )
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(sarimax['TIMESTAMP']), y=sarimax['y_pred'], name='SARIMAX', mode='lines+markers',
                   marker=dict(size=3), line_color='#f4a261'))
    fig.add_trace(
        go.Scatter(x=pd.to_datetime(lstm['TIMESTAMP']), y=lstm['y_pred'], name='Stacked Bi-LSTM', mode='lines+markers',
                   marker=dict(size=3), line_color='RebeccaPurple'))
    fig.update_traces(line={'width': 1})
    fig.update_layout(title_text='')
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=15,
            # color="RebeccaPurple"
        ),
        autosize=False,
        width=700,
        height=500,
        title={
            'text': f'Forecasting results',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            # 'font': {'size': 15},
        },
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='#edede9',
            gridwidth=1,
            zerolinecolor='#edede9',
            zerolinewidth=2,
        ),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Time [h]")
    # Set y-axes titles
    fig.update_yaxes(title_text="AC Power [kW]")
    fig.write_image(f"models.pdf", format="pdf")
    fig.show()




if __name__ == '__main__':
    #cross_validation_results({"SARIMAX": "sarima/cval", "LSTM OPT": "lstm/optimized/cval"})
    handout_results()
    plot_lstm_sarimax_results()







