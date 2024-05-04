import os
import sys

import numpy as np
from matplotlib.pyplot import subplots
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from utils import manage_directory

from preferences_module import decode_users, decode_DASP, Weekday

pio.kaleido.scope.mathjax = None


def compare_bess(df, y_name, x_name, title):
    df.index.name = 'BESS type'
    f = px.scatter(df, x=x_name, y=y_name)
    fig = go.Figure(f)
    fig.update_traces(marker=dict(size=12,
                                  color='#2a9d8f',
                                  line=dict(width=2, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_xaxes(title=dict(text=x_name,
                                font=dict(size=18, family="Computer Modern")),
                     ticklabelposition="inside bottom", )
    fig.update_yaxes(title=dict(text=y_name,
                                font=dict(size=18, family="Computer Modern")),
                     autorange=True,
                     showgrid=True,
                     zeroline=True,
                     gridcolor='#edede9',
                     gridwidth=1,
                     zerolinecolor='#edede9',
                     zerolinewidth=2, )
    fig.update_annotations(font=dict(size=18, family="Computer Modern"))

    fig.update_layout(showlegend=False, height=310, width=500, font=dict(
        family="Computer Modern", size=18), legend=dict(font=dict(family="Computer Modern", size=16)),
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      hoverlabel=dict(
                          font_size=18,
                      ),

                      title=title
                      )

    fig.show()
    fig.write_html(f'{title}.html')
    fig.write_image(f"{title}.pdf")


def where_stops_increasing(ori_array, tresh):
    idx = 0
    for i in range(len(ori_array) - 1):
        if abs(ori_array[i + 1] - ori_array[i]) <= tresh:
            idx = i
            break
    return idx


def where_max_diff(a1, a2):
    result = [abs(a - b) for a, b in zip(a1, a2)]
    idx = np.argmax(result)
    return idx


def intersect(res, y_name, x_name, title):
    fig = go.Figure()
    colors = ['teal', '#f4a261']
    i = 0
    for key in res:
        df = res[key]
        fig.add_trace(go.Scatter(x=df[x_name], y=df[y_name], name=key, marker=dict(size=10,
                                                                                   color=colors[i],
                                                                                   line=dict(width=1.1,
                                                                                             color='DarkSlateGrey'))))

        i += 1

    fig.update_xaxes(title=dict(text=x_name, font=dict(size=18, family="Computer Modern")),
                     ticklabelposition="inside bottom", )
    fig.update_yaxes(title=dict(text=y_name, font=dict(size=18, family="Computer Modern")),
                     autorange=True,
                     showgrid=True,
                     zeroline=True,
                     gridcolor='#edede9',
                     gridwidth=1,
                     zerolinecolor='#edede9',
                     zerolinewidth=2, )
    fig.update_annotations(font=dict(size=18, family="Computer Modern"))

    fig.update_layout(showlegend=True, height=310, width=600, font=dict(
        family="Computer Modern", size=18), legend=dict(font=dict(family="Computer Modern", size=16)),
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      hoverlabel=dict(
                          font_size=18,
                      ),

                      title=title
                      )

    fig.show()
    fig.write_image(f"{title.replace(' ', '_')}_clean.pdf")

    keys = list(res.keys())
    idx = where_max_diff(list(res[keys[0]][y_name]), res[keys[1]][y_name])
    fig.add_vline(x=res[keys[0]][x_name][idx], line_color='DarkSlateGrey', line_width=1, line_dash="dash")

    idx = where_stops_increasing(list(res['Combined'][y_name]), 0.1)
    fig.add_vline(x=res['Combined'][x_name][idx], line_color='DarkSlateGrey', line_width=1, line_dash="dash")
    fig.show()
    fig.write_html(f'{title}.html')
    fig.write_image(f"{title.replace(' ', '_')}.pdf")


def compute_score(fname):
    res = pd.read_csv(fname, sep=';')
    s = []
    for i in res.index:
        s.append((res['GAIN'][i]))
    return sum(s)


def BESS_dimensioning_comparison():
    scale_factors = [100, 75, 50, 25]
    manage_directory("bess_dimensioning/data", delete_existing=True)
    for scale in scale_factors:
        cases = pd.read_csv("BESS configurations.csv", sep=',', index_col=0).sort_values(
            by=['capacity', 'dis_power', 'ch_power'],
            inplace=False)

        results = {}
        for scenario in ['Combined', 'BESS Only']:
            ch_l = []
            dis_l = []
            cap_l = []
            score = []

            for bess, row in cases.iterrows():
                cap_l.append(row['capacity'])
                dis_l.append(row['dis_power'])
                ch_l.append(row['ch_power'])
                s = compute_score(f"p{scale}_{scenario}{os.sep}{bess}_results{os.sep}simulation_results.csv")
                score.append(round(s, 2))

            pd.set_option('display.max_columns', None)
            res = pd.DataFrame({'Configuration': cases.index, 'Capacity [kWh]': cap_l,
                                'Max discharging power [kW]': dis_l, 'Max charging power [kW]': ch_l,
                                'Cumulative gain [€]': score})
            res.sort_values(
                by=['Capacity [kWh]', 'Max discharging power [kW]', 'Max charging power [kW]', 'Cumulative gain [€]'],
                inplace=True)

            results[scenario] = res

        for key in results:
            results[key].rename(columns={'Cumulative gain [€]': f'Cumulative gain [€] - {key}'}, inplace=True)

        results = list(results.values())
        results[1].drop(columns=['Capacity [kWh]', 'Max charging power [kW]', 'Max discharging power [kW]'],
                        inplace=True)
        res = results[0].merge(results[1], on='Configuration')
        print(res)

        res.to_csv(f'bess_dimensioning/data/p{scale}_bess_comparison.csv', sep=';', index=False)


def compare_scenarios():
    cases = {
        'sim_none': {
            'BESS': False,
            'SCHED': False,
        },
        'sim_bess_only': {
            'BESS': True,
            'SCHED': False
        },
        'sim_sched_only': {
            'BESS': False,
            'SCHED': True
        },
        'sim_complete': {
            'BESS': True,
            'SCHED': True
        },
    }

    scenario = {
        'expected': False,
        'real': True
    }

    scale_factors = [100, 75, 50, 25]
    for scale in scale_factors:
        for s in scenario:
            gain_l = []
            scr_l = []
            ssr_l = []
            names = []

            for c in cases.keys():
                print(f'\n\nSIMULATING P{scale} {s} {c}')

                names.append(c)
                res = pd.read_csv(f"p{scale}_{s}{os.sep}{c}_results{os.sep}simulation_results.csv", sep=";")

                gain_l.append(round(np.sum(res['GAIN']), 2))
                scr_l.append(round(np.mean(res['SCR']), 2))
                ssr_l.append(round(np.mean(res['SSR']), 2))

            res = pd.DataFrame(
                {'Scenario': names, 'Cumulative Gain [€]': gain_l, 'Mean SCR': scr_l, 'Mean SSR': ssr_l})

            res.sort_values(by=['Cumulative Gain [€]', 'Mean SSR', 'Mean SCR'], inplace=True)
            res.to_csv(f'p{scale}_{s}_comparison.csv', sep=';', index=False)


def bess_comparison(df, y_name, x_name, title, min_value, max_value):
    fig = go.Figure()
    colors = ['teal', '#f4a261']
    fig.add_trace(go.Scatter(x=df[x_name], y=df[f'{y_name} - Combined'], name='Combined', marker=dict(size=10,
                                                                                                      color=colors[0],
                                                                                                      line=dict(
                                                                                                          width=1.1,
                                                                                                          color='DarkSlateGrey'))))

    fig.add_trace(go.Scatter(x=df[x_name], y=df[f'{y_name} - BESS Only'], name='BESS Only', marker=dict(size=10,
                                                                                                        color=colors[1],
                                                                                                        line=dict(
                                                                                                            width=1.1,
                                                                                                            color='DarkSlateGrey'))))

    fig.update_xaxes(title=dict(text=x_name, font=dict(size=18, family="Computer Modern")),
                     ticklabelposition="inside bottom", )
    fig.update_yaxes(title=dict(text=y_name, font=dict(size=18, family="Computer Modern")),
                     autorange=False,
                     showgrid=True,
                     zeroline=True,
                     gridcolor='#edede9',
                     gridwidth=1,
                     zerolinecolor='#edede9',
                     zerolinewidth=2, )
    fig.update_annotations(font=dict(size=18, family="Computer Modern"))

    fig.update_layout(showlegend=True, height=310, width=600, font=dict(
        family="Computer Modern", size=18), legend=dict(font=dict(family="Computer Modern", size=16)),
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      hoverlabel=dict(
                          font_size=18,
                      ), title=title)

    fig.update_layout(yaxis_range=[min_value, max_value])
    fig.write_image(f"bess_dimensioning{os.sep}{title.replace(' ', '_')}_clean.pdf")
    return fig


def plot_bess_dimensioning():
    datasets = []
    max_value = -10 ^ 9
    min_value = 10 ^ 9
    for f in os.listdir(f'bess_dimensioning{os.sep}data'):
        df = pd.read_csv(f"bess_dimensioning{os.sep}data{os.sep}{f}", sep=";")
        max_cur = max(np.array(df.iloc[:, -2:]).flatten())
        if max_cur > max_value: max_value = max_cur
        min_cur = min(np.array(df.iloc[:, -2:]).flatten())
        if min_cur < min_value: min_value = min_cur
        datasets.append((f'SF {f.split("_")[0][1:]}%', df))

    for df in datasets:
        fig = bess_comparison(df[1], 'Cumulative gain [€]', 'Configuration', df[0],
                              min_value - 100, max_value + 100)
        fig.show()


def DASP_visualization(day):
    source = f'DASP{os.sep}expected{os.sep}sim_complete_results{os.sep}'
    DASP, fixed_load = decode_DASP(day, source)

    values = []
    for i in range(len(DASP)):
        for t in range(24):
            values.append(list(DASP.values())[i][t] + list(fixed_load.values())[i][t])

    max_value = max(values)
    print(max_value)

    cols = 3
    users = sorted(list(DASP.keys()))
    names = []
    profile = []
    p = 0
    y = 0
    f = 0
    for name in users:
        if "Pensionate" in name:
            names.append(f"P {p}")
            profile.append("P")
            p += 1
        if "Young Couple" in name:
            names.append(f"YC {y}")
            profile.append("YC")
            y += 1
        if "Family" in name:
            names.append(f"F {f}")
            profile.append("F")
            f += 1

    fig = make_subplots(rows=int(len(DASP) / cols), cols=cols, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=names, horizontal_spacing=0.03, vertical_spacing=0.18)
    fig.update_layout(xaxis_tickmode='array', bargap=0.02, bargroupgap=0)

    u = -1
    for row in range(int(len(DASP) / cols)):
        for col in range(cols):
            u += 1

            showlegend = False
            if row == 0 and col == 0:
                showlegend = True

            fig.add_trace(go.Bar(y=DASP[users[u]], name='Shiftable Load',
                                 legendgroup='Shiftable Load',
                                 showlegend=showlegend,
                                 marker_color='lightpink'
                                 ), row=row + 1, col=col + 1)

            fig.add_trace(go.Bar(y=fixed_load[users[u]], name='Fixed Load',
                                 marker_color='crimson',
                                 legendgroup='Fixed Load',
                                 showlegend=showlegend
                                 ), row=row + 1, col=col + 1)

            fig.update_xaxes(title=dict(text="Time [h]", font=dict(size=16, family="Computer Modern")),
                             ticklabelposition="inside bottom", showticklabels=True,
                             row=row + 1, col=col + 1)

            if col == 0:
                fig.update_yaxes(showticklabels=True,
                                 range=[0, max_value],
                                 title=dict(text="Energy [kWh]",
                                            font=dict(size=16, family="Computer Modern")),
                                 # autorange=True,
                                 showgrid=True,
                                 zeroline=True,
                                 gridcolor='#edede9',
                                 gridwidth=1,
                                 zerolinecolor='#edede9',
                                 zerolinewidth=2,
                                 row=row + 1, col=col + 1
                                 )
            else:
                fig.update_yaxes(showticklabels=False, range=[0, max_value],
                                 # autorange=True,
                                 showgrid=True,
                                 zeroline=True,
                                 gridcolor='#edede9',
                                 gridwidth=1,
                                 zerolinecolor='#edede9',
                                 zerolinewidth=2,
                                 row=row + 1, col=col + 1
                                 )

            fig.update_xaxes(showticklabels=True,
                             tickmode='auto',
                             row=row + 1, col=col + 1)

    w = 300 * cols
    fig.update_layout(showlegend=True, height=80 * len(DASP), width=w,
                      barmode='stack',
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      title=dict(text='DASP, Monday, 30th April 2018',
                                 font=dict(size=18, family="Computer Modern")),
                      font=dict(family="Computer Modern"),
                      legend=dict(
                          orientation="h",
                          yanchor="top",
                          y=-0.13,
                          xanchor="center",
                          x=0.5,
                          font=dict(size=18)
                      )
                      )
    fig.show()
    fig.write_image("DASP.pdf")


def print_bess_dim():
    manage_directory("bess_dimensioning/values", delete_existing=True)
    scale_factors = [100, 75, 50, 25]
    p_load_max = [6000, 6000, 8000, 6000, 8000, 8000, 6000, 6000, 8000, 8000, 6000, 8000, 8000, 8000, 8000, 8000, 8000, 6000, 8000, 8000, 8000]
    bc = pd.read_csv(f'BESS configurations.csv', sep=',')
    peak = sum(p_load_max)/1000
    print(peak)
    for scale in scale_factors:
        df = pd.DataFrame(columns=['BESS', 'C[kWh]', 'Ch.[kWh]', 'Dis.[kWh]'])
        print(f"Scale {scale} {scale/100 * peak}")
        for i, row in bc.iterrows():
            c = row['capacity'] * scale/100 * peak
            print(f"{row['Configurations']} {c} {c*row['ch_power']/100}")
            df.loc[i] = [row['Configurations'], c, c*row['ch_power']/100, c*row['dis_power']/100]
        df.to_csv(f'bess_dimensioning/values/{scale}.csv', index=False)


if __name__ == '__main__':
    # BESS_dimensioning_comparison()
    # plot_bess_dimensioning()
    compare_scenarios()
