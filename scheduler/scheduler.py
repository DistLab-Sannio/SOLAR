import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import xpress as xp
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas
import plotly.io as pio
from forecaster_module import get_production
from tariff_scraper import get_tariffs
from utils import manage_directory

pio.kaleido.scope.mathjax = None
from preferences_module import load_preferences, Weekday

xp.init(r"xpauth.xpr")


def show_dasp(df, source, title, x_title, y_title):
    n_rows = int(len(df.columns) / 2) + 1
    n_cols = 2
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=df.columns)
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if i < len(df.columns):
                fig.add_trace(go.Bar(x=df.index, y=df[df.columns[i]], name=df.columns[i]), row=row + 1, col=col + 1)
                fig.update_xaxes(title=dict(text=x_title, font=dict(size=18, family="Computer Modern")),
                                 ticklabelposition="inside bottom", tickmode='linear', row=row + 1, col=col + 1)
                fig.update_yaxes(title=dict(text=y_title[i], font=dict(size=18, family="Computer Modern")),
                                 autorange=True,
                                 showgrid=True,
                                 zeroline=True,
                                 gridcolor='#edede9',
                                 gridwidth=1,
                                 zerolinecolor='#edede9',
                                 zerolinewidth=2,
                                 row=row + 1, col=col + 1)
                fig.update_annotations(font=dict(size=18, family="Computer Modern", ))
                i += 1

    fig.update_layout(showlegend=False, height=200 * n_rows, width=650 * n_cols, font=dict(
        family="Computer Modern",
        size=18),
                      paper_bgcolor='rgb(255, 255, 255)',
                      plot_bgcolor='rgb(255, 255, 255)',
                      title=title
                      )

    manage_directory(f"DASP{os.sep}{source}", delete_existing=False)
    fig.write_image(f"DASP{os.sep}{source}{os.sep}{title}.pdf")
    # fig.show()


def show_prices(df, title, x_title, y_title):
    fig = make_subplots(rows=len(df.columns), cols=1, subplot_titles=df.columns)
    for col in range(len(df.columns)):
        fig.add_trace(go.Bar(x=df.index, y=df[df.columns[col]], name=df.columns[col]), row=col + 1, col=1)
        fig.update_xaxes(title=dict(text=x_title, font=dict(size=18, family="Computer Modern")),
                         ticklabelposition="inside bottom",
                         tickmode='linear', row=col + 1, col=1)
        fig.update_yaxes(title=dict(text=y_title, font=dict(size=18, family="Computer Modern")), row=col + 1, col=1)
    fig.update_layout(font=dict(
        family="Computer Modern",
        size=18),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        title=title, showlegend=False, height=250 * len(df.columns), width=650)
    fig.show()


def produce_DASP(day, date, preferences_dir, production, scale_factor, tariffs, bess, bess_features, res_dir):
    T = 24
    D_T = 1  # Temporal Slot Duration
    N_BESS = 1 if bess else 0

    cur_date = date.strftime("%A, %B %dth, %Y")

    # PUN, CEI, inc [€/Wh]
    l_buy = [float(x) / 1000000 for x in tariffs['PUN']]
    l_sell = [float(x) / 1000000 for x in tariffs['PZ']]
    inc = [float(x) / 1000000 for x in tariffs['INC']]

    alpha = 0.999  # 0.10  # Selling taxation [%]
    avoided_loss = 7 / 1000000  # [€]

    # prices = pd.DataFrame().from_dict({"Zonal Price": l_sell, "PUN": l_buy, "Incentive": inc})
    # show_prices(prices, f"Tariffs {cur_date}", "Time [h]", "Value [€/Wh]")

    active_scheduler, user_id, p_load_max, p_load_nsh, all_appliances, user_appliances, all_intervals, appliance_intervals, \
        all_slots, appliance_slots, all_interval_sets, all_int_appliances, \
        int_counter, user_int_appliances, all_int_interval_sets, all_int_intervals, int_appliance_intervals, \
        = load_preferences(day, preferences_dir)

    production = scale(production, sum(p_load_max),scale_factor)

    max_n_interval_sets = 0
    if active_scheduler:
        for u in range(len(user_id)):
            for v in appliance_intervals[u].values():
                if len(v) > max_n_interval_sets:
                    max_n_interval_sets = len(v)

    max_n_intervals = 0
    if active_scheduler:
        for u in range(len(user_id)):
            for s in appliance_intervals[u]:
                for i in range(len(appliance_intervals[u][s])):
                    if len(appliance_intervals[u][s][i]) > max_n_intervals:
                        max_n_intervals = len(appliance_intervals[u][s][i])

    max_n_slots = 0
    if active_scheduler:
        for u in range(len(user_id)):
            for s in appliance_slots[u]:
                for interval_set in range(len(appliance_slots[u][s])):
                    for interval in range(len(appliance_slots[u][s][interval_set])):
                        if len(appliance_slots[u][s][interval_set][interval]) > max_n_slots:
                            max_n_slots = len(appliance_slots[u][s][interval_set][interval])

    max_n_int_interval_sets = 0
    if active_scheduler:
        for u in range(len(user_id)):
            for v in int_appliance_intervals[u].values():
                if len(v) > max_n_int_interval_sets:
                    max_n_int_interval_sets = len(v)

    max_n_int_intervals = 0
    if active_scheduler:
        for u in range(len(user_id)):
            for k in int_appliance_intervals[u]:
                for i in range(len(int_appliance_intervals[u][k])):
                    if len(int_appliance_intervals[u][k][i]) > max_n_int_intervals:
                        max_n_int_intervals = len(int_appliance_intervals[u][k][i])

    sched_problem = xp.problem()

    x = [[[xp.var(name=f"{u}-{s}-{t}", vartype=xp.binary) for t in range(T)]
          for s in range(len(all_appliances))]
         for u in range(len(user_id))]

    y = [[[[[xp.var(name=f"{u}-{s}-{i_s}-{i}-{agg_s}", vartype=xp.binary) for agg_s in range(max_n_slots)]
            for i in range(max_n_intervals)]
           for i_s in range(max_n_interval_sets)]
          for s in range(len(all_appliances))]
         for u in range(len(user_id))]

    w = [[[xp.var(name=f"shint-{u}-{k}-{t}", vartype=xp.binary) for t in range(T)]
          for k in range(len(all_int_appliances))]
         for u in range(len(user_id))]

    q = [[[[xp.var(name=f"shint-{u}-{k}-{i_s}-{i}", vartype=xp.binary) for i in range(max_n_int_intervals)]
           for i_s in range(max_n_int_interval_sets)]
          for k in range(len(all_int_appliances))]
         for u in range(len(user_id))]

    p_shared = [xp.var(name=f"p_shared_{t}", vartype=xp.continuous) for t in range(T)]

    # BESS
    C = bess_features['capacity']/100 * sum(p_load_max) * scale_factor/100  # Capacity in WATT
    R_ch = bess_features['ch_power']/100 * C * N_BESS  # MAX charging power
    R_dis = bess_features['dis_power']/ 100 * C * N_BESS  # MAX discharging power
    eta_ch = 0.95
    eta_dis = 0.95
    SoC_min = 0.1 * C * N_BESS
    SoC_max = 0.9 * C * N_BESS
    SoC_init = 0.1 * C * N_BESS

    print(f"BESS Capacity {C}")
    print(f"R_ch {R_ch}")
    print(f"R_dis {R_dis}")
    print(f"SoC_max {SoC_max}")

    P_load = [xp.var(name=f"P_load_{t}", vartype=xp.continuous) for t in range(T)]
    P_dis = [xp.var(name=f"p_dis_{t}", vartype=xp.continuous) for t in range(T)]
    P_ch = [xp.var(name=f"p_ch_{t}", vartype=xp.continuous) for t in range(T)]
    P_inj = [xp.var(name=f"p_inj_{t}", vartype=xp.continuous) for t in range(T)]
    P_grid = [xp.var(name=f"p_grid_{t}", vartype=xp.continuous) for t in range(T)]

    SoC = [xp.var(name=f"SoC_{t}", vartype=xp.continuous) for t in range(T)]
    m_ess = [xp.var(name=f"m_ess_{t}", vartype=xp.binary) for t in range(T)]

    sched_problem.addVariable(x, y, w, p_shared, q, P_load, P_dis,
                              P_ch, P_grid, P_inj, SoC, m_ess, )

    sched_problem.addConstraint([0 <= P_ch[t] <= R_ch * (1 - m_ess[t]) for t in range(T)])
    print(len(sched_problem.getConstraint()))

    sched_problem.addConstraint([0 <= P_dis[t] <= R_dis * m_ess[t] for t in range(T)])
    sched_problem.addConstraint([0 <= P_dis[t] <= SoC[t] for t in range(T)])
    print(len(sched_problem.getConstraint()))

    sched_problem.addConstraint(
        [SoC[t] == (SoC[t - 1] + (eta_ch * P_ch[t - 1] - P_dis[t - 1] / eta_dis) * D_T) for t in range(1, T)])

    print(len(sched_problem.getConstraint()))
    sched_problem.addConstraint([SoC[0] == SoC_init])
    sched_problem.addConstraint([SoC[T - 1] == SoC[0]])
    print(len(sched_problem.getConstraint()))
    sched_problem.addConstraint([SoC_min <= SoC[t] <= SoC_max for t in range(T)])
    print(len(sched_problem.getConstraint()))

    ##################################################################################################################
    sched_problem.addConstraint(
        [P_dis[t] + production[t] + P_grid[t] == P_load[t] + P_ch[t] + P_inj[t] for t in range(T)])
    print(len(sched_problem.getConstraint()))

    # SHARED ENERGY
    sched_problem.addConstraint([p_shared[t] <= P_load[t] + P_ch[t] for t in range(T)])
    sched_problem.addConstraint([p_shared[t] <= production[t] for t in range(T)])

    sched_problem.setObjective(D_T * xp.Sum([l_sell[t] * (1 - alpha) * P_inj[t] +
                                             p_shared[t] * (inc[t] + avoided_loss)
                                             - l_buy[t] * P_grid[t]
                                             for t in range(T)]), sense=xp.maximize)

    #LOAD DEFINITION
    sched_problem.addConstraint([P_load[t] == (xp.Sum(
        [x[u][s][t] * all_appliances[s][1] for s in range(len(all_appliances)) for u in range(len(user_id))])
                                               + xp.Sum(
                [w[u][j][t] * all_int_appliances[j][1] for j in range(len(all_int_appliances)) for u in
                 range(len(user_id))])
                                               + xp.Sum([p_load_nsh[u][t] for u in range(len(user_id))])) for t in
                                 range(T)])
    print(len(sched_problem.getConstraint()))

    ########################################################################################################
    if active_scheduler:
        # CONSTRAINT 1
        sched_problem.addConstraint([xp.Sum([x[u][s][t] * all_appliances[s][1] for s in range(len(all_appliances))])
                                     + xp.Sum(
            [w[u][j][t] * all_int_appliances[j][1] for j in range(len(all_int_appliances))])
                                     + p_load_nsh[u][t] <= p_load_max[u]
                                     for t in range(T)
                                     for u in range(len(user_id))])
        print(len(sched_problem.getConstraint()))

        # INTERRUPTIBLES
        # MIN UPTIME
        if len(appliance_intervals) > 0:
            for u in range(len(user_id)):
                if len(int_appliance_intervals[u]):
                    for k in range(min(int_appliance_intervals[u].keys()), max(int_appliance_intervals[u].keys()) + 1):
                        for i_s in range(len(int_appliance_intervals[u][k])):
                            for i in range(len(int_appliance_intervals[u][k][i_s])):
                                sched_problem.addConstraint(
                                    xp.Sum([w[u][k][t]
                                            for t in range(int_appliance_intervals[u][k][i_s][i][0],
                                                           int_appliance_intervals[u][k][i_s][i][1] + 1)])
                                    >= all_int_appliances[k][2] * q[u][k][i_s][i])
        print(len(sched_problem.getConstraint()))

        #exclusive OR
        for u in range(len(user_id)):
            if len(int_appliance_intervals[u]):
                for k in range(min(int_appliance_intervals[u].keys()), max(int_appliance_intervals[u].keys()) + 1):
                    for i_s in range(len(int_appliance_intervals[u][k])):
                        sched_problem.addConstraint(
                            xp.Sum([q[u][k][i_s][i] for i in range(len(int_appliance_intervals[u][k][i_s]))]) == 1)
        print(len(sched_problem.getConstraint()))

        if len(appliance_intervals) > 0:
            for u in range(len(user_id)):
                if len(int_appliance_intervals[u]):
                    for k in range(min(int_appliance_intervals[u].keys()),
                                   max(int_appliance_intervals[u].keys()) + 1):
                        for i_s in range(len(int_appliance_intervals[u][k])):
                            for i in range(len(int_appliance_intervals[u][k][i_s])):
                                for t in range(int_appliance_intervals[u][k][i_s][i][0], int_appliance_intervals[u][k][i_s][i][1] + 1):
                                    sched_problem.addConstraint(w[u][k][t] <= q[u][k][i_s][i])
        print(len(sched_problem.getConstraint()))

        if len(appliance_intervals) > 0:
            for u in range(len(user_id)):
                if len(int_appliance_intervals[u]):
                    for k in range(min(int_appliance_intervals[u].keys()), max(int_appliance_intervals[u].keys()) + 1):
                        sched_slots = []
                        for i_s in range(len(int_appliance_intervals[u][k])):
                            for i in range(len(int_appliance_intervals[u][k][i_s])):
                                for t in range(int_appliance_intervals[u][k][i_s][i][0], int_appliance_intervals[u][k][i_s][i][1] + 1):
                                     sched_slots.append(t)
                        for t in range(T):
                            if t not in sched_slots:
                                #print(f"NOT SCHED {t}")
                                sched_problem.addConstraint(w[u][k][t] == 0)
        print(len(sched_problem.getConstraint()))


        # NON INTERRUPTIBLES
        # VINCOLO 2
        sched_problem.addConstraint(
            [xp.Sum([y[u][s][i_s][i][agg_s] for i in range(max_n_intervals) for agg_s in range(max_n_slots)]) == 1
             for u in range(len(user_id))
             for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
             for i_s in range(max_n_interval_sets) if i_s < len(appliance_intervals[u][s])])
        print(len(sched_problem.getConstraint()))

        # VINCOLO 3
        for u in range(len(user_id)):
            for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1):
                for i_s in range(len(appliance_slots[u][s])):
                    for i in range(len(appliance_slots[u][s][i_s])):
                        for agg_s in range(len(appliance_slots[u][s][i_s][i])):
                            sched_problem.addConstraint(
                                [xp.Sum([x[u][s][t] for t in range(appliance_slots[u][s][i_s][i][agg_s][0],
                                                                   appliance_slots[u][s][i_s][i][agg_s][0] +
                                                                   appliance_slots[u][s][i_s][i][agg_s][1])])
                                 >= appliance_slots[u][s][i_s][i][agg_s][1] * y[u][s][i_s][i][agg_s]]
                            )
        print(len(sched_problem.getConstraint()))

        # VINCOLO 4
        sched_problem.addConstraint([xp.Sum([x[u][s][t]
                                             for k in range(len(appliance_intervals[u][s][i_s]))
                                             for t in range(appliance_intervals[u][s][i_s][k][0],
                                                            appliance_intervals[u][s][i_s][k][1] + 1)])
                                     == appliance_slots[u][s][i_s][0][0][1]
                                     for u in range(len(user_id))
                                     for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for i_s in range(len(appliance_intervals[u][s]))])
        print(len(sched_problem.getConstraint()))

        all_slots_user_appliance = {}
        for u in range(len(user_id)):
            for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1):
                all_slots_user_appliance[u, s] = set()
                for i_s in range(len(appliance_intervals[u][s])):
                    for k in range(len(appliance_intervals[u][s][i_s])):
                        for t in range(appliance_intervals[u][s][i_s][k][0], appliance_intervals[u][s][i_s][k][1] + 1):
                            all_slots_user_appliance[u, s].add(t)

        # VINCOLO 5
        sched_problem.addConstraint([x[u][s][t] == 0
                                     for u in range(len(user_id))
                                     for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for t in range(T) if t not in all_slots_user_appliance[u, s]])
        print(len(sched_problem.getConstraint()))

        sched_problem.addConstraint([x[u][s][t] == 0
                                     for u in range(len(user_id))
                                     for s in range(len(all_appliances)) if
                                     s not in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for t in range(T)])

        print(len(sched_problem.getConstraint()))
        sched_problem.addConstraint([y[u][s][i_s][i][agg_s] == 0
                                     for u in range(len(user_id))
                                     for s in range(len(all_appliances)) if
                                     s not in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for i_s in range(max_n_interval_sets)
                                     for i in range(max_n_intervals)
                                     for agg_s in range(max_n_slots)])
        print(len(sched_problem.getConstraint()))

        sched_problem.addConstraint([y[u][s][i_s][i][agg_s] == 0
                                     for u in range(len(user_id))
                                     for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for i_s in range(max_n_interval_sets) if i_s >= len(appliance_intervals[u][s])
                                     for i in range(max_n_intervals)
                                     for agg_s in range(max_n_slots)])
        print(len(sched_problem.getConstraint()))

        sched_problem.addConstraint([y[u][s][i_s][i][agg_s] == 0
                                     for u in range(len(user_id))
                                     for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for i_s in range(max_n_interval_sets) if i_s < len(appliance_intervals[u][s])
                                     for i in range(max_n_intervals) if i >= len(appliance_intervals[u][s][i_s])
                                     for agg_s in range(max_n_slots)])
        print(len(sched_problem.getConstraint()))

        sched_problem.addConstraint([y[u][s][i_s][i][agg_s] == 0
                                     for u in range(len(user_id))
                                     for s in range(min(appliance_slots[u].keys()), max(appliance_slots[u].keys()) + 1)
                                     for i_s in range(max_n_interval_sets) if i_s < len(appliance_intervals[u][s])
                                     for i in range(max_n_intervals) if i < len(appliance_intervals[u][s][i_s])
                                     for agg_s in range(max_n_slots) if agg_s >= len(appliance_slots[u][s][i_s][i])])
        print(len(sched_problem.getConstraint()))

    # TIMELIMIT	The maximum time in seconds that the Optimizer will run before it terminates.
    # SOLTIMELIMIT	A soft limit (in seconds) on runtime for a MIP solve. The solver stops whenever a feasible MIP solution is found and the runtime exceeds the specified value.
    # MAXNODES	The maximum number of nodes that will be explored by branch and bound.
    # MIPRELSTOP	Stop the MIP search when the gap (difference between the best solution's objective function and the current best solution bound) becomes less than the specified percentage.
    # MIPABSSTOP	Stop the MIP search when the gap (difference between the best solution's objective function and the current best solution bound) becomes less than the specified absolution value.
    # sched_problem.controls.soltimelimit=100
    if N_BESS == 0 and active_scheduler:
        sched_problem.controls.MIPRELSTOP = 0.01
    sched_problem.write(f"{res_dir}_problem", "lps")
    sched_problem.solve()

    df = pandas.DataFrame()
    df["Produced"] = [x * D_T for x in production]
    df['Load'] = [sched_problem.getSolution(f"P_load_{t}") * D_T for t in range(T)]
    df['BESS Discharge'] = [sched_problem.getSolution(f"p_dis_{t}") * D_T for t in range(T)]
    df['BESS Charge'] = [sched_problem.getSolution(f"p_ch_{t}") * D_T for t in range(T)]
    df['BESS SoC'] = [sched_problem.getSolution(f"SoC_{t}") * D_T for t in range(T)]
    df['Shared'] = [sched_problem.getSolution(f"p_shared_{t}") * D_T for t in range(T)]
    df["Injected"] = [sched_problem.getSolution(f"p_inj_{t}") * D_T for t in range(T)]
    df["Withdrawn"] = [sched_problem.getSolution(f"p_grid_{t}") * D_T for t in range(T)]
    df['Gain'] = [l_sell[t] * (1 - alpha) * df['Injected'][t] + (inc[t] + avoided_loss) * df['Shared'][t]
                  - l_buy[t] * df['Withdrawn'][t] for t in range(T)]

    total_gain = sum(list(df['Gain']))
    total_load = sum(list(df['Load'])) + sum(list(df['BESS Charge']))
    print(f'USER LOAD {sum(list(df["Load"]))} + BESS CHARGE {sum(list(df["BESS Charge"]))} = {total_load}')

    format_DASP(res_dir, sched_problem, user_id, user_appliances, all_appliances, user_int_appliances,
                all_int_appliances, T, day.value)

    show_dasp(df, res_dir, f"{cur_date}", "Time [h]", ["Energy [Wh]"] * (len(df.columns) - 1) + ["Value [€/Wh]"])

    scr = (sum(list(df['Shared'])) / sum(list(production))) * 100
    ssr = (sum(list(df['Shared'])) / total_load) * 100

    print(f"BESS: {N_BESS}")
    print(f"GAIN: {total_gain}")
    print(f"SELF-CONSUMPTION: {scr}")
    print(f"SELF-SUFFICIENCY: {ssr}")

    return total_gain, scr, ssr, np.min(list(df['Load'])), np.max(list(df['Load']))

class DASP():
    def __init__(self, weekday, users, activations, bess_activations):
        self.weekday = weekday
        self.users = users
        self.activations = activations
        self.bess_activations = bess_activations

    def to_json(self):
        activations_dict = {}
        for u in self.users:
            activations_dict[u] = []

        for a in self.activations:
            activations_dict[a.user].append({'appliance': a.appliance, 'interval': a.intervals})

        bess_activations_dict = {}
        for a in self.bess_activations:
            if a.id not in bess_activations_dict.keys():
                bess_activations_dict[a.id] = []
            bess_activations_dict[a.id].append({'at': a.at, 'amount': a.amount})

        return {
            'weekday': self.weekday,
            'activations': activations_dict,
            'bess_activations': bess_activations_dict
        }


class Activation():
    def __init__(self, user, appliance, intervals):
        self.user = user
        self.appliance = appliance
        self.intervals = intervals


class BESS_Activation():
    def __init__(self, id, at, amount):
        self.id = id
        self.at = at
        self.amount = amount


class Interval():
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def to_json(self):
        if self.start != self.end:
            return {
                'start': self.start,
                'end': self.end
            }
        else:
            return {
                'at': self.start
            }


def find_intervals(sequence):
    subsequences = []
    intervals = []
    start = 0
    for i in range(1, len(sequence)):
        # Check for discontinuity (difference greater than 1)
        if sequence[i] - sequence[i - 1] > 1:
            subsequences.append((start, i - 1))  # Add previous subsequence
            start = i  # Update start for next subsequence
    # Add the last subsequence if it exists
    if start < len(sequence):
        subsequences.append((start, len(sequence) - 1))

    for start, end in subsequences:
        intervals.append(Interval(int(sequence[start]), int(sequence[end])))

    return intervals


def format_DASP(dest, sched_problem, user_id, user_appliances, all_appliances, user_int_appliances,
                all_int_appliances, T, day):
    activations = []

    if len(user_appliances) > 0:
        for u in range(len(user_id)):
            u_appliances = user_appliances[u]
            for s in range(len(all_appliances)):
                a_name = all_appliances[s][0]
                if a_name in u_appliances:
                    sequence = []
                    for t in range(T):
                        if sched_problem.getSolution(f"{u}-{s}-{t}") > 0:
                            sequence.append(t)
                    if len(sequence) > 0:
                        activations.append(Activation(user_id[u], a_name.split('- ')[1], find_intervals(sequence)))

            u_appliances = user_int_appliances[u]
            for k in range(0, len(all_int_appliances)):
                a_name = all_int_appliances[k][0]
                if a_name in u_appliances:
                    sequence = []
                    for t in range(T):
                        if sched_problem.getSolution(f"shint-{u}-{k}-{t}") > 0:
                            sequence.append(t)
                    if len(sequence) > 0:
                        print(f"{a_name} in {sequence}")
                        activations.append(Activation(user_id[u], a_name.split("- ")[1], find_intervals(sequence)))

    bess_activations = []
    for t in range(T):
        amount = sched_problem.getSolution(f"p_dis_{t}")
        if amount > 0:
            bess_activations.append(BESS_Activation('BESS', t, f"-{amount} W"))
        amount = sched_problem.getSolution(f"p_ch_{t}")
        if amount > 0:
            bess_activations.append(BESS_Activation('BESS', t, f"+{amount} W"))

    dasp = DASP(weekday=day, users=user_id, activations=activations, bess_activations=bess_activations)
    manage_directory(f"DASP{os.sep}{dest}{os.sep}", delete_existing=False)
    with open(f'DASP{os.sep}{dest}{os.sep}{os.sep}{day}.json', 'w') as f:
        f.write(json.dumps(dasp, default=lambda o: o.to_json(), indent=4))


def simulate(bess_conf):
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

    bess_features = getBess(bess_conf)

    scenario = {
        #'expected': False,
         'real': True
    }

    scale_factors = [100, 75, 50, 25]

    for scale_factor in scale_factors:
        for s in scenario:
            for c in cases.keys():
                print(f'\n\nSIMULATING {s} {c}')
                source = f"p{scale_factor}_{s}{os.sep}{c}_results"
                print(f"SIMULATING -> {source}")
                manage_directory(source, delete_existing=True)
                #date = pd.to_datetime("2018-04-30", format='mixed')
                date = pd.to_datetime("2018-11-26", format='mixed')
                dates = []
                gain_l = []
                scr_l = []
                ssr_l = []
                min_l = []
                max_l = []

                for day in Weekday:
                    print(f"Schedule for {day}")
                    dates.append(date.strftime("%Y-%m-%d"))
                    production = get_production(date, expected=scenario[s])

                    tariffs = get_tariffs(date)

                    preferences_dir = 'preferences' if cases[c]['SCHED'] else 'actual_behavior'
                    gain, scr, ssr, min_load, max_load = produce_DASP(day, date, preferences_dir, production,
                                                                      scale_factor,
                                                                      tariffs, cases[c]['BESS'], bess_features, source)
                    gain_l.append(round(gain, 2))
                    scr_l.append(round(scr, 2))
                    ssr_l.append(round(ssr, 2))

                    min_l.append(round(min_load, 2))
                    max_l.append(round(max_load, 2))
                    date = date + timedelta(days=1)
                    print("*" * 100)

                pd.DataFrame(
                    {'DAYS': [x.value for x in Weekday], 'DATE': dates, 'GAIN': gain_l, 'SCR': scr_l, 'SSR': ssr_l,
                     "MIN LOAD": min_l, "MAX LOAD": max_l}) \
                    .to_csv(f'{source}{os.sep}simulation_results.csv', sep=';', index=False)


def BESS_dimensioning():
    scenarios = {
        'BESS Only': {
            'SCHED': False
        },
        'Combined': {
            'SCHED': True
        }
    }

    cases = pd.read_csv("BESS configurations.csv", sep=',', index_col=0).sort_values(
        by=['capacity', 'dis_power', 'ch_power'],
        inplace=False).to_dict(orient='index')

    print(cases)
    scale_factors = [100, 75, 50, 25]

    for scale_factor in scale_factors:
        for scenario in scenarios:
            for c in cases.keys():
                #date = pd.to_datetime("2018-04-30", format='mixed')
                date = pd.to_datetime("2018-11-26", format='mixed')
                dates = []
                gain_l = []
                scr_l = []
                ssr_l = []
                min_l = []
                max_l = []
                source = f"p{scale_factor}_{scenario}{os.sep}{c}_results"
                manage_directory(source, delete_existing=True)
                for day in Weekday:
                    print(f"SCENARIO: p{scale_factor} {scenario} {c} {day}")
                    dates.append(date.strftime("%Y-%m-%d"))
                    print(f"SIMULATING {scenario} {c} {day}")
                    production = get_production(date, expected=False)
                    tariffs = get_tariffs(date)

                    preferences_dir = 'preferences' if scenarios[scenario]['SCHED'] else 'actual_behavior'
                    gain, scr, ssr, min_load, max_load = produce_DASP(day, date, preferences_dir, production,
                                                                      scale_factor,
                                                                      tariffs, True, cases[c], source)

                    gain_l.append(round(gain, 2))
                    scr_l.append(round(scr, 2))
                    ssr_l.append(round(ssr, 2))

                    min_l.append(round(min_load, 2))
                    max_l.append(round(max_load, 2))

                    date = date + timedelta(days=1)
                    print(f"*" * 100)

                pd.DataFrame(
                    {'DAYS': [x.value for x in Weekday], 'DATE': dates, 'GAIN': gain_l, 'SCR': scr_l, 'SSR': ssr_l,
                     "MIN LOAD": min_l, "MAX LOAD": max_l}) \
                    .to_csv(f'{source}{os.sep}simulation_results.csv', sep=';', index=False)


def getBess(conf):
    cases = pd.read_csv("BESS configurations.csv", sep=',', index_col=0).sort_values(
        by=['capacity', 'dis_power', 'ch_power'],
        inplace=False).to_dict(orient='index')
    return cases[conf]


def scale(production, exp_peak, scale_factor=100):
    peak = max(production)
    print(f"PEAK: {peak}")
    print(f"EXP_PEAK: {exp_peak}")
    scaled = [((x * exp_peak) / peak) * (scale_factor / 100) for x in production]
    print(f"NEW PEAK: {max(scaled)}")
    return scaled


if __name__ == '__main__':
    #BESS_dimensioning()
    bess_conf = 'A1'
    simulate(bess_conf)
