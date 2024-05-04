import os
import random
import json
import uuid
from enum import Enum
import numpy as np
import pandas as pd
# from old_version.data_import import read_user_prefs_complete
from utils import manage_directory, randomize_start_end


class Weekday(Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"

    def to_json(self):
        return self.value


class Appliance():
    def __init__(self, class_name, model, nominal_power_W):
        self.class_name = class_name
        self.model = model
        self.nominal_power_W = nominal_power_W

    def get_ID(self):
        return self.class_name + " " + self.model


class NSH_Appliance(Appliance):
    def __init__(self, class_name, model, nominal_power_W):
        super().__init__(class_name, model, nominal_power_W)
        self.type = 'NSH'

    def to_json(self):
        return {
            'class_name': self.class_name,
            'model': self.model,
            'nominal_power_W': self.nominal_power_W,
            'type': self.type
        }


class SH_Appliance(Appliance):
    def __init__(self, class_name, model, nominal_power_W, duration):
        super().__init__(class_name, model, nominal_power_W)
        self.type = 'SH'
        self.duration = duration

    def to_json(self):
        return {
            'class_name': self.class_name,
            'model': self.model,
            'nominal_power_W': self.nominal_power_W,
            'type': self.type,
            'duration': self.duration
        }


class SI_Appliance(Appliance):
    def __init__(self, class_name, model, nominal_power_W, min_uptime):
        super().__init__(class_name, model, nominal_power_W)
        self.type = 'SI'
        self.min_uptime = min_uptime

    def to_json(self):
        return {
            'class_name': self.class_name,
            'model': self.model,
            'nominal_power_W': self.nominal_power_W,
            'type': self.type,
            'min_uptime': self.min_uptime
        }


class Refrigerator(NSH_Appliance):
    def __init__(self):
        domestic_refrigerators = {
            'Haier HRF-342D6AWB': 120,  # Assuming a medium-sized model
            'Samsung RT25H5955BS': 100,  # Assuming a mid-range refrigerator
            'LG LFXS2699SW': 130,  # Assuming a large-capacity model
            'Whirlpool WRF535SMBHD': 110,  # Assuming a standard refrigerator
            'Bosch KAN93NXUSD': 180,  # Assuming a high-end model with extra features
            'Electrolux EI32BE300': 90,  # Assuming a compact refrigerator
            'Kenmore 73042': 140,  # Placeholder - Replace with a specific Kenmore model
            'Maytag MFI2278VR': 125,  # Placeholder - Replace with a specific Maytag model
            'GE GWE0935SMSS': 80,  # Assuming an energy-efficient model
            'KitchenAid KFIS20ET': 150  # Assuming a counter-depth refrigerator
        }

        model = random.choice(list(domestic_refrigerators.keys()))
        npw = domestic_refrigerators[model]
        super().__init__("Refrigerator", model, npw)

    def to_json(self):
        return super().to_json()


class Television(NSH_Appliance):
    def __init__(self):
        domestic_Televisions = {
            'Samsung QN85B': 150,  # Assuming a 55-inch QLED model
            'LG C2': 120,  # Assuming a 65-inch OLED model
            'Sony X90K': 100,  # Assuming a 50-inch LED model
            'TCL 6-Series': 80,  # Assuming a 55-inch LED model
            'Hisense U8H': 70,  # Assuming a 50-inch LED model
            'Vizio M-Series': 60,  # Assuming a 43-inch LED model
            'Toshiba M5500': 50,  # Assuming a 43-inch LED model
            'Insignia NS-55F301NA21': 45,  # Assuming a 55-inch LED model
            'Element ELWD4317': 40,  # Assuming a 43-inch LED model
            'Sceptre S515WV-F': 35  # Assuming a 32-inch LED model
        }

        model = random.choice(list(domestic_Televisions.keys()))
        npw = domestic_Televisions[model]
        super().__init__(f"Television", model, npw)

    def to_json(self):
        return super().to_json()


class Laptop(NSH_Appliance):
    def __init__(self):
        laptops = {
            'Apple MacBook Air M1': 30,  # Assuming a low-power model
            'Dell XPS 13': 45,  # Assuming a mid-range ultrabook
            'Lenovo ThinkPad X1 Carbon': 50,  # Assuming a business-oriented laptop
            'Microsoft Surface Laptop 4': 35,  # Assuming a productivity-focused model
            'Asus ROG Zephyrus G14': 150,  # Assuming a high-performance gaming laptop
            'MSI Creator 15': 120,  # Assuming a content creation-oriented laptop
            'Razer Blade 15': 170,  # Assuming a powerful gaming laptop
            'HP Spectre x360': 40,  # Assuming a convertible laptop
            'Acer Swift 3': 25,  # Assuming a budget-friendly option
            'Samsung Galaxy Book Pro': 38  # Assuming a productivity-focused model
        }

        model = random.choice(list(laptops.keys()))
        npw = laptops[model]
        super().__init__(f"Laptop", model, npw)

    def to_json(self):
        return super().to_json()


class Oven(SH_Appliance):
    def __init__(self):
        domestic_ovens = {
            'Whirlpool WOC5ER150': 2200,  # Assuming a standard electric oven
            'Samsung NE59N5950SS': 1800,  # Assuming a mid-range electric oven
            'LG LREL6386D': 2400,  # Assuming a large electric oven with convection
            'KitchenAid KODE500ESS': 2000,  # Assuming a high-end wall oven
            'Bosch HBN3551UC': 2300,  # Assuming a self-cleaning electric oven
            'Electrolux EI30BF01': 1500,  # Assuming a countertop convection oven
            'Kenmore Elite 50363': 2800,  # Placeholder - Replace with a specific Kenmore model
            'Maytag MEC8800FZ': 2000,  # Placeholder - Replace with a specific Maytag model
            'GE Cafe CGB715SKSS': 2500,  # Assuming a high-powered double oven
            'Black decker TO3217SS': 1200  # Assuming a smaller toaster oven
        }
        model = random.choice(list(domestic_ovens.keys()))
        npw = domestic_ovens[model]
        duration = random.choices([1, 2], [95, 5])[0]
        super().__init__(f"Oven", model, npw, duration)

    def to_json(self):
        return super().to_json()


class Dishwasher(SH_Appliance):
    def __init__(self):
        dishwashers = {
            'Bosch SMS40E08EU': 1950,  # Assuming a high-power European model
            'Whirlpool WDT730PAHS': 1800,  # Assuming a standard American model
            'Samsung DW80M9550US': 1500,  # Assuming a water-efficient model
            'LG LDFN4542TM': 1900,  # Assuming a large-capacity model
            'Electrolux ESF6175PW': 1200,  # Assuming a compact model
            'KitchenAid KDFE104KP': 1700,  # Assuming a counter-depth model
            'Maytag MDB8959SKW': 1200,  # Placeholder - Replace with a specific Maytag model
            'Kenmore 66513': 1300,  # Placeholder - Replace with a specific Kenmore model
            'GE GTS5550SCS': 1200,  # Assuming a basic model
            'Haier HDI-2T52FSS': 1100  # Assuming an energy-efficient model
        }

        model = random.choice(list(dishwashers.keys()))
        npw = dishwashers[model]
        duration = random.choices([1, 2], [95, 5])[0]
        super().__init__(f"Dishwasher", model, npw, duration)

    def to_json(self):
        return super().to_json()


class Washing_Machine(SH_Appliance):
    def __init__(self):
        washing_machines = {
            'Samsung WF4&BBM9040AP': 1800,  # Assuming a high-capacity washer
            'LG WM3900HVA': 1500,  # Assuming a high-efficiency washer
            'Whirlpool WTW79ES7HZ': 1200,  # Assuming a standard washer
            'Maytag MHW6630HC': 1400,  # Placeholder - Replace with a specific Maytag model
            'Kenmore 79092': 1300,  # Placeholder - Replace with a specific Kenmore model
            'GE GTD42PASMVH': 1100,  # Assuming a basic washer
            'Electrolux EFLS617SI': 1000,  # Assuming a compact washer
            'Bosch WAT28PHGB': 900,  # Assuming a European washer with water-saving features
            'Haier HLC1700EMFW': 800,  # Assuming an energy-efficient model
            'KitchenAid KTLX1042AC': 1600  # Assuming a washer with extra features
        }

        model = random.choice(list(washing_machines.keys()))
        npw = washing_machines[model]
        duration = random.choices([1, 2], [90, 10])[0]
        super().__init__(f"Washing machine", model, npw, duration)

    def to_json(self):
        return super().to_json()


class Electric_Car(SI_Appliance):
    def __init__(self):
        electric_cars = {
            "Dacia Spring Electric 45": 26800,
            "Mini Cooper E": 32000,
            "Renault Twingo Electric": 21700,
            "Fiat 500e 3+1 24 kWh": 21300,
            "Mazda MX-30": 30000
        }
        model = random.choice(list(electric_cars.keys()))
        min_uptime = random.choices([3, 4, 5], [20, 50, 60])[0]
        npw = (0.8 * electric_cars[model]) / min_uptime
        super().__init__(f"Electric car", model, npw, min_uptime)

    def to_json(self):
        return super().to_json()


class Reacharcheable_Vacuum(SI_Appliance):
    def __init__(self):
        rechargeable_vacuums = {
            'Shark NV391': 120,  # Assuming a mid-range upright vacuum
            'Dyson V8 Absolute': 250,  # Assuming a high-powered cordless stick vacuum
            'BISSELL CrossWave Pet Pro': 500,  # Assuming a powerful wet/dry vacuum
            'Ecovacs DEEBOT Ozmo T8 AIVI': 200,  # Assuming a robotic vacuum cleaner
            'Samsung Jet Bot AI+': 180,  # Assuming a smart robotic vacuum
            'Roborock S7': 150,  # Assuming a mid-range robotic vacuum
            'Hoover ONEPWR Cordless Reach': 130,  # Assuming a cordless stick vacuum
            'Bissell PowerEdge Slim Lightweight': 80,  # Assuming a compact and lightweight vacuum
            'Black+Decker DUSTBUSTER Handheld': 60,  # Assuming a small handheld vacuum
            'Dirt Devil SB700 AccuCharge Cordless Stick': 100  # Assuming a basic cordless stick vacuum
        }

        model = random.choice(list(rechargeable_vacuums.keys()))
        npw = rechargeable_vacuums[model]
        min_uptime = random.choices([1, 2], [95, 5])[0]
        super().__init__(f"Vacuum", model, npw, min_uptime)

    def to_json(self):
        return super().to_json()


class User():
    def __init__(self, name, max_abs_power_W, appliances, profile):
        self.name = name
        self.max_abs_power_W = max_abs_power_W
        self.appliances = appliances
        self.profile = profile

    def to_json(self):
        return {
            'name': self.name,
            'max_abs_power_W': self.max_abs_power_W,
            'appliances': self.appliances,
            'profile': self.profile
        }

    def index_appliance_by_name(self):
        appliance_dict = {}
        for a in self.appliances:
            appliance_dict[a.get_ID()] = a
        return appliance_dict

    def index_appliance_by_class(self):
        appliance_dict = {}
        for a in self.appliances:
            appliance_dict[a.class_name] = a
        return appliance_dict


class Preference():
    def __init__(self, user, appliance, weekday, interval_sets):
        self.user = user
        self.appliance = appliance
        self.weekday = weekday
        self.interval_sets = interval_sets

    def to_json(self):
        return {
            'user': self.user,
            'appliance': self.appliance,
            'weekday': self.weekday,
            'interval_sets': self.interval_sets
        }


class Interval_Set():
    def __init__(self, interval_set):
        self.interval_set = interval_set
        if len(self.interval_set) > 0:
            self.interval_set = sorted(self.interval_set, key=lambda x: x.start)

    def to_json(self):
        return self.interval_set


class Interval():
    def __init__(self, start=0, end=23):
        self.start = start
        self.end = end

    def to_json(self):
        return {
            'start': self.start,
            'end': self.end
        }


class EarlyMorning(Interval):
    def __init__(self):
        super().__init__(0, 5)


class Morning(Interval):
    def __init__(self):
        super().__init__(6, 10)


class LateMorning(Interval):
    def __init__(self):
        super().__init__(11, 13)


class Afternoon(Interval):
    def __init__(self):
        super().__init__(14, 17)


class Evening(Interval):
    def __init__(self):
        super().__init__(18, 20)


class Nigth(Interval):
    def __init__(self):
        super().__init__(21, 23)


def decode_users():
    users = []
    for fname in os.listdir('users'):
        with open(f'users/{fname}', 'r') as f:
            users.append(User(**json.load(f)))
            f.close()

    for user in users:
        appliances = []
        for key in user.appliances:
            type = user.appliances[key].pop('type')
            if type == 'SH':
                appliances.append(SH_Appliance(**user.appliances[key]))
            elif type == 'NSH':
                appliances.append(NSH_Appliance(**user.appliances[key]))
            elif type == 'SI':
                appliances.append(SI_Appliance(**user.appliances[key]))

        user.appliances = appliances
    return users


def decode_preferences(day, source):
    users = decode_users()
    preferences = {}
    for u in users:
        preferences[u.name] = []
        with open(f'{source}{os.sep}{day.value}{os.sep}{u.name}.json', 'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                p = Preference(**data[i])
                interval_sets = []
                for interval_set in p.interval_sets:
                    intervals = []
                    for interval in interval_set:
                        intervals.append(Interval(**interval))
                    interval_sets.append(intervals)
                p.interval_sets = interval_sets
                preferences[u.name].append(p)
            f.close()
    return users, preferences


def format_preferences(users, preferences):
    T = 24
    user_id = []
    p_load_max = []
    p_load_nsh = []

    all_appliances = []
    appliances_counter = 0
    user_appliances = []
    all_interval_sets = []
    all_intervals = []
    appliance_intervals = []
    all_slots = []
    appliance_slots = []

    all_int_appliances = []
    int_counter = 0
    user_int_appliances = []
    all_int_interval_sets = []
    all_int_intervals = []
    int_appliance_intervals = []

    for u in range(len(users)):
        user_id.append(users[u].name)
        p_load_max.append(users[u].max_abs_power_W)
        p_load_nsh.append([0] * T)

        user_appliances.append([])
        appliance_intervals.append({})
        appliance_slots.append({})

        user_int_appliances.append([])
        int_appliance_intervals.append({})

        u_prefs = preferences[users[u].name]
        u_appliances = users[u].index_appliance_by_name()

        for p in range(len(u_prefs)):
            u_appliance = u_appliances[u_prefs[p].appliance]
            if u_appliance.type == 'SH':
                all_appliances.append(
                    (user_id[u] + " - " + u_appliance.get_ID(), u_appliance.nominal_power_W, u_appliance.duration))
                user_appliances[u].append(users[u].name + " - " + u_appliance.get_ID())

                for interval_set in u_prefs[p].interval_sets:
                    interval_set_temp = []
                    interval_set_slot_temp = []

                    for interval in interval_set:
                        all_intervals.append((interval.start, interval.end))
                        interval_set_temp.append((interval.start, interval.end))

                        agg_slot_temp = []
                        for agg_slot in range(interval.start, interval.end - u_appliance.duration + 2):
                            all_slots.append((agg_slot, u_appliance.duration))
                            agg_slot_temp.append((agg_slot, u_appliance.duration))

                        interval_set_slot_temp.append(agg_slot_temp)

                    if appliances_counter not in appliance_intervals[u]:
                        appliance_intervals[u][appliances_counter] = []
                    appliance_intervals[u][appliances_counter].append(interval_set_temp)

                    if appliances_counter not in appliance_slots[u]:
                        appliance_slots[u][appliances_counter] = []
                    appliance_slots[u][appliances_counter].append(interval_set_slot_temp)
                    all_interval_sets.append(interval_set_temp)
                appliances_counter += 1
            elif u_appliance.type == 'SI':
                all_int_appliances.append((user_id[u] + " - " + u_appliance.get_ID(),
                                           u_appliance.nominal_power_W,
                                           u_appliance.min_uptime))

                user_int_appliances[u].append(user_id[u] + " - " + u_appliance.get_ID())

                for interval_set in u_prefs[p].interval_sets:
                    interval_set_temp = []

                    for interval in interval_set:
                        all_int_intervals.append((interval.start, interval.end))
                        interval_set_temp.append((interval.start, interval.end))

                    if int_counter not in int_appliance_intervals[u]:
                        int_appliance_intervals[u][int_counter] = []
                    int_appliance_intervals[u][int_counter].append(interval_set_temp)
                    all_int_interval_sets.append(interval_set_temp)
                int_counter += 1
            elif u_appliance.type == 'NSH':
                for interval_set in u_prefs[p].interval_sets:
                    for interval in interval_set:
                        for i in range(interval.start, interval.end + 1):
                            p_load_nsh[u][i] += u_appliance.nominal_power_W

    return True, user_id, p_load_max, p_load_nsh, all_appliances, user_appliances, all_intervals, appliance_intervals, \
        all_slots, appliance_slots, all_interval_sets, all_int_appliances, int_counter, \
        user_int_appliances, all_int_interval_sets, all_int_intervals, int_appliance_intervals


def format_behavior(users, behavior):
    T = 24
    user_id = []
    p_load_max = []
    p_load_nsh = []

    for u in range(len(users)):
        user_id.append(users[u].name)
        p_load_max.append(users[u].max_abs_power_W)
        p_load_nsh.append([0] * T)

        u_behavior = behavior[users[u].name]
        u_appliances = users[u].index_appliance_by_name()
        for b in range(len(u_behavior)):
            for interval_set in u_behavior[b].interval_sets:
                for interval in interval_set:
                    for t in range(interval.start, interval.end + 1):
                        p_load_nsh[u][t] += u_appliances[u_behavior[b].appliance].nominal_power_W

    return False, user_id, p_load_max, p_load_nsh, [], [], [], [], [], [], [], [], 0, 0, [], [], []


def load_preferences(day, source='preferences'):
    users, preferences = decode_preferences(day, source)
    if source == 'preferences': return format_preferences(users, preferences)
    if source == 'actual_behavior': return format_behavior(users, preferences)


def test_generate_young_couple_profile(day):
    random.seed = 11
    appliances = {'refrigerator': Refrigerator(),
                  'dishwasher': Dishwasher(),
                  'washing_machine': Washing_Machine(),
                  'television': Television(),
                  'oven': Oven(),
                  'vacuum': Reacharcheable_Vacuum(),
                  'ev': Electric_Car()}

    user = User("Young Couple", max_abs_power_W=6, appliances=appliances, profile='Young Couple')

    prefs = []
    prefs.append(Preference(user.name, user.appliances['refrigerator'].get_ID(), day.value,
                            [Interval_Set([Interval(0, 23)])]))

    user.appliances['dishwasher'].duration = 1
    prefs.append(Preference(user.name, user.appliances['dishwasher'].get_ID(), day.value,
                            [Interval_Set([Interval(20, 22)])]))

    user.appliances['ev'].min_uptime = 3
    prefs.append(Preference(user.name, user.appliances['ev'].get_ID(), day.value,
                            [Interval_Set([Interval(00, 5), Interval(9, 13)])]))

    manage_directory(f"preferences/{day.value}", delete_existing=True)

    manage_directory(f"users", delete_existing=True)

    with open(f'users/{user.name}.json', 'w') as f:
        f.write(json.dumps(user, default=lambda o: o.to_json(), indent=4))

    with open(f'preferences/{day.value}/{user.name}.json', 'w') as f:
        f.write(json.dumps(prefs, default=lambda o: o.to_json(), indent=4))

    return user, prefs


# def test_scheduler_connector():
#     test_generate_young_couple_profile(Weekday.MONDAY)
#
#     user_id, p_load_max, p_load_nsh, all_appliances, user_appliances, all_intervals, appliance_intervals, \
#     all_slots, appliance_slots, all_interval_sets, all_int_appliances, \
#     int_counter, user_int_appliances, all_int_interval_sets, all_int_intervals, int_appliance_intervals, \
#     = load_preferences(Weekday.MONDAY)
#
#     current_output = (
#         f"user_id: {user_id}\n"
#         f"p_load_max: {p_load_max}\n"
#         f"p_load_nsh: {p_load_nsh}\n"
#         f"all_appliances: {all_appliances}\n"
#         f"user_appliances: {user_appliances}\n"
#         f"all_intervals: {all_intervals}\n"
#         f"appliance_intervals: {appliance_intervals}\n"
#         f"all_slots: {all_slots}\n"
#         f"appliance_slots: {appliance_slots}\n"
#         f"all_interval_sets: {all_interval_sets}\n"
#         f"all_int_appliances: {all_int_appliances}\n"
#         f"int_counter: {int_counter}\n"
#         f"user_int_appliances: {user_int_appliances}\n"
#         f"all_int_interval_sets: {all_int_interval_sets}\n"
#         f"all_int_intervals: {all_int_intervals}\n"
#         f"int_appliance_intervals: {int_appliance_intervals}"
#     )
#
#     user_id, p_load_max, p_load_nsh, all_appliances, user_appliances, all_intervals, appliance_intervals, \
#         all_slots, appliance_slots, all_interval_sets, all_int_appliances, \
#         int_counter, user_int_appliances, all_int_interval_sets, all_int_intervals, int_appliance_intervals, \
#         = read_user_prefs_complete('old_version/user_prefs/' + 'case_test.json', 24, 'Monday')
#
#     expected_output = (
#         f"user_id: {user_id}\n"
#         f"p_load_max: {p_load_max}\n"
#         f"p_load_nsh: {p_load_nsh}\n"
#         f"all_appliances: {all_appliances}\n"
#         f"user_appliances: {user_appliances}\n"
#         f"all_intervals: {all_intervals}\n"
#         f"appliance_intervals: {appliance_intervals}\n"
#         f"all_slots: {all_slots}\n"
#         f"appliance_slots: {appliance_slots}\n"
#         f"all_interval_sets: {all_interval_sets}\n"
#         f"all_int_appliances: {all_int_appliances}\n"
#         f"int_counter: {int_counter}\n"
#         f"user_int_appliances: {user_int_appliances}\n"
#         f"all_int_interval_sets: {all_int_interval_sets}\n"
#         f"all_int_intervals: {all_int_intervals}\n"
#         f"int_appliance_intervals: {int_appliance_intervals}"
#     )
#
#     compare(expected_output, current_output)

def compare(str1, str2):
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    # Find the shorter list to avoid extra iteration
    shorter_lines = min(lines1, lines2, key=len)
    longer_lines = lines1 if len(lines1) > len(lines2) else lines2

    # Compare lines up to the length of the shorter list
    for i in range(len(shorter_lines)):
        if lines1[i] != lines2[i]:
            print(f"Line {i + 1}:")
            print(f"  Exp 1: {lines1[i]}")
            print(f"  Cur 2: {lines2[i]}")

    if len(longer_lines) > len(shorter_lines):
        for i in range(len(shorter_lines), len(longer_lines)):
            print(f"Line {i + 1} (present only in String {longer_lines[0][:2]}): {longer_lines[i]}")


def get_appliance(a_type):
    if a_type == "Refrigerator":
        return Refrigerator()
    if a_type == "Dishwasher":
        a = Dishwasher()
        # a.duration = random.choices([1, 2], [95, 5])[0]
        return a
    if a_type == "Washing machine":
        a = Washing_Machine()
        # a.duration = random.choices([1, 2], [90, 10])[0]
        return a
    if a_type == "Oven":
        a = Oven()
        # a.duration = random.choices([1, 2], [95, 5])[0]
        return a
    if a_type == "Vacuum":
        a = Reacharcheable_Vacuum()
        # a.min_uptime = random.choices([1, 2], [95, 5])[0]
        return a
    if a_type == "Electric car":
        a = Electric_Car()
        # a.min_uptime = random.choices([3, 4, 5], [30, 60, 30])[0]
        return a
    if a_type == "Television":
        return Television()
    if a_type == "Laptop":
        return Laptop()


time_slots = ['EARLY MORNING', 'MORNING', 'LATE MORNING', 'AFTERNOON', 'EVENING', 'NIGTH']


def get_interval_OR(prob):
    n = np.count_nonzero(prob)
    selection = random.choices(time_slots, prob, k=n)
    selection = list(np.unique(selection))

    intervals = []
    for i in range(len(selection)):
        if selection[i] == 'EARLY MORNING': intervals.append(EarlyMorning())
        if selection[i] == 'MORNING': intervals.append(Morning())
        if selection[i] == 'LATE MORNING': intervals.append(LateMorning())
        if selection[i] == 'AFTERNOON': intervals.append(Afternoon())
        if selection[i] == 'EVENING': intervals.append(Evening())
        if selection[i] == 'NIGTH': intervals.append(Nigth())
    return Interval_Set(intervals)


def get_interval_AND(i_name, prob):
    if prob == 0: return None
    activate = random.choices([True, False], [prob, 100 - prob])
    if activate:
        if i_name == 'EARLY MORNING': return EarlyMorning()
        if i_name == 'MORNING': return Morning()
        if i_name == 'LATE MORNING': return LateMorning()
        if i_name == 'AFTERNOON': return Afternoon()
        if i_name == 'EVENING': return Evening()
        if i_name == 'NIGTH': return Nigth()


def generate_daily_preferences(user, day):
    source_dir = 'consumption_profile'
    u_appliances = user.index_appliance_by_class()

    profile = pd.read_csv(f"{source_dir}{os.sep}{user.profile}.csv", sep=';')
    if day.value is not (Weekday.SUNDAY.value or Weekday.SATURDAY.value):
        profile = profile[profile['WEEKDAY'] == 1]
    else:
        profile = profile[profile['WEEKDAY'] == 0]

    prefs = []
    for i in profile.index:
        is_active = random.choices([True, False], [profile['ACTIVATION'][i], 100 - profile['ACTIVATION'][i]])
        if is_active:
            interval_sets = []
            if profile['INTERPRETATION'][i] == 'OR':
                interval_set = get_interval_OR([profile[x][i] for x in time_slots])
                if interval_set is not None:
                    interval_sets.append(interval_set)
            elif profile['INTERPRETATION'][i] == 'AND':
                for i_name in time_slots:
                    interval = get_interval_AND(i_name, profile[i_name][i])
                    if interval is not None:
                        interval_sets.append(Interval_Set([interval]))

            prefs.append(
                Preference(user.name, u_appliances[profile['DEVICE'][i].capitalize()].get_ID(), day.value,
                           interval_sets))

    with open(f'preferences{os.sep}{day.value}{os.sep}{user.name}.json', 'w') as f:
        daily_prefs = [x for x in prefs if x.weekday == day.value]
        f.write(json.dumps(daily_prefs, default=lambda o: o.to_json(), indent=4))


def generate_users(n):
    source_dir = 'consumption_profile'
    manage_directory(f"users", delete_existing=True)
    user_per_profile = n
    for i in range(user_per_profile):
        for fname in os.listdir(source_dir):
            profile_name = fname.split('.csv')[0]
            profile = pd.read_csv(f"{source_dir}{os.sep}{fname}", sep=';')
            appliance_types = np.unique(profile['DEVICE'])
            appliances = {}
            max_abs_power_W = 6000
            for a_type in appliance_types:
                a = get_appliance(a_type.capitalize())
                appliances[a_type.capitalize()] = a
                if a_type == "ELECTRIC CAR":
                    max_abs_power_W += 2000
                    base = round(a.nominal_power_W / 1000) * 1000
                    if base >= max_abs_power_W:
                        max_abs_power_W = base + 2000

            user = User(f"{profile_name} {uuid.uuid4()}", max_abs_power_W=max_abs_power_W, appliances=appliances,
                        profile=profile_name)
            with open(f'users{os.sep}{user.name}.json', 'w') as f:
                f.write(json.dumps(user, default=lambda o: o.to_json(), indent=4))


def generate_daily_actual_behavior(user, day):
    preferences = list(decode_user_preferences(day, 'preferences', user).values())[0]
    u_appliances = user.index_appliance_by_name()

    daily_behavior = []
    for pref in preferences:
        app = u_appliances[pref.appliance]
        b_interval_sets = []
        for interval_set in pref.interval_sets:
            selected = random.choice(interval_set)
            duration = None
            if type(app) is not NSH_Appliance:
                if type(app) is SH_Appliance:
                    duration = app.duration
                if type(app) is SI_Appliance:
                    i_len = selected.end - selected.start
                    duration = random.choice(range(app.min_uptime, i_len)) if app.min_uptime < i_len else app.min_uptime
                start, end = randomize_start_end(selected.start, selected.end, duration)
                selected.start = start
                selected.end = end
            b_interval_sets.append(Interval_Set([selected]))
        daily_behavior.append(Preference(user.name, app.get_ID(), day.value, b_interval_sets))

    with open(f'actual_behavior{os.sep}{day.value}{os.sep}{user.name}.json', 'w') as f:
        daily_prefs = [x for x in daily_behavior if x.weekday == day.value]
        f.write(json.dumps(daily_prefs, default=lambda o: o.to_json(), indent=4))


def generate_preferences(n):
    generate_users(n)
    users = decode_users()
    for day in Weekday:
        manage_directory(f"preferences{os.sep}{day.value}", delete_existing=True)
        print(f"Generates preferences for {day.value}")
        for user in users:
            generate_daily_preferences(user, day)


def generate_actual_behavior():
    users = decode_users()
    for day in Weekday:
        manage_directory(f"actual_behavior{os.sep}{day.value}", delete_existing=True)
        print(f"Generates actual behavior for {day.value}")
        for user in users:
            generate_daily_actual_behavior(user, day)
            attempts = 0
            limit = 50
            feasibility = False
            while attempts < limit and feasibility is False:
                feasibility = check_single_user_feasibility(user, day)
                if feasibility is False:
                    generate_daily_actual_behavior(user, day)
                    attempts += 1
                    if attempts > limit:
                        print(f"Try {attempts}: No feasile solution found :(. Try again...")


def decode_user_preferences(day, source, user):
    preferences = {}
    preferences[user.name] = []
    with open(f'{source}{os.sep}{day.value}{os.sep}{user.name}.json', 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            p = Preference(**data[i])
            interval_sets = []
            for interval_set in p.interval_sets:
                intervals = []
                for interval in interval_set:
                    intervals.append(Interval(**interval))
                interval_sets.append(intervals)
            p.interval_sets = interval_sets
            preferences[user.name].append(p)
        f.close()
    return preferences


def check_single_user_feasibility(user, day):
    source = 'actual_behavior'
    T = 24
    preferences = decode_user_preferences(day, source, user)
    u_appliances = user.index_appliance_by_name()

    feasible = True
    for t in range(T):
        load_t = 0
        for pref in preferences[user.name]:
            for interval_set in pref.interval_sets:
                for interval in interval_set:
                    if interval.start <= t <= interval.end:
                        load_t = load_t + u_appliances[pref.appliance].nominal_power_W

        # if day == Weekday.MONDAY:
        #     print(f"{user.name} Load({t}) = {load_t} <= {user.max_abs_power_W} ? {load_t <= user.max_abs_power_W}")
        if load_t > user.max_abs_power_W:
            feasible = False
            break

    if feasible:
        print(f"{user.name} - Feasible behavior :)")
    else:
        print(f"{user.name} - No feasible behavior :(")
    return feasible


def decode_DASP(day, source):
    dasp = {}
    T = 24
    users = decode_users()
    with open(f'{source}{os.sep}{day.value}.json', 'r') as f:
        data = json.load(f)
        for user in users:
            u_appliances = user.index_appliance_by_name()
            consumption = [0] * T
            d = data['activations'][user.name]
            for i in range(len(d)):
                if d[i]['appliance'] in u_appliances.keys():
                    npw = u_appliances[d[i]['appliance']].nominal_power_W / 1000
                    for j in range(len(d[i]['interval'])):
                        if 'at' in d[i]['interval'][j]:
                            consumption[d[i]['interval'][j]['at']] += npw
                        else:
                            for k in range(d[i]['interval'][j]['start'], d[i]['interval'][j]['end'] + 1):
                                consumption[k] += npw
            dasp[user.name] = consumption
    f.close()

    fixed_load = {}
    source = f'preferences'
    for user in users:
        u_appliances = user.index_appliance_by_name()
        consumption = [0] * T
        preferences = decode_user_preferences(day, source, user)
        for pref in preferences[user.name]:
            a = u_appliances[pref.appliance]
            if type(a) is NSH_Appliance:
                for interval_set in pref.interval_sets:
                    for interval in interval_set:
                        for t in range(interval.start, interval.end):
                            consumption[t] += a.nominal_power_W / 1000
        fixed_load[user.name] = consumption

    return dasp, fixed_load


if __name__ == '__main__':
    n = 7  # users per profiles
    generate_preferences(n)
    generate_actual_behavior()
    
