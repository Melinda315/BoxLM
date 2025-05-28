from datetime import datetime
import pickle
import torch
import torch.nn as nn

def parse_datetimes(datetime_strings):
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M") for dt_str in datetime_strings]

def timedelta_to_str(tdelta):
    days = tdelta.days
    seconds = tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return days * 1440 + hours * 60 + minutes

def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)

    return [abs(timedelta_to_str(datetimes[-1]-datetimes[-2]))]

mimic = pickle.load(open('mimic3_box_dataset_0.5.pkl', 'rb'))

trainset = torch.load('trainset.pt')
validset = torch.load('validset.pt')
testset = torch.load('testset.pt')
train_visits = [mimic.samples[i] for i in trainset]
valid_visits = [mimic.samples[i] for i in validset]
test_visits = [mimic.samples[i] for i in testset]

time1 = []
for record in train_visits:
    x = convert_to_relative_time(record['adm_time'])
    time1 += x
for record in valid_visits:
    x = convert_to_relative_time(record['adm_time'])
    time1 += x
for record in test_visits:
    x = convert_to_relative_time(record['adm_time'])
    time1 += x

torch.save(time1, 'visit_time.pt')