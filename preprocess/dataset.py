import pickle
import pandas as pd
import torch

ccs9 = pickle.load(open('../ccs9.pkl','rb'))

df = pd.read_csv('ICD9CM_to_CCSCM.csv')
c1 = df['ICD9CM'].tolist()
c2 = df['CCSCM'].tolist()

d_to_ccs9 = {}
for diagnose, ccs in zip(c1,c2):
    d_to_ccs9[diagnose] = ccs

def diagnose_to_ccs(diagnose):
    if diagnose.startswith('V'):
        icd9 = diagnose[:3]
        if len(diagnose) > 3:
            icd9 = icd9 + '.' + diagnose[3:]
    elif diagnose.startswith('E'):
        icd9 = diagnose[:4]
        if len(diagnose) > 4:
            icd9 = icd9 + '.' + diagnose[4:]
    else:
        icd9 = diagnose[:3]
        if len(diagnose) > 3:
            icd9 = icd9 + '.' + diagnose[3:]

    if icd9 not in d_to_ccs9:
        print(icd9)
        return ''
    else:
        return d_to_ccs9[icd9]

with open('mimic3_0.5.pkl', 'rb') as f:
    mimic = pickle.load(f)


df = pd.read_csv('CCSCM.csv')
ccs9 = df['code'].astype(str).tolist()


visit2index = dict()
diagnose2index = dict()
ccs2index = dict()


trainset = torch.load('trainset.pt')
validset = torch.load('validset.pt')
testset = torch.load('testset.pt')

train_visits = [mimic.samples[i] for i in trainset]
valid_visits = [mimic.samples[i] for i in validset]
test_visits = [mimic.samples[i] for i in testset]

diagnose_index_list = mimic.get_all_tokens('cond_hist')
ccs_index_list = ccs9

print('unique diagnoses:', len(diagnose_index_list))
print('unique ccs:', len(ccs_index_list))
print('unique ccs:', len(ccs9))


with open('visits.txt', 'w') as f:
    f.write('org_id\tremap_id\n')
    visit_index = 0
    for visit in train_visits:
        visit2index[visit['visit_id']] = visit_index
        f.write(visit['visit_id'] + '\t' + str(visit_index) + '\n')
        visit_index += 1
    for visit in valid_visits:
        visit2index[visit['visit_id']] = visit_index
        f.write(visit['visit_id'] + '\t' + str(visit_index) + '\n')
        visit_index += 1
    for visit in test_visits:
        visit2index[visit['visit_id']] = visit_index
        f.write(visit['visit_id'] + '\t' + str(visit_index) + '\n')
        visit_index += 1

with open('ccss.txt', 'w') as f:
    f.write('org_id\tremap_id\n')
    ccs_index = 0

    for i, ccs in enumerate(ccs9):
        ccs2index[str(ccs)] = ccs_index
        f.write(str(ccs) + '\t' + str(ccs_index) + '\n')
        ccs_index += 1

with open('icds.txt', 'w') as f:
    f.write('org_id\tremap_id\n')
    icd_index = 0
    for i, diagnose in enumerate(diagnose_index_list):
        diagnose2index[diagnose] = icd_index
        f.write(diagnose + '\t' + str(icd_index) + '\n')
        icd_index += 1

ccs2icd_set = set()
visit2icd_set = set()
visit_ccs_icd_set = set()

for visit in train_visits:

    for diagnose_list in visit['cond_hist']:
        for diagnose in diagnose_list:
            ccs = str(diagnose_to_ccs(diagnose))
            if ccs == '':
                continue
            ccs2icd_set.add((ccs2index[ccs], diagnose2index[diagnose]))
            visit2icd_set.add((visit2index[visit['visit_id']], diagnose2index[diagnose]))
            visit_ccs_icd_set.add((visit2index[visit['visit_id']], ccs2index[ccs], diagnose2index[diagnose]))
with open('visit2icd.txt', 'w') as f:
    f.write('visitID\ticdID\n')
    for visit_icd_pair in visit2icd_set:
        f.write(str(visit_icd_pair[0]) + '\t' + str(visit_icd_pair[1]) + '\n')
with open('ccs2icd.txt', 'w') as f:
    f.write('ccsID\ticdID\n')
    for ccs_icd_pair in ccs2icd_set:
        f.write(str(ccs_icd_pair[0]) + '\t' + str(ccs_icd_pair[1]) + '\n')

print(len(visit_ccs_icd_set))

with open('train.txt', 'w') as f:
    f.write('visitID\tccsID\ticdID\n')
    for visit_ccs_icd_pair in visit_ccs_icd_set:
        f.write(str(visit_ccs_icd_pair[0]) + '\t' + str(visit_ccs_icd_pair[1]) + '\t' + str(visit_ccs_icd_pair[2]) + '\n')

#valid
ccs2icd_set = set()
visit2icd_set = set()
visit_ccs_icd_set = set()

for visit in valid_visits:

    for diagnose_list in visit['cond_hist']:
        for diagnose in diagnose_list:
            ccs = str(diagnose_to_ccs(diagnose))
            if ccs == '':
                continue
            ccs2icd_set.add((ccs2index[ccs], diagnose2index[diagnose]))
            visit2icd_set.add((visit2index[visit['visit_id']], diagnose2index[diagnose]))
            visit_ccs_icd_set.add((visit2index[visit['visit_id']], ccs2index[ccs], diagnose2index[diagnose]))
with open('visit2icd_valid.txt', 'w') as f:
    f.write('visitID\ticdID\n')
    for visit_icd_pair in visit2icd_set:
        f.write(str(visit_icd_pair[0]) + '\t' + str(visit_icd_pair[1]) + '\n')

print(len(visit_ccs_icd_set))

with open('valid.txt', 'w') as f:
    f.write('visitID\tccsID\ticdID\n')
    for visit_ccs_icd_pair in visit_ccs_icd_set:
        f.write(str(visit_ccs_icd_pair[0]) + '\t' + str(visit_ccs_icd_pair[1]) + '\t' + str(visit_ccs_icd_pair[2]) + '\n')

#test
ccs2icd_set = set()
visit2icd_set = set()
visit_ccs_icd_set = set()

for visit in test_visits:

    for diagnose_list in visit['cond_hist']:
        for diagnose in diagnose_list:
            ccs = str(diagnose_to_ccs(diagnose))
            if ccs == '':
                continue
            ccs2icd_set.add((ccs2index[ccs], diagnose2index[diagnose]))
            visit2icd_set.add((visit2index[visit['visit_id']], diagnose2index[diagnose]))
            visit_ccs_icd_set.add((visit2index[visit['visit_id']], ccs2index[ccs], diagnose2index[diagnose]))
with open('visit2icd_test.txt', 'w') as f:
    f.write('visitID\ticdID\n')
    for visit_icd_pair in visit2icd_set:
        f.write(str(visit_icd_pair[0]) + '\t' + str(visit_icd_pair[1]) + '\n')

print(len(visit_ccs_icd_set))

with open('test.txt', 'w') as f:
    f.write('visitID\tccsID\ticdID\n')
    for visit_ccs_icd_pair in visit_ccs_icd_set:
        f.write(str(visit_ccs_icd_pair[0]) + '\t' + str(visit_ccs_icd_pair[1]) + '\t' + str(visit_ccs_icd_pair[2]) + '\n')
