from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch

def diagnose_norm(diagnose):
    if diagnose.startswith('V'):
        icd9 = diagnose[:3]
        if len(diagnose) > 3:
            icd9 = icd9 + '.' + diagnose[3:]
    elif diagnose.startswith('E'):
        icd9 = diagnose[:4]
        if len(diagnose) > 4:
            icd9 = icd9 + '.' + diagnose[4:]
    elif '-' in diagnose:
        icd9 = diagnose
    else:
        icd9 = diagnose[:3]
        if len(diagnose) > 3:
            icd9 = icd9 + '.' + diagnose[3:]
    return icd9

df = pd.read_csv('ICD9CM.csv')
c1 = df['code'].tolist()
c2 = df['name'].tolist()
icd_name = {}
for icd, name in zip(c1,c2):
    icd_name[icd] = name


icd_name_list = []
with open('icds.txt') as f:
    icds = f.readlines()[1:]
    for icd in icds:
        icd = icd.split('\t')
        icd = icd[0]

        if icd in icd_name:
            name = icd_name[icd]
            icd_name_list.append(name)
        elif diagnose_norm(icd) in icd_name:
            icd = diagnose_norm(icd)
            name = icd_name[icd]
            icd_name_list.append(name)
        else:
            if len(icd) > 3:
                icd = icd[:3]+'.'+icd[3:]
            name = icd_name[icd]
            icd_name_list.append(name)


df2 = pd.read_csv('CCSCM.csv')
c12 = df2['code'].tolist()
c22 = df2['name'].tolist()
ccs_name = {}
for ccs, name in zip(c12,c22):
    ccs_name[ccs] = name

ccs_name_list = []
with open('ccss.txt') as f:
    ccss = f.readlines()[1:]
    for ccs in ccss:
        ccs = ccs.split('\t')
        ccs = ccs[0]

        if ccs in ccs_name:
            name = ccs_name[ccs]
            ccs_name_list.append(name)
        else:
            ccs = int(ccs)
            name = ccs_name[ccs]
            ccs_name_list.append(name)


tokenizer = AutoTokenizer.from_pretrained("../biobert")
model = AutoModel.from_pretrained("../biobert")


model.eval()

def get_sentence_embeddings(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[:, 0, :].detach().numpy()
    return cls_embedding

ccs_embeddings = get_sentence_embeddings(ccs_name_list)
torch.save(ccs_embeddings, 'ccs_embeddings.pt')

icd_embeddings = get_sentence_embeddings(icd_name_list)
torch.save(icd_embeddings, 'icd_embeddings.pt')