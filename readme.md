# BoxLM

This repository contains a implementation of our "BoxLM : Unifying Structures and Semantics of Medical Concepts for Diagnosis Prediction in Healthcare" accepted by ICML 2025.

## Requirements

Python

pytorch

Pyhealth

## data

The initial data follows https://github.com/The-Real-JerryChen/TRANS. 

The data files used by our model need to be preprocessed through the code in the "preprocess" folder.

Thanks to the [TRANS](https://github.com/The-Real-JerryChen/TRANS) repo for sharing their data processing.

## Example to run the codes

```bash
python main.py --dataset mimic3_0.5
```

