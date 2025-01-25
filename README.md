# FairAgent - Introdcution
This is an anonymous repository about FairAgent, corresponding to the paper:**Enhancing New-item Fairness in Dynamic Recommender Systems**.
# Instruction
1. Install the related environment configuration and dependency packages.
2. Process data refer to the step in ./data/process_kuairec.ipynb
3. Train backbone model using the following command:
```
# MF / LightGCN:
python drs-backbone.py --save=2 --dataset='kuairec' --algo_name='mf' --debias_method='backbone' --gpu=0
# ALDI:
python drs-aldi.py --save=2 --dataset='kuairec' --algo_name='mf' --debias_method='backbone' 
```
4. Train FairAgent using the following command
```
# MF / LightGCN:
python drs-fairagent.py --save=2 --dataset='kuairec' --algo_name='mf' --alpha=2 --beta=0.5 --gama=0.1 --gpu=0 --batch_size=8192
# ALDI:
python drs-fairagent-ALDI.py --save=2 --dataset='kuairec' --algo_name='mf' --alpha=2 --beta=0.5 --gama=0.1 --gpu=1
```
# Files

- **daisy** : Folder: Open Library for Recommender Systems.
- **data**: Folder: Pre-process dataset.
  -- **process_kuairec** Jupyter File: instrction of processing related dataset.
  -- **data_utilis** Python File: Funtions of processing dataset.
- **drs-backbone.py**: Python File: Train backbone models.
- **drs-ALDI.py**: Python File: Train ALDI backbone models.
- **drs-fairagent.py** : Python File: Train FairAgent based MF/LightGCN backbone.
- **drs-fairagent-ALDI.py**: Python File: Train FairAgent based on ALDI backbone.
- **FairAgent.py**: Python File: Model of FairAgent.
- **ALDI.py**: Python File: Model of ALDI backbone baased Pytorch.
- **calculate_unf.ipynb**: Jupyter File: An example of calculating UNF metric.
- **Plot.ipynb**: Jupyter File: An example of figures.
- **utilis.py** : General purpose functions.
# Environment
This project is based on **Python 3.8.1** , **Pytorch 1.8.0** and **DaisyRec-v2.0**.

Packages: Refer to requirement.txt


