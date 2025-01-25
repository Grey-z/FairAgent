# FairAgent - Introdcution
This is an anonymous repository about FairAgent, corresponding to the paper:**Enhancing New-item Fairness in Dynamic Recommender Systems**.
# Instruction
1. Install the related environment configuration and dependency packages.
2. Process data refer to the step in ./data/process_kuairec.ipynb
3. Train backbone model using the following command:
```
python drs_train.py --save=2 --dataset='kuairec' --algo_name='mf' --debias_method='backbone' --gpu=0
```
4. Train FairAgent using the following command
```
python drs-fairagent.py --save=2 --dataset='kuairec' --algo_name='mf' --alpha=2 --beta=0.5 --gama=0.1 --gpu=0 --batch_size=8192
```
# Files

- **daisy** : Folder: Open Library for Recommender Systems.
- **data**: Folder: Pre-process dataset.
  -- **process_kuairec** Jupyter File: instrction of processing related dataset.
  -- **data_utilis** Python File: Funtions of processing dataset.
- *drs-backbone.py**: Python File: Train backbone models.
- **drs-fairagent.py** : Python File: Train FairAgent.
- **utilis.py** : General purpose functions.
# Environment
This project is based on **Python 3.8.1** , **Pytorch 1.8.0** and **DaisyRec-v2.0**.

Packages: Refer to requirement.txt


