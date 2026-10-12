# Fraud Detection System

An intelligent system for detecting fraudulent bank transactions
(AI mini-project).

## Team
- BELAANES Erij
- BEN HAJ SAAD Safa
- KHELIFI Souha

## Goal
Detect potentially fraudulent transactions using supervised classification
models, then analyze the network of transactions between accounts to
prioritize suspicious situations.

## Installation
```bash
git clone https://github.com/Bermoth/Fraud-Detection-System.git
cd Fraud-Detection-System
python -m venv .venv
#Windows
.venv\Scripts\activate
#Linux ou Mac
#source .venv/bin/activate
pip install -r requirements.txt
```

## Data
The dataset is not versioned (too large). See `data/README.md` for
instructions on downloading it into `data/raw/`.

## Project structure
- `notebooks/`: exploration, models, comparison, network analysis
- `src/`: reusable code (preprocessing, evaluation, network)
- `models/`: trained models
- `predictions/`: model predictions
- `reports/`: report and figures
- `app/`: demo interface (bonus)

## Methodology
1. Data exploration and preprocessing
2. Training of several models (one per team member)
3. Comparison on the validation set and selection of the best model
4. Construction and analysis of the transaction network
5. Report and presentation