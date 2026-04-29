# JetNet
The repo is meant to train the SimCLR model for classifying/detecting radio source as a jetted source or not.

## Steps to Run the code
### 0. Install the dependencies in requrements.txt
```bash
pip install -r requirements.txt
```

### 1. Run Simulations
```bash
python simulation.py
``` 

### 2. Train the Model
```bash
python train_SSL.py
```

### 3. Evaluate the Model
```bash
python evaluate_SSL.py
```