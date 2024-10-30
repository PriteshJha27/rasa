### Steps to train model

#----------------------------------------------------------------------------------------

rasa init
### This will create default files like domain.yml, config.yml etc.
### Choose 'n' when asked to train the model

#----------------------------------------------------------------------------------------

### Step 1: Initial RASA training with default config
rasa train

### Step 2: Run hyperparameter tuning
python diet-hyperparameter-tuning.py
### This will create best_diet_params.json with optimal parameters

### Step 3: Apply the tuned parameters
python apply-tuned-parameters.py. This will create config_tuned.yml

### Step 4: Train the final model with tuned parameters
rasa train --config config_tuned.yml

###----------------------------------------------------------------------------------------

### Expected directory
your_project/
├── data/
│   └── nlu.yml
├── models/                    # Created by RASA
├── diet_tuning.db            # Created during tuning
├── best_diet_params.json     # Created by tuning
├── config_tuned.yml          # Created by apply_tuned_parameters.py
├── config.yml                # From rasa init
├── domain.yml                # From rasa init
├── diet-hyperparameter-tuning.py
└── apply-tuned-parameters.py


###----------------------------------------------------------------------------------------

### Test the model
rasa shell nlu
