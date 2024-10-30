import os
import json
import yaml
import optuna
import subprocess
from typing import Dict, Any
from rasa.model_training import train
from rasa.shared.constants import DEFAULT_CONFIG_PATH
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.nlu.training_data.loading import load_data

class DIETHyperparameterTuner:
    def __init__(
        self,
        nlu_data_path: str = "data/nlu.yml",
        config_path: str = "config.yml",
        n_trials: int = 20,
        study_name: str = "diet_optimization"
    ):
        self.nlu_data_path = nlu_data_path
        self.config_path = config_path
        self.n_trials = n_trials
        self.study_name = study_name
        
    def load_config(self) -> Dict[str, Any]:
        """Load the RASA configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self, config: Dict[str, Any], path: str) -> None:
        """Save the configuration to a file."""
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def objective(self, trial: optuna.Trial) -> float:
        """Optimization objective function."""
        # Load base config
        config = self.load_config()
        
        # Get DIETClassifier config
        diet_config = None
        for item in config['pipeline']:
            if isinstance(item, dict) and item.get('name') == 'DIETClassifier':
                diet_config = item
                break
        
        if not diet_config:
            raise ValueError("DIETClassifier not found in config")

        # Define hyperparameters to optimize
        diet_config.update({
            'epochs': trial.suggest_int('epochs', 50, 300),
            'batch_size': trial.suggest_int('batch_size', 16, 256),
            'embedding_dimension': trial.suggest_int('embedding_dimension', 16, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'hidden_layers_sizes': {
                'text': [
                    trial.suggest_int('hidden_layer_1', 32, 512),
                    trial.suggest_int('hidden_layer_2', 16, 256)
                ]
            },
            'number_of_transformer_layers': trial.suggest_int('transformer_layers', 1, 4),
            'transformer_size': trial.suggest_int('transformer_size', 128, 512),
            'connection_density': trial.suggest_uniform('connection_density', 0.1, 0.5),
            'weight_sparsity': trial.suggest_uniform('weight_sparsity', 0.5, 0.9)
        })

        # Save temporary config
        temp_config_path = f"config_trial_{trial.number}.yml"
        self.save_config(config, temp_config_path)

        try:
            # Train model with current hyperparameters
            output_dir = f"models/trial_{trial.number}"
            trained_model = train(
                domain="domain.yml",
                config=temp_config_path,
                training_files=["data"],
                output=output_dir,
                force_training=True
            )

            # Evaluate model
            result = subprocess.run(
                ["rasa", "test", "nlu", 
                 "-u", self.nlu_data_path,
                 "-m", trained_model,
                 "--out", "results"],
                capture_output=True,
                text=True
            )

            # Parse results
            results_file = "results/intent_report.json"
            with open(results_file, 'r') as f:
                results = json.load(f)
                f1_score = results['weighted avg']['f1-score']

            # Cleanup
            os.remove(temp_config_path)
            
            return f1_score

        except Exception as e:
            print(f"Error during trial {trial.number}: {str(e)}")
            return 0.0

    def optimize(self) -> None:
        """Run the optimization process."""
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            storage="sqlite:///diet_optimization.db",
            load_if_exists=True
        )

        study.optimize(self.objective, n_trials=self.n_trials)

        print("\nBest trial:")
        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Save best configuration
        best_config = self.load_config()
        for item in best_config['pipeline']:
            if isinstance(item, dict) and item.get('name') == 'DIETClassifier':
                item.update({
                    'epochs': trial.params['epochs'],
                    'batch_size': trial.params['batch_size'],
                    'embedding_dimension': trial.params['embedding_dimension'],
                    'learning_rate': trial.params['learning_rate'],
                    'hidden_layers_sizes': {
                        'text': [
                            trial.params['hidden_layer_1'],
                            trial.params['hidden_layer_2']
                        ]
                    },
                    'number_of_transformer_layers': trial.params['transformer_layers'],
                    'transformer_size': trial.params['transformer_size'],
                    'connection_density': trial.params['connection_density'],
                    'weight_sparsity': trial.params['weight_sparsity']
                })
                break

        self.save_config(best_config, 'config_optimized.yml')

if __name__ == "__main__":
    tuner = DIETHyperparameterTuner()
    tuner.optimize()
