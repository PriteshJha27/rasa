import os
import json
import logging
import optuna
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.model import get_latest_model
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.core.agent import Agent

class DIETTuner:
    def __init__(
        self,
        model_path: str = None,
        nlu_data_path: str = "data/nlu.yml",
        n_trials: int = 20,
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.model_path = model_path or get_latest_model()
        self.nlu_data_path = nlu_data_path
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = self._setup_logger()
        
        # Initialize model storage
        self.model_storage = LocalModelStorage("models")
        
        # Load the model and training data
        self.load_model_and_data()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DIETTuner")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_model_and_data(self) -> None:
        try:
            # Load the agent with the trained model
            self.agent = Agent.load(self.model_path)
            
            # Load training data
            importer = RasaFileImporter(
                domain_path="domain.yml",
                training_data_paths=[self.nlu_data_path]
            )
            self.training_data = importer.get_nlu_data()
            self.logger.info(f"Loaded training data with {len(self.training_data.intent_examples)} examples")
            
        except Exception as e:
            self.logger.error(f"Error loading model or data: {str(e)}")
            raise

    def prepare_cross_validation(self) -> List[Tuple[TrainingData, TrainingData]]:
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        examples = self.training_data.intent_examples
        cv_splits = []
        
        for train_idx, val_idx in kf.split(examples):
            train_examples = [examples[i] for i in train_idx]
            val_examples = [examples[i] for i in val_idx]
            
            train_data = TrainingData(training_examples=train_examples)
            val_data = TrainingData(training_examples=val_examples)
            
            cv_splits.append((train_data, val_data))
            
        return cv_splits

    def create_diet_classifier(self, trial: optuna.Trial) -> DIETClassifier:
        """Create a DIET classifier with trial parameters."""
        # Define hyperparameters
        epochs = trial.suggest_int('epochs', 50, 300)
        batch_size = trial.suggest_int('batch_size', 16, 256)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        hidden_layer_1 = trial.suggest_int('hidden_layer_1', 32, 256)
        hidden_layer_2 = trial.suggest_int('hidden_layer_2', 16, 128)
        embedding_dim = trial.suggest_int('embedding_dimension', 16, 256)
        transformer_layers = trial.suggest_int('transformer_layers', 1, 4)
        transformer_size = trial.suggest_int('transformer_size', 128, 512)
        weight_sparsity = trial.suggest_float('weight_sparsity', 0.5, 0.9)

        # Create classifier with updated initialization
        classifier = DIETClassifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_layers_sizes={'text': [hidden_layer_1, hidden_layer_2]},
            embedding_dimension=embedding_dim,
            number_of_transformer_layers=transformer_layers,
            transformer_size=transformer_size,
            weight_sparsity=weight_sparsity,
            model_storage=self.model_storage,
            resource=Resource("diet_classifier")
        )
        
        return classifier

    def evaluate_parameters(self, classifier: DIETClassifier, 
                          train_data: TrainingData, 
                          val_data: TrainingData) -> float:
        try:
            # Train the classifier
            classifier.train(train_data)
            
            # Predict on validation set
            correct = 0
            total = 0
            
            for example in val_data.intent_examples:
                result = classifier.process([example])[0]
                if result.get("intent", {}).get("name") == example.get("intent"):
                    correct += 1
                total += 1
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            return 0.0

    def objective(self, trial: optuna.Trial) -> float:
        cv_splits = self.prepare_cross_validation()
        scores = []
        
        for fold_idx, (train_data, val_data) in enumerate(cv_splits, 1):
            self.logger.info(f"Evaluating fold {fold_idx}/{self.n_folds}")
            
            try:
                # Create classifier with trial parameters
                classifier = self.create_diet_classifier(trial)
                
                # Evaluate current parameters
                fold_score = self.evaluate_parameters(classifier, train_data, val_data)
                scores.append(fold_score)
                
                self.logger.info(f"Fold {fold_idx} score: {fold_score:.4f}")
            
            except Exception as e:
                self.logger.error(f"Error in fold {fold_idx}: {str(e)}")
                return 0.0
        
        mean_score = np.mean(scores)
        self.logger.info(f"Trial {trial.number} mean score: {mean_score:.4f}")
        
        return mean_score

    def optimize(self) -> Dict[str, Any]:
        study = optuna.create_study(
            study_name="diet_tuning",
            direction="maximize",
            storage="sqlite:///diet_tuning.db",
            load_if_exists=True
        )
        
        self.logger.info("Starting hyperparameter optimization...")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.logger.info("\nOptimization completed!")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        self.logger.info("Best hyperparameters:")
        for param, value in study.best_trial.params.items():
            self.logger.info(f"  {param}: {value}")
            
        # Save best parameters
        self.save_best_parameters(study.best_trial.params)
        
        return study.best_trial.params

    def save_best_parameters(self, params: Dict[str, Any]) -> None:
        output_file = "best_diet_params.json"
        with open(output_file, 'w') as f:
            json.dump(params, f, indent=2)
        self.logger.info(f"Best parameters saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    tuner = DIETTuner(
        nlu_data_path="data/nlu.yml",
        n_trials=20,
        n_folds=5
    )
    
    best_params = tuner.optimize()