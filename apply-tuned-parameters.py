import json
import yaml
import os
from typing import Dict, Any
import logging

class ConfigUpdater:
    def __init__(self, 
                 config_path: str = "config.yml",
                 params_path: str = "best_diet_params.json",
                 output_path: str = "config_tuned.yml"):
        self.config_path = config_path
        self.params_path = params_path
        self.output_path = output_path
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConfigUpdater")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_default_config(self) -> Dict[str, Any]:
        """Create default RASA configuration."""
        return {
            'pipeline': [
                {
                    'name': 'WhitespaceTokenizer',
                    'intent_tokenization_flag': False,
                    'intent_split_symbol': '_',
                    'token_pattern': None
                },
                {
                    'name': 'LexicalSyntacticFeaturizer',
                    'features': [
                        ["low", "title", "upper"],
                        ["BOS", "EOS", "low", "upper", "title", "digit"],
                        ["low", "title", "upper"]
                    ]
                },
                {
                    'name': 'CountVectorsFeaturizer',
                    'analyzer': 'word',
                    'min_ngram': 1,
                    'max_ngram': 1
                },
                {
                    'name': 'CountVectorsFeaturizer',
                    'analyzer': 'char_wb',
                    'min_ngram': 1,
                    'max_ngram': 4
                }
            ],
            'policies': [
                {
                    'name': 'MemoizationPolicy',
                    'max_history': 5
                },
                {
                    'name': 'RulePolicy'
                },
                {
                    'name': 'TEDPolicy',
                    'max_history': 5,
                    'epochs': 100
                }
            ]
        }

    def ensure_valid_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the configuration has all required components."""
        if not config or not isinstance(config, dict):
            self.logger.warning("Invalid config structure. Using default configuration.")
            return self.get_default_config()
            
        if 'pipeline' not in config or not config['pipeline']:
            self.logger.warning("No pipeline found in config. Adding default pipeline.")
            config['pipeline'] = self.get_default_config()['pipeline']
            
        if 'policies' not in config or not config['policies']:
            self.logger.warning("No policies found in config. Adding default policies.")
            config['policies'] = self.get_default_config()['policies']
            
        return config

    def adjust_transformer_size(self, size: int, n_heads: int = 4) -> int:
        """Adjust transformer size to be multiple of number of attention heads."""
        return round(size / n_heads) * n_heads

    def load_files(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load configuration and parameters files."""
        try:
            config = self.get_default_config()
            
            if os.path.exists(self.config_path):
                self.logger.info(f"Loading config from {self.config_path}")
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        config = loaded_config

            config = self.ensure_valid_config(config)
            
            with open(self.params_path, 'r') as f:
                tuned_params = json.load(f)
            self.logger.info(f"Loaded tuned parameters from {self.params_path}")
            
            return config, tuned_params
            
        except Exception as e:
            self.logger.error(f"Error loading files: {str(e)}")
            raise

    def create_diet_config(self, tuned_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create DIET classifier configuration."""
        adjusted_transformer_size = self.adjust_transformer_size(tuned_params['transformer_size'])
        
        return {
            'name': 'DIETClassifier',
            'component_config': {
                'epochs': tuned_params['epochs'],
                'batch_size': tuned_params['batch_size'],
                'learning_rate': tuned_params['learning_rate'],
                'hidden_layers_sizes': {
                    'text': [
                        tuned_params['hidden_layer_1'],
                        tuned_params['hidden_layer_2']
                    ]
                },
                'embedding_dimension': tuned_params['embedding_dimension'],
                'number_of_transformer_layers': tuned_params['transformer_layers'],
                'transformer_size': adjusted_transformer_size,
                'number_of_attention_heads': 4,
                'use_masked_language_model': False,
                'intent_classification': True,
                'entity_recognition': True,
                'constrain_similarities': True,
                'similarity_type': 'inner',
                'loss_type': 'cross_entropy',
                'evaluate_every_number_of_epochs': 10,
                'evaluate_on_number_of_examples': 0,
                'checkpoint_model': True,
                'tensorboard_log_directory': './logs',
                'weight_sparsity': tuned_params['weight_sparsity'],
                'dense_dimension': 128,
                'concat_dimension': None,
                'drop_rate': 0.2,
                'drop_rate_attention': 0.0,
                'random_seed': 42,
                'learning_rate_warmup': True,
                'warmup_steps': 100
            }
        }

    def update_config(self, config: Dict[str, Any], tuned_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with tuned parameters."""
        try:
            # Create DIET configuration
            diet_config = self.create_diet_config(tuned_params)
            
            # Find and update or add DIETClassifier
            diet_found = False
            for idx, component in enumerate(config['pipeline']):
                if isinstance(component, dict) and component.get('name') == 'DIETClassifier':
                    config['pipeline'][idx] = diet_config
                    diet_found = True
                    self.logger.info("Updated existing DIETClassifier configuration")
                    break

            if not diet_found:
                self.logger.info("Adding DIETClassifier to pipeline")
                config['pipeline'].append(diet_config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            raise

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration."""
        try:
            if os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.backup"
                with open(self.config_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                self.logger.info(f"Created backup of original config at {backup_path}")

            with open(self.output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Saved updated config to {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise

    def display_changes(self, tuned_params: Dict[str, Any]) -> None:
        """Display the tuned parameters."""
        self.logger.info("\nTuned Parameters Summary:")
        self.logger.info("-" * 40)
        for param, value in tuned_params.items():
            self.logger.info(f"{param}: {value}")
        self.logger.info("-" * 40)

    def update_config_with_tuned_params(self) -> None:
        """Main method to update configuration with tuned parameters."""
        try:
            config, tuned_params = self.load_files()
            self.display_changes(tuned_params)
            updated_config = self.update_config(config, tuned_params)
            self.save_config(updated_config)
            
            self.logger.info("\nNext steps:")
            self.logger.info("1. Review the updated config in: " + self.output_path)
            self.logger.info("2. Train your model with:")
            self.logger.info(f"   rasa train --config {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to update config: {str(e)}")
            raise

if __name__ == "__main__":
    updater = ConfigUpdater()
    updater.update_config_with_tuned_params()