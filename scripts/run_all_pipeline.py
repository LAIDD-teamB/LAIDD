#!/usr/bin/env python3
"""
LAIDD Unified Pipeline
======================
This script runs the complete LAIDD pipeline:
1. Pretraining: Multi-affinity model pretraining
2. Finetuning: GSK3β-specific finetuning
3. Generative Model: LSTM-based molecule generation

Usage:
    python run_all_pipeline.py --config config.json

    or run with default settings:
    python run_all_pipeline.py
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Import modules will be done lazily to avoid import errors when just creating config


class LAIDDPipeline:
    """Unified pipeline for LAIDD workflow"""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create main output directory
        self.output_root = config.get("output_root", "./pipeline_results")
        os.makedirs(self.output_root, exist_ok=True)

        # Create subdirectories for each stage
        self.pretrain_dir = os.path.join(self.output_root, "1_pretraining")
        self.finetune_dir = os.path.join(self.output_root, "2_finetuning")
        self.generative_dir = os.path.join(self.output_root, "3_generative")

        for d in [self.pretrain_dir, self.finetune_dir, self.generative_dir]:
            os.makedirs(d, exist_ok=True)

    def log(self, message, level="INFO"):
        """Print and log messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

        # Write to log file
        log_file = os.path.join(self.output_root, "pipeline.log")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")

    def run_pretraining(self):
        """Step 1: Run pretraining"""
        self.log("="*80)
        self.log("STEP 1: PRETRAINING - Multi-Affinity Model")
        self.log("="*80)

        if not self.config.get("run_pretraining", True):
            self.log("Pretraining skipped (disabled in config)")
            return None

        # Lazy import
        try:
            from pretraining_code import main_multi_affinity_bindingdb
        except ImportError as e:
            self.log(f"Failed to import pretraining module: {e}", "ERROR")
            raise

        pretrain_config = self.config.get("pretraining", {})

        # Check if data file exists
        data_path = pretrain_config.get("data_path", "./data/pretraining_data.tsv")
        if not os.path.exists(data_path):
            self.log(f"Pretraining data not found: {data_path}", "ERROR")
            raise FileNotFoundError(f"Pretraining data not found: {data_path}")

        self.log(f"Data path: {data_path}")
        self.log(f"Output directory: {self.pretrain_dir}")

        try:
            model, results = main_multi_affinity_bindingdb(
                tsv_path=data_path,
                batch_size=pretrain_config.get("batch_size", 32),
                max_len_smiles=pretrain_config.get("max_len_smiles", 256),
                max_len_seq=pretrain_config.get("max_len_seq", 1500),
                epochs=pretrain_config.get("epochs", 30),
                lr=pretrain_config.get("lr", 1e-3),
                weight_decay=pretrain_config.get("weight_decay", 1e-5),
                out_dir=self.pretrain_dir,
                use_scaffold=pretrain_config.get("use_scaffold", True),
                grad_accum=pretrain_config.get("grad_accum", 4),
                warmup_steps=pretrain_config.get("warmup_steps", 1000),
                save_every=pretrain_config.get("save_every", 5),
                target_seq_column=pretrain_config.get("target_seq_column", "BindingDB Target Chain Sequence")
            )

            self.results["pretraining"] = {
                "status": "success",
                "best_model_path": os.path.join(self.pretrain_dir, "best_model.pt"),
                "test_metrics": results.get("test_overall", {})
            }

            self.log("Pretraining completed successfully!")
            self.log(f"Best model saved: {self.results['pretraining']['best_model_path']}")

            return model, results

        except Exception as e:
            self.log(f"Pretraining failed: {e}", "ERROR")
            self.results["pretraining"] = {"status": "failed", "error": str(e)}
            raise

    def run_finetuning(self, pretrained_model_path=None):
        """Step 2: Run finetuning"""
        self.log("="*80)
        self.log("STEP 2: FINETUNING - GSK3β Specialization")
        self.log("="*80)

        if not self.config.get("run_finetuning", True):
            self.log("Finetuning skipped (disabled in config)")
            return None

        # Lazy import
        try:
            from finetuning_code import fine_tune_gsk3b, fine_tune_gsk3b_5fold
        except ImportError as e:
            self.log(f"Failed to import finetuning module: {e}", "ERROR")
            raise

        finetune_config = self.config.get("finetuning", {})

        # Determine pretrained model path
        if pretrained_model_path is None:
            pretrained_model_path = finetune_config.get("pretrained_model_path")
            if pretrained_model_path is None:
                # Use the model from pretraining step
                pretrained_model_path = os.path.join(self.pretrain_dir, "best_model.pt")

        if not os.path.exists(pretrained_model_path):
            self.log(f"Pretrained model not found: {pretrained_model_path}", "ERROR")
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

        # Check finetuning data
        data_path = finetune_config.get("data_path", "./data/finetuning_data.tsv")
        if not os.path.exists(data_path):
            self.log(f"Finetuning data not found: {data_path}", "ERROR")
            raise FileNotFoundError(f"Finetuning data not found: {data_path}")

        self.log(f"Pretrained model: {pretrained_model_path}")
        self.log(f"Finetuning data: {data_path}")
        self.log(f"Output directory: {self.finetune_dir}")

        # Choose finetuning method
        use_5fold = finetune_config.get("use_5fold", False)

        try:
            if use_5fold:
                self.log("Using 5-fold cross-validation finetuning")
                model_paths, results = fine_tune_gsk3b_5fold(
                    pretrained_model_path=pretrained_model_path,
                    gsk3b_data_path=data_path,
                    output_dir=self.finetune_dir,
                    epochs=finetune_config.get("epochs", 50),
                    lr=finetune_config.get("lr", 1e-4),
                    weight_decay=finetune_config.get("weight_decay", 1e-4),
                    batch_size=finetune_config.get("batch_size", 16),
                    early_stopping_patience=finetune_config.get("early_stopping_patience", 10),
                    test_split=finetune_config.get("test_split", 0.1),
                    freeze_encoder=finetune_config.get("freeze_encoder", False),
                    data_augmentation=finetune_config.get("data_augmentation", True)
                )

                self.results["finetuning"] = {
                    "status": "success",
                    "method": "5-fold CV",
                    "model_paths": model_paths,
                    "ensemble_metrics": results.get("ensemble_test_metrics", {})
                }
            else:
                self.log("Using single model finetuning")
                model, results = fine_tune_gsk3b(
                    pretrained_model_path=pretrained_model_path,
                    gsk3b_data_path=data_path,
                    output_dir=self.finetune_dir,
                    epochs=finetune_config.get("epochs", 50),
                    lr=finetune_config.get("lr", 1e-4),
                    weight_decay=finetune_config.get("weight_decay", 1e-4),
                    batch_size=finetune_config.get("batch_size", 16),
                    early_stopping_patience=finetune_config.get("early_stopping_patience", 15),
                    validation_split=finetune_config.get("validation_split", 0.2),
                    test_split=finetune_config.get("test_split", 0.1),
                    freeze_encoder=finetune_config.get("freeze_encoder", False),
                    data_augmentation=finetune_config.get("data_augmentation", True)
                )

                self.results["finetuning"] = {
                    "status": "success",
                    "method": "single model",
                    "best_model_path": os.path.join(self.finetune_dir, "best_gsk3b_model.pt"),
                    "test_metrics": results.get("test_metrics", {})
                }

            self.log("Finetuning completed successfully!")
            return results

        except Exception as e:
            self.log(f"Finetuning failed: {e}", "ERROR")
            self.results["finetuning"] = {"status": "failed", "error": str(e)}
            raise

    def run_generative(self):
        """Step 3: Run generative model training"""
        self.log("="*80)
        self.log("STEP 3: GENERATIVE MODEL - LSTM Molecule Generation")
        self.log("="*80)

        if not self.config.get("run_generative", True):
            self.log("Generative model training skipped (disabled in config)")
            return None

        gen_config = self.config.get("generative", {})

        # Check if the generative training script exists
        gen_script = gen_config.get("training_script", None)

        if gen_script and os.path.exists(gen_script):
            self.log(f"Running generative model training script: {gen_script}")
            try:
                # Run the generative training as a subprocess
                result = subprocess.run(
                    [sys.executable, gen_script],
                    cwd=os.path.dirname(gen_script) or ".",
                    capture_output=True,
                    text=True,
                    timeout=gen_config.get("timeout", 3600)
                )

                if result.returncode == 0:
                    self.log("Generative model training completed successfully!")
                    self.results["generative"] = {
                        "status": "success",
                        "output": result.stdout
                    }
                else:
                    self.log(f"Generative model training failed with code {result.returncode}", "ERROR")
                    self.log(f"Error output: {result.stderr}", "ERROR")
                    self.results["generative"] = {
                        "status": "failed",
                        "error": result.stderr
                    }

            except subprocess.TimeoutExpired:
                self.log("Generative model training timed out", "ERROR")
                self.results["generative"] = {
                    "status": "failed",
                    "error": "Timeout"
                }
            except Exception as e:
                self.log(f"Generative model training failed: {e}", "ERROR")
                self.results["generative"] = {
                    "status": "failed",
                    "error": str(e)
                }
        else:
            self.log("Generative model training script not specified or not found", "WARN")
            self.log("Please implement generative model training separately using bs_denovo modules")
            self.results["generative"] = {
                "status": "skipped",
                "reason": "No training script specified"
            }

    def run_all(self):
        """Run the complete pipeline"""
        self.log("="*80)
        self.log("LAIDD UNIFIED PIPELINE")
        self.log(f"Started at: {self.timestamp}")
        self.log("="*80)

        # Save configuration
        config_path = os.path.join(self.output_root, "pipeline_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        self.log(f"Configuration saved: {config_path}")

        try:
            # Step 1: Pretraining
            if self.config.get("run_pretraining", True):
                pretrain_results = self.run_pretraining()

            # Step 2: Finetuning
            if self.config.get("run_finetuning", True):
                finetune_results = self.run_finetuning()

            # Step 3: Generative model
            if self.config.get("run_generative", True):
                generative_results = self.run_generative()

            # Save final results
            results_path = os.path.join(self.output_root, "pipeline_results.json")
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2)

            self.log("="*80)
            self.log("PIPELINE COMPLETED SUCCESSFULLY!")
            self.log(f"Results saved: {results_path}")
            self.log("="*80)

            # Print summary
            self.print_summary()

            return self.results

        except Exception as e:
            self.log(f"Pipeline failed: {e}", "ERROR")

            # Save partial results
            results_path = os.path.join(self.output_root, "pipeline_results_partial.json")
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2)

            self.log(f"Partial results saved: {results_path}")
            raise

    def print_summary(self):
        """Print a summary of the pipeline results"""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)

        for stage, result in self.results.items():
            print(f"\n{stage.upper()}:")
            print(f"  Status: {result.get('status', 'unknown')}")

            if result.get("status") == "success":
                if stage == "pretraining":
                    metrics = result.get("test_metrics", {})
                    print(f"  Test RMSE: {metrics.get('RMSE', 'N/A'):.3f}")
                    print(f"  Test R²: {metrics.get('R2', 'N/A'):.3f}")
                    print(f"  Model: {result.get('best_model_path', 'N/A')}")

                elif stage == "finetuning":
                    method = result.get("method", "unknown")
                    print(f"  Method: {method}")

                    if method == "5-fold CV":
                        metrics = result.get("ensemble_metrics", {})
                        print(f"  Ensemble Test RMSE: {metrics.get('RMSE', 'N/A'):.3f}")
                        print(f"  Ensemble Test R²: {metrics.get('R2', 'N/A'):.3f}")
                    else:
                        metrics = result.get("test_metrics", {})
                        print(f"  Test RMSE: {metrics.get('RMSE', 'N/A'):.3f}")
                        print(f"  Test R²: {metrics.get('R2', 'N/A'):.3f}")
                        print(f"  Model: {result.get('best_model_path', 'N/A')}")

                elif stage == "generative":
                    print(f"  Output: {result.get('output', 'N/A')[:100]}...")

            elif result.get("status") == "failed":
                print(f"  Error: {result.get('error', 'Unknown error')}")

        print("\n" + "="*80)
        print(f"Full results saved in: {self.output_root}")
        print("="*80 + "\n")


def load_config(config_path):
    """Load configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        print(f"[WARN] Config file not found: {config_path}")
        print("[INFO] Using default configuration")
        return get_default_config()


def get_default_config():
    """Get default configuration"""
    return {
        "output_root": "./pipeline_results",
        "run_pretraining": True,
        "run_finetuning": True,
        "run_generative": False,  # Disabled by default as it requires separate implementation

        "pretraining": {
            "data_path": "./data/pretraining_data.tsv",
            "batch_size": 32,
            "max_len_smiles": 256,
            "max_len_seq": 1500,
            "epochs": 30,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "use_scaffold": True,
            "grad_accum": 4,
            "warmup_steps": 1000,
            "save_every": 5,
            "target_seq_column": "BindingDB Target Chain Sequence"
        },

        "finetuning": {
            "data_path": "./data/finetuning_data.tsv",
            "pretrained_model_path": None,  # Will use pretraining output by default
            "use_5fold": False,
            "epochs": 50,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "early_stopping_patience": 15,
            "validation_split": 0.2,
            "test_split": 0.1,
            "freeze_encoder": False,
            "data_augmentation": True
        },

        "generative": {
            "training_script": None,  # Path to generative model training script
            "timeout": 3600  # Timeout in seconds
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="LAIDD Unified Pipeline - Pretraining, Finetuning, and Generative Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="pipeline_config.json",
        help="Path to configuration JSON file (default: pipeline_config.json)"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit"
    )

    parser.add_argument(
        "--skip-pretraining",
        action="store_true",
        help="Skip pretraining step"
    )

    parser.add_argument(
        "--skip-finetuning",
        action="store_true",
        help="Skip finetuning step"
    )

    parser.add_argument(
        "--enable-generative",
        action="store_true",
        help="Enable generative model training (disabled by default)"
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        config = get_default_config()
        with open("pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Default configuration file created: pipeline_config.json")
        print("Please edit this file to customize the pipeline settings.")
        return

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.skip_pretraining:
        config["run_pretraining"] = False
    if args.skip_finetuning:
        config["run_finetuning"] = False
    if args.enable_generative:
        config["run_generative"] = True

    # Run pipeline
    pipeline = LAIDDPipeline(config)

    try:
        results = pipeline.run_all()
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
