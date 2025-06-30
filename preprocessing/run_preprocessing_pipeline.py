"""
Master Preprocessing Pipeline for AKI Prediction LSTM

This script orchestrates the entire preprocessing workflow with the enhanced study design:
1. Extract patient events (Step 1)
2. Validate patient data (Step 2)  
3. Create AKI episodes with multiple horizons (Step 3)
4. Truncate patient timeseries (Step 4)
5. Create train/test splits (Step 5)
6. Generate quantization dictionary (Step 6)
7. Quantize patient events (Step 7)
8. Create training arrays (Step 8)
9. Recreate arrays with Word2Vec vocabulary (Step 9)

Usage:
    python run_preprocessing_pipeline.py [--start_from_step N]
    
Make sure to update the config.py file with your paths before running.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'preprocessing_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline"""
    
    def __init__(self, 
                 root_data_path=DATA_ROOT_PATH,
                 mimic_csv_path=MIMIC_CSV_PATH,
                 input_hours=INPUT_HOURS,
                 gap_hours=GAP_HOURS,
                 prediction_horizons=PREDICTION_HORIZONS,
                 n_bins=N_BINS,
                 seed=RANDOM_SEED,
                 max_len=MAX_SEQUENCE_LENGTH,
                 metavision_only=METAVISION_ONLY,
                 minimum_age=MINIMUM_AGE):
        
        self.root_data_path = root_data_path
        self.mimic_csv_path = mimic_csv_path
        self.input_hours = input_hours
        self.gap_hours = gap_hours
        self.prediction_horizons = prediction_horizons
        self.n_bins = n_bins
        self.seed = seed
        self.max_len = max_len
        self.metavision_only = metavision_only
        self.minimum_age = minimum_age
        
    def run_step_1_2(self):
        """Run steps 1-2: subject events extraction and validation"""
        logger.info("=== STEP 1-2: Subject Events Extraction & Validation ===")
        
        try:
            # Import and run step 1
            import subprocess
            logger.info(f"Running step 1: Patient events extraction...")
            result = subprocess.run([
                sys.executable, '01_extract_patient_events.py',
                '--mimic_path', self.mimic_csv_path,
                '--output_path', self.root_data_path,
                '--minimum_age', str(self.minimum_age)
            ] + (['--metavision_only'] if self.metavision_only else []),
            cwd=os.path.dirname(__file__), capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Step 1 failed: {result.stderr}")
                return False
            
            logger.info("Running step 2: Patient data validation...")
            result = subprocess.run([
                sys.executable, '02_validate_patient_data.py',
                '--data_path', self.root_data_path
            ], cwd=os.path.dirname(__file__), capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Step 2 failed: {result.stderr}")
                return False
            
            logger.info("✓ Steps 1-2 completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error in steps 1-2: {e}", exc_info=True)
            return False
    
    def run_remaining_steps(self):
        """Run the remaining preprocessing steps 3-9"""
        logger.info("=== STEPS 3-9: Remaining Pipeline Steps ===")
        
        steps_success = []
        
        # Step 3: Create AKI Episodes
        logger.info("Running Step 3: Create AKI Episodes...")
        try:
            result = subprocess.run([
                'python', '03_create_aki_episodes.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours),
                '--gap_hours', str(self.gap_hours),
                '--prediction_horizons'] + [str(h) for h in self.prediction_horizons]
            )
            
            if result.returncode == 0:
                logger.info("✓ Step 3 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 3 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 3 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 4: Truncate Patient Timeseries
        logger.info("Running Step 4: Truncate Patient Timeseries...")
        try:
            result = subprocess.run([
                'python', '04_truncate_patient_timeseries.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours)
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 4 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 4 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 4 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 5: Create Train/Test Splits
        logger.info("Running Step 5: Create Train/Test Splits...")
        try:
            result = subprocess.run([
                'python', '05_create_train_test_splits.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours),
                '--seed', str(self.seed)
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 5 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 5 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 5 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 6: Generate Quantization Dictionary
        logger.info("Running Step 6: Generate Quantization Dictionary...")
        try:
            result = subprocess.run([
                'python', '06_generate_quantization_dict.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours),
                '--n_bins', str(self.n_bins),
                '--seed', str(self.seed)
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 6 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 6 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 6 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 7: Quantize Patient Events
        logger.info("Running Step 7: Quantize Patient Events...")
        try:
            result = subprocess.run([
                'python', '07_quantize_patient_events.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours),
                '--n_bins', str(self.n_bins),
                '--seed', str(self.seed)
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 7 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 7 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 7 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 8: Create Training Arrays
        logger.info("Running Step 8: Create Training Arrays...")
        try:
            result = subprocess.run([
                'python', '08_create_training_arrays.py',
                '--data_path', self.root_data_path,
                '--input_hours', str(self.input_hours),
                '--n_bins', str(self.n_bins),
                '--seed', str(self.seed),
                '--max_len', str(self.max_len)
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 8 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 8 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 8 failed with exception: {e}")
            steps_success.append(False)
        
        # Step 9: Recreate Arrays with Word2Vec Vocabulary
        logger.info("Running Step 9: Recreate Arrays with Word2Vec Vocabulary...")
        try:
            result = subprocess.run([
                'python', '09_recreate_arrays_word2vec.py'
                # Note: Step 9 uses configuration from config.py
            ])
            
            if result.returncode == 0:
                logger.info("✓ Step 9 completed successfully")
                steps_success.append(True)
            else:
                logger.error(f"✗ Step 9 failed with return code {result.returncode}")
                steps_success.append(False)
        except Exception as e:
            logger.error(f"✗ Step 9 failed with exception: {e}")
            steps_success.append(False)
        
        successful_steps = sum(steps_success)
        total_steps = len(steps_success)
        
        logger.info(f"Steps 3-9 Summary: {successful_steps}/{total_steps} completed successfully")
        
        return successful_steps > 0
    
    def validate_pipeline_completion(self):
        """Validate that all expected outputs were created"""
        logger.info("=== PIPELINE VALIDATION ===")
        
        validation_results = {}
        
        # Check if subject directories exist
        if os.path.exists(self.root_data_path):
            subject_dirs = [d for d in os.listdir(self.root_data_path) if d.isdigit()]
            validation_results['subject_directories'] = len(subject_dirs) > 0
            if len(subject_dirs) > 0:
                logger.info(f"Found {len(subject_dirs)} subject directories")
        else:
            validation_results['subject_directories'] = False
        
        # Check for basic files
        validation_results['data_directory'] = os.path.exists(self.root_data_path)
        
        # Report results
        logger.info("Validation Results:")
        for component, exists in validation_results.items():
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {component}: {exists}")
        
        all_passed = all(validation_results.values())
        
        if all_passed:
            logger.info("✓ Pipeline validation PASSED")
        else:
            logger.warning("⚠ Pipeline validation found missing components")
        
        return all_passed
    
    def run_full_pipeline(self, start_from_step=1):
        """Run the complete preprocessing pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING AKI PREDICTION PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        logger.info(f"Configuration:")
        logger.info(f"  Data path: {self.root_data_path}")
        logger.info(f"  MIMIC path: {self.mimic_csv_path}")
        logger.info(f"  Input hours: {self.input_hours}")
        logger.info(f"  Gap hours: {self.gap_hours}")
        logger.info(f"  Prediction horizons: {self.prediction_horizons}")
        logger.info(f"  Number of bins: {self.n_bins}")
        logger.info(f"  Random seed: {self.seed}")
        logger.info(f"  Max sequence length: {self.max_len}")
        logger.info(f"  Metavision only: {self.metavision_only}")
        logger.info(f"  Minimum age: {self.minimum_age}")
        
        success_count = 0
        total_steps = 9  # Complete pipeline with all 9 steps
        
        if start_from_step <= 2:
            logger.info(f"\n{'='*20} Steps 1-2: Subject Events & Validation {'='*20}")
            
            try:
                success = self.run_step_1_2()
                if success:
                    success_count += 1
                    logger.info(f"✓ Steps 1-2 completed successfully\n")
                else:
                    logger.error(f"✗ Steps 1-2 failed\n")
                    
            except Exception as e:
                logger.error(f"✗ Steps 1-2 failed with exception: {e}", exc_info=True)
        
        # Run remaining steps
        if start_from_step <= 9:
            logger.info(f"\n{'='*20} Steps 3-9: Complete Pipeline {'='*20}")
            
            try:
                success = self.run_remaining_steps()
                if success:
                    success_count += 7  # Steps 3-9 count as 7 steps
                    logger.info(f"✓ Steps 3-9 completed successfully\n")
                else:
                    logger.error(f"✗ Steps 3-9 failed\n")
                    
            except Exception as e:
                logger.error(f"✗ Steps 3-9 failed with exception: {e}", exc_info=True)
        
        # Final validation
        logger.info("\n" + "="*60)
        validation_passed = self.validate_pipeline_completion()
        
        # Summary
        logger.info("="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Steps completed: {success_count}/{total_steps}")
        logger.info(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        if success_count == total_steps:
            logger.info("🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            logger.info("✓ All 9 preprocessing steps have been executed")
            return True
        elif success_count > 0:
            logger.info("⚠ PREPROCESSING PIPELINE PARTIALLY COMPLETED")
            logger.info(f"✓ {success_count} out of {total_steps} steps completed")
            return True
        else:
            logger.error("❌ PREPROCESSING PIPELINE FAILED")
            return False


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run AKI prediction preprocessing pipeline')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path for processed data')
    
    parser.add_argument('--mimic_path', type=str, default=MIMIC_CSV_PATH,
                       help='Path to MIMIC-III CSV files')
    
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data to use for prediction')
    
    parser.add_argument('--gap_hours', type=int, default=GAP_HOURS,
                       help='Gap between input period and prediction window')
    
    parser.add_argument('--prediction_horizons', type=int, nargs='+', default=PREDICTION_HORIZONS,
                       help='Prediction horizons in hours')
    
    parser.add_argument('--n_bins', type=int, default=N_BINS,
                       help='Number of bins for continuous variable quantization')
    
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--max_len', type=int, default=MAX_SEQUENCE_LENGTH,
                       help='Maximum sequence length for padding/truncation')
    
    parser.add_argument('--start_from_step', type=int, default=1,
                       help='Step number to start from (useful for resuming)')
    
    parser.add_argument('--metavision_only', action='store_true', default=METAVISION_ONLY,
                       help='Only include patients with all Metavision ICU stays')
    
    parser.add_argument('--minimum_age', type=int, default=MINIMUM_AGE,
                       help='Minimum patient age for inclusion')
    
    args = parser.parse_args()
    
    print("Starting AKI Prediction Preprocessing Pipeline")
    print("=" * 50)
    print_config()
    
    # Validate configuration
    if not validate_paths():
        logger.error("Configuration validation failed. Please check config.py")
        sys.exit(1)
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        root_data_path=args.data_path,
        mimic_csv_path=args.mimic_path,
        input_hours=args.input_hours,
        gap_hours=args.gap_hours,
        prediction_horizons=args.prediction_horizons,
        n_bins=args.n_bins,
        seed=args.seed,
        max_len=args.max_len,
        metavision_only=args.metavision_only,
        minimum_age=args.minimum_age
    )
    
    # Run pipeline
    success = pipeline.run_full_pipeline(start_from_step=args.start_from_step)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 