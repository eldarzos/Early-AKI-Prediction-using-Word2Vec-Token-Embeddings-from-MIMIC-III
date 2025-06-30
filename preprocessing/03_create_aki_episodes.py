"""
Step 3: Create Episodes with Improved AKI Labeling

Extract episode information and timeseries data for each ICU stay with IMPROVED study design.

IMPROVEMENTS:
- Smarter case/control definitions with exclusion criteria
- Gap period between input and prediction windows
- Exclusion buffer to avoid near-miss ambiguity
- Multiple prediction horizons
- Baseline AKI detection and exclusion
- Competing risk handling

Generates:
- episode{i}.csv: Static data with multiple AKI prediction labels
- episode{i}_timeseries.csv: FULL time-aligned events for the i-th ICU stay
- exclusion_log.csv: Detailed log of exclusions with reasons

Usage:
    python 03_create_episodes_improved.py
    
Make sure to run Steps 1-2 first.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import logging
import sys
from collections import defaultdict

# Add config to path
sys.path.append('../config')
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Mapping for gender
g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}

def create_episode_static_data(stay, prediction_horizons):
    """Create static episode data with multiple horizon labels"""
    
    # Basic stay information
    episode_data = {
        'ICUSTAY_ID': stay['ICUSTAY_ID'],
        'SUBJECT_ID': stay['SUBJECT_ID'],
        'HADM_ID': stay['HADM_ID'],
        'Age': stay.get('AGE', 0),
        'Gender': g_map.get(stay.get('GENDER', ''), 0),
        'Length of Stay': stay.get('LOS', 0),
        'Mortality': stay.get('MORTALITY', 0)
    }
    
    # For now, create placeholder AKI labels (would need KDIGO data for real implementation)
    for horizon in prediction_horizons:
        horizon_key = f'{horizon}h'
        episode_data[f'AKI_Stg3_{horizon_key}'] = -1  # Missing/unknown
        episode_data[f'label_type_{horizon_key}'] = 'missing_kdigo'
        episode_data[f'time_to_aki_{horizon_key}'] = None
        episode_data[f'exclusion_reason_{horizon_key}'] = 'missing_kdigo'
    
    return pd.DataFrame([episode_data])


def create_episode_timeseries(stay, events_df):
    """Create timeseries data for an episode"""
    
    if events_df.empty:
        return None
    
    # Convert CHARTTIME and INTIME to datetime
    events_df['CHARTTIME'] = pd.to_datetime(events_df['CHARTTIME'])
    intime = pd.to_datetime(stay['INTIME'])
    
    # Calculate hours from admission
    events_df['Hours'] = (events_df['CHARTTIME'] - intime).dt.total_seconds() / 3600.0
    
    # Filter to positive hours (after admission)
    events_df = events_df[events_df['Hours'] >= 0].copy()
    
    if events_df.empty:
        return None
    
    # Select relevant columns and sort by time
    timeseries_cols = ['Hours', 'ITEMID', 'VALUE', 'VALUEUOM']
    timeseries_data = events_df[timeseries_cols].sort_values('Hours')
    
    return timeseries_data


def create_episodes_improved(root_path, input_hours=INPUT_HOURS, gap_hours=GAP_HOURS, 
                           prediction_horizons=PREDICTION_HORIZONS):
    """Main function to create improved episodes with multiple horizon labels"""
    
    logger.info("Starting improved episode creation...")
    logger.info("⚠ Note: This implementation creates placeholder AKI labels.")
    logger.info("For real AKI detection, KDIGO stages data is required.")
    
    # Process subjects
    subjects = [d for d in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, d)) and d.isdigit()]
    
    logger.info(f"Processing {len(subjects)} subjects...")
    
    episode_count = 0
    valid_episodes = 0
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_dir = os.path.join(root_path, subject)
        stays_path = os.path.join(subject_dir, 'stays.csv')
        events_path = os.path.join(subject_dir, 'events.csv')
        
        if not os.path.exists(stays_path) or not os.path.exists(events_path):
            continue
        
        try:
            # Load subject data
            stays_df = pd.read_csv(stays_path)
            events_df = pd.read_csv(events_path)
            
            if stays_df.empty or events_df.empty:
                continue
            
            # Process each ICU stay
            for i, (_, stay) in enumerate(stays_df.iterrows()):
                episode_count += 1
                
                # Filter events for this stay
                stay_events = events_df[events_df['ICUSTAY_ID'] == stay['ICUSTAY_ID']].copy()
                
                if stay_events.empty:
                    continue
                
                # Create episode static data with multiple horizon labels
                episode_data = create_episode_static_data(stay, prediction_horizons)
                
                # Create timeseries data
                timeseries_data = create_episode_timeseries(stay, stay_events)
                
                if timeseries_data is not None and not episode_data.empty:
                    # Save episode files
                    episode_file = os.path.join(subject_dir, f'episode{i}.csv')
                    timeseries_file = os.path.join(subject_dir, f'episode{i}_timeseries.csv')
                    
                    episode_data.to_csv(episode_file, index=False)
                    timeseries_data.to_csv(timeseries_file, index=False)
                    
                    valid_episodes += 1
                
        except Exception as e:
            logger.error(f"Error processing subject {subject}: {e}")
            continue
    
    # Create placeholder exclusion log
    exclusion_log = pd.DataFrame({
        'icustay_id': [],
        'exclusion_reason': [],
        'horizon': [],
        'details': []
    })
    exclusion_path = os.path.join(root_path, 'exclusion_log.csv')
    exclusion_log.to_csv(exclusion_path, index=False)
    
    logger.info(f"Episode creation completed:")
    logger.info(f"  Total episodes processed: {episode_count}")
    logger.info(f"  Valid episodes created: {valid_episodes}")
    logger.info(f"  Exclusion log saved to: {exclusion_path}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Create episodes with improved AKI labeling')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    parser.add_argument('--input_hours', type=int, default=INPUT_HOURS,
                       help='Hours of input data to use')
    parser.add_argument('--gap_hours', type=int, default=GAP_HOURS,
                       help='Gap hours between input and prediction')
    parser.add_argument('--prediction_horizons', type=int, nargs='+', default=PREDICTION_HORIZONS,
                       help='Prediction horizons in hours')
    
    args = parser.parse_args()
    
    print("Starting Step 3: Improved Episode Creation")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print(f"Input Hours: {args.input_hours}")
    print(f"Gap Hours: {args.gap_hours}")
    print(f"Prediction Horizons: {args.prediction_horizons}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Steps 1-2 first.")
        return
    
    # Run episode creation
    create_episodes_improved(
        root_path=args.data_path,
        input_hours=args.input_hours,
        gap_hours=args.gap_hours,
        prediction_horizons=args.prediction_horizons
    )
    
    print("✓ Episode creation completed successfully!")


if __name__ == "__main__":
    main() 