"""
Step 2: Validate Events

Validates events have sufficient information (HADM_ID, ICUSTAY_ID) to be used.
Ensures events can be reliably linked to a specific ICU stay defined in stays.csv.
Drops events that cannot be linked or have conflicting information.

Usage:
    python 02_validate_events.py
    
Make sure to run Step 1 (extract_subject_events.py) first.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import sys

# Add config to path
sys.path.append('../config')
from config import *

def is_subject_folder(x):
    """Verify directory name is numeric (likely a subject ID)."""
    return x.isdigit()

def validate_events(subjects_root_path):
    """Main function to validate events against stay information."""

    n_events_initial = 0           # total number of events read
    n_events_final = 0             # total number of events after validation
    empty_hadm = 0                 # HADM_ID is empty in events.csv.
    no_hadm_in_stay = 0            # HADM_ID from event does not appear in stays.csv for that subject.
    no_icustay = 0                 # ICUSTAY_ID is empty in events.csv.
    recovered = 0                  # empty ICUSTAY_IDs recovered using HADM_ID match.
    could_not_recover = 0          # empty ICUSTAY_IDs that could not be recovered.
    icustay_id_mismatch = 0        # ICUSTAY_ID in event doesn't match ICUSTAY_ID in stays.csv for the same HADM_ID.

    subdirectories = os.listdir(subjects_root_path)
    subjects = sorted([d for d in subdirectories if is_subject_folder(d) and \
                       os.path.isdir(os.path.join(subjects_root_path, d))])

    print(f"Found {len(subjects)} potential subject directories.")

    for subject in tqdm(subjects, desc="Validating events per subject"):
        subject_dir = os.path.join(subjects_root_path, subject)
        stays_path = os.path.join(subject_dir, 'stays.csv')
        events_path = os.path.join(subject_dir, 'events.csv')

        # Check if required files exist
        if not os.path.exists(stays_path):
            continue
        if not os.path.exists(events_path):
            continue

        try:
            # Read stays, ensuring IDs are strings for merging
            stays_df = pd.read_csv(
                stays_path,
                index_col=False,
                dtype={'HADM_ID': str, "ICUSTAY_ID": str}
            )
            # Standardize column names
            stays_df.columns = stays_df.columns.str.upper()
            # Keep only essential columns for validation
            stays_df = stays_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].astype(str)

            # Assert that stays file has unique HADM_ID and ICUSTAY_ID per subject
            if not stays_df['HADM_ID'].is_unique:
                 print(f"Warning: Duplicate HADM_ID found in stays.csv for subject {subject}. Check script 1 filtering.")
                 stays_df = stays_df.drop_duplicates(subset=['HADM_ID'], keep='first')
            if not stays_df['ICUSTAY_ID'].is_unique:
                 print(f"Warning: Duplicate ICUSTAY_ID found in stays.csv for subject {subject}. Check script 1 filtering.")
                 stays_df = stays_df.drop_duplicates(subset=['ICUSTAY_ID'], keep='first')

            # Read events, ensuring IDs are strings
            events_df = pd.read_csv(
                events_path,
                index_col=False,
                dtype={'HADM_ID': str, "ICUSTAY_ID": str, 'SUBJECT_ID': str}
            )
            events_df.columns = events_df.columns.str.upper()
            n_events_initial += events_df.shape[0]

            # 1. Drop events where HADM_ID is missing or invalid (e.g., '-1' from script 1)
            original_count = events_df.shape[0]
            events_df = events_df.dropna(subset=['HADM_ID'])
            events_df = events_df[events_df['HADM_ID'] != '-1']
            current_empty_hadm = original_count - events_df.shape[0]
            empty_hadm += current_empty_hadm

            if events_df.empty:
                # Overwrite with empty file if all events were invalid
                pd.DataFrame(columns=events_df.columns).to_csv(events_path, index=False)
                continue

            # 2. Merge events with stays based on HADM_ID
            # Use 'inner' merge to keep only events matching a HADM_ID in stays.csv
            merged_df = events_df.merge(
                stays_df[['HADM_ID', 'ICUSTAY_ID']], # Only need these from stays
                on='HADM_ID',
                how='inner', # Keep only events with matching HADM_ID in stays
                suffixes=['', '_r'] # _r denotes the ICUSTAY_ID from stays.csv
            )
            current_no_hadm_in_stay = events_df.shape[0] - merged_df.shape[0]
            no_hadm_in_stay += current_no_hadm_in_stay

            if merged_df.empty:
                pd.DataFrame(columns=events_df.columns).to_csv(events_path, index=False)
                continue

            # 3. Handle ICUSTAY_ID: Recover if missing, validate if present
            # Count events where ICUSTAY_ID was initially missing or placeholder '-1'
            cur_no_icustay = merged_df['ICUSTAY_ID'].isnull().sum() + (merged_df['ICUSTAY_ID'] == '-1').sum()
            no_icustay += cur_no_icustay

            # Attempt recovery: If ICUSTAY_ID is missing/placeholder, use ICUSTAY_ID_r from stays
            missing_or_placeholder = merged_df['ICUSTAY_ID'].isnull() | (merged_df['ICUSTAY_ID'] == '-1')
            merged_df.loc[missing_or_placeholder, 'ICUSTAY_ID'] = merged_df.loc[missing_or_placeholder, 'ICUSTAY_ID_r']

            # Check how many were successfully recovered
            recovered_count = cur_no_icustay - (merged_df['ICUSTAY_ID'].isnull().sum() + (merged_df['ICUSTAY_ID'] == '-1').sum())
            recovered += recovered_count
            
            # Check if any could still not be recovered
            current_could_not_recover = merged_df['ICUSTAY_ID'].isnull().sum() + (merged_df['ICUSTAY_ID'] == '-1').sum()
            could_not_recover += current_could_not_recover

            # Drop events where recovery failed (if any)
            merged_df = merged_df.dropna(subset=['ICUSTAY_ID'])
            merged_df = merged_df[merged_df['ICUSTAY_ID'] != '-1']

            # 4. Validate existing ICUSTAY_ID against the one from stays.csv
            # Keep only events where the event's ICUSTAY_ID matches the stay's ICUSTAY_ID_r
            mismatch_df = merged_df[merged_df['ICUSTAY_ID'] != merged_df['ICUSTAY_ID_r']]
            current_icustay_id_mismatch = mismatch_df.shape[0]
            icustay_id_mismatch += current_icustay_id_mismatch

            # Keep only matching rows
            merged_df = merged_df[merged_df['ICUSTAY_ID'] == merged_df['ICUSTAY_ID_r']]

            # Select and order columns for the final validated events file
            final_columns = [
                'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID',
                'VALUE', 'VALUEUOM'
            ]
            
            # Ensure SUBJECT_ID is present
            if 'SUBJECT_ID' not in merged_df.columns:
                 # If SUBJECT_ID was dropped, try to get it from stays_df based on HADM_ID
                 subject_map = stays_df.set_index('HADM_ID')['SUBJECT_ID']
                 merged_df['SUBJECT_ID'] = merged_df['HADM_ID'].map(subject_map)

            # Handle potential missing SUBJECT_ID if merge failed unexpectedly
            merged_df = merged_df.dropna(subset=['SUBJECT_ID'])
            merged_df['SUBJECT_ID'] = merged_df['SUBJECT_ID'].astype(int).astype(str)

            to_write = merged_df[final_columns]
            n_events_final += to_write.shape[0]

            # Overwrite the original events.csv with the validated data
            to_write.to_csv(events_path, index=False)

        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            continue

    print("\n--- Validation Summary ---")
    print(f"Total events read: {n_events_initial}")
    print(f"Events removed due to missing/invalid HADM_ID: {empty_hadm}")
    print(f"Events removed due to HADM_ID not in stays.csv: {no_hadm_in_stay}")
    print(f"Events initially missing ICUSTAY_ID: {no_icustay}")
    print(f"  Successfully recovered ICUSTAY_ID: {recovered}")
    print(f"  Failed to recover ICUSTAY_ID: {could_not_recover}")
    print(f"Events removed due to ICUSTAY_ID mismatch: {icustay_id_mismatch}")
    print(f"Total events remaining after validation: {n_events_final}")
    print(f"Percentage of events kept: {n_events_final / n_events_initial * 100 if n_events_initial > 0 else 0:.2f}%")
    
    if could_not_recover > 0:
        print("Warning: Some ICUSTAY_IDs could not be recovered. Check data integrity.")

    print("Event validation completed successfully!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Validate events against stay information')
    
    parser.add_argument('--data_path', type=str, default=DATA_ROOT_PATH,
                       help='Root path where subject data is stored')
    
    args = parser.parse_args()
    
    print("Starting Step 2: Event Validation")
    print("=" * 50)
    print(f"Data Path: {args.data_path}")
    print("=" * 50)
    
    # Validate path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("Please run Step 1 (extract_subject_events.py) first.")
        return
    
    # Run validation
    validate_events(args.data_path)


if __name__ == "__main__":
    main() 