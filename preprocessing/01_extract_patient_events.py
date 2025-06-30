"""
Step 1: Extract Subject Events from MIMIC-III

This script extracts subjects and their associated events from MIMIC-III tables.
It applies initial filtering criteria and saves individual subject data.

Usage:
    python 01_extract_subject_events.py
    
Make sure to update the config.py file with your MIMIC-III and data paths before running.
"""

import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import sys
import argparse

# Add config to path
sys.path.append('../config')
from config import *

def extract_subject_data(mimic3_path, output_path, event_tables, metavision_only=True, minimum_age=18):
    """
    Extract subjects and their associated events from MIMIC-III tables.
    
    Args:
        mimic3_path: Path to MIMIC-III CSV files
        output_path: Path to save processed subject data
        event_tables: List of event tables to process
        metavision_only: If True, only keep patients with all Metavision ICU stays
        minimum_age: Minimum patient age for inclusion
    """

    # --- Read and Prepare Metadata Tables ---
    print('Reading metadata tables (PATIENTS, ADMISSIONS, ICUSTAYS)...')
    try:
        print('Reading PATIENTS.csv...')
        pats = pd.read_csv(
            os.path.join(mimic3_path, 'PATIENTS.csv'),
            header=0, index_col=None, dtype={'SUBJECT_ID': int},
            usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD'])
        pats['DOB'] = pd.to_datetime(pats['DOB'], errors='coerce')
        pats['DOD'] = pd.to_datetime(pats['DOD'], errors='coerce')

        print('Reading ADMISSIONS.csv...')
        admits = pd.read_csv(
            os.path.join(mimic3_path, 'ADMISSIONS.csv'),
            header=0, index_col=None, dtype={'SUBJECT_ID': int, 'HADM_ID': int},
            usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'])
        admits['ADMITTIME'] = pd.to_datetime(admits['ADMITTIME'], errors='coerce')
        admits['DISCHTIME'] = pd.to_datetime(admits['DISCHTIME'], errors='coerce')
        admits['DEATHTIME'] = pd.to_datetime(admits['DEATHTIME'], errors='coerce')

        stays = pd.read_csv(
            os.path.join(mimic3_path, 'ICUSTAYS.csv'),
            header=0, index_col=None, dtype={'SUBJECT_ID': int, 'HADM_ID': int, 'ICUSTAY_ID': int})
        required_stay_cols = {'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME',
                              'FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'LOS', 'DBSOURCE'}
        if not required_stay_cols.issubset(stays.columns):
            missing = required_stay_cols - set(stays.columns)
            print(f"Missing required columns in ICUSTAYS.csv: {missing}")
            return
        stays['INTIME'] = pd.to_datetime(stays['INTIME'], errors='coerce')
        stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'], errors='coerce')

    except FileNotFoundError as e:
        print(f"Error reading input CSV file: {e}. Please check mimic3_path.")
        return
    except Exception as e:
        print(f"Error during table reading or initial date conversion: {e}", exc_info=True)
        return

    print(f"Initial unique counts: ICUSTAY_IDs={stays['ICUSTAY_ID'].nunique()}, HADM_IDs={stays['HADM_ID'].nunique()}, SUBJECT_IDs={stays['SUBJECT_ID'].nunique()}")

    # --- Metavision Filtering (if enabled) ---
    if metavision_only:
        print('Filtering to Metavision-only patients...')
        original_subjects = stays['SUBJECT_ID'].nunique()
        original_stays = len(stays)
        
        # Group by subject and check if ALL stays are Metavision
        subject_dbsource_check = stays.groupby('SUBJECT_ID')['DBSOURCE'].apply(
            lambda x: (x == 'metavision').all()
        )
        metavision_subjects = subject_dbsource_check[subject_dbsource_check].index
        
        # Filter to only Metavision subjects
        stays = stays[stays['SUBJECT_ID'].isin(metavision_subjects)].copy()
        
        print(f" Subjects after Metavision filter: {stays['SUBJECT_ID'].nunique()} (removed {original_subjects - stays['SUBJECT_ID'].nunique()})")
        print(f" Stays after Metavision filter: {len(stays)} (removed {original_stays - len(stays)})")
        
        if stays.empty:
            print("No Metavision-only patients found. Stopping.")
            return
    
    # --- Filter Stays ---
    print('Filtering stays: Removing transfers...')
    original_rows = stays.shape[0]
    stays = stays.loc[(stays['FIRST_WARDID'] == stays['LAST_WARDID']) & (stays['FIRST_CAREUNIT'] == stays['LAST_CAREUNIT'])]
    stays = stays.drop(columns=['FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT'], errors='ignore')
    print(f" Stays after removing transfers: {stays.shape[0]} (removed {original_rows - stays.shape[0]})")

    print("Merging stays with admissions and patients...")
    try:
        stays = stays.merge(admits, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        stays = stays.merge(pats, on=['SUBJECT_ID'], how='inner')
        print(f" Stays after merging: {stays.shape[0]}")
    except Exception as e:
        print(f"Error during merging of tables: {e}", exc_info=True)
        return

    print('Filtering stays: Keeping only admissions with exactly 1 ICU stay...')
    icu_counts = stays.groupby('HADM_ID')['ICUSTAY_ID'].transform('count')
    stays = stays[icu_counts == 1].copy()
    print(f" Stays after keeping HADM_IDs with 1 ICUSTAY: {stays.shape[0]}")

    print(f"Filtering stays: Calculating and filtering by age (>={minimum_age})...")
    valid_dates_mask = stays['INTIME'].notna() & stays['DOB'].notna()
    stays['AGE'] = np.nan
    if valid_dates_mask.any():
        try:
            time_diff_valid = stays.loc[valid_dates_mask, 'INTIME'] - stays.loc[valid_dates_mask, 'DOB']
            valid_diff_mask = valid_dates_mask & time_diff_valid.notna()
            if valid_diff_mask.any():
                 age_in_days_valid = time_diff_valid[valid_diff_mask].dt.days
                 stays.loc[valid_diff_mask, 'AGE'] = age_in_days_valid / 365.25
        except OverflowError as e: 
            print(f"OverflowError during age calculation: {e}.")
        except Exception as e: 
            print(f"Unexpected error during age calculation: {e}", exc_info=True)
    
    # Handle age >89 (anonymized as negative in MIMIC)
    age_ge_89_mask = stays['AGE'] < 0
    if age_ge_89_mask.any(): 
        stays.loc[age_ge_89_mask, 'AGE'] = 91.4
    
    # Impute missing ages
    nan_age_mask = stays['AGE'].isna()
    if nan_age_mask.any(): 
        stays['AGE'].fillna(91.4, inplace=True)
    
    original_count_before_age_filter = stays.shape[0]
    stays = stays.loc[stays['AGE'] >= minimum_age].copy()
    print(f" Stays after age filter (>={minimum_age}): {stays.shape[0]} (removed {original_count_before_age_filter - stays.shape[0]})")

    if stays.empty:
        print("No stays remaining after all filtering. Stopping.")
        return

    print('Adding in-hospital mortality info...')
    date_cols_for_mort = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD']
    for col in date_cols_for_mort:
        if col in stays.columns: 
            stays[col] = pd.to_datetime(stays[col], errors='coerce')
    
    died_in_hosp_dod = (stays['DOD'].notna()) & (stays['ADMITTIME'].notna()) & (stays['DISCHTIME'].notna()) & (stays['ADMITTIME'] <= stays['DOD']) & (stays['DISCHTIME'] >= stays['DOD'])
    died_in_hosp_deathtime = (stays['DEATHTIME'].notna()) & (stays['ADMITTIME'].notna()) & (stays['DISCHTIME'].notna()) & (stays['ADMITTIME'] <= stays['DEATHTIME']) & (stays['DISCHTIME'] >= stays['DEATHTIME'])
    stays['MORTALITY'] = (died_in_hosp_dod | died_in_hosp_deathtime).astype(int)
    print(f" Calculated in-hospital mortality for {len(stays)} final valid stays.")

    # --- Save stays.csv per subject ---
    print('Saving filtered stays per subject...')
    subjects_with_stays_ids = stays['SUBJECT_ID'].unique()
    print(f"Total unique subjects with stays after all filters: {len(subjects_with_stays_ids)}")

    subjects_to_process_events_for = set(str(s) for s in subjects_with_stays_ids)

    for subject_id_int in tqdm(subjects_with_stays_ids, desc="Saving stays.csv"):
        subject_id_str = str(subject_id_int)
        subject_stays = stays.loc[stays['SUBJECT_ID'] == subject_id_int].sort_values(by='INTIME')
        dn = os.path.join(output_path, subject_id_str)
        os.makedirs(dn, exist_ok=True)
        try:
            for col in subject_stays.select_dtypes(include=['datetime64[ns]']).columns:
                 subject_stays[col] = subject_stays[col].astype(str)
            subject_stays.to_csv(os.path.join(dn, 'stays.csv'), index=False)
        except Exception as e:
            print(f"Error saving stays.csv for subject {subject_id_str}: {e}")

    # --- Process and Save events.csv per subject ---
    print('Reading event tables and processing events per subject...')
    event_columns = {
        'CHARTEVENTS': ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'],
        'LABEVENTS': ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'],
        'OUTPUTEVENTS': ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    }
    event_dtypes = {
        'SUBJECT_ID': 'int32', 'HADM_ID': 'float64', 'ICUSTAY_ID': 'float64',
        'ITEMID': 'int32', 'VALUE': 'object', 'VALUEUOM': 'object'
    }
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']

    # Dictionary to hold list of event dataframes for each subject
    subject_events_data = {subj_id: [] for subj_id in subjects_to_process_events_for}
    total_events_processed = 0

    for table in tqdm(event_tables, desc="Processing Event Tables"):
        print(f" Reading {table}...")
        tn = os.path.join(mimic3_path, table + '.csv')
        if not os.path.exists(tn):
            print(f" {table}.csv not found in {mimic3_path}. Skipping.")
            continue

        try:
            # Read the entire table - REQUIRES SIGNIFICANT MEMORY
            df_table = pd.read_csv(
                tn,
                usecols=event_columns[table],
                dtype=event_dtypes,
                parse_dates=['CHARTTIME'],
                low_memory=False
            )
            print(f"  Read {len(df_table)} rows from {table}.")

            # Filter to only eligible subjects
            df_table['SUBJECT_ID'] = df_table['SUBJECT_ID'].astype(str)
            df_filtered = df_table[df_table['SUBJECT_ID'].isin(subjects_to_process_events_for)].copy()
            print(f"  Kept {len(df_filtered)} rows after filtering for eligible subjects.")
            del df_table  # Free memory

            if df_filtered.empty: 
                continue

            # Data Cleaning
            df_filtered['HADM_ID'] = df_filtered['HADM_ID'].fillna(-1).astype(int)
            if 'ICUSTAY_ID' in df_filtered.columns:
                df_filtered['ICUSTAY_ID'] = df_filtered['ICUSTAY_ID'].fillna(-1).astype(int)
            else:
                df_filtered['ICUSTAY_ID'] = -1
            
            # Convert IDs to string for consistency
            df_filtered['SUBJECT_ID'] = df_filtered['SUBJECT_ID'].astype(str)
            df_filtered['HADM_ID'] = df_filtered['HADM_ID'].astype(str)
            df_filtered['ICUSTAY_ID'] = df_filtered['ICUSTAY_ID'].astype(str)
            df_filtered['ITEMID'] = df_filtered['ITEMID'].astype(str)
            df_filtered['VALUEUOM'] = df_filtered['VALUEUOM'].fillna('').astype(str)
            df_filtered['VALUE'] = df_filtered['VALUE'].astype(str)

            # Clean timestamps
            df_filtered['CHARTTIME'] = pd.to_datetime(df_filtered['CHARTTIME'], errors='coerce')
            df_filtered.dropna(subset=['CHARTTIME'], inplace=True)

            df_filtered = df_filtered[obs_header]  # Reorder/select columns

            # Append processed data to the dictionary keyed by subject_id
            for subject_id_str, group in df_filtered.groupby('SUBJECT_ID'):
                subject_events_data[subject_id_str].append(group)
                total_events_processed += len(group)

        except MemoryError:
             print(f"MEMORY ERROR while processing {table}! Cannot load full table. Please use chunking.")
             raise
        except Exception as e:
             print(f"Error processing {table}: {e}", exc_info=True)

    # --- Concatenate and Save Events for each subject ---
    print(f"Finished reading event tables. Processed {total_events_processed} potential events.")
    print(f"Saving events.csv for {len(subjects_to_process_events_for)} subjects...")

    for subject_id_str in tqdm(subjects_to_process_events_for, desc="Saving events.csv"):
        dn = os.path.join(output_path, subject_id_str)
        fn = os.path.join(dn, 'events.csv')
        list_of_dfs = subject_events_data.get(subject_id_str, [])

        if list_of_dfs:
            try:
                final_df = pd.concat(list_of_dfs, ignore_index=True)
                final_df.sort_values(by='CHARTTIME', inplace=True)
                # Convert CHARTTIME to string before saving
                final_df['CHARTTIME'] = final_df['CHARTTIME'].astype(str)
                final_df.to_csv(fn, index=False, quoting=csv.QUOTE_MINIMAL)
            except Exception as e:
                print(f"Error during concat/sort/save of events.csv for {subject_id_str}: {e}", exc_info=True)
        else:
            print(f"No events found for subject {subject_id_str} after processing all tables.")

    print("Subject event extraction completed successfully!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Extract subject events from MIMIC-III')
    
    parser.add_argument('--mimic_path', type=str, default=MIMIC_CSV_PATH,
                       help='Path to MIMIC-III CSV files')
    parser.add_argument('--output_path', type=str, default=DATA_ROOT_PATH,
                       help='Path to save processed subject data')
    parser.add_argument('--event_tables', nargs='+', default=EVENT_TABLES,
                       help='Event tables to process')
    parser.add_argument('--metavision_only', action='store_true', default=METAVISION_ONLY,
                       help='Only include patients with all Metavision ICU stays')
    parser.add_argument('--minimum_age', type=int, default=MINIMUM_AGE,
                       help='Minimum patient age for inclusion')
    
    args = parser.parse_args()
    
    print("Starting Step 1: Subject Event Extraction")
    print("=" * 50)
    print(f"MIMIC-III Path: {args.mimic_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Event Tables: {args.event_tables}")
    print(f"Metavision Only: {args.metavision_only}")
    print(f"Minimum Age: {args.minimum_age}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.mimic_path):
        print(f"Error: MIMIC-III path does not exist: {args.mimic_path}")
        return
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run extraction
    extract_subject_data(
        mimic3_path=args.mimic_path,
        output_path=args.output_path,
        event_tables=args.event_tables,
        metavision_only=args.metavision_only,
        minimum_age=args.minimum_age
    )


if __name__ == "__main__":
    main() 