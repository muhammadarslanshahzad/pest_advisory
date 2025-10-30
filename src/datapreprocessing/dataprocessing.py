import pandas as pd
import numpy as np


class DataCleaning():

    def __init__(self):
        pass

    def clean_cotton_pest_data(self, csv_path):
        """
        Clean Punjab cotton pest survey data
        """
        
        print("🧹 CLEANING COTTON PEST SURVEY DATA\n")
        print("="*60)
        
        # ============================================================
        # STEP 1: LOAD DATA
        # ============================================================
        df = pd.read_csv(csv_path)
        initial_rows = len(df)
        print(f"📂 Loaded: {initial_rows:,} rows, {len(df.columns)} columns")
        
        # ============================================================
        # STEP 2: NORMALIZE COLUMN NAMES
        # ============================================================
        # Remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Convert to uppercase for consistency
        df.columns = df.columns.str.upper()
        
        print("✅ Column names normalized")
        
        # ============================================================
        # STEP 3: IDENTIFY COLUMN TYPES
        # ============================================================
        # Core columns
        core_cols = ['DISTRICT', 'TEHSIL', 'DATE']
        
        # Survey metadata
        survey_cols = ['TOTAL SPOTS VISITED', 'TOTAL AREA VISITED']
        
        # Pest columns
        above_etl_cols = [col for col in df.columns if 'ABOVE ETL' in col]
        below_etl_cols = [col for col in df.columns if 'BELOW ETL' in col]
        clcv_cols = [col for col in df.columns if 'CLCV' in col]
        wilt_cols = [col for col in df.columns if 'WILT' in col]
        
        all_pest_cols = above_etl_cols + below_etl_cols + clcv_cols + wilt_cols
        
        print(f"   📊 Found {len(above_etl_cols)} pests above ETL")
        print(f"   📊 Found {len(below_etl_cols)} pests below ETL")
        print(f"   📊 Found {len(clcv_cols)} CLCV measures")
        print(f"   📊 Found {len(wilt_cols)} Wilt measures")
        
        # ============================================================
        # STEP 4: HANDLE CRITICAL MISSING VALUES
        # ============================================================
        # Drop rows without district (can't analyze)
        if 'DISTRICT' in df.columns:
            before = len(df)
            df = df.dropna(subset=['DISTRICT'])
            dropped = before - len(df)
            if dropped > 0:
                print(f"⚠️  Dropped {dropped} rows with missing DISTRICT")
        
        # ============================================================
        # STEP 5: FIX INVALID SURVEY METADATA
        # ============================================================
        # Remove rows with 0 spots visited (invalid survey)
        if 'TOTAL SPOTS VISITED' in df.columns:
            invalid_spots = df['TOTAL SPOTS VISITED'] <= 0
            if invalid_spots.sum() > 0:
                print(f"⚠️  Found {invalid_spots.sum()} rows with TOTAL SPOTS ≤ 0")
                df = df[~invalid_spots]
                print(f"   → Removed invalid rows")
        
        # Remove rows with 0 area visited (invalid survey)
        if 'TOTAL AREA VISITED' in df.columns:
            invalid_area = df['TOTAL AREA VISITED'] <= 0
            if invalid_area.sum() > 0:
                print(f"⚠️  Found {invalid_area.sum()} rows with TOTAL AREA ≤ 0")
                df = df[~invalid_area]
                print(f"   → Removed invalid rows")
        
        # ============================================================
        # STEP 6: FILL MISSING PEST COUNTS
        # ============================================================
        # Missing pest count = not observed = 0
        for col in all_pest_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    df[col] = df[col].fillna(0)
        
        print(f"✅ Filled {len(all_pest_cols)} pest columns (missing → 0)")
        
        # ============================================================
        # STEP 7: CONVERT DATA TYPES
        # ============================================================
        # Convert pest counts to numeric
        for col in all_pest_cols:
            if col in df.columns:
                # Convert to numeric, force errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill conversion errors with 0
                df[col] = df[col].fillna(0)
                
                # Ensure no negative values
                df[col] = df[col].abs()
        
        # Convert survey metadata to numeric
        for col in survey_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).abs()
        
        print("✅ Converted all numeric columns")
        
        # ============================================================
        # STEP 8: HANDLE DATES
        # ============================================================
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            missing_dates = df['DATE'].isnull().sum()
            if missing_dates > 0:
                print(f"⚠️  {missing_dates} rows with invalid dates")
        
        # ============================================================
        # STEP 9: HANDLE OUTLIERS IN PEST COUNTS
        # ============================================================
        # Based on your data: max = 100 for most pests (likely data entry limit)
        # These are NOT outliers, they're real caps
        # But let's check for unrealistic spikes
        
        print("\n📊 OUTLIER ANALYSIS:")
        
        for col in above_etl_cols:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                extreme = df[col] > q99 * 2  # More than 2x the 99th percentile
                
                if extreme.sum() > 0:
                    print(f"   {col}: {extreme.sum()} extreme values (>{q99*2:.1f})")
                    # For your data, 100 is the cap, so don't remove these
                    # Just flag for awareness
        
        # ============================================================
        # STEP 10: REMOVE DUPLICATES
        # ============================================================
        before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before_dedup - len(df)
        
        if duplicates_removed > 0:
            print(f"✅ Removed {duplicates_removed} duplicate rows")
        
        # ============================================================
        # STEP 11: CREATE USEFUL DERIVED COLUMNS
        # ============================================================
        # Total pests above ETL (useful for severity scoring)
        df['TOTAL_PESTS_ABOVE_ETL'] = df[above_etl_cols].sum(axis=1)
        
        # Total pests below ETL
        df['TOTAL_PESTS_BELOW_ETL'] = df[below_etl_cols].sum(axis=1)
        
        # Pest pressure per acre
        if 'TOTAL AREA VISITED' in df.columns:
            df['PEST_PRESSURE_PER_ACRE'] = (
                df['TOTAL_PESTS_ABOVE_ETL'] / df['TOTAL AREA VISITED']
            ).replace([np.inf, -np.inf], 0)
        
        print("✅ Created derived columns")
        
        # ============================================================
        # STEP 12: FINAL VALIDATION
        # ============================================================
        print("\n" + "="*60)
        print("🔍 FINAL DATA QUALITY CHECK:")
        print("="*60)
        
        # Check missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            print("\n⚠️  Remaining missing values:")
            print(missing_summary[missing_summary > 0])
        else:
            print("✅ No missing values")
        
        # Check data ranges
        print(f"\n📊 Survey Coverage:")
        # print(f"   TEHSILS: {df['TEHSIL'].nunique()}")
        print(f"   Total spots: {df['TOTAL SPOTS VISITED'].sum():,.0f}")
        print(f"   Total area: {df['TOTAL AREA VISITED'].sum():,.0f} acres")
        
        if 'DATE' in df.columns:
            print(f"   Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        
        # Summary
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        print("\n" + "="*60)
        print("✨ CLEANING COMPLETE!")
        print("="*60)
        print(f"   Initial rows:  {initial_rows:,}")
        print(f"   Final rows:    {final_rows:,}")
        print(f"   Removed:       {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
        print(f"   Remaining:     {final_rows/initial_rows*100:.1f}%")
        
        return df

if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaning()
    cleaned_df = cleaner.clean_cotton_pest_data('data/data.csv')
    cleaned_df.to_csv('data/pest_survey_cleaned1.csv', index=False)
    print("\n✅ Cleaned data saved to 'data/pest_survey_cleaned1.csv'")