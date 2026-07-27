import os
import pandas as pd
import numpy as np
import re

DATA_PATH = "Dataset.xlsx"

COLUMN_NAMES = [
    "sample_id", "woman_id", "age", "height", "weight", "lmp_date",
    "conception_method", "test_date", "blood_draw_number", "gestational_age_raw",
    "bmi", "raw_reads", "alignment_ratio", "duplicate_ratio", "unique_reads",
    "gc_content", "z13", "z18", "z21", "zx", "zy", "y_concentration",
    "x_concentration", "gc13", "gc18", "gc21", "filtered_read_ratio",
    "aneuploidy", "pregnancy_count", "delivery_count", "foetal_health"
]

DEMOGRAPHIC_COLS = ["age", "height", "weight", "bmi", "pregnancy_count", "delivery_count"]

SEQUENCING_COLS = ["raw_reads", "alignment_ratio", "duplicate_ratio", "unique_reads",
                   "gc_content", "filtered_read_ratio"]

ZSCORE_COLS = ["z13", "z18", "z21", "zx", "zy"]

GC_REGION_COLS = ["gc13", "gc18", "gc21"]


def load_data():
    male = pd.read_excel(
        DATA_PATH, sheet_name="Male foetuses",
        header=None, names=COLUMN_NAMES, skiprows=1
    )
    female = pd.read_excel(
        DATA_PATH, sheet_name="Female foetuses",
        header=None, names=COLUMN_NAMES, skiprows=1
    )
    male["sex"] = "male"
    female["sex"] = "female"
    return male, female

def parse_gestational_age(raw):
    if pd.isna(raw):
        return np.nan
    
    raw = str(raw).strip().lower()
    
    m = re.match(r"(\d+)w\+(\d+)", raw)
    if m:
        weeks = int(m.group(1))
        days = int(m.group(2))
        return weeks + days/7
    
    try:
        return float(raw)
    except ValueError:
        return np.nan   

def clean_data(df):
    df = df.copy()
    
    df["gestational_age_weeks"] = df["gestational_age_raw"].apply(parse_gestational_age)
    
    numeric_cols = DEMOGRAPHIC_COLS + SEQUENCING_COLS + ZSCORE_COLS + GC_REGION_COLS + ["x_concentration", "y_concentration"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["test_date_dt"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["lmp_date_dt"] = pd.to_datetime(df["lmp_date"], errors="coerce")
    
    return df

def flag_quality(df):
    df = df.copy()
    
    df["gc_out_of_range"] = ((df["gc_content"] < 0.37) | (df["gc_content"] > 0.43)).astype(int)
    df["low_reads"] = (df["raw_reads"] < 3_000_000).astype(int)
    df["low_alignment"] = (df["alignment_ratio"] < 0.70).astype(int)
    df["high_duplicates"] = (df["duplicate_ratio"] > 0.10).astype(int)
    
    df["quality_flag"] = (df["gc_out_of_range"] | df["low_reads"] | 
                          df["low_alignment"] | df["high_duplicates"]).astype(int)
    
    return df

if __name__ == "__main__":
    male_raw, female_raw = load_data()
    
    male = clean_data(male_raw)
    male = flag_quality(male)
    
    female = clean_data(female_raw)
    female = flag_quality(female)
    
    df = pd.concat([male, female], ignore_index=True)
    
    print(f"Total samples: {len(df)}")
    print(f"  Male: {len(male)}, Female: {len(female)}")
    print(f"  Missing GA: {df['gestational_age_weeks'].isna().sum()}")
    print(f"  Quality flags: {df['quality_flag'].sum()}")
    print(f"  GA range: {df['gestational_age_weeks'].min():.1f} - {df['gestational_age_weeks'].max():.1f} weeks")
    print(f"  BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    
    os.makedirs("outputs", exist_ok=True)
    male.to_csv("outputs/cleaned_male.csv", index=False)
    female.to_csv("outputs/cleaned_female.csv", index=False)
    print("\nSaved: outputs/cleaned_male.csv")
    print("Saved: outputs/cleaned_female.csv")

    