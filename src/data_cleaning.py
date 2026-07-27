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
    
    m = re.match(r"(\d+)\+(\d+)", raw)
    if m:
        weeks = int(m.group(1))
        days = int(m.group(2))
        return weeks + days/7
    
    try:
        return float(raw)
    except ValueError:
        return np.nan

if __name__ == "__main__":
    male, female = load_data()
    print("Male shape:", male.shape)
    print("Female shape:", female.shape)
    print("Male columns:", male.columns.tolist())

    