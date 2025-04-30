#!/usr/bin/env python3
import pandas as pd
import os

def export_rows_to_txt(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
    # for idx, row in df.head(2).iterrows():
        filename = os.path.join(output_dir, f"{idx}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {row['Title']}\n")
            f.write(f"Tag: {row['Tag']}\n\n")
            f.write(row['Content'])

    print(f"Exported {len(df)} files to '{output_dir}'")

if __name__ == "__main__":
    csv_path = "Financial.csv"
    output_dir = "output_texts"
    export_rows_to_txt(csv_path, output_dir)

