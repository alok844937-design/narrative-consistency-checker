import argparse 
import pandas as pd 
from src.pipeline import run_pipeline 

def main(input_path, output_path):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    outputs = []

    for _, row in df.iterrows():
        pred = run_pipeline(row["novel_text"], row["backstory"])
        outputs.append(pred)

    df["prediction"] = outputs 
    df[["prediction"]].to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)