#! /usr/bin/env python3

import pandas as pd
import sys
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",
                        dest="input_path",
                        default="./prediction.csv",
                        help="Path to the original file.\n\tdefault : './prediction.csv'")
    parser.add_argument("--out",
                        dest="output_path",
                        default="./ready_for_submission.csv",
                        help="Path to the output file.\n\tdefault : './ready_for_submission.csv'")
    parser.add_argument("--pred",
                        dest="prediction_label",
                        default="Prediction",
                        help="Prediction's column's name.\n\tdefault : 'Prediction'")
    parser.add_argument("--byday",
                        dest="by_day",
                        default=False,
                        help="Whether the dataset is already aggregated by day of not.\n\tdefault : False")
    parser.add_argument("--addone",
                        dest="add_one",
                        default=True,
                        help="Whether to add one to the Prediction columns or not.\n\tdefault : True")
    return parser


def main(input_path="", output_path="", prediction_label="Prediction", by_day="False", add_one="True"):
    print("Loading",input_path)
    df = pd.read_csv(input_path)

    print("Columns checks : ", end="")
    try:
        if "Id" not in df.columns:
            raise KeyError
        if prediction_label not in df.columns:
            raise KeyError
    except KeyError:
        print("ERROR : {} or {} key not found id file {}.".format("Id", prediction_label, input_path))
        exit(-2)
    print("passed.")

    print("Keep only 'Id' and",prediction_label)
    df = df[["Id", prediction_label]]

    if not by_day :
        print("Aggregate data by day.")
        df["Id"] = df["Id"].astype("category")
        df = df.groupby("Id").agg({prediction_label: pd.Series.sum})
        # print("Set Id as index.")
        # df.set_index("Id", inplace=True)

    print("Get Baseline.")
    baseline = pd.read_csv("../Test/Test/Baselines/Baseline_observation_test.csv")

    print("Remove Ids not in Baseline.")
    submission = baseline.drop("Prediction", axis=1).merge(df, how="left", on="Id")

    print(f"\nSum of NaNs :\n\n{submission.isna().sum()}\n\n")
    if submission[prediction_label].isna().sum() > 0:
        print("fill nans with average.")
        submission[prediction_label].fillna(submission[prediction_label].mean(), inplace=True)

    if len(submission) != 85140:
        print("Warning : len(df) != len(Baseline) i.e. {} != {}".format(len(submission), 183498))

    print("Write output to :", output_path)

    if add_one :
        submission[prediction_label] = submission[prediction_label] + 1

    submission.to_csv(output_path, index=False)
    print(f"File save as {output_path}.")
    return submission


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    main(input_path=args.input_path,
         output_path=args.output_path,
         prediction_label=args.prediction_label,
         by_day=args.by_day,
         add_one=args.add_one)