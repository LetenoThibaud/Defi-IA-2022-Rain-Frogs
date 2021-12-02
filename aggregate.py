#! /usr/bin/env python3

import pandas as pd
import sys


def main(input_path="", output_path="", prediction_label="Prediction"):
    if input_path == "":
        input_path = "./prediction.csv"
    if output_path == "":
        output_path = "./ready_for_submission.csv"

    df = pd.read_csv(input_path)

    try:
        if "Id" not in df.columns:
            raise KeyError
        if prediction_label not in df.columns:
            raise KeyError
    except KeyError:
        print("ERROR : {} or {} key not found id file {}.".format("Id", prediction_label, input_path))
        exit(-2)

    if len(df) > 100000:  # assuming the data used is not _by_day.csv
        df = df[["Id", prediction_label]]

        df = df.groupby(["Id"]).agg({prediction_label: pd.Series.sum})
        df.set_index("Id", inplace=True)

    baseline = pd.read_csv("../Test/Test/Baselines/Baseline_observation_test.csv")
    submission = baseline.drop("Prediction",axis=1).merge(df, how="left", on="Id")

    print(f"\nSum of NaNs :\n\n{submission.isna().sum()}\n\n")

    if len(submission) != 85140:
        print("Warning : len(df) != len(Baseline) i.e. {} != {}".format(len(submission), 183498))

    submission.to_csv(output_path)
    print(f"File save as {output_path}.", index=False)
    return submission


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        if len(sys.argv) == 2:
            main(sys.argv[1])
        elif len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2])
        else:
            main(sys.argv[1], sys.argv[2], sys.argv[3])
