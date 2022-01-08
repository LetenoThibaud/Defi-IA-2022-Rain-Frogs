#! /usr/bin/env python3

import pandas as pd
import sys


def main(input_path="", output_path="", prediction_label="Prediction"):
    if input_path == "":
        input_path = "./prediction.csv"
    if output_path == "":
        output_path = "./ready_for_submission.csv"

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

    if False :#len(df) > 100000:  # assuming the data used is not _by_day.csv
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

    # df[prediction_label] = df[prediction_label] + 1
    submission[prediction_label] = submission[prediction_label] + 1
    submission.to_csv(output_path, index=False)
    print(f"File save as {output_path}.")
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
