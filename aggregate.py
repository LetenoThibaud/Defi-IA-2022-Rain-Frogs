#! /usr/bin/env python3

import pandas as pd
import sys


def main(input_path="", output_path="", prediction_label="prediction"):
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

    df = df[["Id", prediction_label]]

    df = df.groupby(["Id"]).agg({prediction_label: pd.Series.sum})
    df.set_index("Id", inplace=True)

    if len(df) != 183498:
        print("Warning : len(df) != len(Baseline) i.e. {} != {}".format(len(df), 183498))

    df.to_csv(output_path)
    return df


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
