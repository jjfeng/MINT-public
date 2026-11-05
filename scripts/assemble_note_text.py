"""Collect the clinical note of every train and val patient in a BIF into a csv of
[split, patient_id, encounter_key, note_text], for plot_bif.py.
"""

import argparse

import pandas as pd
import xarray as xr

# BIF patient coordinate of each split
SPLIT_COORDS = {"train": "patient_1", "val": "patient_2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bif", required=True, help="zarr directory written by train_bayes.py")
    parser.add_argument("--train-keys", required=True, help="csv written by subsample_tokens.py")
    parser.add_argument("--val-keys", required=True, help="csv written by subsample_tokens.py")
    parser.add_argument("--visits", required=True, help="raw visits csv with a note_text column")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    bif = xr.open_zarr(args.bif)
    key_files = {"train": args.train_keys, "val": args.val_keys}
    patients = pd.concat(
        [
            pd.read_csv(key_files[split])
            .set_index("patient_id")
            .loc[bif[coord].values]
            .reset_index()
            .assign(split=split)
            for split, coord in SPLIT_COORDS.items()
        ]
    )

    visits = pd.read_csv(args.visits, usecols=["EncounterKey", "note_text"])
    assert visits["EncounterKey"].is_unique, "visits has several notes per encounter"
    notes = patients.merge(
        visits, left_on="encounter_key", right_on="EncounterKey", how="left"
    )
    missing = notes.loc[notes["note_text"].isna(), "encounter_key"]
    assert missing.empty, f"no note for encounters {missing.tolist()}"
    notes[["split", "patient_id", "encounter_key", "note_text"]].to_csv(
        args.out_csv, index=False
    )
    print(f"wrote {len(notes)} notes to {args.out_csv}")


if __name__ == "__main__":
    main()
