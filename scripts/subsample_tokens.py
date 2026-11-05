"""Subsample encounters from a tokenizer feather split and write the (N, 3) uint32
array [patient_id, t, token_id] consumed by train.py (same layout as tokenizer_v2.save_outputs).
"""

import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feather", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--n-encounters", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-npy", required=True)
    args = parser.parse_args()

    df = pd.read_feather(args.feather, columns=["encounter_key", "name", "t"])
    print(f"Number of total encounters: {len(df)}")
    vocab = pd.read_csv(args.vocab)
    token_to_id = dict(zip(vocab["name"], vocab["index"]))

    encounters = df["encounter_key"].unique()
    assert args.n_encounters <= len(
        encounters
    ), f"only {len(encounters)} encounters available"
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(encounters, size=args.n_encounters, replace=False)
    df = df[df["encounter_key"].isin(keep)].copy()

    df["token_id"] = df["name"].map(token_to_id)
    assert df["token_id"].notna().all(), "feather contains tokens missing from vocab"
    # get_p2i requires each encounter's rows to be contiguous, in time order
    df["patient_id"] = pd.factorize(df["encounter_key"])[0] + 1
    df = df.sort_values(["patient_id", "t"], kind="stable")

    data = df[["patient_id", "t", "token_id"]].to_numpy().astype(np.uint32)
    np.save(args.out_npy, data)
    print(f"{args.feather}: kept {args.n_encounters} encounters, {len(data):,} tokens")


if __name__ == "__main__":
    main()
