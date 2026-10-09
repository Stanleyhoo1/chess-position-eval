"""Sample positions from the Lichess evaluation dataset on Hugging Face and
write an encoded training CSV.

The dataset is 20 parquet shards (~2.2 GB each, ~1M rows per row group). Rows
are read directly with pyarrow, one row group at a time and only the columns
we need, which keeps memory under ~1 GB. Row groups are visited in a seeded
random order and a limited number of positions is taken from each, so the
sample is spread across the whole dataset instead of one slice.

Each output row is one position:

    fen            FEN string: board, side to move, castling rights, en-passant square
    evaluation     centipawns from White's point of view, clipped to +-CP_CAP;
                   forced mates become +CP_CAP (White mates) or -CP_CAP (Black mates)

Only the FEN is stored. Decoding it into model inputs (8x8 piece-ID board,
side to move, castling flags, en-passant ID) happens at training time via
chess_eval/encoding.py, so the encoding can change without regenerating this file.

Usage (from the repository root):
    python -m training.get_training_data                       # 500k rows -> data/training_data.csv
    python -m training.get_training_data --rows 50000 --min-depth 25 --out data/small.csv
    python -m training.get_training_data --resume              # continue an interrupted run
    python -m training.get_training_data --rows 100000 --skip-per-group 40000 --seed 7 --out data/test_data.csv
"""

import argparse
import csv
import os
import random
import sys
import time

DATASET = "Lichess/chess-position-evaluations"
CP_CAP = 2500
COLUMNS_NEEDED = ["fen", "depth", "cp", "mate"]

CSV_COLUMNS = ["fen", "evaluation"]


# ---------------------------------------------------------------------------
# Row encoding
# ---------------------------------------------------------------------------

def evaluation_from_row(row):
    """Map a dataset row to a clipped centipawn score, or None if unusable."""
    mate = row.get("mate")
    cp = row.get("cp")
    if mate is not None:
        if mate > 0:
            return CP_CAP
        if mate < 0:
            return -CP_CAP
        return None            # mate == 0 should not happen; skip defensively
    if cp is None:
        return None
    return max(-CP_CAP, min(CP_CAP, int(cp)))


def encode_row(row):
    """Dataset row -> CSV row (list), or None if the row should be skipped."""
    evaluation = evaluation_from_row(row)
    if evaluation is None:
        return None
    return [row["fen"], evaluation]


# ---------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------

def list_row_groups(fs, shard_paths):
    """Return [(shard_path, row_group_index), ...] for every row group."""
    import pyarrow.parquet as pq

    pairs = []
    for path in shard_paths:
        with fs.open(path, "rb") as f:
            n = pq.ParquetFile(f).metadata.num_row_groups   # footer read only
        pairs.extend((path, i) for i in range(n))
    return pairs


def iter_row_groups(order, fs):
    """Yield (shard_path, row_group_index, pyarrow.Table) with only the needed
    columns. Shard files are kept open and reused between row groups."""
    import pyarrow.parquet as pq

    handles = {}
    try:
        for path, rg in order:
            if path not in handles:
                f = fs.open(path, "rb", block_size=8 << 20)
                handles[path] = (f, pq.ParquetFile(f))
            _, pf = handles[path]
            yield path, rg, pf.read_row_group(rg, columns=COLUMNS_NEEDED)
    finally:
        for f, _ in handles.values():
            f.close()


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

def load_existing(path):
    """Read a partial output file; drop a trailing partial line if the previous
    run was killed mid-write. Returns the set of FENs it contains."""
    with open(path, "r", newline="") as f:
        lines = f.read().split("\n")
    header, body = lines[0], lines[1:]
    if header.split(",") != CSV_COLUMNS:
        raise SystemExit(f"{path} has an unexpected header; refusing to resume")
    if body and body[-1] == "":
        body.pop()
    good = [ln for ln in body if ln.count(",") == len(CSV_COLUMNS) - 1 and ln.split(",")[-1].strip() != ""]
    if len(good) != len(body):
        with open(path, "w", newline="") as f:
            f.write(header + "\n" + "\n".join(good) + ("\n" if good else ""))
        print(f"dropped {len(body) - len(good)} incomplete line(s) from {path}", flush=True)
    return {ln.split(",", 1)[0] for ln in good}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=500_000, help="number of positions to write")
    parser.add_argument("--min-depth", type=int, default=20, help="skip evaluations shallower than this")
    parser.add_argument("--per-group", type=int, default=5_000,
                        help="max positions taken from each ~1M-row row group (spreads the sample)")
    parser.add_argument("--skip-per-group", type=int, default=0,
                        help="skip this many depth-eligible rows at the start of every row group before "
                             "taking any; use it to build a test set disjoint from an earlier export, "
                             "which always took rows from the start of each group")
    parser.add_argument("--out", default="data/training_data.csv")
    parser.add_argument("--seed", type=int, default=42, help="seed for the row-group visiting order")
    parser.add_argument("--no-shuffle", action="store_true", help="visit row groups in stored order")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing output file, skipping FENs it already contains")
    args = parser.parse_args()

    import pyarrow.compute as pc
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shard_paths = sorted(fs.glob(f"datasets/{DATASET}/data/*.parquet"))
    if not shard_paths:
        raise SystemExit("no parquet shards found")
    order = list_row_groups(fs, shard_paths)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(order)
    print(f"{len(shard_paths)} shards, {len(order)} row groups", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    seen_fens = set()
    written = skipped_depth = skipped_dup = skipped_eval = 0
    started = time.time()

    resuming = args.resume and os.path.exists(args.out)
    if resuming:
        seen_fens = load_existing(args.out)
        written = len(seen_fens)
        print(f"resuming: {written:,} rows already in {args.out}", flush=True)
        if written >= args.rows:
            print("nothing to do")
            return 0

    with open(args.out, "a" if resuming else "w", newline="") as f:
        writer = csv.writer(f)
        if not resuming:
            writer.writerow(CSV_COLUMNS)

        for path, rg, table in iter_row_groups(order, fs):
            n_before = table.num_rows
            table = table.filter(pc.greater_equal(table["depth"], args.min_depth))
            skipped_depth += n_before - table.num_rows

            fens = table["fen"].to_pylist()
            cps = table["cp"].to_pylist()
            mates = table["mate"].to_pylist()

            taken = 0
            skipped_lead = 0
            for fen, cp, mate in zip(fens, cps, mates):
                if skipped_lead < args.skip_per_group:
                    skipped_lead += 1         # leave the rows an earlier export may have used
                    continue
                if fen in seen_fens:
                    skipped_dup += 1          # same position, another engine line
                    continue
                encoded = encode_row({"fen": fen, "cp": cp, "mate": mate})
                if encoded is None:
                    skipped_eval += 1
                    continue
                seen_fens.add(fen)
                writer.writerow(encoded)
                written += 1
                taken += 1
                if taken >= args.per_group or written >= args.rows:
                    break

            f.flush()
            elapsed = time.time() - started
            print(f"{written:>8,} rows  {elapsed:6.0f}s  {os.path.basename(path)} rg {rg:<3} "
                  f"(skipped: depth {skipped_depth:,}, dup {skipped_dup:,}, no eval {skipped_eval:,})",
                  flush=True)
            if written >= args.rows:
                break

    elapsed = time.time() - started
    print(f"done: wrote {written:,} rows to {args.out} in {elapsed:.0f}s")
    print(f"skipped: depth<{args.min_depth} {skipped_depth:,}, duplicate FEN {skipped_dup:,}, no eval {skipped_eval:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
