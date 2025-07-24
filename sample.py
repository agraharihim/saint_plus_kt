#!/usr/bin/env python3
"""
Resumable, memory‑efficient loader for EdNet KT datasets (KT‑1 … KT‑4).

Key points
----------
* NO os.listdir(): probes only the filenames you care about.
* Commit after --batch-size files (default 500) ⇒ fewer DB round‑trips.
* Optional --start-id / --end-id limit the user‑ID range to load.
* Idempotent via INSERT IGNORE + per‑dataset <dataset>_load_history table.
* Streams each CSV in CHUNK_SIZE rows (default 50 000), keeping RAM low.
"""

import argparse
import os
import sys
import logging
import re
import json
import pandas as pd
import mysql.connector
from tqdm import tqdm

# -------------------------------------------------------------------------
CHUNK_SIZE = 50_000
UID_RE     = re.compile(r"u(\d+)\.csv$", re.I)

DATASET_SCHEMAS = {
    "kt1": {
        "csv_cols": ["timestamp", "solving_id", "question_id",
                     "user_answer", "elapsed_time"],
        "rename":   {"timestamp": "ts_ms", "elapsed_time": "elapsed_ms"},
        "timestamp_col": "ts_ms",
        "table":    "kt1_users",
        "history":  "kt1_load_history"
    },
    "kt2": {
        "csv_cols": ["timestamp", "action_type", "item_id", "source",
                     "user_answer", "platform"],
        "rename":   {"timestamp": "ts_ms"},
        "timestamp_col": "ts_ms",
        "table":    "kt2_users",
        "history":  "kt2_load_history"
    },
    # Add entries for kt3 / kt4 here when needed
}

# -------------------------------------------------------------------------
def connect_mysql(**kw):
    """Return an autocommit‑OFF connection."""
    return mysql.connector.connect(charset="utf8mb4", autocommit=False, **kw)

def get_loaded(cur, history_tbl):
    cur.execute(f"SELECT filename FROM {history_tbl}")
    return {row[0] for row in cur.fetchall()}

def mark_done(cur, history_tbl, fname):
    cur.execute(
        f"INSERT INTO {history_tbl} (filename, processed_at) "
        f"VALUES (%s, UTC_TIMESTAMP()) "
        f"ON DUPLICATE KEY UPDATE processed_at = VALUES(processed_at)",
        (fname,)
    )

def insert_user_data(cur, table, user_id, json_data):
    sql = f"INSERT INTO {table} (user_id, data) VALUES (%s, %s) ON DUPLICATE KEY UPDATE data = VALUES(data)"
    cur.execute(sql, (user_id, json_data))

def process_file(fpath, schema, cur):
    import numpy as np
    user_id = os.path.splitext(os.path.basename(fpath))[0]  # 'u123'
    
    # Read entire file at once and process
    df = pd.read_csv(fpath, header=None,
                     names=schema["csv_cols"],
                     keep_default_na=True,
                     na_values=['', 'NULL', 'null', 'None'])
    
    if df.empty:
        # Skip empty files silently
        return
    
    # Apply column renaming
    if schema["rename"]:
        df.rename(columns=schema["rename"], inplace=True)
    
    # Handle NaN values by replacing with None
    df_clean = df.replace({np.nan: None})
    
    # Sort by timestamp
    timestamp_col = schema["timestamp_col"]
    if timestamp_col in df_clean.columns:
        df_clean = df_clean.sort_values(by=timestamp_col)
    
    # Convert to list of dictionaries for JSON storage
    records = df_clean.to_dict('records')
    
    # Convert to JSON string
    json_data = json.dumps(records, default=str)
    
    # Insert into database
    insert_user_data(cur, schema["table"], user_id, json_data)

# -------------------------------------------------------------------------
def iter_user_ids(start_id, end_id):
    """Generator for ascending user‑IDs between the inclusive bounds."""
    uid = start_id if start_id is not None else 1
    while True:
        if end_id is not None and uid > end_id:
            break
        yield uid
        uid += 1

def build_worklist(folder, schema, cur,
                   start_id=None, end_id=None,
                   quit_after_misses=1_000):
    """
    Yield filenames that still need loading WITHOUT directory listing.
      • If start_id/end_id given → probe exactly that range.
      • Else keep probing sequential IDs until we hit 'quit_after_misses'
        consecutive absent files ⇒ assume we've passed dataset tail.
    """
    done = get_loaded(cur, schema["history"])
    logging.info("Found %d files already processed in history table", len(done))
    misses = 0
    found_files = 0

    for uid in iter_user_ids(start_id, end_id):
        fname = f"u{uid}.csv"
        if fname in done:
            continue
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            misses = 0
            found_files += 1
            yield fname
        else:
            misses += 1
            if (end_id is None) and (misses >= quit_after_misses):
                logging.info("Hit %d consecutive missing files; "
                             "assuming dataset end.", quit_after_misses)
                break
    
    logging.info("Found %d files to process", found_files)

# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Efficient, resumable EdNet KT loader")
    ap.add_argument("folder", help="Directory containing KT CSV files")
    ap.add_argument("--dataset", required=True,
                    choices=DATASET_SCHEMAS.keys())
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=3306)
    ap.add_argument("--database", default="ednet")

    ap.add_argument("--batch-size", type=int, default=500,
                    help="commit after this many files (default 500)")
    ap.add_argument("--start-id", type=int,
                    help="first user‑ID to load (u<id>.csv)")
    ap.add_argument("--end-id", type=int,
                    help="last user‑ID to load (inclusive)")
    ap.add_argument("--quit-after-misses", type=int, default=1_000,
                    help="open‑ended scan stops after this many "
                         "consecutive missing IDs (default 1000)")
    args = ap.parse_args()

    schema = DATASET_SCHEMAS[args.dataset]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    conn = connect_mysql(host=args.host, port=args.port,
                         user=args.user, password=args.password,
                         database=args.database)
    cur = conn.cursor()

    try:
        filenames_iter = build_worklist(
            args.folder, schema, cur,
            start_id=args.start_id, end_id=args.end_id,
            quit_after_misses=args.quit_after_misses
        )

        bar = tqdm(unit="file", desc=f"Loading {args.dataset.upper()}")
        batch_count = 0

        for fname in filenames_iter:
            try:
                process_file(os.path.join(args.folder, fname), schema, cur)
                mark_done(cur, schema["history"], fname)
                batch_count += 1
                bar.update(1)

                if batch_count >= args.batch_size:
                    conn.commit()
                    batch_count = 0
            except Exception:
                conn.rollback()
                logging.exception("Rolled back due to error on %s", fname)
                raise

        if batch_count:
            conn.commit()
        bar.close()
        logging.info("Finished.")
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
