import pandas as pd
import os

col_names = [
    'ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service',
    'duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp',
    'missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes',
    'tunnel_parents','label','detailed-label'
]

CHUNK_SIZE = 200_000
file_path = "../data/raw/conn_log_52.labeled"
top_ip = "192.168.1.197"   # already confirmed from previous run
output_path = "../data/stage_labeled/iot23_52_with_stage.csv"

stage_map = {
    '-': 'Benign',
    'PartOfAHorizontalPortScan': 'Scan',
    'C&C': 'C2',
    'FileDownload': 'Infect',
    'C&C-FileDownload': 'Infect',
    'C&C-HeartBeat': 'C2',
    'C&C-Torii': 'C2',
    'DDoS': 'Impact',
    'Attack': 'Infect'
}

# Remove old output file if it exists, so we start fresh
if os.path.exists(output_path):
    os.remove(output_path)

reader = pd.read_csv(
    file_path, sep=r"\s+", comment="#", names=col_names,
    engine="python", dtype=str, chunksize=CHUNK_SIZE,
    keep_default_na=False
)

first_write = True
stage_totals = {}
unmapped_total = 0

for i, chunk in enumerate(reader):
    matching = chunk[chunk["id.orig_h"] == top_ip].copy()
    if len(matching) == 0:
        continue

    matching["stage"] = matching["detailed-label"].map(stage_map)
    unmapped_total += matching["stage"].isna().sum()

    # Update running stage counts
    counts = matching["stage"].value_counts(dropna=True)
    for stage_name, count in counts.items():
        stage_totals[stage_name] = stage_totals.get(stage_name, 0) + count

    # Append this chunk to the output CSV
    matching.to_csv(output_path, mode="a", header=first_write, index=False)
    first_write = False

    print(f"Chunk {i+1} processed, {len(matching)} matching rows written so far this chunk")

print()
print("DONE.")
print("Unmapped rows total:", unmapped_total)
print("Final stage totals:")
for stage_name, count in stage_totals.items():
    print(f"  {stage_name}: {count}")