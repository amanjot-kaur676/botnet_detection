import pandas as pd
import os
import glob

#  actual extracted IoT-23 folder path
SCENARIOS_ROOT = r"D:\iot_23_datasets_small\opt\Malware-Project\BigDataset\IoTScenarios" 

CHUNK_SIZE = 200_000
col_names = [
    'ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service',
    'duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp',
    'missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes',
    'tunnel_parents','label','detailed-label'
]

stage_map = {
    '-': 'Benign',
    'PartOfAHorizontalPortScan': 'Scan',
    'C&C': 'C2',
    'FileDownload': 'Infect',
    'C&C-FileDownload': 'Infect',
    'C&C-HeartBeat': 'C2',
    'C&C-HeartBeat-FileDownload': 'Infect',
    'C&C-Torii': 'C2',
    'DDoS': 'Impact',
    'Attack': 'Infect',
    'Okiru': 'Impact',
    'Okiru-Attack': 'Impact',
    'PartOfAHorizontalPortScan-Attack': 'Scan',
    'C&C-PartOfAHorizontalPortScan': 'Scan',
    'C&C-HeartBeat-Attack': 'Impact',
}

output_dir = "../data/stage_labeled"
os.makedirs(output_dir, exist_ok=True)

# Find every scenario folder automatically
scenario_folders = sorted(glob.glob(os.path.join(SCENARIOS_ROOT, "CTU-IoT-Malware-Capture-*")))
print(f"Found {len(scenario_folders)} scenario folders.\n")

summary = []

for folder in scenario_folders:
    scenario_name = os.path.basename(folder)

    # Locate the conn.log.labeled file inside (path can vary slightly)
    candidates = glob.glob(os.path.join(folder, "**", "conn.log.labeled"), recursive=True)
    if not candidates:
        print(f"[SKIP] {scenario_name}: no conn.log.labeled found")
        summary.append((scenario_name, "SKIPPED - file not found", 0))
        continue

    file_path = candidates[0]
    output_path = os.path.join(output_dir, f"iot23_{scenario_name}_with_stage.csv")

    try:
        # Pass 1: find infected device via Malicious label count
        malicious_ip_counts = pd.Series(dtype="int64")
        reader = pd.read_csv(
            file_path, sep=r"\s+", comment="#", names=col_names,
            engine="python", dtype=str, chunksize=CHUNK_SIZE,
            keep_default_na=False
        )
        for chunk in reader:
            mal_chunk = chunk[chunk["label"] == "Malicious"]
            counts = mal_chunk["id.orig_h"].value_counts()
            malicious_ip_counts = malicious_ip_counts.add(counts, fill_value=0)

        if len(malicious_ip_counts) == 0:
            print(f"[SKIP] {scenario_name}: no Malicious rows found at all")
            summary.append((scenario_name, "SKIPPED - no malicious rows", 0))
            continue

        top_ip = malicious_ip_counts.idxmax()

        # Pass 2: label and write incrementally (memory-safe)
        if os.path.exists(output_path):
            os.remove(output_path)

        reader = pd.read_csv(
            file_path, sep=r"\s+", comment="#", names=col_names,
            engine="python", dtype=str, chunksize=CHUNK_SIZE,
            keep_default_na=False
        )

        first_write = True
        total_rows = 0
        unmapped = 0

        for chunk in reader:
            matching = chunk[chunk["id.orig_h"] == top_ip].copy()
            if len(matching) == 0:
                continue
            matching["stage"] = matching["detailed-label"].map(stage_map)
            matching["scenario"] = scenario_name
            unmapped += matching["stage"].isna().sum()
            total_rows += len(matching)
            matching.to_csv(output_path, mode="a", header=first_write, index=False)
            first_write = False

        print(f"[OK] {scenario_name}: infected device {top_ip}, {total_rows} rows, {unmapped} unmapped")
        summary.append((scenario_name, "OK", total_rows))

    except Exception as e:
        print(f"[ERROR] {scenario_name}: {e}")
        summary.append((scenario_name, f"ERROR - {e}", 0))

print("\n=== SUMMARY ===")
for name, status, rows in summary:
    print(f"{name}: {status} ({rows} rows)")