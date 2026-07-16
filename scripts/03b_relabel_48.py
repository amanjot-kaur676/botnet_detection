import pandas as pd
import os

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

file_path = r"D:\iot_23_datasets_small\opt\Malware-Project\BigDataset\IoTScenarios\CTU-IoT-Malware-Capture-48-1\bro\conn.log.labeled"
output_path = "../data/stage_labeled/iot23_CTU-IoT-Malware-Capture-48-1_with_stage.csv"
top_ip = "192.168.1.200"  # already known from previous run

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
    matching["scenario"] = "CTU-IoT-Malware-Capture-48-1"
    unmapped += matching["stage"].isna().sum()
    total_rows += len(matching)
    matching.to_csv(output_path, mode="a", header=first_write, index=False)
    first_write = False

print(f"Done: {total_rows} rows, {unmapped} unmapped")