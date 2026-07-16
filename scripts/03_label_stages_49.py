import pandas as pd

col_names = [
    'ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service',
    'duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp',
    'missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes',
    'tunnel_parents','label','detailed-label'
]

df = pd.read_csv(
    "../data/raw/conn_log_49.labeled",
    sep=r"\s+", comment="#", names=col_names, engine="python"
)

device = df[df["id.orig_h"] == "192.168.1.193"].copy()
device = device.sort_values(by="ts")

stage_map = {
    '-': 'Benign',
    'PartOfAHorizontalPortScan': 'Scan',
    'C&C': 'C2',
    'FileDownload': 'Infect',
    'C&C-FileDownload': 'Infect'
}
device["stage"] = device["detailed-label"].map(stage_map)

device.to_csv("../data/stage_labeled/iot23_49_with_stage.csv", index=False)
print(device["stage"].value_counts())