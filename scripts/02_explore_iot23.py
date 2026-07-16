import pandas as pd

col_names = [
    'ts','uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p','proto','service',
    'duration','orig_bytes','resp_bytes','conn_state','local_orig','local_resp',
    'missed_bytes','history','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes',
    'tunnel_parents','label','detailed-label'
]

df = pd.read_csv(
    "../data/raw/conn_log_34-1.labeled",
    sep=r"\s+", comment="#", names=col_names, engine="python"
)

print("Total rows:", len(df))
print(df["id.orig_h"].value_counts())
print(df["detailed-label"].value_counts())