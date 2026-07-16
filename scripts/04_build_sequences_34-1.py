import pandas as pd

df = pd.read_csv("../data/stage_labeled/iot23_34-1_with_stage.csv")
df = df.sort_values(by="ts").reset_index(drop=True)

WINDOW_SIZE = 20

windows = []
for start in range(0, len(df), WINDOW_SIZE):
    chunk = df.iloc[start:start+WINDOW_SIZE]
    if len(chunk) == 0:
        continue
    windows.append({
        "window_id": start // WINDOW_SIZE,
        "start_ts": chunk["ts"].min(),
        "end_ts": chunk["ts"].max(),
        "n_flows": len(chunk),
        "dominant_stage": chunk["stage"].value_counts().idxmax(),
        "avg_duration": pd.to_numeric(chunk["duration"], errors="coerce").mean(),
        "total_orig_bytes": pd.to_numeric(chunk["orig_bytes"], errors="coerce").sum(),
        "total_resp_bytes": pd.to_numeric(chunk["resp_bytes"], errors="coerce").sum(),
        "total_orig_pkts": pd.to_numeric(chunk["orig_pkts"], errors="coerce").sum(),
        "unique_dst_ips": chunk["id.resp_h"].nunique(),
    })

seq_df = pd.DataFrame(windows)
seq_df.to_csv("../data/sequences/sequences_34-1.csv", index=False)
print("Total windows:", len(seq_df))
print(seq_df["dominant_stage"].value_counts())