import pandas as pd

df = pd.read_csv("../data/stage_labeled/iot23_49_with_stage.csv", low_memory=False)
df = df.sort_values(by="ts").reset_index(drop=True)

WINDOW_SECONDS = 30
df["time_bucket"] = (df["ts"] // WINDOW_SECONDS) * WINDOW_SECONDS

windows = []
for bucket_val, chunk in df.groupby("time_bucket"):
    stage_counts = chunk["stage"].value_counts()
    total = len(chunk)
    windows.append({
        "time_bucket": bucket_val,
        "n_flows": total,
        "dominant_stage": stage_counts.idxmax(),
        "pct_scan": stage_counts.get("Scan", 0) / total,
        "pct_c2": stage_counts.get("C2", 0) / total,
        "pct_infect": stage_counts.get("Infect", 0) / total,
        "pct_benign": stage_counts.get("Benign", 0) / total,
        "avg_duration": pd.to_numeric(chunk["duration"], errors="coerce").mean(),
        "total_orig_bytes": pd.to_numeric(chunk["orig_bytes"], errors="coerce").sum(),
        "unique_dst_ips": chunk["id.resp_h"].nunique(),
    })

seq_df = pd.DataFrame(windows)
seq_df.to_csv("../data/sequences/sequences_49-1_timebased.csv", index=False)
print("Total time windows:", len(seq_df))
print(seq_df["dominant_stage"].value_counts())