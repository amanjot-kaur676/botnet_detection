import pandas as pd

WINDOW_SECONDS = 30
CHUNK_SIZE = 200_000
input_path = "../data/stage_labeled/iot23_52_with_stage.csv"
output_path = "../data/sequences/sequences_52-1_standard.csv"

# Dictionary to accumulate stats per time_bucket across all chunks
bucket_data = {}

reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False)

for chunk in reader:
    chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
    chunk["time_bucket"] = (chunk["ts"] // WINDOW_SECONDS) * WINDOW_SECONDS

    for bucket_val, group in chunk.groupby("time_bucket"):
        if bucket_val not in bucket_data:
            bucket_data[bucket_val] = {
                "n_flows": 0, "scan": 0, "infect": 0, "c2": 0, "impact": 0, "benign": 0,
                "orig_bytes_sum": 0.0, "orig_pkts_sum": 0.0,
                "duration_sum": 0.0, "duration_count": 0,
                "dst_ips": set()
            }

        d = bucket_data[bucket_val]
        stage_counts = group["stage"].value_counts()

        d["n_flows"] += len(group)
        d["scan"] += stage_counts.get("Scan", 0)
        d["infect"] += stage_counts.get("Infect", 0)
        d["c2"] += stage_counts.get("C2", 0)
        d["impact"] += stage_counts.get("Impact", 0)
        d["benign"] += stage_counts.get("Benign", 0)

        d["orig_bytes_sum"] += pd.to_numeric(group["orig_bytes"], errors="coerce").sum()
        d["orig_pkts_sum"] += pd.to_numeric(group["orig_pkts"], errors="coerce").sum()

        durations = pd.to_numeric(group["duration"], errors="coerce").dropna()
        d["duration_sum"] += durations.sum()
        d["duration_count"] += len(durations)

        d["dst_ips"].update(group["id.resp_h"].unique())

# Convert accumulated dict into the final windows table
rows = []
for bucket_val, d in bucket_data.items():
    total = d["n_flows"]
    stage_totals = {"Scan": d["scan"], "Infect": d["infect"], "C2": d["c2"], "Impact": d["impact"], "Benign": d["benign"]}
    dominant = max(stage_totals, key=stage_totals.get)

    rows.append({
        "scenario": "52-1",
        "time_bucket": bucket_val,
        "n_flows": total,
        "dominant_stage": dominant,
        "pct_scan": d["scan"] / total,
        "pct_infect": d["infect"] / total,
        "pct_c2": d["c2"] / total,
        "pct_impact": d["impact"] / total,
        "pct_benign": d["benign"] / total,
        "avg_duration": d["duration_sum"] / d["duration_count"] if d["duration_count"] > 0 else None,
        "total_orig_bytes": d["orig_bytes_sum"],
        "total_orig_pkts": d["orig_pkts_sum"],
        "unique_dst_ips": len(d["dst_ips"]),
    })

seq_df = pd.DataFrame(rows).sort_values(by="time_bucket").reset_index(drop=True)
seq_df.to_csv(output_path, index=False)

print("52-1:", len(seq_df), "windows saved to", output_path)
print(seq_df["dominant_stage"].value_counts())