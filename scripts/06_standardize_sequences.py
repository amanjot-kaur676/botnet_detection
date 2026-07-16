import pandas as pd

WINDOW_SECONDS = 30

def build_timebased_sequence(stage_csv_path, output_path, scenario_name):
    df = pd.read_csv(stage_csv_path, low_memory=False)
    df = df.sort_values(by="ts").reset_index(drop=True)
    df["time_bucket"] = (df["ts"] // WINDOW_SECONDS) * WINDOW_SECONDS

    windows = []
    for bucket_val, chunk in df.groupby("time_bucket"):
        stage_counts = chunk["stage"].value_counts()
        total = len(chunk)
        windows.append({
            "scenario": scenario_name,
            "time_bucket": bucket_val,
            "n_flows": total,
            "dominant_stage": stage_counts.idxmax(),
            "pct_scan": stage_counts.get("Scan", 0) / total,
            "pct_infect": stage_counts.get("Infect", 0) / total,
            "pct_c2": stage_counts.get("C2", 0) / total,
            "pct_impact": stage_counts.get("Impact", 0) / total,
            "pct_benign": stage_counts.get("Benign", 0) / total,
            "avg_duration": pd.to_numeric(chunk["duration"], errors="coerce").mean(),
            "total_orig_bytes": pd.to_numeric(chunk["orig_bytes"], errors="coerce").sum(),
            "total_orig_pkts": pd.to_numeric(chunk["orig_pkts"], errors="coerce").sum(),
            "unique_dst_ips": chunk["id.resp_h"].nunique(),
        })

    seq_df = pd.DataFrame(windows)
    seq_df.to_csv(output_path, index=False)
    print(f"{scenario_name}: {len(seq_df)} windows saved to {output_path}")
    print(seq_df["dominant_stage"].value_counts())
    print()

build_timebased_sequence(
    "../data/stage_labeled/iot23_34-1_with_stage.csv",
    "../data/sequences/sequences_34-1_standard.csv",
    "34-1"
)

build_timebased_sequence(
    "../data/stage_labeled/iot23_49_with_stage.csv",
    "../data/sequences/sequences_49-1_standard.csv",
    "49-1"
)

build_timebased_sequence(
    "../data/stage_labeled/iot23_52_with_stage.csv",
    "../data/sequences/sequences_52-1_standard.csv",
    "52-1"
)