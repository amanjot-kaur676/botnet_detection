# botnet_detection
# IoT Botnet (Mirai-Lineage) Multi-Stage Attack Detection

A sequence/state-based detection framework for identifying and characterizing the attack lifecycle of Mirai and Mirai-derived IoT botnets, using a combination of network flow data and host telemetry.

## Motivation

Most existing IoT botnet detection approaches treat traffic classification as a flat, binary problem: benign vs. malicious. This project instead models the **full attack lifecycle as a sequence of states**:

```
Scan → Infect → C2 (Command & Control) → Impact (DoS / DDoS / Exfiltration / other)
```

Rather than only asking "is this traffic malicious?", the goal is to answer:
- **What stage** of the attack is currently happening?
- Once at the Impact stage, **what type of impact** is it — a denial-of-service flood, data exfiltration, or something else?
- How many distinct attack impacts can a Mirai-family botnet actually be shown to produce, using real, verifiable evidence from public datasets?

## Datasets

No single public dataset contains every feature needed for this approach, so two complementary, Zeek/Bro-based datasets are combined:

| Dataset | Contribution |
|---|---|
| **IoT-23** (Stratosphere Lab, CTU) | Real network captures from IoT devices infected with Mirai and Mirai-lineage malware variants. Provides ground-truth stage/attack labels via its `detailed-label` field. |
| **TON_IoT** (UNSW Canberra Cyber) | Network flow data *and* host-level telemetry (CPU, memory, process activity) collected under labeled attack conditions (DoS, DDoS, and others). Used to validate that host-side signals (e.g. CPU spikes) meaningfully correlate with specific attack types. |

Both datasets use Bro/Zeek (an open-source IDS) log conventions, which allows core network features (`conn_state`, `duration`, `orig_bytes`/`resp_bytes`, `history`, etc.) to be aligned across both sources.

## Methodology

1. **Feature study** — catalogued generic botnet features, features that distinguish botnet families, and known Mirai variants (Okiru/Satori, Masuta, Hakai, Wicked, IoTroop/Reaper, OMG, Mukashi, etc.).
2. **Stage labeling** — built a crosswalk mapping each dataset's native label columns to four lifecycle stages: Scan, Infect, C2, Impact.
3. **Mirai-lineage filtering** — IoT-23 contains multiple unrelated IoT botnet families (Torii, Hajime, Muhstik, Gafgyt, and others). Traffic was filtered down to scenarios confirmed as Mirai or a documented Mirai-derived variant (Okiru, Hakai), based on the dataset's official per-scenario malware documentation.
4. **Sequence construction** — flows were grouped into fixed-length ordered sequences per scenario, capped and balanced across scenarios to avoid any single capture dominating the dataset.
5. **Impact sub-typing** — the Impact stage is being broken down into specific attack types:
   - DDoS: directly labeled in both IoT-23 (`DDoS`) and TON_IoT (`ddos`)
   - DoS: labeled in TON_IoT (`dos`) — not present as a distinct label in IoT-23
   - Exfiltration: not natively labeled in either dataset — detected via engineered features (outward data volume) rather than a ground-truth label, per the project's original design goal
6. **Host telemetry feature engineering** — TON_IoT's Linux process-level CPU data has no timestamp or session key, so per-flow fusion with network data isn't possible. Instead, windowed aggregate features (max CPU, 95th-percentile CPU, count of high-load processes) are engineered to characterize host behavior differences between attack types and normal operation, used as corroborating evidence rather than row-level fused input.

## Current Status

- [x] Literature/feature research and Mirai variant cataloguing
- [x] Dataset selection and acquisition (IoT-23, TON_IoT network + Linux host subsets)
- [x] Stage labeling pipeline (Scan/Infect/C2/Impact) applied to IoT-23
- [x] Sequence construction, capping, and cross-scenario balancing
- [x] Mirai-lineage filtering (10 of 20 IoT-23 scenarios confirmed as Mirai-lineage)
- [x] TON_IoT host CPU data inspected and engineered features prototyped
- [ ] Impact-stage sub-typing (DoS / DDoS / Exfiltration) finalized across both datasets
- [ ] Outward-byte-volume exfiltration feature built and validated
- [ ] Final feature-enriched sequence dataset assembled
- [ ] State-sequence model design and training
- [ ] Evaluation and results
- [ ] Paper writeup

## Repository Structure

```
data/
  raw/                  # original downloaded dataset files
  stage_labeled/         # IoT-23 files with stage labels applied
  sequences/              # windowed sequence arrays (.npz) per scenario, combined, and Mirai-lineage filtered
  processed/              # engineered feature outputs (e.g. TON_IoT CPU window features)
scripts/                  # numbered pipeline scripts, run in order
```

## Notes on Data Limitations

This project aims to be transparent about dataset constraints rather than overstate findings:
- IoT-23 provides no host telemetry; TON_IoT's Linux host files provide no timestamp/session key for exact per-flow fusion with its own network data. Host-behavior claims are therefore based on aggregate/statistical comparison across labeled attack types, not literal per-event fusion.
- Some IoT-23 labels (e.g. `Okiru`) denote botnet family rather than attack behavior, and required separate verification before being used in lifecycle-stage or impact-type classification.
- Not all IoT-23 capture scenarios represent Mirai-lineage malware; scenario-level filtering was applied based on the dataset's official documentation.
