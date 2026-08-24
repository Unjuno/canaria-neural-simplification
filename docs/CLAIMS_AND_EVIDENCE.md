# Claims and evidence

| Claim | Current status | Evidence class | Main limitation |
|---|---|---|---|
| Simplification exists outside high-Canary regions | **Supported** | Blinded confirmatory, n=8 | Original confirmatory family is residual CNN |
| Canary is a necessary local condition | **Rejected** | Blinded confirmatory | Canary definition may be improved |
| Canary adds useful prediction beyond span width | **Uncertain / weak** | Confirmatory LOSO | ΔAUC CI crosses zero |
| Composition complexity is frequently subadditive | **Supported in tested setting** | Confirmatory | Operational candidate grammar |
| Repair time increases simplification frequency | **Supported in original setting** | Confirmatory phase structure | Not monotone across all later tasks |
| Downstream location itself causes better repair | **Rejected** | Multiple equal-capacity interventions | Other topologies not tested |
| Adaptive recovery is explained by parameter count alone | **Rejected / incomplete** | Capacity-matched pilots/controls | Functional tangent dimension not fully measured |
| Local compression is entirely complexity relocation | **Rejected under measured codecs** | New-seed whole-network accounting | Codec-dependent complexity |
| Whole-network description length can be reduced ~26–29% in the residual-CNN setting | **Supported under tested codecs** | New-seed matched accounting | Not codec-independent MDL |
| 44.5-byte core can preserve task after repair | **Supported** | Independent holdout | Core only; requires shell |
| ~28-byte exact core serialization can preserve predictions | **Supported as serialization** | Exact codec roundtrip | Same 44.5-byte model, not new functional compression |
| Whole network can be serialized below 10 KB in the residual-CNN setting | **Supported** | Independent n=8 + exact codec | Specific task/architecture |
| Current best residual-CNN whole model is 9,926 B | **Supported** | Exact 8-seed codec | Not proven globally minimal |
| Simplification transfers from the residual-CNN family to a small Vision Transformer on the same digits task | **Supported as adapted transfer** | v20 preregistered n=8 architecture-shift confirmatory | Same dataset/task; zero-shot transfer failed |
| A 4-block small ViT core can be replaced by 2 smaller Transformer blocks with >60% whole-model reduction while retaining matched-control utility | **Supported** | v20 confirmatory + q8/zlib follow-up | Small ViT; one task and one adaptation recipe |
| One fixed compiler recipe transfers zero-shot from the residual CNN to the small ViT | **Rejected under tested recipe** | v20 confirmatory | Other generic grammars may exist |
| The G3 Transformer compiler transfers zero-shot from image-token ViT to a discrete non-image sequence encoder | **Supported** | v21 preregistered n=8 confirmatory | Synthetic sequence-order classification, not language |
| A 4-block sequence Transformer can be replaced by 2 smaller blocks without task repair while preserving utility | **Supported** | v21 zero-shot confirmatory | Controlled synthetic task; small model |
| Sequence-Transformer q8 state-stream size can fall by ~58% without task repair and retain utility | **Supported under declared codec** | v21 q8 follow-up | Shared architecture/decoder code not charged per model |
| Simplification transfers to a synthetic causal decoder-only Transformer after bounded repair | **Supported as adapted transfer** | v22 preregistered n=8 confirmatory | Controlled deterministic language; not natural prose |
| Teacher-forced PPL alone is sufficient to certify decoder simplification | **Rejected** | v22 + v23 paired PPL/rollout confirmatory evidence | Small causal models |
| Decoder zero-shot compilation can preserve PPL while still failing free-running generation | **Supported and independently reinforced** | v22 synthetic decoder + v23 real-text decoder | Short greedy rollout horizons |
| A 4-block synthetic causal decoder can be replaced by 2 smaller causal blocks with ~58% parameter/state reduction after repair while retaining PPL and generation utility | **Supported** | v22 confirmatory + q8 follow-up | Small synthetic task |
| The same bounded 4->2 causal simplification transfers to held-out natural English character modeling | **Rejected under tested budget** | **v23 preregistered n=8 negative confirmatory** | Small character LM; limited corpus; alternative objectives not tested |
| On v23 real text, tau0 compilation preserves teacher-forced PPL (~0.997 utility) while strongly changing autoregressive rollouts (~0.633 agreement) | **Supported** | v23 n=8 confirmatory | Greedy rollout fidelity metric; 24-character horizon |
| The v23 prespecified joint-repair adaptation rescues real-text autoregressive fidelity | **Rejected** | v23 n=8 confirmatory | Other repair/objective families remain open |
| Transformer simplification is uniformly positive across tasks once architecture-specific adaptation is allowed | **Rejected by current transfer map** | G3/G5/G6/G6b mixed outcomes | Limited architecture/task panel |
| Unlimited recursive compilation reaches arbitrarily small models | **Not supported** | Pilot negative results | Grammar-dependent |
| A finite universal mechanism dictionary exists | **Open** | Exploratory | Candidate-grammar dependence |
| Findings generalize to CIFAR/ResNet/pretrained subword LMs/arbitrary subgraphs | **Open** | Not yet tested | External validity |
