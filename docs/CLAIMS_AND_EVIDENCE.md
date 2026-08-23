# Claims and evidence

| Claim | Current status | Evidence class | Main limitation |
|---|---|---|---|
| Simplification exists outside high-Canary regions | **Supported** | Blinded confirmatory, n=8 | Original confirmatory family is residual CNN |
| Canary is a necessary local condition | **Rejected** | Blinded confirmatory | Canary definition may be improved |
| Canary adds useful prediction beyond span width | **Uncertain / weak** | Confirmatory LOSO | ΔAUC CI crosses zero |
| Composition complexity is frequently subadditive | **Supported in tested setting** | Confirmatory | Operational candidate grammar |
| Repair time increases simplification frequency | **Supported** | Confirmatory phase structure | Mechanism not fully identified |
| Downstream location itself causes better repair | **Rejected** | Multiple equal-capacity interventions | Other topologies not tested |
| Adaptive recovery is explained by parameter count alone | **Rejected / incomplete** | Capacity-matched pilots/controls | Functional tangent dimension not fully measured |
| Local compression is entirely complexity relocation | **Rejected under measured codecs** | New-seed whole-network accounting | Codec-dependent complexity |
| Whole-network description length can be reduced ~26–29% in the residual-CNN setting | **Supported under tested codecs** | New-seed matched accounting | Not codec-independent MDL |
| 44.5-byte core can preserve task after repair | **Supported** | Independent holdout | Core only; requires shell |
| ~28-byte exact core serialization can preserve predictions | **Supported as serialization** | Exact codec roundtrip | Same 44.5-byte model, not new functional compression |
| Whole network can be serialized below 10 KB in the residual-CNN setting | **Supported** | Independent n=8 + exact codec | Specific task/architecture |
| Current best residual-CNN whole model is 9,926 B | **Supported** | Exact 8-seed codec | Not proven globally minimal |
| Simplification transfers from the residual-CNN family to a small Vision Transformer on the same digits task | **Supported as adapted transfer** | Preregistered n=8 architecture-shift confirmatory | Same dataset/task; zero-shot transfer failed |
| A 4-block small ViT core can be replaced by 2 smaller Transformer blocks with >60% whole-model reduction while retaining matched-control utility | **Supported** | v20 confirmatory + q8/zlib follow-up | Small ViT; one task and one adaptation recipe |
| One fixed compiler recipe transfers zero-shot from the residual CNN to the small ViT | **Rejected under tested recipe** | v20 confirmatory | Other generic grammars may exist |
| Unlimited recursive compilation reaches arbitrarily small models | **Not supported** | Pilot negative results | Grammar-dependent |
| A finite universal mechanism dictionary exists | **Open** | Exploratory | Candidate-grammar dependence |
| Findings generalize to CIFAR/ResNet/language Transformers/arbitrary subgraphs | **Open** | Not yet tested | External validity |
