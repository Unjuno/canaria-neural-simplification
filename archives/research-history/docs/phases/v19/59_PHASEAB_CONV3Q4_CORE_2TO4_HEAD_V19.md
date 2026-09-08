# Phase AB — Less aggressive core under 10 KB (exploratory)

Purpose: test whether whole-model instability below 10 KB comes from the 44.5 B aggressive core rather than head compression.

Condition: replace full 8-block span by fitted dense Conv3; quantize Conv3 weight and bias separately to calibrated signed 4-bit with FP16 scale (296 B core). Repair shell tau=8 as in v18. Then prune head first linear to magnitude 2:4 without coefficient refit; quantize retained head and remaining shell using Phase-AA2 channelwise 4-bit/FP16-bias codec. Expected whole storage 9,926 B including support pattern and metadata.

Initial cohort: seeds 3000-3007, exploratory. Matched continued-training control unchanged.