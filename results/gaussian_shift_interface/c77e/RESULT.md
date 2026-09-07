# C77E absolute utility after head-rank prediction

PROSPECTIVE_EXPLORATORY. Main unchanged.

Decision: `N512_VALID_GAIN_NOT_ESTABLISHED`

| Cell | Accuracy mean | Teacher gap pp | 95%CI | Absolute safeguard |
|---|---:|---:|---|---|
| N384_direct | 0.825000 | -3.240741 | [-4.074074327945709, -2.3611120879650116] | PASS |
| N384_head_rank9 | 0.818981 | -3.842593 | [-4.629630222916603, -3.0787046998739243] | PASS |
| N384_p64 | 0.818750 | -3.865741 | [-4.675925150513649, -3.101852163672447] | PASS |
| N512_direct | 0.830556 | -2.685186 | [-3.449074923992157, -1.9675932824611664] | PASS |
| N512_head_rank9 | 0.822222 | -3.518519 | [-4.351852089166641, -2.708333730697632] | PASS |
| N512_p64 | 0.823611 | -3.379630 | [-4.120370373129845, -2.662038430571556] | PASS |

## Relative head/P64 gates
{
  "384": {
    "hidden_nmse_ratio": {
      "bootstrap_se": 0.0064175012974890585,
      "ci95": [
        1.1299352533281666,
        1.154962967235417
      ],
      "margin": 1.25,
      "mean": 1.142565034280696,
      "median": 1.1441204765896602,
      "status": "PASS"
    },
    "utility_pp": {
      "bootstrap_se": 0.25055923204161035,
      "ci95": [
        -0.4629630595445633,
        0.5092594772577286
      ],
      "margin": -2,
      "mean": 0.023147836327552795,
      "median": 0.18518269062042236,
      "status": "PASS"
    }
  },
  "512": {
    "hidden_nmse_ratio": {
      "bootstrap_se": 0.005791417482242097,
      "ci95": [
        1.160152392670658,
        1.1827594720018741
      ],
      "margin": 1.25,
      "mean": 1.1715928787290704,
      "median": 1.1797691035630884,
      "status": "PASS"
    },
    "utility_pp": {
      "bootstrap_se": 0.22687033018786612,
      "ci95": [
        -0.5787033587694168,
        0.3009263426065445
      ],
      "margin": -2,
      "mean": -0.13888850808143616,
      "median": -0.18518269062042236,
      "status": "PASS"
    }
  }
}

## Paired N512-minus-N384 gains
{
  "direct": {
    "bootstrap_se": 0.3404830327424395,
    "ci95": [
      -0.11574067175388336,
      1.2268505990505219
    ],
    "margin": 0,
    "mean": 0.5555551499128342,
    "median": 0.37037134170532227,
    "status": "UNCERTAIN"
  },
  "head_rank9": {
    "bootstrap_se": 0.2543974018627638,
    "ci95": [
      -0.1620359718799591,
      0.8101858198642731
    ],
    "margin": 0,
    "mean": 0.32407455146312714,
    "median": 0.37037134170532227,
    "status": "UNCERTAIN"
  },
  "p64": {
    "bootstrap_se": 0.2188866407099174,
    "ci95": [
      0.04629641771316528,
      0.902777910232544
    ],
    "margin": 0,
    "mean": 0.4861108958721161,
    "median": 0.740736722946167,
    "status": "PASS"
  }
}

## Boundaries
The N384 baseline is not assumed intrinsically invalid. All cells keep4096 parameters,600 updates,batch128,76800 draws; N512 appends128 new calibration examples with an exact N384 prefix. Head rank is chosen before training, not by dimension sweeps. The new cohort is independent only in model seeds, not the dataset. No universal512 minimum or measured hardware/communication saving. Any new512 repair remains exploratory until separate fresh confirmation.
