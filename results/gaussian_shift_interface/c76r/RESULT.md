# C76R head-rank confirmation

PROSPECTIVE_CONFIRMATORY. Main/public claims unchanged.

Decision: `C76R_CONFIRMATORY_PASS`

The selected correction dimension is the numerical rank of the class-centered fixed affine head, computed before fitting. No reduced dimension grid was trained. This is a sufficient-dimension rule, not a minimum-dimension theorem for learned networks.

## Primary gates
{
  "utility": {
    "bootstrap_se": 0.29020349181667815,
    "ci95": [
      -0.7870376110076904,
      0.3472212702035904
    ],
    "margin": -2,
    "mean": -0.23148208856582642,
    "median": -0.18518567085266113,
    "status": "PASS"
  },
  "hidden_nmse_ratio": {
    "bootstrap_se": 0.010276892797974663,
    "ci95": [
      1.1190577259672065,
      1.1594257177440785
    ],
    "margin": 1.25,
    "mean": 1.1390641622394864,
    "median": 1.1352506185957578,
    "status": "PASS"
  }
}

## Reference prerequisites
{
  "target": {
    "bootstrap_se": 0.430359259574642,
    "ci95": [
      -12.800925225019455,
      -11.11111119389534
    ],
    "margin": -20,
    "mean": -11.921295523643494,
    "median": -11.481481790542603,
    "status": "PASS"
  },
  "references": {
    "direct": {
      "bootstrap_se": 0.26850054915705684,
      "ci95": [
        -4.16666716337204,
        -3.125
      ],
      "margin": -5,
      "mean": -3.6342602223157883,
      "median": -3.7037014961242676,
      "status": "PASS"
    },
    "p64": {
      "bootstrap_se": 0.37370557841347707,
      "ci95": [
        -4.745371267199516,
        -3.2870370894670486
      ],
      "margin": -5,
      "mean": -4.027778282761574,
      "median": -3.8888931274414062,
      "status": "PASS"
    }
  }
}

## Descriptive seed success
{
  "interpretation": "conditional iid model-seed assumption, not independent dataset validation",
  "n": 16,
  "successes": 16,
  "two_sided95_clopper_pearson": [
    0.7940927857921773,
    1.0
  ]
}

## Boundaries
Fixed digits split/calibration, repaired Residual-MLP sigma.36,384 samples,4096-parameter mapping. Full teacher residuals are computed at calibration, so no measured communication saving. Ideal probabilities and fitted pipeline are distinct. No independent dataset, CNN/SmallViT, hardware or universal-minimum claim. Source/count/logit/NMSE audit is not independent scientific peer review.
