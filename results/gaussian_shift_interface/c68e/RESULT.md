# C68E result — teacher validity repair at Gaussian sigma=.36

## Status

**PROSPECTIVE EXPLORATORY**. Protocol was locked before outcomes at commit `78f382f9b8c4d469c54de4b5cc7741dc7e1e92c0`. Fresh model seeds were `66400–66415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`ADVANCE_REPAIRED_TEACHER_TO_C69E`

A single locked Gaussian-augmentation recipe (50% of training examples independently noised at sigma=.36, same 60-epoch optimizer/permutation schedule, paired initialization) passed all three preregistered repair gates.

| gate | estimate | bootstrap 95% | margin | result |
|---|---:|---:|---:|---|
| augmented clean − clean-only clean accuracy | -0.3935 pp | [-0.8796, +0.0926] pp | lower > -4 pp | PASS |
| augmented shifted − clean-only shifted accuracy | +16.2500 pp | [+14.6296, +17.9167] pp | lower > 0 pp | PASS |
| augmented shifted − augmented clean accuracy | -11.4352 pp | [-12.4537, -10.4398] pp | lower > -20 pp | PASS |

Descriptively, clean-only teachers averaged `98.0324%` clean validation accuracy and `69.9537%` at sigma=.36, a `-28.0787 pp` shift drop. The augmented teachers averaged `97.6389%` clean and `86.2037%` shifted accuracy.

## Scientific interpretation

C68E provides exploratory evidence that the **target-validity problem identified in C67E can be repaired by the one prospectively specified augmentation recipe** in this Residual-MLP testbed. The repair incurs little average clean-accuracy cost and substantially improves sigma=.36 validation accuracy under the locked fresh cohort.

This does not answer the interface-complexity question. Robust training can change internal representation and residual geometry, so the earlier P0/P2 findings cannot simply be transferred to this repaired teacher.

## Safe statement

> For the repository Residual-MLP at Gaussian sigma=.36, the locked 50%-Gaussian-augmentation teacher recipe passed the preregistered clean-accuracy, shifted-superiority, and task-validity gates on fresh seeds, making it an exploratory candidate target for a separate interface experiment.

## Not supported

- the augmented teacher is optimal;
- Gaussian augmentation universally repairs robustness;
- P0 remains sufficient for the repaired teacher;
- any prior interface dimension transfers unchanged to the repaired teacher;
- any claim about the imported Residual CNN C59/C60 line.

## Next gate

C69E should treat the repaired teacher as a **new target geometry** and re-open the interface frontier prospectively. It must use fresh model seeds and a sufficiently strong reference correction rather than assuming the clean-teacher P2 reference remains adequate.
