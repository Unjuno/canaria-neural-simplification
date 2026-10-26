"""Standard-library-only cohort evidence assembly; no numerical evaluation."""
from pathlib import Path
import json,hashlib,datetime,subprocess,math
ROOT=Path('/workspace/shared/canaria_cifar_replacement_runs_20261026')
OLD=Path('/workspace/shared/canaria_cifar_replacement_audit_20261026')
NEW=Path('/workspace/shared/canaria_cifar_replacement_audit_seed03_20261026')
COMMIT='125126f01dd509f93e9eced9febebf859a382987'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
assert sha(OLD/'AUDIT_MANIFEST.json')=='3f230a91acb00f7520ff838ea04c12eba37495e461678af942a0f623ea8e8539'
for name,meta in load(OLD/'AUDIT_MANIFEST.json').items():
    assert sha(OLD/name)==meta['sha256'] and (OLD/name).stat().st_size==meta['bytes']
# Recheck every original input used by both independent audits, including old01/02 evidence.
inputs={}
for directory in (OLD,NEW):
    for p,h in load(directory/'evidence_sha256.json').items():
        assert p not in inputs or inputs[p]==h
        assert sha(p)==h
        inputs[p]=h
old_audit=load(OLD/'audit_result.json');new_audit=load(NEW/'audit_result.json')
assert old_audit['status']==new_audit['status']=='PASS'
assert old_audit['official_test_member_opened'] is new_audit['official_test_member_opened'] is False
all_results={**old_audit['results'],**new_audit['results']}
assert set(all_results)=={'26102601','26102602','26102603'}
EX=ROOT/'research/cifar10_replacements/execution'
owner_path=EX/'results_all_three.json'
assert sha(owner_path)=='9f1ec13b8947c0ac657817055d9330ddd1f5568d839bb15155478c9ed5796d76'
owner=load(owner_path);assert owner['status']=='all_three_complete' and owner['official_test_decoded'] is False
assert {x['seed'] for x in owner['teachers']}=={26102601,26102602,26102603}
cohort=[]
for row in owner['teachers']:
    seed=row['seed'];directory=ROOT/f'research/cifar10_replacements/artifacts/seed_{seed}'
    audit=all_results[str(seed)]
    planpath=EX/f'plan_seed_{seed}.APPROVED.json';plan=load(planpath);run=plan['runs'][str(seed)]
    for name,meta in row['files'].items():
        p=directory/name;assert sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes']
    val=load(directory/'validation.json');assert val==row['validation']
    assert load(EX/f'seed{seed}.process_exit.json')==row['process_exit']
    assert row['process_exit']['exit_code']==0 and row['process_exit']['process_exited'] is True
    assert row['all_optimizer_steps']==row['checkpoint_cursor']==1000 and row['initial_cursor']==0 and row['finite'] is True
    assert row['parameter_counts']=={n:5120 for n in ('Comp-joint','Sep-local','Sep-joint')}
    assert audit['checks']['final']['file_sha256']==val['checkpoint_sha256']
    assert audit['checks']['teacher']['file_sha256']==val['teacher_sha256']
    for arm,metric in audit['arms'].items():
        orig=val['arms'][arm]
        assert metric['correct']==orig['correct'] and audit['teacher_correct']==orig['teacher_correct']
        assert metric['utility_drop_le_2pp']==orig['utility_drop_le_2pp'] is False
        assert math.isclose(metric['centered_span_nmse'],orig['centered_span_nmse'],rel_tol=1e-12,abs_tol=1e-12)
    auditdir=NEW if seed==26102603 else OLD
    cohort.append(dict(seed=seed,status='complete',independent_validation_audit='PASS',
        teacher_correct=audit['teacher_correct'],n=5000,arms=audit['arms'],
        teacher_sha256=sha(run['teacher_checkpoint']),replacement_sha256=sha(directory/'final_update1000.ckpt'),
        recipe_sha256=sha(directory/'recipe_lock.json'),validation_sha256=sha(directory/'validation.json'),
        plan_sha256=sha(planpath),initial_checkpoint_sha256=sha(directory/'initial.ckpt'),
        independent_result_path=str(auditdir/f'result_seed_{seed}.json'),independent_result_sha256=sha(auditdir/f'result_seed_{seed}.json'),
        independent_predictions_path=str(auditdir/f'predictions_seed_{seed}.npz'),independent_predictions_sha256=sha(auditdir/f'predictions_seed_{seed}.npz'),
        saved_updates=1000,parameters_per_arm=5120,schedule=audit['checks']['schedule']))
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==COMMIT
assert subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()==''
report=dict(schema=1,status='all_three_independently_verified',purpose='validation-cohort evidence lock only; NOT official-test release authorization',
    result_commit=COMMIT,planned_seeds=[26102601,26102602,26102603],all_intended_final_statuses_present=True,
    official_test_member_opened=False,official_test_decoded=False,official_test_release_authorized=False,
    source_sha256=load(EX/'EXECUTION_PROTOCOL.json')['source_sha256'],execution_protocol_sha256=sha(EX/'EXECUTION_PROTOCOL.json'),
    owner_results_path=str(owner_path),owner_results_sha256=sha(owner_path),
    prior_audit_path=str(OLD/'audit_result.json'),prior_audit_sha256=sha(OLD/'audit_result.json'),
    prior_audit_manifest_sha256=sha(OLD/'AUDIT_MANIFEST.json'),prior_audit_unchanged=True,
    seed03_audit_path=str(NEW/'audit_result.json'),seed03_audit_sha256=sha(NEW/'audit_result.json'),
    all_original_evidence_rehashed_unchanged=True,cohort=cohort,
    descriptive_utility_pass_count={n:0 for n in ('Comp-joint','Sep-joint','Sep-local')},descriptive_utility_total_per_arm=3,
    max_nmse_absolute_difference=max(a['nmse_absolute_difference_from_original'] for r in all_results.values() for a in r['arms'].values()),
    limitations=[
        'Every final prediction and metric was independently recomputed on the fixed validation split; none is official-test evidence.',
        'No historical training replay or per-update ID/gradient event log verification. Exact final counters, source, schedule, initialization, hashes and optimizer states are consistent with the locked recipe.',
        'No final fit-time teacher snapshot exists. Frozen teacher/suffix/BN are supported by pinned source and optimizer isolation; unchanged original teacher was used for independent inference.',
        'All three intended seeds retained, with all nine descriptive <=2pp outcomes failing; no population-generalization or statistical noninferiority conclusion follows.',
        'Comp versus Sep has architecture/receptive-field/depth confounds; only Sep-local versus Sep-joint fixes architecture and initialization.',
        'Hashes establish integrity and binding, not independent authorship or absence of hypothetical unlogged activity.'
    ])
(NEW/'cohort_audit_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(NEW/'cohort_input_sha256.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
rows=[]
for r in cohort:
    for arm in ('Comp-joint','Sep-joint','Sep-local'):
        a=r['arms'][arm]
        rows.append(f"| {r['seed']} | {r['teacher_correct']} | {arm} | {a['correct']} | {a['drop_pp']:.2f} | {a['centered_span_nmse']:.12f} | Fail |")
(NEW/'README.md').write_text('''# All three replacement seeds independently verified

Seed03 independent audit PASS, exit0. Counts: teacher4545/5000, Comp4352, Sep-joint4240, Sep-local4223. Drops:3.86,6.10,6.44 pp; all fail the descriptive <=2pp criterion. Separate functional inference reproduces centered NMSE within2.78e-16. Hash/schedule/initialization/5120-parameter/1000-step optimizer checks passed. Official test was not opened or decoded. No training or optimizer steps occurred.

The prior01/02 audit is preserved unchanged at `/workspace/shared/canaria_cifar_replacement_audit_20261026`. Its manifest and every listed output/input hash were rechecked. `cohort_audit_manifest.json` binds all three independently recomputed results to the original teacher, replacement, initialization, recipe, validation, approved-plan and prediction hashes. This is an evidence manifest, NOT official-test release authorization.

## Entire intended cohort (correct counts /5000)

| Seed | Teacher | Arm | Correct | Drop (pp) | Centered NMSE | <=2pp |
|---|---:|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

All nine outcomes fail <=2pp. This is a descriptive three-seed pilot, not statistical noninferiority or a population-generalization result. Comp/Sep architecture, receptive field and depth differ; only Sep-local/Sep-joint share architecture and exact initialization.

## Evidence and limits

`audit.py` imports no production experiment code. It independently reconstructs the allowed train-only data split/normalization, seeded initializations and schedules, then uses a separate functional forward and NumPy float64 aggregate statistics. Saved NPZs retain IDs/labels/predictions/logits and per-feature teacher statistics. All original evidence bytes remained unchanged.

Teacher parameters/BN/suffix are frozen by the pinned source and excluded from saved replacement optimizers; independent inference uses unchanged original teacher state. Historical every-step behavior was not replayed and no final fit-time teacher snapshot exists, so the final counters/source cannot prove every historical update or historical immutability by themselves. See the full manifest limitations and the preserved prior SOURCE_REVIEW.md.

The4-thread numerical audit process exited0 and the slot was explicitly released at14:17 UTC. Cohort assembly afterward used standard library only. No network/CI or original evidence writes.
''')
(NEW/'process_exit.json').write_text(json.dumps(dict(exit_code=0,process_exited=True,slot_threads_released=4,release_announced_utc='2026-10-26T14:17:00Z',cohort_sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
# Seal outputs after all computations and summary writes; avoid cyclic manifest hashing.
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(NEW.iterdir()) if p.is_file() and p.name!='AUDIT_MANIFEST.json'}
(NEW/'AUDIT_MANIFEST.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
print('COHORT_AUDIT_MANIFEST_SHA256',sha(NEW/'cohort_audit_manifest.json'))
print('AUDIT_MANIFEST_SHA256',sha(NEW/'AUDIT_MANIFEST.json'))
print('SEED03_AUDIT_RESULT_SHA256',sha(NEW/'audit_result.json'))
print('ALL_THREE_COMPLETE_INDEPENDENT_AUDIT_PASS')
