"""Synthetic controls only. No real clock measurements, ML imports, models or data.
Supplied for separately authorized reviewer execution; not executed during packaging.
"""
import contextlib,hashlib,json,pathlib,subprocess,tempfile,types,unittest
from unittest import mock
import benchmark as b
import latency_contract as c
import supervise
import timing_core as t


def artificial_rows():
    return [{'id':s,'batch':batch,'round':r,'position':p,'arm':arm,
             'iterations':20,'total_ns':{'teacher':2000,'C':1000,'J':1200}[arm],
             'ns_per_forward':{'teacher':100,'C':50,'J':60}[arm]}
            for s in range(16) for batch in ((1,128) if s%2==0 else (128,1))
            for r in range(30) for p,arm in enumerate(t.ordered(s,r))]


def artificial_manifest():
    def descriptor(name):return {'path':name,'sha256':'a'*64}
    return {'schema':c.SCHEMA,'selection':descriptor('selection.json'),
            'csv':{'path':'digits.csv.gz','sha256':c.CSV_SHA256},
            'origin_selection':{'path':'origin-selection.json','sha256':c.ORIGIN_SELECTION_SHA256},
            'historical_reference_manifest':{'path':'historical-references.json','sha256':c.REFERENCE_MANIFEST_SHA256},
            'reference_selection':{'path':'reference-selection.json','sha256':c.REFERENCE_SELECTION_SHA256},
            'storage':{'directory':'storage','artifacts':descriptor('storage/ARTIFACTS.json'),'rows':descriptor('storage/ROWS.json')},
            'validation':[{'id':i,'split':descriptor(f'seed_{i}/split.npz'),
                'logits':{a:{**descriptor(f'seed_{i}/validation.npz'),'key':'teacher_logits' if a=='teacher' else a+'_logits'} for a in t.ARMS}}
                for i in range(16)]}


class Controls(unittest.TestCase):
    def test_exact_literal_permutation_schedule(self):
        self.assertEqual(t.ORDERS,[('teacher','C','J'),('teacher','J','C'),('C','teacher','J'),('C','J','teacher'),('J','teacher','C'),('J','C','teacher')])
        for seed in range(16):
            orders=[t.ordered(seed,r) for r in range(30)]
            for order in t.ORDERS:self.assertEqual(orders.count(order),5)
            for arm in t.ARMS:
                for pos in range(3):self.assertEqual(sum(o[pos]==arm for o in orders),10)

    def test_fake_timer(self):
        ticks=iter([100,500]);calls=[]
        self.assertEqual(t.timed_block(lambda x:calls.append(x),'artificial',clock=lambda:next(ticks)),400)
        self.assertEqual(calls,['artificial']*20)

    def test_summary_all_pairs_and_no_outlier_removal(self):
        rows=artificial_rows();rows[0]['total_ns']=200000000;rows[0]['ns_per_forward']=10000000
        self.assertEqual(len(t.validate_raw(rows)),2880)
        result=t.summarize(rows)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(result['paired']),32)
        self.assertEqual(len(rows),2880);self.assertEqual(rows[0]['total_ns'],200000000)
        self.assertAlmostEqual(result['ratios']['batch1_C_over_J']['geomean'],5/6)
        self.assertEqual(result['ratios']['batch128_C_over_teacher']['smaller'],16)
        self.assertEqual(t.summarize(rows[:-1])['status'],'INCOMPLETE')

    def test_strict_duplicate_and_order_rejection(self):
        rows=artificial_rows();rows[1]=dict(rows[0])
        with self.assertRaises(ValueError):t.validate_raw(rows)
        with self.assertRaises(ValueError):t.validate_raw(artificial_rows()[:-1])

    def test_evidence_cap_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);writer=t.EvidenceWriter([root],cap=100000,reserve=70000)
            writer.write(root/'a',b'x'*30000)
            with self.assertRaises(RuntimeError):writer.write(root/'b',b'x')
            writer.write(root/'c',b'x'*4000,final=True)
            with self.assertRaises(RuntimeError):writer.write(root/'d',b'x'*1000,final=True)

    def test_full_failure_census(self):
        census=t.full_census();census[0]['status']='STARTED';t.failure_census(census)
        self.assertEqual(len(census),96);self.assertEqual(census[0]['status'],'FAILED')
        self.assertEqual(sum(x['status']=='NOT_STARTED' for x in census),95)

    def test_supervisor_fake_native_hang(self):
        proc=mock.Mock(pid=123);proc.wait.side_effect=[subprocess.TimeoutExpired('fake',590),subprocess.TimeoutExpired('fake',5),-9]
        with mock.patch.object(supervise.os,'killpg') as kill:
            self.assertEqual(supervise.bounded_wait(proc,590),(-9,True))
            self.assertEqual(kill.call_count,2)
            self.assertEqual([x.kwargs['timeout'] for x in proc.wait.call_args_list],[590,5,5])

    def test_supervisor_zero_exit_is_not_false_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);reg=root/'registry';reg.mkdir();out=root/'out'
            event=supervise.terminal(reg,out,0,False,0)
            self.assertEqual(event['status'],'FAILED');self.assertEqual(len(event['census']),96)

    def test_supervisor_rescues_truncated_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);reg=root/'registry';reg.mkdir()
            (reg/'CENSUS.json').write_text('[{')
            event=supervise.terminal(reg,root/'out',-9,True,0)
            self.assertEqual(event['status'],'FAILED');self.assertEqual(len(event['census']),96)
            self.assertIsNotNone(event['metadata_error_type'])
            self.assertTrue(all(c['status']=='ABORTED_BY_SUPERVISOR' for c in event['census']))

    def test_exceptional_wait_terminates_fake_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);inp=root/'inputs';control=root/'control';inp.mkdir();control.mkdir()
            release={'attempt_id':'synthetic','control_root_sha256':hashlib.sha256(str(control.resolve()).encode()).hexdigest()}
            proc=mock.Mock(pid=123);proc.poll.return_value=None
            proc.wait.side_effect=[RuntimeError('artificial wait failure'),-15]
            argv=['--out',str(root/'out'),'--release','artificial','--manifest','artificial',
                  '--manifest-sha256','a'*64,'--input-root',str(inp),'--control-root',str(control)]
            original_reader=c.read_json
            def read(path,*args):return release if str(path)=='artificial' else original_reader(path,*args)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(c,'read_json',side_effect=read))
                stack.enter_context(mock.patch.object(c,'package_hashes',return_value={}))
                stack.enter_context(mock.patch.object(c,'git_identity',return_value='c'*40))
                stack.enter_context(mock.patch.object(c,'validate_release',return_value=1000))
                stack.enter_context(mock.patch.object(c,'sha',return_value='b'*64))
                stack.enter_context(mock.patch.object(supervise.time,'time',return_value=100))
                stack.enter_context(mock.patch.object(supervise.time,'monotonic',return_value=0))
                stack.enter_context(mock.patch.object(supervise.subprocess,'Popen',return_value=proc))
                kill=stack.enter_context(mock.patch.object(supervise.os,'killpg'))
                with self.assertRaises(RuntimeError):supervise.main(argv)
                kill.assert_called_once_with(123,supervise.signal.SIGTERM)
            event=json.loads((control/'attempt-synthetic'/'SUPERVISOR_EXIT.json').read_text())
            self.assertEqual(event['status'],'FAILED');self.assertEqual(len(event['census']),96)
            self.assertEqual(proc.wait.call_args_list[-1].kwargs,{'timeout':5})

    def test_manifest_rejects_traversal_and_missing_seed(self):
        c.validate_inputs(artificial_manifest())
        manifest=artificial_manifest();manifest['csv']['path']='../data.gz'
        with self.assertRaises(ValueError):c.validate_inputs(manifest)
        manifest=artificial_manifest();manifest['validation'][15]['id']=14
        with self.assertRaises(ValueError):c.validate_inputs(manifest)

    def test_existing_output_and_consumed_attempt_are_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d)
            inp=root/'inputs';pkg=root/'package';control=root/'control';out=root/'output'
            for path in (inp,pkg,control):path.mkdir()
            out.mkdir();(out/'marker').write_text('keep')
            with self.assertRaises(ValueError):c.paths(out,inp,control,pkg,'synthetic')
            self.assertEqual((out/'marker').read_text(),'keep')
            (control/'attempt-synthetic').mkdir()
            with self.assertRaises(ValueError):c.paths(root/'fresh',inp,control,pkg,'synthetic')
            self.assertFalse((root/'fresh').exists())

    def test_no_symlink_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);(root/'x').write_text('synthetic');(root/'alias').symlink_to(root/'x')
            with self.assertRaises(ValueError):c.contained(root,'alias')

    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'x.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):c.read_json(path)

    def test_full_worker_with_only_fake_callables_and_fake_clock(self):
        class ArtificialInput:
            def astype(self,_):return self
            def __truediv__(self,_):return self
            def __getitem__(self,index):return {'batch':index.stop}
        counts={};ticks=iter(range(0,1000000,100));load_events=[]
        def load(selection,*_):
            seed=selection['id'];load_events.append(seed)
            return {arm:(lambda x,s=seed,a=arm:counts.__setitem__((s,x['batch'],a),counts.get((s,x['batch'],a),0)+1)) for arm in t.ARMS}
        fake_np=types.SimpleNamespace(float32='artificial-float32',random=types.SimpleNamespace(default_rng=lambda seed:types.SimpleNamespace(integers=lambda *a,**k:ArtificialInput())),save=lambda buf,x,allow_pickle:buf.write(b'artificial-only'))
        fake_torch=types.SimpleNamespace(no_grad=contextlib.nullcontext,from_numpy=lambda x:x)
        runtime=types.SimpleNamespace(np=fake_np,torch=fake_torch,load_models=load,validation=lambda *a:{'artificial':True})
        inputs={'models':[{'selection':{'id':i},'expected':{},'validation':{},'sources':{},'references':{},'split':'artificial'} for i in range(16)],'artifacts':{},'storage_root':'artificial','csv':'artificial'}
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);reg=root/'registry';reg.mkdir();out=root/'out'
            args=types.SimpleNamespace(out=str(out),registry=str(reg),release='artificial',manifest='artificial',manifest_sha256='a'*64,input_root='artificial')
            request={'release_sha256':'b'*64,'input_manifest_sha256':'a'*64}
            with contextlib.ExitStack() as stack:
                for name,value in [('read_json',request),('package_hashes',{}),('git_identity','c'*40),('validate_release',1000),('sha','b'*64),('load_inputs',inputs)]:
                    stack.enter_context(mock.patch.object(b.contract,name,return_value=value))
                stack.enter_context(mock.patch.dict(b.os.environ,{'CANARIA_LATENCY_SUPERVISOR':str(b.os.getppid())}))
                stack.enter_context(mock.patch.object(b,'configure_runtime',return_value=runtime))
                stack.enter_context(mock.patch.object(b,'environment',return_value={'artificial':True}))
                stack.enter_context(mock.patch.object(b.resource,'setrlimit'))
                stack.enter_context(mock.patch.object(b.signal,'signal'));stack.enter_context(mock.patch.object(b.signal,'alarm'))
                stack.enter_context(mock.patch.object(b.time,'time',return_value=100))
                stack.enter_context(mock.patch.object(b,'timed_block',side_effect=lambda fn,x:t.timed_block(fn,x,clock=lambda:next(ticks))))
                b.run(args)
            raw=[json.loads(line) for line in (out/'RAW.jsonl').read_text().splitlines()]
            self.assertEqual(load_events,list(range(16)));self.assertEqual(len(t.validate_raw(raw)),2880)
            self.assertEqual(len(counts),96);self.assertTrue(all(n==20+30*20 for n in counts.values()))
            self.assertEqual(json.loads((reg/'EXIT.json').read_text())['status'],'COMPLETE')


if __name__=='__main__':unittest.main()
