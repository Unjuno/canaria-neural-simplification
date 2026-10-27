"""Pure artificial-array review tests; no historical model/data/artifact reads.
These tests are supplied as source and were NOT executed by the packager.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
import numpy as np
import manifest_contract as contract
import measure
import storage_codec as codec


def descriptor():
    return {'schema':contract.SCHEMA, 'cohort_size':16,
            'selection_origin_sha256':hashlib.sha256(b'artificial selection').hexdigest(),
            'models':[{'id':sid,
                'teacher':{'path':f'inputs/seed_{sid}_teacher.npz','file_sha256':'1'*64},
                'arms':{arm:{'path':f'inputs/seed_{sid}_{arm}.npz', 'file_sha256':'2'*64,
                              'state_sha256':'3'*64, 'budget':512 if arm=='C' else 1024}
                        for arm in ('C','J')}} for sid in range(16)]}


def arrays(shapes, offset=0):
    return {key:(np.arange(np.prod(shape),dtype='<f4').reshape(shape)/np.float32(1024)+np.float32(offset))
            for key,shape in shapes.items()}


def artificial_cohort(root):
    manifest = descriptor()
    (root/'inputs').mkdir()
    for row in manifest['models']:
        for arm,source in [('teacher',row['teacher']),*row['arms'].items()]:
            shapes = codec.shapes_teacher() if arm=='teacher' else codec.shapes_arm(arm,source['budget'])
            values = arrays(shapes,row['id'])
            path = root/source['path']
            codec.serialize(path,values,zipfile.ZIP_STORED)
            source['file_sha256'] = codec.sha(path)
            if arm!='teacher':
                source['state_sha256'] = codec.state_hash(values)
    return manifest


def reference_bytes(values, compression):
    """Independent restatement of original write semantics, not codec calls."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=compression,
                         compresslevel=6 if compression==zipfile.ZIP_DEFLATED else None) as archive:
        for name in sorted(values):
            payload = io.BytesIO()
            np.lib.format.write_array(payload,np.ascontiguousarray(values[name]),version=(1,0),allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy',(1980,1,1,0,0,0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.compress_type = compression
            archive.writestr(info,payload.getvalue(),compress_type=compression,
                             compresslevel=6 if compression==zipfile.ZIP_DEFLATED else None)
    return stream.getvalue()


class ManifestTests(unittest.TestCase):
    def test_valid_artificial_manifest(self):
        value=descriptor()
        self.assertIs(contract.validate_manifest(value),value)

    def test_id_count_duplicate_bool_and_unknown_fields(self):
        for mutate in [lambda m:m['models'].pop(),
                       lambda m:m['models'][1].update(id=0),
                       lambda m:m['models'][0].update(id=True),
                       lambda m:m.update(extra='forbidden')]:
            with self.subTest(mutate=mutate):
                value=descriptor();mutate(value)
                with self.assertRaises(ValueError):contract.validate_manifest(value)

    def test_path_and_digest_mismatches(self):
        for path in ('../bad.npz','/bad.npz','x/../bad.npz','x//bad.npz','x\\bad.npz','https://host/a.npz','x.npy'):
            value=descriptor();value['models'][0]['teacher']['path']=path
            with self.subTest(path=path),self.assertRaises(ValueError):contract.validate_manifest(value)
        value=descriptor();value['models'][1]['teacher']['path']=value['models'][0]['teacher']['path']
        with self.assertRaises(ValueError):contract.validate_manifest(value)
        value=descriptor();value['models'][0]['teacher']['file_sha256']='0'*63
        with self.assertRaises(ValueError):contract.validate_manifest(value)

    def test_budget_and_arm_census(self):
        for budget in (True,0,513,8192,512.0):
            value=descriptor();value['models'][0]['arms']['C']['budget']=budget
            with self.subTest(budget=budget),self.assertRaises(ValueError):contract.validate_manifest(value)
        value=descriptor();del value['models'][0]['arms']['J']
        with self.assertRaises(ValueError):contract.validate_manifest(value)

    def test_manifest_digest_and_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'manifest.json'
            path.write_text(json.dumps(descriptor()))
            self.assertEqual(contract.read_manifest(path,codec.sha(path)),descriptor())
            with self.assertRaises(ValueError):contract.read_manifest(path,'0'*64)
            path.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaises(ValueError):contract.read_manifest(path,codec.sha(path))

    def test_symlinks_forbidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'real.npz').write_bytes(b'artificial')
            (root/'alias.npz').symlink_to(root/'real.npz')
            with self.assertRaises(ValueError):contract.source_path(root,'alias.npz')


class CodecTests(unittest.TestCase):
    def test_determinism_reference_roundtrip_and_metadata(self):
        values={'z':np.array([0.0,-0.0,1.0],dtype='<f4'),
                'a':np.arange(12,dtype='<f4').reshape(3,4)[:,::2]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
                with self.subTest(compression=compression):
                    one=root/f'{compression}_one.npz';two=root/f'{compression}_two.npz'
                    codec.serialize(one,values,compression)
                    codec.serialize(two,dict(reversed(list(values.items()))),compression)
                    self.assertEqual(one.read_bytes(),two.read_bytes())
                    self.assertEqual(one.read_bytes(),reference_bytes(values,compression))
                    with zipfile.ZipFile(one) as archive:
                        self.assertEqual(archive.namelist(),['a.npy','z.npy'])
                        self.assertEqual(archive.comment,b'')
                        for member in archive.infolist():
                            self.assertEqual(member.date_time,(1980,1,1,0,0,0))
                            self.assertEqual(member.create_system,3)
                            self.assertEqual(member.external_attr,0o600<<16)
                            self.assertEqual(member.compress_type,compression)
                            self.assertEqual(member.extra,b'')
                            self.assertEqual(archive.read(member)[:8],b'\x93NUMPY\x01\x00')
                    with np.load(one,allow_pickle=False) as saved:
                        for key,value in values.items():
                            self.assertEqual(saved[key].dtype,value.dtype)
                            self.assertEqual(saved[key].shape,value.shape)
                            self.assertEqual(saved[key].tobytes(),value.tobytes())

    def test_archive_collision_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'present.npz';path.write_bytes(b'untouched')
            with self.assertRaises(FileExistsError):codec.serialize(path,{'a':np.zeros(1,dtype='<f4')},zipfile.ZIP_STORED)
            self.assertEqual(path.read_bytes(),b'untouched')

    def test_source_hash_shape_dtype_finite_keys(self):
        expected={'a':(2,)}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            good=root/'good.npz';codec.serialize(good,{'a':np.ones(2,dtype='<f4')},zipfile.ZIP_STORED)
            self.assertEqual(codec.load_source(good,codec.sha(good),expected)['a'].tolist(),[1,1])
            with self.assertRaises(ValueError):codec.load_source(good,'0'*64,expected)
            bad_values=[{'a':np.ones(3,dtype='<f4')},{'a':np.ones(2,dtype='<f8')},
                        {'a':np.array([np.nan,1],dtype='<f4')},{'b':np.ones(2,dtype='<f4')}]
            for index,value in enumerate(bad_values):
                path=root/f'bad_{index}.npz';codec.serialize(path,value,zipfile.ZIP_STORED)
                with self.subTest(index=index),self.assertRaises(ValueError):codec.load_source(path,codec.sha(path),expected)

    def test_assembly_whole_model_denominator(self):
        teacher=arrays(codec.shapes_teacher())
        self.assertEqual(sum(a.size for a in teacher.values()),71626)
        self.assertEqual(len(teacher),28)
        for arm in ('C','J'):
            replacement=arrays(codec.shapes_arm(arm,512))
            whole=codec.assemble(teacher,replacement,arm,512)
            self.assertEqual(sum(a.size for a in whole.values()),38218+512)
            self.assertEqual(len(whole),18 if arm=='C' else 20)
            self.assertFalse(any(k.startswith(('blocks.0.','blocks.1.')) for k in whole))
            for key in teacher:
                if not key.startswith(('blocks.0.','blocks.1.')):
                    self.assertIs(whole[key],teacher[key])


class OutputTests(unittest.TestCase):
    def test_existing_output_rejected_before_manifest_or_numerical_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';inputs.mkdir();out=root/'results';out.mkdir()
            marker=out/'keep';marker.write_text('untouched')
            with mock.patch.object(contract,'read_manifest',side_effect=AssertionError('must not read manifest')):
                with self.assertRaises(ValueError):
                    measure.main(['--manifest',str(root/'absent.json'),'--manifest-sha256','0'*64,
                                  '--input-root',str(inputs),'--output',str(out)])
            self.assertEqual(marker.read_text(),'untouched')
            self.assertEqual(list(out.iterdir()),[marker])

    def test_output_not_in_input_or_package_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';inputs.mkdir();package=root/'package';package.mkdir()
            for output in (inputs/'new',package/'new'):
                with self.subTest(output=output),self.assertRaises(ValueError):contract.check_output(output,inputs,package)

    def test_complete_artificial_cohort_48_models_96_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'source';inputs.mkdir()
            manifest=artificial_cohort(inputs);out=root/'fresh'
            summary=measure.measure(manifest,inputs,out,'4'*64)
            self.assertEqual(summary['status'],'COMPLETE')
            self.assertEqual(summary['complete'],16)
            self.assertEqual(summary['produced_archives'],96)
            self.assertEqual(len(list(out.glob('*.npz'))),96)
            rows=json.loads((out/'ROWS.json').read_text())
            sources=json.loads((out/'SOURCES.json').read_text())
            self.assertEqual(len(sources),48)
            self.assertEqual({r['id'] for r in rows},set(range(16)))
            for row in rows:
                self.assertEqual(row['teacher_parameters'],71626)
                self.assertEqual(row['C_over_teacher_parameters'],(38218+512)/71626)
                self.assertEqual(row['J_over_teacher_parameters'],(38218+1024)/71626)
            self.assertAlmostEqual(summary['ratios']['C_over_J_parameters']['geomean'],(38218+512)/(38218+1024))
            with self.assertRaises(ValueError):measure.measure(manifest,inputs,out,'4'*64)

    def test_logical_mismatch_retains_failure_and_incomplete_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'source';inputs.mkdir()
            manifest=artificial_cohort(inputs)
            manifest['models'][0]['arms']['C']['state_sha256']='0'*64
            summary=measure.measure(manifest,inputs,root/'fresh','4'*64)
            self.assertEqual(summary['status'],'INCOMPLETE')
            self.assertEqual(summary['cohort'],16)
            self.assertEqual(summary['complete'],15)
            self.assertEqual(summary['produced_archives'],90)
            self.assertEqual(summary['failures'],[{'id':0,'status':'FAIL','error_type':'ValueError'}])
            self.assertEqual(len(json.loads((root/'fresh'/'CENSUS.json').read_text())),16)

    def test_geometric_not_arithmetic_summary(self):
        rows=[{'C_over_J_parameters':0.5},{'C_over_J_parameters':2.0}]
        summary=measure.summarize(rows,[{'id':0,'status':'PASS'},{'id':1,'status':'PASS'}])
        self.assertEqual(summary['ratios']['C_over_J_parameters']['geomean'],1.0)
        self.assertEqual(summary['ratios']['C_over_J_parameters']['smaller'],1)
        self.assertEqual(summary['ratios']['C_over_J_parameters']['larger'],1)


if __name__=='__main__':
    unittest.main()
