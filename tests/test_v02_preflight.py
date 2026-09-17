"""#RSNA #Kaggle #Testes — bounded cache and train-only pilot gates."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from scripts.preflight_v02_cache import check_image, MANIFEST_SHA, PLANES
from scripts.prepare_v02_pilot import validate,build


class V02Tests(unittest.TestCase):
    def test_pixels_indices_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'array.npz';image=np.arange(3*224*224,dtype=np.uint8).reshape(3,224,224)
            def write(x,indices):np.savez_compressed(p,image=x,sample_indices=np.array(indices,dtype=np.int16))
            write(image,[7,14,22]);s={'bytes':p.stat().st_size,'slices_recorded':30};r={'sample_indices':[7,14,22]}
            out=check_image(p,s,r);self.assertEqual(out['sample_gaps'],[7,8])
            for x,ind in [(image[:1],[7,14,22]),(image.astype(float),[7,14,22]),
                          (image,[0,1,2]),(np.zeros_like(image),[7,14,22])]:
                write(x,ind);s['bytes']=p.stat().st_size
                with self.assertRaises(ValueError):check_image(p,s,r)
            write(image,[7,14,22]);s['bytes']=p.stat().st_size+1
            with self.assertRaises(ValueError):check_image(p,s,r)

    def test_train_and_reserved_gates(self):
        train=[{'StudyInstanceUID':str(i),'labels':[.5]*12,'series':[{'plane':p} for p in PLANES]} for i in range(299)]
        pilot=copy.deepcopy(train[:12])
        a={'status':'PASSED_TRAIN_CACHE_PIXELS','manifest_sha256':MANIFEST_SHA,'train_studies':299,'series':897,
           'development_pixels_read':0,'confirmation_pixels_read':0,'pilot':pilot,'studies':train}
        self.assertEqual(validate(a),pilot)
        for key,value in [('status','FAILED'),('manifest_sha256','wrong'),('series',896),('confirmation_pixels_read',1)]:
            bad=copy.deepcopy(a);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(a);bad['pilot'][0]['labels'][0]=float('nan')
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(a);bad['pilot'][0]['StudyInstanceUID']='reserved'
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(a);bad['studies'].pop()
        with self.assertRaises(ValueError):validate(bad)
        source=build(a,'pass\n','pass\n');ast.parse(source)
        self.assertIn('pilot_only',source);self.assertNotIn('submission.csv',source)

    def test_runtime_parses_and_never_publishes(self):
        s=Path('scripts/v02_pilot_runtime.py').read_text();ast.parse(s)
        self.assertNotIn('submission.csv',s)
        self.assertIn('weights_only=True',s)
        self.assertIn("'resume_exact':True",s)
        self.assertIn('Rebuilt pixels differ from HD cache',s)

    def test_pixel_mismatch_persists_evidence_and_stops(self):
        tree=ast.parse(Path('scripts/v02_pilot_runtime.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','rebuild_series'}]
        namespace={'np':np,'json':json,'hashlib':hashlib,'Path':Path}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<pixel-gate>','exec'),namespace)
        pixels=np.arange(16,dtype=np.float32).reshape(4,4)
        channel=np.arange(224*224,dtype=np.uint8).reshape(224,224)
        helper={'PIXEL_TAGS':(),'_pixel_array':lambda ds:ds,
                'normalize_slice':lambda x:(x/15,0.,15.),'resize_slice':lambda x,size:channel}
        reader=SimpleNamespace(dcmread=lambda *args,**kwargs:pixels)
        expected=np.stack([channel]*3);digest=hashlib.sha256(expected.tobytes()).hexdigest()
        series={'series_uid':'series','selected_files':['a.dcm','b.dcm','c.dcm'],'image_sha256':digest}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'train_series'/'train-study'/'series';data.mkdir(parents=True)
            for name in series['selected_files']:(data/name).write_bytes(b'synthetic fixture, not DICOM')
            output=root/'diagnostic'
            image,receipt=namespace['rebuild_series'](root,'train-study',series,helper,reader,output)
            np.testing.assert_array_equal(image,expected)
            self.assertEqual(receipt['sha256'],digest);self.assertFalse(output.exists())
            series['image_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'Rebuilt pixels differ'):
                namespace['rebuild_series'](root,'train-study',series,helper,reader,output)
            evidence=json.loads((output/'v02_pixel_mismatch.json').read_text())
            self.assertEqual(evidence['status'],'FAILED_EXACT_PIXEL_PARITY')
            self.assertEqual(evidence['observed_sha256'],digest)
            self.assertEqual(len(evidence['stages']),3)
            with np.load(output/'v02_pixel_mismatch.npz',allow_pickle=False) as archive:
                self.assertEqual(len(archive.files),7)
                np.testing.assert_array_equal(archive['pixels_0'],pixels)
            self.assertFalse((output/'v02_pilot_features.npz').exists())


if __name__=='__main__':unittest.main()
