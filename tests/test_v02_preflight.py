"""#RSNA #Kaggle #Testes — bounded cache and train-only pilot gates."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
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


if __name__=='__main__':unittest.main()
