"""Release changes must preserve user inputs and reconnect identical assets only."""
import json
from pathlib import Path
import tempfile
import unittest
from gui.release import sync_release


class ReleaseProjectTests(unittest.TestCase):
    def test_reconnects_identical_shipped_files_but_not_a_changed_custom_model(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);old=base/'old';new=base/'new';project=base/'project';project.mkdir()
            for root in (old,new):
                (root/'data').mkdir(parents=True);(root/'examples').mkdir()
                (root/'data/aif.csv').write_text('time,aif\n0,1\n')
            (old/'examples/custom.py').write_text('old model')
            (new/'examples/custom.py').write_text('different model')
            own=base/'my_data.csv';own.write_text('my study')
            cfg={'aif_csv':str(old/'data/aif.csv'),'model_file':str(old/'examples/custom.py'),'my_data':str(own)}
            path=project/'protocol.json';path.write_text(json.dumps(cfg))
            sync_release(old,project)
            sync_release(new,project)
            actual=json.loads(path.read_text())
            self.assertEqual(Path(actual['aif_csv']).resolve(),(new/'data/aif.csv').resolve())
            self.assertEqual(actual['model_file'],cfg['model_file'])
            self.assertEqual(actual['my_data'],str(own));self.assertEqual(own.read_text(),'my study')
            self.assertTrue((old/'data/aif.csv').exists())


if __name__=='__main__':unittest.main()
