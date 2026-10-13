"""Every GUI page must render before starting a computation."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Optional GUI dependency is not installed')
class GuiTests(unittest.TestCase):
    def test_imported_custom_dataset_can_review_without_simulation_protocol(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'imported.npz'
            arrays={'t_meas':np.array([.5,1.5]),'frame_durations_min':np.ones(2),'parameter_names':np.asarray(['K1','k2','Vb'])}
            for split in ('train','val','test'):
                arrays['x_'+split]=np.array([[.1,.2,.3,0],[.2,.3,.4,0]])
                arrays['y_'+split]=np.array([[1,2,3,4],[2,3,4,5]])
            np.savez_compressed(path,**arrays)
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
            app.sidebar.text_input[0].set_value('gui_import_fixture').run()
            app.sidebar.radio[0].set_value('3 · Fine-tune').run()
            next(w for w in app.text_input if w.label=='Labelled dataset (.npz)').set_value(str(path)).run()
            next(w for w in app.button if w.label=='Review configuration').click().run()
            self.assertFalse(list(app.exception))
            self.assertTrue(any('Configuration validated' in w.value for w in app.success))
            cfg=json.loads(app.json[0].value)
            self.assertEqual((cfg['x_dim'],cfg['y_dim'],cfg['kinetic_model']),(3,4,'custom'))
            self.assertNotIn('simulation_protocol',cfg)

    def test_ucdavis_is_default_and_sixth_page_is_available(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        self.assertEqual(next(w for w in app.sidebar.selectbox if w.label=='Starting template').value,'UC Davis Scenario 1')
        self.assertIn('6 · Parametric imaging',app.sidebar.radio[0].options)
        app.sidebar.radio[0].set_value('3 · Fine-tune').run()
        self.assertFalse(list(app.exception))
        self.assertEqual(next(w for w in app.number_input if w.label=='Rank').value,4)
        self.assertEqual(next(w for w in app.number_input if w.label=='Alpha').value,8.)

    def test_all_pages_render(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        self.assertFalse(list(app.exception))
        for page in app.sidebar.radio[0].options:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(list(app.exception),page)

    def test_scratch_page_derives_schema_and_caps_small_batch(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'small.npz'
            arrays={'t_meas':np.array([.5,1.5]),'frame_durations_min':np.ones(2),
                    'parameter_names':np.asarray(['K1','k2','k3','k4','Vb'])}
            for split in ('train','val','test'):
                arrays['x_'+split]=np.array([[.1,.2,.3,.4,.1,0],[.2,.3,.4,.5,.2,0]])
                arrays['y_'+split]=np.array([[1,2,3,4],[2,3,4,5]])
            np.savez_compressed(path,**arrays)
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
            app.sidebar.radio[0].set_value('CM from scratch').run()
            source=next(w for w in app.text_input if w.label=='Labelled dataset (.npz)')
            source.set_value(str(path)).run()
            self.assertFalse(list(app.exception))
            self.assertEqual([w.value for w in app.metric],['2','5','4'])
            self.assertEqual(next(w for w in app.number_input if w.label=='Batch size').value,2)
            self.assertEqual(next(w for w in app.selectbox if w.label=='Dataset kinetic model').value,'2TC')
            self.assertEqual(next(w for w in app.number_input if w.label=='Embedding dimension · embed_dim').value,1024)


if __name__=='__main__':unittest.main()
