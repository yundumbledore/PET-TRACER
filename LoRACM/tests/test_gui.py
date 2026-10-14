"""Every GUI page must render before starting a computation."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import os
from unittest.mock import patch
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Optional GUI dependency is not installed')
class GuiTests(unittest.TestCase):
    def setUp(self):
        self.workspace=tempfile.TemporaryDirectory()
        self.environment=patch.dict(os.environ,{'PET_TRACER_WORKSPACES':self.workspace.name})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.workspace.cleanup()

    def test_drafts_survive_arbitrary_navigation_and_project_switches(self):
        from streamlit.testing.v1 import AppTest
        file=str(Path(__file__).resolve().parents[1]/'gui/app.py')
        app=AppTest.from_file(file,default_timeout=45).run()
        def page(name):
            app.sidebar.radio[0].set_value(name).run()
            self.assertFalse(list(app.exception))
        def number(label,value): next(w for w in app.number_input if w.label==label).set_value(value).run()
        page('1 · Model & protocol')
        number('Isotope half-life (minutes)',123.4)
        next(w for w in app.text_input if w.label=='Parameter names, in order').set_value('K1, k2, k3, k4, Vb   ').run()
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        self.assertFalse(list(app.exception))
        page('2 · Dataset');number('Training pairs',1234);number('Simulation random seed',42)
        page('3 · Fine-tune');number('Epochs',7);number('Rank',6)
        page('6 · Parametric imaging');page('2 · Dataset')
        self.assertEqual(next(w for w in app.number_input if w.label=='Training pairs').value,1234)
        self.assertEqual(next(w for w in app.number_input if w.label=='Simulation random seed').value,42)
        page('5 · Single TAC & NLS');number('Reverse sampling steps',5)
        page('1 · Model & protocol')
        self.assertEqual(next(w for w in app.number_input if w.label=='Isotope half-life (minutes)').value,123.4)
        self.assertEqual(next(w for w in app.text_input if w.label=='Parameter names, in order').value,'K1, k2, k3, k4, Vb   ')
        page('3 · Fine-tune')
        self.assertEqual(next(w for w in app.number_input if w.label=='Epochs').value,7)
        self.assertEqual(next(w for w in app.number_input if w.label=='Rank').value,6)
        app.sidebar.text_input[0].set_value('independent').run()
        self.assertEqual(next(w for w in app.number_input if w.label=='Epochs').value,5000)
        app.sidebar.text_input[0].set_value('my_project').run()
        self.assertEqual(next(w for w in app.number_input if w.label=='Epochs').value,7)
        # A new browser session restores the project-owned draft from disk.
        fresh=AppTest.from_file(file,default_timeout=45).run()
        fresh.sidebar.radio[0].set_value('3 · Fine-tune').run()
        self.assertEqual(next(w for w in fresh.number_input if w.label=='Epochs').value,7)
        self.assertFalse(list(fresh.exception))

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

    def test_pending_voxel_never_falls_back_to_an_older_result(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        workspace=Path(self.workspace.name)/'my_project'
        old=workspace/'analyses/old';old.mkdir(parents=True)
        draws=np.random.default_rng(42).uniform(.01,.5,(1,16,5))
        names=np.asarray(['K1','k2','k3','k4','Vb'])
        np.savez_compressed(old/'posterior.npz',posterior_samples=draws,parameter_names=names)
        (old/'analysis.json').write_text('{"state":"complete"}')
        pending=workspace/'analyses/new';pending.mkdir()
        drafts=app.session_state['_project_drafts_'+str(workspace)]
        drafts['current_analysis']=str(pending)
        app.sidebar.radio[0].set_value('5 · Single TAC & NLS').run()
        self.assertFalse(list(app.exception))
        self.assertFalse(any(w.value=='Kinetic parameter posteriors' for w in app.subheader))
        self.assertTrue(any('new voxel analysis is pending' in w.value for w in app.info))
        np.savez_compressed(pending/'posterior.npz',posterior_samples=draws,parameter_names=names)
        np.savez_compressed(pending/'input.npz',tac=np.array([[1.,2.]]),aif=np.array([3.,4.]),
                            t_meas=np.array([.5,1.5]),voxel_columns=np.asarray(['new_voxel']))
        (pending/'analysis.json').write_text('{"state":"complete"}')
        app.run()
        self.assertFalse(list(app.exception))
        self.assertTrue(any(w.value=='Kinetic parameter posteriors' for w in app.subheader))
        self.assertTrue(any('new_voxel' in w.value for w in app.caption))

    def test_ucdavis_is_default_and_sixth_page_is_available(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        self.assertEqual(next(w for w in app.sidebar.selectbox if w.label=='Example setup').value,'UC Davis Scenario 1')
        self.assertIn('6 · Parametric imaging',app.sidebar.radio[0].options)
        app.sidebar.radio[0].set_value('3 · Fine-tune').run()
        self.assertFalse(list(app.exception))
        self.assertEqual(next(w for w in app.number_input if w.label=='Rank').value,4)
        self.assertEqual(next(w for w in app.number_input if w.label=='Alpha').value,8.)

    def test_original_aif_defaults_and_coverage_checked_before_saving(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        self.assertEqual(Path(next(w for w in app.text_input if w.label=='Measured AIF CSV').value).name,'UCDavis_1H35.csv')
        self.assertTrue(next(w for w in app.checkbox if w.label.startswith('If the first AIF')).value)
        tail=next(w for w in app.selectbox if w.label=='After the final AIF sample')
        self.assertEqual(tail.value,'hold_last')
        tail.set_value('require_coverage').run()
        self.assertTrue(any('last measurement midpoint' in w.value for w in app.warning))
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        workspace=Path(self.workspace.name)/'my_project'
        self.assertFalse((workspace/'protocol.json').exists())
        self.assertTrue(any('57.5 min' in w.value for w in app.error))
        next(w for w in app.selectbox if w.label=='After the final AIF sample').set_value('hold_last').run()
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        self.assertFalse(list(app.exception))
        cfg=json.loads((workspace/'protocol.json').read_text())
        self.assertEqual((cfg['aif_before_first'],cfg['aif_after_last']),('zero_at_origin','hold_last'))
        app.sidebar.radio[0].set_value('2 · Dataset').run()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        self.assertEqual(next(w for w in app.selectbox if w.label=='After the final AIF sample').value,'hold_last')

    def test_dataset_preflight_rejects_incomplete_saved_aif_before_launch(self):
        from streamlit.testing.v1 import AppTest
        from ucdavis import defaults
        workspace=Path(self.workspace.name)/'my_project';workspace.mkdir()
        cfg,_=defaults(workspace,small=True)
        cfg['aif_after_last']='require_coverage'
        (workspace/'protocol.json').write_text(json.dumps(cfg))
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        app.sidebar.radio[0].set_value('2 · Dataset').run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Generate dataset').click().run()
            start.assert_not_called()
        self.assertFalse(list(app.exception))
        self.assertTrue(any('last measurement midpoint' in w.value for w in app.error))

    def test_all_pages_render(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45).run()
        next(w for w in app.sidebar.checkbox if w.label=='Enable CM from scratch comparison').check().run()
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
            next(w for w in app.sidebar.checkbox if w.label=='Enable CM from scratch comparison').check().run()
            app.sidebar.radio[0].set_value('CM from scratch').run()
            source=next(w for w in app.text_input if w.label=='Labelled dataset (.npz)')
            source.set_value(str(path)).run()
            self.assertFalse(list(app.exception))
            self.assertEqual([w.value for w in app.metric],['2','5','4'])
            self.assertEqual(next(w for w in app.number_input if w.label=='Batch size').value,2)
            self.assertEqual(next(w for w in app.selectbox if w.label=='Dataset kinetic model').value,'2TC')
            self.assertEqual(next(w for w in app.number_input if w.label=='Embedding dimension · embed_dim').value,1024)


if __name__=='__main__':unittest.main()
