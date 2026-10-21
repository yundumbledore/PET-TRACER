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

    def open_app(self,module='lora',file=None):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(file or str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45)
        app.query_params['module']=module
        return app.run()

    def prediction_fixture(self,mode='lora',name='scenario1'):
        workspace=Path(self.workspace.name)/('cm' if mode=='cm' else '')/'my_project'
        run=workspace/'runs'/name;run.mkdir(parents=True,exist_ok=True)
        cfg={'training_mode':mode,'x_dim':5,'y_dim':4,'parameter_names':['K1','k2','k3','k4','Vb'],'kinetic_model':'2TC'}
        (run/'model_params.json').write_text(json.dumps(cfg))
        (run/'scaling_params.json').write_text(json.dumps({'t_meas':[.5,1.5],'frame_durations_min':[1,1]}))
        (run/'best.pth').touch()
        source=workspace/'patient.npz'
        np.savez(source,tac=np.ones((3,2)),aif=np.ones(2),t_meas=[.5,1.5],frame_durations_min=[1,1],voxel_columns=['a','b','c'])
        return workspace,run,source,cfg

    def saved_prediction(self,output,source,cfg):
        from common import file_sha256
        meta={'input':str(source),'input_sha256':file_sha256(source),'config':cfg}
        np.savez(output,parameter_names=cfg['parameter_names'],voxel_columns=['a','b','c'],params_mean=np.ones((3,5)),
                 params_ci95_width=np.ones((3,5)),Ki_mean=[1,2,3],Ki_ci95_width=[.1,.2,.3],metadata_json=json.dumps(meta))
        output.with_suffix('.json').write_text(json.dumps(meta))

    def test_scratch_dataset_has_independent_size_protocol_and_training_selection(self):
        lora=self.open_app()
        lora.sidebar.radio[0].set_value('1 · Model & protocol').run()
        next(w for w in lora.button if w.label=='Save and validate protocol').click().run()
        workspace=Path(self.workspace.name)/'my_project';original=(workspace/'protocol.json').read_bytes()
        lora.sidebar.radio[0].set_value('2 · Dataset').run()
        next(w for w in lora.number_input if w.label=='Training pairs').set_value(1234).run()
        app=self.open_app('cm')
        self.assertNotIn('3 · Fine-tune',app.sidebar.radio[0].options)
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        app.sidebar.radio[0].set_value('CM · Dataset').run()
        self.assertEqual(next(w for w in app.number_input if w.label=='Training pairs').value,1000000)
        next(w for w in app.number_input if w.label=='Training pairs').set_value(9).run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Generate CM dataset').click().run()
            self.assertTrue(start.called)
        cfg=json.loads((Path(self.workspace.name)/'cm/my_project/cm_protocol.json').read_text())
        self.assertEqual(cfg['n_train'],9);self.assertEqual(Path(cfg['output']).name,'cm_dataset.npz')
        self.assertEqual((workspace/'protocol.json').read_bytes(),original)
        app.sidebar.radio[0].set_value('CM from scratch').run()
        self.assertEqual(Path(next(w for w in app.text_input if w.label=='Labelled dataset (.npz)').value).name,'cm_dataset.npz')
        lora.sidebar.radio[0].set_value('2 · Dataset').run()
        self.assertEqual(next(w for w in lora.number_input if w.label=='Training pairs').value,1234)
        self.assertFalse(list(app.exception))

    def test_repeated_gui_predictions_preserve_files_and_cm_route_filters_models(self):
        from streamlit.testing.v1 import AppTest
        workspace,run,source,cfg=self.prediction_fixture()
        _,cm,_,_=self.prediction_fixture('cm','cm_scratch')
        old=run/'predictions.npz';self.saved_prediction(old,source,cfg);original=old.read_bytes()
        app=self.open_app()
        app.sidebar.radio[0].set_value('4 · Predict').run()
        self.assertIn('LoRACM · scenario1',next(w for w in app.selectbox if w.label=='Model to use').options)
        next(w for w in app.selectbox if w.label=='Prediction input').set_value('My own file').run()
        next(w for w in app.text_input if w.label=='Testing dataset').set_value(str(source)).run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Run prediction').click().run()
            args=start.call_args.args[1]
            destination=Path(args[args.index('--output')+1]);self.assertEqual(destination.name,'predictions_002.npz')
        self.assertFalse(any(w.value.startswith('Posterior summaries') for w in app.subheader))
        self.saved_prediction(destination,source,cfg)
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Run prediction').click().run()
            args=start.call_args.args[1]
            self.assertEqual(Path(args[args.index('--output')+1]).name,'predictions_003.npz')
        self.assertEqual(old.read_bytes(),original)
        app=self.open_app('cm')
        app.sidebar.radio[0].set_value('CM · Predict').run()
        self.assertEqual(next(w for w in app.selectbox if w.label=='Model to use').options,['CM from scratch · cm_scratch'])
        self.assertTrue(any(w.value=='CM · Predict with your scratch model' for w in app.title))
        self.assertFalse(list(app.exception))

    def test_own_scan_requires_mask_and_carries_verified_geometry_to_imaging(self):
        from streamlit.testing.v1 import AppTest
        workspace,run,source,cfg=self.prediction_fixture()
        mask=workspace/'patient_mask.npy';np.save(mask,np.array([[0,1],[1,1]]))
        app=self.open_app()
        app.sidebar.radio[0].set_value('4 · Predict').run()
        next(w for w in app.selectbox if w.label=='Prediction input').set_value('My own patient scan').run()
        next(w for w in app.text_input if w.label=='Testing dataset').set_value(str(source)).run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Run prediction').click().run();start.assert_not_called()
        self.assertTrue(any('corresponding mask' in w.value for w in app.error))
        next(w for w in app.checkbox if w.label.startswith('I verified the TAC voxel order')).check().run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Run prediction').click().run()
            self.assertTrue(start.called)
            args=start.call_args.args[1];prediction=Path(args[args.index('--output')+1])
        self.saved_prediction(prediction,source,cfg)
        app.sidebar.radio[0].set_value('6 · Parametric imaging').run()
        self.assertFalse(list(app.exception))
        geometry=Path(next(w for w in app.text_input if w.label=='Voxel geometry NPZ').value)
        self.assertIn('patient_scans',geometry.parts)
        self.assertIn('Ki',next(w for w in app.selectbox if w.label=='Parameter to image').options)
        self.assertIn('K1',next(w for w in app.selectbox if w.label=='Parameter to image').options)
        next(w for w in app.selectbox if w.label=='Map value').set_value('95% credible-interval width').run()
        self.assertFalse(list(app.exception))

    def test_drafts_survive_arbitrary_navigation_and_project_switches(self):
        from streamlit.testing.v1 import AppTest
        file=str(Path(__file__).resolve().parents[1]/'gui/app.py')
        app=self.open_app(file=file)
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
        fresh=self.open_app(file=file)
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
            app=self.open_app()
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
        app=self.open_app()
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
        app=self.open_app()
        self.assertEqual(next(w for w in app.sidebar.selectbox if w.label=='Example setup').value,'UC Davis Scenario 1')
        self.assertIn('6 · Parametric imaging',app.sidebar.radio[0].options)
        app.sidebar.radio[0].set_value('3 · Fine-tune').run()
        self.assertFalse(list(app.exception))
        self.assertEqual(next(w for w in app.number_input if w.label=='Rank').value,4)
        self.assertEqual(next(w for w in app.number_input if w.label=='Alpha').value,8.)

    def test_corrected_ucdavis_prior_defaults_and_project_edits_persist(self):
        from streamlit.testing.v1 import AppTest
        app=self.open_app()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        prior=next(w for w in app.text_area if w.label=='Parameter priors (uniform ranges)')
        expected=json.loads(prior.value)
        self.assertEqual({name:spec['max'] for name,spec in expected.items()},dict(K1=2.,k2=2.,k3=.5,k4=.2,Vb=1.))
        changed=json.loads(prior.value);changed['K1']['max']=1.1
        prior.set_value(json.dumps(changed)).run()
        next(w for w in app.number_input if w.label=='Isotope half-life (minutes)').set_value(100.).run()
        app.sidebar.radio[0].set_value('2 · Dataset').run()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        self.assertEqual(json.loads(next(w for w in app.text_area if w.label=='Parameter priors (uniform ranges)').value)['K1']['max'],1.1)
        self.assertFalse(any(w.label=='Load UC Davis source priors' for w in app.button))
        self.assertEqual(next(w for w in app.number_input if w.label=='Isotope half-life (minutes)').value,100.)
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        cfg=json.loads((Path(self.workspace.name)/'my_project/protocol.json').read_text())
        self.assertEqual(cfg['kinetic_priors'],changed)
        self.assertEqual(cfg['half_life_min'],100.)
        self.assertFalse(list(app.exception))

    def test_protocol_page_uses_midpoints_without_convention_controls(self):
        from streamlit.testing.v1 import AppTest
        app=self.open_app()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        self.assertEqual(Path(next(w for w in app.text_input if w.label=='Measured AIF CSV').value).name,'UCDavis_1H35.csv')
        labels=[w.label for w in app.selectbox]
        self.assertNotIn('Simulation convention',labels)
        self.assertNotIn('How are your measured curves sampled?',labels)
        self.assertFalse(any('Paper convention:' in w.value for w in app.info))
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        self.assertFalse(list(app.exception))
        workspace=Path(self.workspace.name)/'my_project'
        cfg=json.loads((workspace/'protocol.json').read_text())
        self.assertEqual((cfg['simulation_mode'],cfg['observation'],cfg['random_stream']),('hpc_euler','midpoint','jax_hpc'))
        from ucdavis import defaults
        original,_=defaults(workspace)
        self.assertEqual(cfg['t_meas'],original['t_meas'])
        np.testing.assert_allclose(cfg['prediction_t_meas'],original['prediction_t_meas'],atol=1e-8,rtol=0)
        self.assertEqual(cfg['aif_sampling'],'source_samples')
        app.sidebar.radio[0].set_value('2 · Dataset').run()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        self.assertFalse(list(app.exception))

    def test_custom_schedule_is_saved_without_hidden_ucdavis_override(self):
        from streamlit.testing.v1 import AppTest
        app=self.open_app()
        app.sidebar.radio[0].set_value('1 · Model & protocol').run()
        # Restore an edited schedule draft, as on return from another page.
        drafts=app.session_state['_project_drafts_'+str(Path(self.workspace.name)/'my_project')]
        key='1 · Model & protocol:data_editor:schedule_my_project'
        frame=json.loads(drafts[key]['_dataframe']);frame['data'][-1][0]=57.51
        drafts[key]={'_dataframe':json.dumps(frame)}
        app.run()
        next(w for w in app.button if w.label=='Save and validate protocol').click().run()
        self.assertFalse(list(app.exception))
        cfg=json.loads((Path(self.workspace.name)/'my_project/protocol.json').read_text())
        self.assertEqual(cfg['t_meas'][-1],57.51)
        self.assertNotIn('prediction_t_meas',cfg)
        self.assertEqual(cfg['aif_sampling'],'interpolate_to_schedule')

    def test_dataset_preflight_rejects_incomplete_saved_aif_before_launch(self):
        from streamlit.testing.v1 import AppTest
        from ucdavis import defaults
        workspace=Path(self.workspace.name)/'my_project';workspace.mkdir()
        cfg,_=defaults(workspace,small=True)
        cfg.update(simulation_mode='rk4',t_meas=cfg.pop('prediction_t_meas'),aif_before_first='zero_at_origin',aif_after_last='require_coverage')
        (workspace/'protocol.json').write_text(json.dumps(cfg))
        app=self.open_app()
        app.sidebar.radio[0].set_value('2 · Dataset').run()
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Generate dataset').click().run()
            start.assert_not_called()
        self.assertFalse(list(app.exception))
        self.assertTrue(any('last measurement midpoint' in w.value for w in app.error))

    def test_paper_checkpoint_is_available_without_dataset_or_training(self):
        from streamlit.testing.v1 import AppTest
        from common import file_sha256
        app=self.open_app()
        app.sidebar.radio[0].set_value('5 · Single TAC & NLS').run()
        workspace=Path(self.workspace.name)/'my_project'
        paper=workspace/'runs/paper_ucdavis'
        self.assertTrue((paper/'best.pth').exists())
        self.assertEqual(file_sha256(paper/'best.pth'),'f4eef814a56b78e639a4398fcf6eaa29f2526f3afb018b5fbd55b986ae04b420')
        self.assertFalse((workspace/'dataset.npz').exists())
        self.assertFalse((workspace/'train.json').exists())
        self.assertEqual(Path(next(w for w in app.text_input if w.label=='Model run folder').value),paper)
        app.sidebar.radio[0].set_value('4 · Predict').run()
        next(w for w in app.button if w.label=='Validate input').click().run()
        self.assertTrue(any('24237' in w.value for w in app.success))
        app.sidebar.radio[0].set_value('3 · Fine-tune').run()
        self.assertEqual(next(w for w in app.number_input if w.label=='Scheduler patience (epochs)').value,50)
        next(w for w in app.button if w.label=='Use supplied paper checkpoint').click().run()
        self.assertTrue(any('Paper checkpoint ready' in w.value for w in app.success))
        self.assertFalse(list(app.exception))

    def test_all_pages_render(self):
        from streamlit.testing.v1 import AppTest
        app=self.open_app()
        self.assertFalse(list(app.exception))
        from gui.navigation import WORKFLOW_PAGES
        for page in WORKFLOW_PAGES['lora']:
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
            app=self.open_app('cm')
            app.sidebar.radio[0].set_value('CM from scratch').run()
            source=next(w for w in app.text_input if w.label=='Labelled dataset (.npz)')
            source.set_value(str(path)).run()
            self.assertFalse(list(app.exception))
            self.assertEqual([w.value for w in app.metric],['2','5','4'])
            self.assertEqual(next(w for w in app.number_input if w.label=='Batch size').value,2)
            self.assertEqual(next(w for w in app.selectbox if w.label=='Dataset kinetic model').value,'2TC')
            self.assertEqual(next(w for w in app.number_input if w.label=='Embedding dimension · embed_dim').value,1024)


if __name__=='__main__':unittest.main()
