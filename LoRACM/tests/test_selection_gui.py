"""Navigation, saved sampling controls and browser-viewer rendering contracts."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('streamlit'),'Optional GUI dependency is not installed')
class SelectionGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{'PET_TRACER_WORKSPACES':self.temp.name});self.env.start()
    def tearDown(self):
        from gui.model_selection import result_data
        result_data.clear()
        self.env.stop();self.temp.cleanup()
    def app(self,module='selection'):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'gui/app.py'),default_timeout=45)
        app.query_params['module']=module
        return app.run()
    def test_home_has_three_distinct_entry_points_and_back_navigation(self):
        app=self.app('home')
        for label in ['Open CM workflow','Open LoRACM workflow','Open model-selection workflow']:
            self.assertTrue(any(b.label==label for b in app.button))
        next(b for b in app.button if b.label=='Open CM workflow').click().run()
        value=app.query_params['module']
        self.assertEqual(value[0] if isinstance(value,list) else value,'cm')
        self.assertNotIn('3 · Fine-tune',app.sidebar.radio[0].options)
        self.assertFalse(any(b.label=='Enable CM from scratch comparison' for b in app.checkbox))
        next(b for b in app.button if b.label=='← All modules').click().run()
        self.assertTrue(any(b.label=='Open model-selection workflow' for b in app.button))
        self.assertFalse(list(app.exception))
    def test_settings_survive_navigation_and_control_the_inference_command(self):
        app=self.app();radio=app.sidebar.radio[0]
        radio.set_value('2 · Inference settings').run()
        for name,value in [('Posterior draws per voxel',17),('Reverse sampling steps',5),('Voxel curve batch size',3),('Posterior sample chunk',4),('Inference random seed',123)]:
            next(w for w in app.number_input if w.label==name).set_value(value).run()
        next(w for w in app.selectbox if w.label=='Inference device').set_value('cpu').run()
        app.sidebar.radio[0].set_value('3 · Run inference').run()
        cfg=json.loads((Path(self.temp.name)/'model_selection/my_project/inference_settings.json').read_text())
        self.assertEqual((cfg['samples'],cfg['steps'],cfg['batch_size'],cfg['sample_chunk'],cfg['seed']),(17,5,3,4,123))
        with patch('gui.jobs.Jobs.start') as start:
            next(w for w in app.button if w.label=='Run P001 model selection').click().run()
            args=list(map(str,start.call_args.args[1]))
            for flag,value in [('--samples','17'),('--steps','5'),('--batch-size','3'),('--sample-chunk','4'),('--seed','123'),('--device','cpu')]:
                self.assertEqual(args[args.index(flag)+1],value)
        app.sidebar.radio[0].set_value('2 · Inference settings').run()
        self.assertEqual(next(w for w in app.number_input if w.label=='Posterior draws per voxel').value,17)
        self.assertFalse(list(app.exception))
    def test_completed_real_checkpoint_result_renders_map_and_conditional_posteriors(self):
        from model_selection.infer import parser,run
        from gui.model_selection import result_data
        target=Path(self.temp.name)/'model_selection/my_project/results/test'
        run(parser().parse_args(['--output',str(target),'--device','cpu','--limit','4','--samples','16','--steps','3']))
        app=self.app();app.sidebar.radio[0].set_value('4 · Explore results').run()
        self.assertFalse(list(app.exception))
        self.assertTrue(any('kinetic parameter posteriors' in w.value for w in app.subheader))
        self.assertTrue(any(w.label=='Download model selection CSV' for w in app.get('download_button')))
        result=result_data(str(target),(target/'selection.npz').stat().st_mtime_ns)
        result['posteriors']._mmap.close();result_data.clear()

    def test_curve_plot_uses_joint_draw_mean_band_and_observed_tac_scale(self):
        import numpy as np
        from model_selection.infer import parser,run
        from model_selection.core import load_demo,forward_posterior
        from gui.model_selection import result_data
        target=Path(self.temp.name)/'model_selection/my_project/results/band'
        run(parser().parse_args(['--output',str(target),'--device','cpu','--limit','4','--samples','16','--steps','3']))
        draws=np.tile([.3,.2,.04,.005,0.,0.,.06,0.,0.,3.,0.],(4,16,1))
        draws[:,:,0]=np.linspace(.1,.5,16);draws[:,:,10]=np.linspace(0.,.1,16)
        np.save(target/'posteriors.npy',draws)
        with np.load(target/'selection.npz',allow_pickle=False) as f:summary={key:f[key] for key in f.files}
        summary['probabilities']=np.zeros((4,7));summary['probabilities'][:,3]=1.
        summary['map_models']=np.full(4,3);summary['confidences']=np.ones(4)
        np.savez(target/'selection.npz',**summary)
        plots=[]
        def capture(figure,*args,**kwargs):
            ax=figure.axes[0]
            if ax.get_title()!='TAC and forward simulation':return
            line=next(line for line in ax.lines if line.get_label()=='2TC posterior mean')
            band=next(collection for collection in ax.collections if collection.get_label()=='95% credible band')
            observed=next(line for line in ax.lines if line.get_label()=='Observed TAC')
            plots.append((line.get_ydata().copy(),band.get_alpha(),ax.get_ylim(),observed.get_ydata().copy(),[text.get_text() for text in ax.get_legend().get_texts()]))
        with patch('gui.model_selection.st.pyplot',side_effect=capture):
            app=self.app();app.sidebar.radio[0].set_value('4 · Explore results').run()
        self.assertFalse(list(app.exception));self.assertEqual(len(plots),1)
        mean,alpha,limits,tac,labels=plots[0];data=load_demo()
        expected=forward_posterior(draws[0],3,data['times'],data['aif'])
        np.testing.assert_allclose(mean,expected['mean']);self.assertGreater(alpha,0);self.assertLess(alpha,1)
        low=min(0.,float(tac.min()));high=max(0.,float(tac.max()));pad=.08*max(high-low,1.)
        np.testing.assert_allclose(limits,[low-pad,high+pad])
        self.assertIn('95% credible band',labels);self.assertIn('Aorta input',labels)
        result=result_data(str(target),(target/'selection.npz').stat().st_mtime_ns)
        result['posteriors']._mmap.close();result_data.clear()


if __name__=='__main__':unittest.main()
