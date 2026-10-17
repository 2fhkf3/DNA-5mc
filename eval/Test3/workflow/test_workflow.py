#!/usr/bin/env python3
"""CPU integration tests in temporary directories; never run the real full evaluation."""
from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from common import *
import step1_audit_predictions as step1
import step2_build_common_pool as step2
import step3_sample_site_reads as step3
import step4_evaluate_coverage as step4
import step5_summarize as step5

sys.path.insert(0,str(PROJECT/'train'))
from model import build_model


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dna5mc-test3-'); self.base=Path(self.tmp.name)
        self.c=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.root=self.base/'protocol'
        self.c.update(enabled_versions=['V011','V554'],models={},output_dir=str(self.root),repeats=4,cpu_threads=1,batch_size=16,site_chunk_size=2)
        self.c['plots']['enabled']=False; self.c['bootstrap']['repeats']=30
        for version,seeds,drop in [('V011',[4,5],None),('V554',[1,2],0)]: self.make_source(version,seeds,drop)

    def tearDown(self): self.tmp.cleanup()

    def make_source(self,version,seeds,drop):
        source=self.base/'source'/version; source.mkdir(parents=True)
        rows=[]; xs=[]; site_y=[]
        widths=[20,21,22,19,23,20,24]; refs=['AACGC','TACGC','AACGC','CACGT','TCCGC','TACGC','TCCGC']
        for j,count in enumerate(widths):
            label=j%2; site_y.append(label)
            for k in range(count):
                if j==0 and k==drop: continue
                rows.append(dict(read_id=f'r{j:02}_{k:03}',chrom='chr16',center_genome_pos0=j*1000000+100,strand='+',source='mod',
                                 label=label,seq=refs[j],ref=refs[j],split='test',signal_len=128))
                x=np.zeros((5,4),dtype='float16'); sign=(1 if label else -1)*(-1 if k%7==0 else 1)
                x[2,0]=sign*(2+k/80) if k%11 else 0.; xs.append(x)
        x=np.asarray(xs,dtype='float16'); y=np.asarray([r['label'] for r in rows],dtype='float32'); n=len(y)
        x.tofile(source/'test.features.f16.bin'); y.tofile(source/'test.labels.f32.bin'); tsv(source/'metadata_test.tsv',rows,list(rows[0]))
        torch.save(dict(storage_format='streaming_memmap_v1',split='test',feature_shape=[n,5,4],records=n,feature_dtype='float16',
                        label_dtype='float32',feature_path='test.features.f16.bin',label_path='test.labels.f32.bin'),source/'test.pt')
        architecture='diffusion' if version=='V011' else 'bert'
        write_json(source/'feature_summary.json',dict(model_architecture=architecture,window_size=5,limit=None,preprocess='none'))
        root=self.base/'training'/version/'runs'
        self.c['models'][version]=dict(architecture=architecture,feature_dir=str(source),runs_root=str(root.parent))
        for strategy,selected_seeds in [('center_base_linear_raw768',seeds),('center_5mer_linear_raw768',[seeds[0]])]:
            for seed in selected_seeds:
                run=root/f'{strategy}_seed{seed}'; run.mkdir(parents=True)
                model=build_model(strategy,4,5,{})
                with torch.no_grad():
                    for param in model.parameters(): param.zero_()
                    model.base_norm.weight.fill_(1.)
                    model.head.weight[0,0 if strategy.startswith('center_base') else 8]=1.
                    model.head.bias.fill_((seed%3-1)*.1)
                model.eval()
                with torch.inference_mode(): probs=torch.sigmoid(model(torch.tensor(x.astype('float32')))).numpy().astype(float)
                cfg=dict(experiment=version+'-v2',strategy=strategy,seed=seed,fingerprint=version+str(seed)+strategy,feature_dir=str(source),
                    model={},training=dict(threshold=.5,deterministic=True),source_window_size=5,hidden_dim=4)
                cfg['input_signature']={'test':[signature(source/name) for name in ('test.pt','metadata_test.tsv','test.features.f16.bin','test.labels.f32.bin')]}
                write_json(run/'config.json',cfg)
                torch.save(dict(config=cfg,model=model.state_dict(),epoch=1),run/'best.pt')
                predictions=[dict(row,prob=float(p)) for row,p in zip(rows,probs)]; tsv(run/'test_predictions.tsv',predictions,list(predictions[0]))
                read_counts=dict(zip(('tp','tn','fp','fn'),map(int,confusion(y,probs,.5))))
                groups={j:[float(p) for r,p in zip(rows,probs) if int(r['center_genome_pos0'])==j*1000000+100] for j in range(7)}
                site_counts=dict(zip(('tp','tn','fp','fn'),map(int,confusion(np.asarray(site_y),np.asarray([np.mean(groups[j]) for j in range(7)]),.5))))
                write_json(run/'metrics.json',dict(strategy=strategy,seed=seed,fingerprint=cfg['fingerprint'],best_epoch=1,
                                                   test_read_level=read_counts,test_site_level=site_counts))

    def through(self,number=5,maximum=None):
        step1.run(self.c,self.root,maximum)
        if number>=2: step2.run(self.c,self.root)
        if number>=3: step3.run(self.c,self.root)
        if number>=4: step4.run(self.c,self.root)
        return step5.run(self.c,self.root) if number>=5 else None

    def test_complete_pipeline_and_unequal_training_seeds(self):
        output=self.through()
        m=read_json(self.root/'common/manifest.json')
        self.assertEqual((m['n_sites'],m['n_label0'],m['n_label1']),(5,3,2))
        self.assertEqual(len(output['model_summary']),2*2*3*len(METRICS))
        rows=[r for r in output['model_differences'] if r['strategy']=='center_base_linear_raw768']
        self.assertTrue(all(r['matching_training_seeds']=='' for r in rows))
        self.assertTrue(all(r['n_sites']==5 for r in output['model_summary']))
        self.assertTrue((self.root/'summary/bootstrap_model_draws.npz').is_file())

    def test_common_coverage_and_nested_shared_sampling(self):
        self.through(3)
        offsets=np.load(self.root/'common/offsets.npy'); selected=np.load(self.root/'sampling/selected_indices.npy')
        self.assertEqual(selected.shape,(4,5,20))
        for r in range(4):
            for j in range(5):
                ids=selected[r,j]
                self.assertEqual(len(set(ids.tolist())),20)
                self.assertTrue(np.all((ids>=offsets[j])&(ids<offsets[j+1])))
                self.assertTrue(set(ids[:5]).issubset(ids[:10])); self.assertTrue(set(ids[:10]).issubset(ids))
        sizes=np.diff(offsets); site=int(np.where(sizes==20)[0][0])
        for r in range(1,4): self.assertEqual(set(selected[0,site]),set(selected[r,site]))
        excluded=list(read_tsv(self.root/'common/excluded_sites.tsv'))
        self.assertEqual(len(excluded),2)

    def test_equal_balance_is_fixed_once(self):
        self.c['site_balance']='equal'; self.through(4)
        m=read_json(self.root/'common/manifest.json')
        self.assertEqual((m['n_sites'],m['n_label0'],m['n_label1']),(4,2,2))
        for p in (self.root/'results').glob('*/*/metrics_by_repeat.tsv'):
            self.assertTrue(all((r['n_label0'],r['n_label1'])==('2','2') for r in read_tsv(p)))

    def test_inference_and_saved_probabilities_agree(self):
        self.c['prediction_source']='inference'; self.through(1)
        manifest=read_json(self.root/'manifests/audit_manifest.json')
        for item in manifest['runs']:
            old=np.asarray([float(r['prob']) for r in read_tsv(Path(item['run_dir'])/'test_predictions.tsv')])
            actual=np.load(self.root/'audit/runs'/item['version']/item['run_name']/'probabilities.npy')
            np.testing.assert_allclose(old,actual,atol=1e-7,rtol=0)

    def test_sampling_independent_of_input_order(self):
        rows=[dict(obs_index=i,library_id='l',read_id=f'r{i}',chrom='chr16',position=100,strand='+') for i in range(30)]
        a=step3.select_indices(rows,42,0,20); b=step3.select_indices(list(reversed(rows)),42,0,20)
        self.assertEqual(a,b)
        self.assertNotEqual(a,step3.select_indices(rows,42,1,20))

    def test_metrics_match_sklearn_including_ties(self):
        from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,matthews_corrcoef
        y=np.asarray([0,1,0,1,1,0]); p=np.asarray([0,.5,.5,1,.3,.8])
        m=binary_metrics(y,p,.5)
        self.assertAlmostEqual(m['auroc'],roc_auc_score(y,p))
        self.assertAlmostEqual(m['average_precision'],average_precision_score(y,p))
        self.assertAlmostEqual(m['balanced_accuracy'],balanced_accuracy_score(y,p>=.5))
        self.assertAlmostEqual(m['mcc'],matthews_corrcoef(y,p>=.5))
        self.assertEqual(m['tp'],2)
        self.assertIsNone(binary_metrics(np.ones(3),np.asarray([0,.5,1]),.5)['auroc'])

    def test_frozen_config_cannot_change(self):
        self.through(1); self.c['sampling_seed']+=1
        with self.assertRaisesRegex(ValueError,'Configuration changed'): step2.run(self.c,self.root)

    def test_block_metrics_use_pooled_confusion_not_mean_block_rates(self):
        counts=np.asarray([[9,1,99,1],[1,99,1,99]])
        pooled=count_metrics(counts.sum(axis=0))['balanced_accuracy']
        self.assertAlmostEqual(float(pooled),((10/110)+(100/200))/2)
        self.assertNotAlmostEqual(float(pooled),float(count_metrics(counts)['balanced_accuracy'].mean()))
        self.assertEqual(interval(np.zeros(30))['ci_status'],'degenerate')
        self.assertEqual(interval([])['ci_status'],'insufficient_or_disabled')

    def test_truncated_feature_cache_rejected(self):
        p=Path(self.c['models']['V011']['feature_dir'])/'test.features.f16.bin'
        p.write_bytes(p.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError,'Truncated'): step1.run(self.c,self.root)
        self.assertFalse(self.root.exists())

    def test_partial_rejected_by_formal_downstream(self):
        self.through(1,maximum=64)
        with self.assertRaisesRegex(ValueError,'Partial'): step2.run(self.c,self.root)

    def test_duplicate_observation_rejected_atomically(self):
        source=Path(self.c['models']['V011']['feature_dir'])/'metadata_test.tsv'
        rows=list(read_tsv(source)); rows[1]=dict(rows[0]); tsv(source,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'): step1.run(self.c,self.root)
        self.assertFalse(self.root.exists())

    def test_prediction_mismatch_rejected(self):
        file=next(Path(self.c['models']['V011']['runs_root']).glob('runs/*/test_predictions.tsv'))
        file.write_text(file.read_text().replace('r00_000','wrong_id',1))
        with self.assertRaisesRegex(ValueError,'mismatch'): step1.run(self.c,self.root)
        self.assertFalse(self.root.exists())

    def test_nonfinite_probability_rejected(self):
        file=next(Path(self.c['models']['V011']['runs_root']).glob('runs/*/test_predictions.tsv'))
        rows=list(read_tsv(file)); rows[0]['prob']='nan'; tsv(file,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'probability'): step1.run(self.c,self.root)

    def test_missing_and_extra_prediction_rows_rejected(self):
        file=next(Path(self.c['models']['V011']['runs_root']).glob('runs/*/test_predictions.tsv'))
        rows=list(read_tsv(file)); tsv(file,rows[:-1],list(rows[0]))
        with self.assertRaisesRegex(ValueError,'Missing prediction'): step1.run(self.c,self.root)
        tsv(file,[*rows,rows[-1]],list(rows[0]))
        with self.assertRaisesRegex(ValueError,'Extra prediction'): step1.run(self.c,self.root)

    def test_corrupted_stage_file_rejected(self):
        self.through(2)
        p=self.root/'common/labels.npy'; p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'Stage file changed'): step3.run(self.c,self.root)

    def test_missing_run_and_overwrite_rejected(self):
        self.through(4)
        with self.assertRaises(FileExistsError): step4.run(self.c,self.root)
        p=next((self.root/'results').glob('*/*/metrics_by_repeat.tsv')); p.unlink()
        with self.assertRaises(FileNotFoundError): step5.run(self.c,self.root)

    def test_shell_entrypoints_dry_run(self):
        path=self.base/'config.yaml'; path.write_text(yaml.safe_dump(self.c))
        env=dict(os.environ,CONFIG=str(path),PYTHON_BIN=sys.executable)
        p=subprocess.run(['bash',str(Path(__file__).with_name('run_all.sh')),'--dry-run'],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr); self.assertIn('n_runs',p.stdout)
        self.assertFalse(self.root.exists())

    def test_optional_plot_exports(self):
        self.c['plots']['enabled']=True; self.through()
        files=list((self.root/'summary/figures').glob('*'))
        self.assertEqual(len(files),8)
        for p in files:
            self.assertGreater(p.stat().st_size,1000)
            self.assertTrue(p.read_bytes().startswith(b'%PDF') if p.suffix=='.pdf' else p.read_bytes().startswith(b'\x89PNG'))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
