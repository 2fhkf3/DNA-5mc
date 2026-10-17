#!/usr/bin/env python3
"""Temporary CPU integration tests; never write to configured production outputs."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from common import (PROJECT, export_results, insert, observation, prediction_db, read_json,
                    signature, tsv, write_json)
from step1_select_native_positives import run as select
from step2_evaluate import evaluate
from step3_compare import compare

sys.path.insert(0,str(PROJECT/'train'))
from model import build_model


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dna5mc-test2-')
        self.root = Path(self.tmp.name)
        self.c = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.c.update(enabled_versions=['V011','V554'],data_root=str(self.root/'positive'),
                      results_root=str(self.root/'results'),comparison_dir=str(self.root/'comparison'))
        self.c['models']={}
        self.c['statistics']['bootstrap']=30
        self.c['evaluation'].update(cpu_threads=1,batch_size=3)
        self.runs={}
        for v,seed,drop in [('V011',4,None),('V554',1,1)]:
            self.make_source(v,seed,drop)

    def tearDown(self):
        self.tmp.cleanup()

    def make_source(self,v,seed,drop):
        root=self.root/'source'/v
        root.mkdir(parents=True,exist_ok=True)
        # Five reads at site 100, two at 1 Mb, one at 2 Mb, plus two negatives.
        positions=[100]*5+[1000100]*2+[2000100]+[3000100]*2
        rows=[dict(read_id=f'read{i}',source='mod',label=int(i<8),chrom='chr16',center_genome_pos0=pos,
                   strand='+',seq='AACGC',ref='AACGC',split='test',signal_len=128,crop_input_window_size=15)
              for i,pos in enumerate(positions) if i != drop]
        n=len(rows)
        x=np.zeros((n,5,4),dtype='float16')
        x[:,2,0]=np.asarray([-3.,0.,3.,-1.,2.,-2.,2.,-2.,0.,0.])[np.asarray([i for i in range(10) if i != drop])]
        x.tofile(root/'test.features.f16.bin')
        np.asarray([r['label'] for r in rows],dtype='float32').tofile(root/'test.labels.f32.bin')
        tsv(root/'metadata_test.tsv',rows,list(rows[0]))
        m=dict(storage_format='streaming_memmap_v1',split='test',feature_shape=[n,5,4],records=n,
               feature_dtype='float16',label_dtype='float32',feature_path='test.features.f16.bin',label_path='test.labels.f32.bin')
        torch.save(m,root/'test.pt')
        arch='diffusion' if v=='V011' else 'bert'
        summary=dict(model_architecture=arch,window_size=5,feature_source='unit-test-center5',preprocess='none',limit=None)
        write_json(root/'feature_summary.json',summary)
        runs=self.root/'training'/v/'runs'
        strategy='center_base_linear_raw768'
        run=runs/f'{strategy}_seed{seed}'
        run.mkdir(parents=True)
        cfg=dict(experiment=v+'-v2',strategy=strategy,seed=seed,fingerprint=v+'-fixture',feature_dir=str(root),
                 source_window_size=5,hidden_dim=4,model={},training=dict(threshold=.5))
        cfg['input_signature']={'test':[signature(root/k) for k in ['test.pt','metadata_test.tsv','test.features.f16.bin','test.labels.f32.bin']]}
        model=build_model(strategy,4,5,{})
        with torch.no_grad():
            for param in model.parameters(): param.zero_()
            model.base_norm.weight.fill_(1.)
            model.head.weight[0,0]=1.
        model.eval()
        with torch.inference_mode():
            probabilities=torch.sigmoid(model(torch.from_numpy(x.astype('float32')))).numpy()
        preds=[dict(r,prob=float(p)) for r,p in zip(rows,probabilities)]
        tsv(run/'test_predictions.tsv',preds,list(preds[0]))
        pos=[float(p) for r,p in zip(rows,probabilities) if r['label']==1]
        fn=sum(p<.5 for p in pos)
        write_json(run/'metrics.json',dict(strategy=strategy,seed=seed,fingerprint=cfg['fingerprint'],best_epoch=1,
                                         test_read_level=dict(tp=len(pos)-fn,fn=fn)))
        write_json(run/'config.json',cfg)
        torch.save(dict(config=cfg,model=model.state_dict(),epoch=1),run/'best.pt')
        self.c['models'][v]=dict(architecture=arch,feature_dir=str(root),runs_root=str(runs.parent))
        self.runs[v]=run

    def selection_args(self,v='V011',maximum=None):
        return SimpleNamespace(input_dir=Path(self.c['models'][v]['feature_dir']),out_dir=Path(self.c['data_root'])/v,
            version=v,library_id=self.c['library_id'],chromosome='chr16',reference_validation_status='unverified',
            chunk_size=3,log_every_chunks=10,max_records=maximum,dry_run=False)

    def eval(self,v='V011',source='predictions',suffix='',maximum=None):
        output=Path(self.c['results_root'])/v/(self.runs[v].name+suffix)
        return evaluate(self.c,v,self.runs[v],output,source,maximum,device='cpu')

    def test_positive_selection_preserves_bytes_and_mapping(self):
        args=self.selection_args()
        result=select(args)
        self.assertEqual(result['records'],8)
        self.assertEqual(result['label_counts'],{'0':0,'1':8})
        self.assertEqual(result['source_label_counts'],{'0':2,'1':8})
        a=(args.input_dir/'test.features.f16.bin').read_bytes()
        b=(args.out_dir/'test.features.f16.bin').read_bytes()
        self.assertEqual(b,a[:8*5*4*2])
        self.assertTrue(np.all(np.fromfile(args.out_dir/'test.labels.f32.bin',dtype='float32')==1))
        with (args.out_dir/'metadata_test.tsv').open() as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
        self.assertEqual([int(r['source_sample_index']) for r in rows],list(range(8)))
        self.assertTrue(all(r['source']=='mod' for r in rows))
        self.assertFalse(result['partial_test'])
        with self.assertRaises(FileExistsError): select(args)

    def test_saved_and_cpu_inference_agree(self):
        select(self.selection_args())
        raw=self.eval()
        inferred=self.eval(source='features',suffix='-inferred')
        self.assertEqual(raw['read_level'],inferred['read_level'])
        self.assertEqual(raw['site_level'],inferred['site_level'])
        self.assertEqual(raw['observation_sha256'],inferred['observation_sha256'])
        self.assertEqual(raw['read_level']['n'],8)
        self.assertEqual(raw['read_level']['fn'],4)
        self.assertEqual(raw['read_level']['recall'],.5)
        self.assertEqual(raw['coverage_slices']['5']['read_level']['n'],5)
        self.assertTrue(raw['original_positive_counts_verified'])

    def test_unequal_seeds_common_then_coverage(self):
        self.eval('V011'); self.eval('V554')
        data=compare(self.c,['V011','V554'],Path(self.c['comparison_dir']))
        self.assertEqual(len(data['aggregate']),4)
        for r in data['common']:
            self.assertEqual(r['read_n'],7 if r['min_coverage']==1 else 0)
            self.assertEqual(r['site_n'],3 if r['min_coverage']==1 else 0)
        self.assertEqual(data['paired'][0]['status'],'no_matching_training_seed')
        self.assertEqual({r['seeds'] for r in data['aggregate']},{'4','1'})

    def test_paired_same_seed_and_zero_event_interval(self):
        self.make_source('V554',4,None)
        self.eval('V011'); self.eval('V554')
        # Remove the previous seed1 fixture so the frozen experiment has one selected run.
        self.c['evaluation']['seeds']=[4]
        data=compare(self.c,['V011','V554'],Path(self.c['comparison_dir']))
        self.assertEqual(len(data['paired']),4)
        for row in data['paired']:
            self.assertEqual(row['difference_fnr_b_minus_a'],0.)
            self.assertEqual(row['block_bootstrap_ci95'],'null')

    def test_partial_requires_explicit_comparison_opt_in(self):
        result=self.eval(maximum=4)
        self.assertTrue(result['partial_test'])
        self.c['enabled_versions']=['V011']
        with self.assertRaisesRegex(ValueError,'Partial'):
            compare(self.c,['V011'],Path(self.c['comparison_dir']))

    def test_misaligned_prediction_rejected_and_stage_cleaned(self):
        path=self.runs['V011']/'test_predictions.tsv'
        path.write_text(path.read_text().replace('read0','wrong0',1))
        with self.assertRaisesRegex(ValueError,'mismatch'): self.eval()
        self.assertFalse((Path(self.c['results_root'])/'V011'/self.runs['V011'].name).exists())

    def test_missing_prediction_row_rejected(self):
        path=self.runs['V011']/'test_predictions.tsv'
        lines=path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'Missing'): self.eval()

    def test_bad_probability_even_on_negative_row_rejected(self):
        path=self.runs['V011']/'test_predictions.tsv'
        with path.open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
        rows[-1]['prob']='nan'
        tsv(path,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'probability'): self.eval()

    def test_truncated_arrays_rejected(self):
        path=Path(self.c['models']['V011']['feature_dir'])/'test.features.f16.bin'
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError,'Truncated'): self.eval()

    def test_positive_inference_rejects_negative_label(self):
        args=self.selection_args()
        select(args)
        p=args.out_dir/'test.labels.f32.bin'
        labels=np.fromfile(p,dtype='float32'); labels[0]=0; labels.tofile(p)
        with self.assertRaisesRegex(ValueError,'positive labels'):
            self.eval(source='features')

    def test_protocol_and_operating_threshold_mismatch_rejected(self):
        args=self.selection_args()
        select(args)
        p=args.out_dir/'feature_summary.json'
        summary=read_json(p); summary['preprocess']='zscore'; write_json(p,summary)
        with self.assertRaisesRegex(ValueError,'protocol mismatch'):
            self.eval(source='features')
        self.eval('V011')
        self.c['evaluation']['read_threshold']=.6
        self.eval('V554')
        with self.assertRaisesRegex(ValueError,'unequal operating thresholds'):
            compare(self.c,['V011','V554'],Path(self.c['comparison_dir']))

    def test_reference_context_conflict_rejected(self):
        db=prediction_db(self.root/'conflict.sqlite')
        try:
            for i,ref in enumerate(['AACGC','TACGC']):
                row=dict(read_id=str(i),source='mod',label=1,chrom='chr16',center_genome_pos0=100,strand='+',seq='AACGC',ref=ref)
                insert(db,[observation(row,.5,i,self.c['library_id'])])
            with self.assertRaisesRegex(ValueError,'Conflicting reference'):
                export_results(db,self.root,self.c,.5,.5)
        finally:
            db.close()

    def test_duplicate_physical_observation_rejected(self):
        db=prediction_db(self.root/'duplicate.sqlite')
        try:
            row=dict(read_id='r',source='mod',label=1,chrom='chr16',center_genome_pos0=10,strand='+',seq='AACGC',ref='AACGC')
            obs=observation(row,.5,0,self.c['library_id'])
            insert(db,[obs])
            with self.assertRaisesRegex(ValueError,'Duplicate'): insert(db,[obs])
        finally: db.close()

    def test_threshold_equality_and_positive_losses(self):
        db=prediction_db(self.root/'edge.sqlite')
        try:
            for i,p in enumerate([0.,.5,1.]):
                row=dict(read_id=str(i),source='mod',label=1,chrom='chr16',center_genome_pos0=i*1000000,strand='+',seq='AACGC',ref='AACGC')
                insert(db,[observation(row,p,i,self.c['library_id'])])
            stage=self.root/'stats'; stage.mkdir()
            m=export_results(db,stage,self.c,.5,.5)
            self.assertEqual(m['read_level']['fn'],1)
            self.assertEqual(m['site_level']['tp'],2)
            self.assertAlmostEqual(m['read_level']['brier'],1.25/3)
            self.assertTrue(np.isfinite(m['read_level']['bce']))
            self.assertIsNone(m['coverage_slices']['5']['read_level']['fnr'])
            with (stage/'threshold_metrics.tsv').open() as f:
                rows=[r for r in csv.DictReader(f,delimiter='\t') if r['level']=='read' and r['min_coverage']=='1']
            self.assertEqual([int(r['fn']) for r in rows],sorted(int(r['fn']) for r in rows))
        finally: db.close()

    def test_scripts_and_configuration_cli(self):
        config=self.root/'config.yaml'
        config.write_text(yaml.safe_dump(self.c))
        for name in ['step1_select_native_positives','step2_evaluate','step3_compare']:
            result=subprocess.run(['bash',str(Path(__file__).with_name(name+'.sh')),'--config',str(config),'--dry-run'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(result.stdout)
        self.assertFalse(Path(self.c['results_root']).exists())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
