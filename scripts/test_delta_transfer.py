"""Local behavioral checks; synthetic fixtures are not aesthetic evidence."""
from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

import delta_transfer as dt
import relight_layers as common


class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        rng=np.random.default_rng(17)
        self.codes=rng.integers(30,220,(72,96,3),dtype=np.uint8)
        self.source=self.root/'source.png';Image.fromarray(self.codes).save(self.source)
        self.relit=self.root/'relit.png';Image.fromarray(self.codes).save(self.relit)
        self.depth=self.root/'depth.png';Image.fromarray(np.tile(np.linspace(0,255,96).astype(np.uint8),(72,1))).save(self.depth)
        colors=np.zeros((72,96,3),np.uint8);colors[:,:48]=[242,170,133];colors[:,48:]=[128,128,128]
        self.materials=self.root/'materials.png';Image.fromarray(colors).save(self.materials)
        self.coverage=self.root/'coverage.png';Image.fromarray(np.pad(np.full((72,48),255,np.uint8),((0,0),(0,48)))).save(self.coverage)

    def tearDown(self):self.temp.cleanup()

    def pair(self,**kwargs):
        return dt.prepare_pair(self.source,self.relit,depth=self.depth,materials=self.materials,coverage=self.coverage,registration='identity',**kwargs)

    def test_identical_images_have_zero_fields_for_every_method(self):
        pair=self.pair()
        for method in dt.METHODS:
            field=dt.filter_method(pair,method)
            for value in field.values():
                self.assertLess(float(np.abs(value).max()),2e-6,method)

    def test_raw_linear_delta_reconstructs_reference_at_matching_grid(self):
        changed=np.clip(self.codes.astype(int)+np.array([12,-14,7]),0,255).astype(np.uint8)
        Image.fromarray(changed).save(self.relit)
        pair=self.pair();field=dt.filter_method(pair,'D01')
        report=dt.compose_field(pair.native,field,self.root/'reconstructed')
        np.testing.assert_array_equal(np.array(Image.open(self.root/'reconstructed/final.png')),changed)
        self.assertGreater(report['negative_delta_component_fraction'],0)

    def test_strength_zero_is_exact_for_delta_and_log_gain(self):
        fields=[{'delta_linear':np.full((18,24,3),-.12,np.float32)},
                {'log_gain':np.full((18,24),1.5,np.float32),'chroma_delta':np.full((18,24,3),-.04,np.float32)}]
        for i,field in enumerate(fields):
            out=self.root/f'zero{i}';dt.compose_field(self.codes,field,out,strength=0)
            np.testing.assert_array_equal(np.array(Image.open(out/'final.png')),self.codes)

    def test_negative_difference_is_not_discarded(self):
        field={'delta_linear':np.full((18,24,3),-.02,np.float32)}
        report=dt.compose_field(self.codes,field,self.root/'negative')
        result=np.array(Image.open(self.root/'negative/final.png'))
        self.assertTrue(np.all(result<=self.codes));self.assertTrue(np.any(result<self.codes))
        self.assertEqual(report['negative_delta_component_fraction'],1)

    def test_native_grid_and_overlay_reconstruct_signed_result(self):
        native=np.full((90,4300,3),100,np.uint8)
        field={'delta_linear':np.zeros((9,430,3),np.float32)}
        field['delta_linear'][:,:200]=[.08,-.03,.02]
        report=dt.compose_field(native,field,self.root/'wide')
        result=np.array(Image.open(self.root/'wide/final.png'))
        layer=np.array(Image.open(self.root/'wide/effect_overlay.png'))
        self.assertEqual(result.shape,native.shape)
        alpha=layer[:,:,3:4].astype(np.float64)/255
        reconstructed=np.rint(layer[:,:,:3]*alpha+native*(1-alpha)).astype(np.uint8)
        self.assertLessEqual(int(np.abs(reconstructed.astype(int)-result).max()),1)
        self.assertLessEqual(report['overlay']['reconstruction_max_code_error'],1)

    def test_constant_fields_survive_smoothing_families(self):
        pair=self.pair();pair.delta[:]=[.03,-.05,.11]
        for method in ['D01','D02','D03','D04','D05','D06','D07','D08','D09','D11']:
            delta=dt.filter_method(pair,method)['delta_linear']
            np.testing.assert_allclose(delta,np.broadcast_to([.03,-.05,.11],delta.shape),atol=1e-5,err_msg=method)

    def test_guidance_preserves_a_corresponding_depth_boundary(self):
        guide=np.zeros((64,96),np.float32);guide[:,48:]=1
        field=np.zeros((64,96,3),np.float32);field[:,48:]=.2
        edge=dt.guided(field,guide,10,1e-5)
        smooth=dt.gaussian(field,5)
        self.assertLess(abs(float(edge[32,47,0])),abs(float(smooth[32,47,0])))
        self.assertGreater(float(edge[32,48,0]),float(smooth[32,48,0]))

    def test_semantic_normalization_keeps_region_constants_separate(self):
        labels=np.zeros((64,96),np.uint8);labels[:,48:]=1
        field=np.full((64,96,3),-.08,np.float32);field[:,48:]=.15
        result=dt.normalized_semantic(field,labels,8)
        np.testing.assert_allclose(result[:,:48],-.08,atol=2e-6)
        np.testing.assert_allclose(result[:,48:],.15,atol=2e-6)

    def test_d12_soft_coverage_is_only_one_experimental_branch(self):
        pair=self.pair();pair.delta[:]=.08
        raw=dt.filter_method(pair,'D01')['delta_linear']
        gated=dt.filter_method(pair,'D12')['delta_linear']
        np.testing.assert_allclose(raw,.08)
        self.assertTrue(np.all(gated<=raw))
        self.assertGreater(float(gated[36,55,0]),0)  # feather extends past hard midpoint
        self.assertGreater(float(gated[36,20,0]),float(gated[36,85,0]))

    def test_registration_only_moves_reference_and_corrects_small_translation(self):
        rng=np.random.default_rng(5)
        gray=cv2.GaussianBlur(rng.integers(0,256,(256,320),dtype=np.uint8),(0,0),1.5)
        source=np.repeat(gray[:,:,None],3,axis=2)
        shifted=cv2.warpAffine(source,np.float32([[1,0,4],[0,1,-3]]),(320,256),borderMode=cv2.BORDER_REFLECT_101)
        before=source.copy();aligned,report=dt.align_reference(source,shifted,'ecc')
        np.testing.assert_array_equal(source,before)
        self.assertEqual(report['applied'],'ecc-euclidean')
        self.assertLessEqual(report['proposed_shift_pixels'],report['maximum_allowed_shift_pixels'])
        area=(slice(12,-12),slice(12,-12))
        self.assertLess(np.mean(np.abs(aligned[area].astype(float)-source[area])),np.mean(np.abs(shifted[area].astype(float)-source[area]))*.2)

    def test_flow_is_bounded_and_identical_images_skip_it(self):
        codes=common.srgb_to_linear(self.codes)
        same,report=dt.bounded_flow_reference(codes,codes)
        np.testing.assert_array_equal(same,codes);self.assertFalse(report['source_warped'])
        shifted=np.roll(codes,2,axis=1)
        _,report=dt.bounded_flow_reference(codes,shifted,max_displacement=3)
        self.assertLessEqual(report.get('actual_max_displacement_working_px',0),3.001)

    def test_ratio_mismatch_and_missing_guides_are_explicit(self):
        square=self.root/'square.png';Image.new('RGB',(64,64)).save(square)
        with self.assertRaisesRegex(ValueError,'aspect ratio'):dt.prepare_pair(self.source,square)
        with self.assertRaisesRegex(ValueError,'depth'):
            dt.run_lab(self.source,self.relit,self.root/'missing',methods=['D06'])
        self.assertFalse((self.root/'missing/manifest.json').exists())

    def test_exif_orientation_is_normalized_without_rewriting_source(self):
        source=self.root/'oriented.jpg';im=Image.new('RGB',(80,60),(80,100,120));exif=Image.Exif();exif[274]=6;im.save(source,exif=exif)
        ref=self.root/'turned.png';Image.new('RGB',(60,80),(90,110,130)).save(ref)
        before=source.read_bytes();pair=dt.prepare_pair(source,ref,registration='identity')
        self.assertEqual(pair.native.shape[:2],(80,60));self.assertEqual(source.read_bytes(),before)

    def test_cached_field_replays_and_binds_to_original(self):
        field={'delta_linear':np.full((18,24,3),[.03,-.03,.04],np.float32)}
        path=self.root/'field.npz'
        dt.save_working_field(field,path,common.file_hash(self.source),'reference-hash','D02')
        direct=self.root/'direct';reuse=self.root/'reuse'
        dt.compose_field(self.codes,field,direct)
        dt.apply_saved(self.source,path,reuse)
        np.testing.assert_array_equal(np.array(Image.open(direct/'final.png')),np.array(Image.open(reuse/'final.png')))
        other=self.root/'other.png';Image.new('RGB',(96,72),(100,100,100)).save(other)
        with self.assertRaisesRegex(ValueError,'different original'):dt.apply_saved(other,path,self.root/'wrong')

    def test_small_end_to_end_manifest_and_input_bytes(self):
        changed=np.clip(self.codes.astype(int)+[15,-10,5],0,255).astype(np.uint8);Image.fromarray(changed).save(self.relit)
        original_bytes=self.source.read_bytes();reference_bytes=self.relit.read_bytes()
        result=dt.run_lab(self.source,self.relit,self.root/'all',depth=self.depth,materials=self.materials,coverage=self.coverage,
                          registration='identity',methods=list(dt.METHODS),save_working=True)
        self.assertEqual(result['count'],14);self.assertEqual(result['network_calls_by_engine'],0)
        for entry in result['methods']:
            self.assertTrue((self.root/'all'/entry['final']).is_file());self.assertTrue((self.root/'all'/entry['field']).is_file())
            report=json.loads((self.root/'all'/entry['report']).read_text(encoding='utf-8'))
            self.assertEqual(report['aesthetic_evaluation'],'user only; no ranking or beauty rejection')
            self.assertEqual(report['native_size'],[96,72])
            self.assertTrue(report['source_unchanged'])
        self.assertEqual(self.source.read_bytes(),original_bytes);self.assertEqual(self.relit.read_bytes(),reference_bytes)

    def test_bad_strength_nonfinite_data_and_input_overwrite_rejected(self):
        field={'delta_linear':np.zeros((18,24,3),np.float32)}
        for strength in [-1,4,float('nan')]:
            with self.assertRaises(ValueError):dt.compose_field(self.codes,field,self.root/'bad',strength=strength)
        with self.assertRaises(ValueError):dt.compose_field(self.codes,{'delta_linear':np.full((18,24,3),np.nan)},self.root/'bad')
        folder=self.root/'collision';folder.mkdir()
        collision=folder/'original-sRGB.png';Image.fromarray(self.codes).save(collision)
        before=collision.read_bytes()
        with self.assertRaisesRegex(ValueError,'overwrite'):
            dt.run_lab(collision,self.relit,folder,methods=['D01'])
        self.assertEqual(collision.read_bytes(),before)

    def test_apply_protects_a_legal_source_named_preview_jpg(self):
        folder=self.root/'preview-collision';folder.mkdir()
        source=folder/'preview.jpg';Image.fromarray(self.codes).save(source,quality=97)
        cache=folder/'working.npz'
        dt.save_working_field({'delta_linear':np.zeros((18,24,3),np.float32)},cache,
                              common.file_hash(source),'reference','D01')
        source_before=source.read_bytes();cache_before=cache.read_bytes()
        with self.assertRaisesRegex(ValueError,'overwrite'):
            dt.apply_saved(source,cache,folder)
        self.assertEqual(source.read_bytes(),source_before)
        self.assertEqual(cache.read_bytes(),cache_before)
        self.assertEqual({p.name for p in folder.iterdir()},{'preview.jpg','working.npz'})

    def test_apply_protects_cache_named_report_json(self):
        folder=self.root/'cache-report-collision';folder.mkdir()
        cache=folder/'report.json'
        # NPZ is content-detected, so a valid cache can have this filename.
        with cache.open('wb') as handle:
            np.savez_compressed(handle,delta_linear=np.zeros((18,24,3),np.float32),
                                source_sha256=np.asarray(common.file_hash(self.source)))
        before=cache.read_bytes()
        with self.assertRaisesRegex(ValueError,'overwrite'):
            dt.apply_saved(self.source,cache,folder)
        self.assertEqual(cache.read_bytes(),before)
        self.assertEqual({p.name for p in folder.iterdir()},{'report.json'})

    def test_run_protects_reference_maps_and_delta_preview_inputs(self):
        cases=[('relit','_working/relit-unaligned-sRGB.png'),
               ('depth','_working/depth-working.png'),
               ('materials','_working/structure-confidence.png'),
               ('coverage','_working/soft-coverage.png'),
               ('source','D01/delta-signed.png'),
               ('relit','D01/delta-positive.png'),
               ('depth','D01/delta-negative.png')]
        for i,(role,relative) in enumerate(cases):
            folder=self.root/f'aux-collision-{i}';collision=folder/relative
            collision.parent.mkdir(parents=True)
            Image.fromarray(self.codes).save(collision,format='PNG')
            inputs={'source':self.source,'relit':self.relit,'depth':self.depth,
                    'materials':self.materials,'coverage':self.coverage}
            inputs[role]=collision;before=collision.read_bytes()
            with self.subTest(role=role,path=relative):
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    dt.run_lab(**inputs,out=folder,methods=['D01'],save_working=True)
                self.assertEqual(collision.read_bytes(),before)
                self.assertEqual([p for p in folder.rglob('*') if p.is_file()],[collision])


if __name__=='__main__':unittest.main(verbosity=2)
