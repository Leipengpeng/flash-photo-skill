"""Numerical and IO invariants; fixtures are not photographic samples."""
from pathlib import Path
import argparse
import contextlib
import io
import tempfile
import unittest

import numpy as np
from PIL import Image

import native_fusion as native
import _native_fusion_core as core


class NativeFusionTests(unittest.TestCase):
    def test_constant_illumination_is_preserved(self):
        y,x=np.mgrid[0:48,0:64]
        low=np.stack((.2+.08*np.sin(x*.3),.3+.08*np.cos(y*.3),.25+.08*np.sin((x+y)*.3)),axis=2).astype(np.float32)
        for gain in (0.,.5):
            light=np.full((24*32,3),gain,np.float32)
            fields,_=core.make_models(low,(low+.001)*np.float32(2**gain)-.001,light)
            selected=np.tile(np.arange(9),(100,1))
            for m in native.METHODS:
                actual=core.evaluate_gain(m,fields,selected,selected)
                self.assertLess(float(np.max(np.abs(actual-gain))),2e-5)

    def test_exact_area_footprint_reduction(self):
        codes=np.array([[[0,0,0],[255,255,255]],[[255,255,255],[0,0,0]]],np.uint8)
        actual=native.area_reduce_codes(codes,(1,1))
        np.testing.assert_allclose(actual,[[[.5,.5,.5]]],atol=1e-7)

    def test_native_identity_orientation_delta_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);y,x=np.mgrid[0:64,0:80]
            codes=np.stack(((x*3+y)%220+10,(y*3+x)%220+10,(x+y*2)%220+10),axis=2).astype(np.uint8)
            exif=Image.Exif();exif[274]=6
            source=root/'source.png';Image.fromarray(codes).save(source,exif=exif,icc_profile=core.SRGB_ICC)
            original,_=core.load_source(source);reference=root/'reference.png'
            Image.fromarray(original).save(reference,icc_profile=core.SRGB_ICC)
            args=argparse.Namespace(source=str(source),reference=str(reference),out=str(root/'result'),methods=['F06','F04'],strength=1.,max_aspect_error=.01)
            with contextlib.redirect_stdout(io.StringIO()):native.run(args)
            for method in native.METHODS:
                folder=root/'result'/method
                with Image.open(folder/'final.png') as im:np.testing.assert_array_equal(np.array(im),original)
                with np.load(folder/'delta_linear.npz') as z:np.testing.assert_array_equal(z['delta_linear'],np.zeros_like(z['delta_linear']))
                for strength in (1.,0.):
                    ap=argparse.Namespace(source=str(source),model=str(folder/'model.npz'),out=str(root/(method+str(strength))),strength=strength)
                    with contextlib.redirect_stdout(io.StringIO()):native.apply(ap)
                    with Image.open(Path(ap.out)/method/'final.png') as im:np.testing.assert_array_equal(np.array(im),original)
            bad=root/'other.png';Image.fromarray(codes).save(bad)
            ap=argparse.Namespace(source=str(bad),model=str(root/'result/F06/model.npz'),out=str(root/'bad'),strength=1.)
            with self.assertRaisesRegex(ValueError,'another source'):native.apply(ap)
            with self.assertRaisesRegex(ValueError,'already exists'):native.run(args)

    def test_protected_input_and_bad_aspect_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source.png';ref=root/'reference.png'
            Image.new('RGB',(64,48),(100,110,120)).save(src);Image.new('RGB',(48,64),(100,110,120)).save(ref)
            before=src.read_bytes()
            with self.assertRaisesRegex(ValueError,'protected input'):native.preflight([src],root,['F06'])
            args=argparse.Namespace(source=str(src),reference=str(ref),out=str(root/'bad-aspect'),methods=['F06'],strength=1.,max_aspect_error=.01)
            with self.assertRaisesRegex(ValueError,'aspect mismatch'):native.run(args)
            self.assertEqual(src.read_bytes(),before)


if __name__=='__main__':unittest.main()
