"""Numerical and IO invariants; fixtures are not photographic samples."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import io
import tempfile
import unittest

import cv2
import numpy as np
import tifffile
from PIL import Image

import native_fusion as native
import _native_fusion_core as core
from _native_image_io import save_rgb_png


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

    def test_native_16bit_tiff_to_png_preserves_values_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y, x = np.mgrid[0:32, 0:48]
            pixels = np.stack((1000+31*x+501*y, 2000+529*x+37*y, 2500+89*x+271*y), axis=-1)
            pixels = pixels.astype(np.uint16)
            source = root / 'source.tiff'
            tifffile.imwrite(source, pixels, photometric='rgb',
                             extratags=[(274, 'H', 1, 6, False), (34675, 'B', len(core.SRGB_ICC), core.SRGB_ICC, False)])
            oriented = np.rot90(pixels, k=3)
            loaded, info = core.load_source(source)
            np.testing.assert_array_equal(loaded, oriented)
            self.assertEqual(info['bit_depth'], 16)
            reference = root / 'reference.jpg'
            reference_codes = core.linear_to_srgb(core.srgb_to_linear(oriented) * 1.5)
            Image.fromarray(reference_codes).resize((16,24)).save(reference, quality=95)
            script = str(Path(native.__file__).resolve())
            process = subprocess.run(
                [sys.executable, script, 'run', '--source', str(source), '--reference', str(reference),
                 '--out', str(root/'native')], check=True, capture_output=True,
            )
            delivery = json.loads(process.stdout.splitlines()[-1])
            self.assertEqual(delivery['format'], 'PNG')
            self.assertEqual(delivery['size'], [32,48])
            self.assertEqual(delivery['bit_depth'], 16)
            final = Path(delivery['final'])
            self.assertEqual(delivery['sha256'], hashlib.sha256(final.read_bytes()).hexdigest())
            final_codes = cv2.imdecode(np.frombuffer(final.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
            self.assertEqual(final_codes.dtype, np.dtype('uint16'))
            self.assertFalse(np.array_equal(final_codes, oriented))
            self.assertGreater(np.unique(final_codes[...,0]).size, 256)
            with Image.open(final) as exported:
                self.assertTrue(exported.info['icc_profile'])
            with np.load(final.parent/'delta_linear.npz') as delta:
                recovered = core.linear_to_srgb(core.srgb_to_linear(oriented) + delta['delta_linear'], np.uint16)
                np.testing.assert_array_equal(recovered, final_codes)

            for strength, expected in ((0.0, oriented), (1.0, final_codes)):
                args = argparse.Namespace(source=str(source), model=delivery['model'],
                                          out=str(root/f'replay{strength}'), strength=strength)
                with contextlib.redirect_stdout(io.StringIO()):
                    replay = native.apply(args)
                replay_codes, _ = core.load_source(replay['methods'][0]['final'])
                np.testing.assert_array_equal(replay_codes, expected)

            with np.load(delivery['model'], allow_pickle=False) as model:
                legacy = {key:model[key] for key in model.files if key != 'source_bit_depth'}
            legacy_path = root/'legacy.npz'
            np.savez(legacy_path, **legacy)
            args = argparse.Namespace(source=str(source), model=str(legacy_path),
                                      out=str(root/'legacy-replay'), strength=1.0)
            with self.assertRaisesRegex(ValueError, 'different source bit depth'):
                native.apply(args)

    def test_native_16bit_png_zero_strength_preserves_low_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ramp = (np.arange(32*48, dtype=np.uint16).reshape(32,48)*31)
            pixels = np.stack((ramp, ramp+3, ramp+7), axis=-1)
            source = root/'source.png'
            ok, raw = cv2.imencode('.png', pixels[..., ::-1])
            self.assertTrue(ok)
            source.write_bytes(raw.tobytes())
            reference = root/'reference.png'
            save_rgb_png(reference, pixels//2, core.SRGB_ICC)
            args = argparse.Namespace(source=str(source), reference=str(reference), out=str(root/'result'),
                                      methods=['F06'], strength=0.0, max_aspect_error=.01)
            with contextlib.redirect_stdout(io.StringIO()):
                result = native.run(args)['methods'][0]
            final_codes = cv2.imdecode(np.frombuffer(Path(result['final']).read_bytes(), np.uint8),
                                       cv2.IMREAD_UNCHANGED)[..., ::-1]
            np.testing.assert_array_equal(final_codes, pixels)
            self.assertEqual(result['bit_depth'], 16)
            self.assertEqual(result['size'], [48,32])

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
