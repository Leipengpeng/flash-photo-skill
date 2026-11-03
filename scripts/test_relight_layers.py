"""Behavioral checks of the native-resolution lighting compositor (no AI calls)."""

import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

import relight_layers as relight


class CompositorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(17)
        self.original = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        self.source = self.root / "source.png"
        Image.fromarray(self.original).save(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def field(self, value=192, size=(32, 24), name="field.png"):
        path = self.root / name
        Image.new("RGB", size, (value, value, value)).save(path)
        return path

    def render(self, field=None, **kwargs):
        out = self.root / "render"
        report = relight.composite(self.source, field or self.field(), out, **kwargs)
        final = np.array(Image.open(out / "final.png"))
        return final, report, out

    def test_neutral_field_is_exact_and_transparent(self):
        final, report, out = self.render(self.field(128), save_arrays=True)
        np.testing.assert_array_equal(final, self.original)
        self.assertTrue(report["zero_EV_pixels_exact"])
        self.assertEqual(report["overlay"]["alpha_max"], 0)
        with np.load(out / "relight_arrays.npz") as arrays:
            self.assertTrue(np.all(arrays["ev"] == 0))
            self.assertTrue(np.all(arrays["delta_linear"] == 0))

    def test_zero_strength_is_exact_even_for_non_neutral_field(self):
        final, report, _ = self.render(strength=0)
        np.testing.assert_array_equal(final, self.original)
        self.assertEqual(report["applied_ev_max"], 0)

    def test_mask_zero_pixels_are_exact(self):
        mask = np.full((96, 128), 255, dtype=np.uint8)
        mask[:, :64] = 0
        mask_path = self.root / "mask.png"
        Image.fromarray(mask).save(mask_path)
        final, report, _ = self.render(mask_path=mask_path)
        np.testing.assert_array_equal(final[:, :64], self.original[:, :64])
        self.assertTrue(np.any(final[:, 64:] > self.original[:, 64:]))
        self.assertTrue(report["mask_zero_pixels_exact"])

    def test_default_add_only_never_reduces_ambient_channels(self):
        field = np.full((24, 32, 3), [230, 80, 145], dtype=np.uint8)
        path = self.root / "colored.png"
        Image.fromarray(field).save(path)
        final, report, _ = self.render(path)
        self.assertTrue(np.all(final >= self.original))
        np.testing.assert_array_equal(final[:, :, 1], self.original[:, :, 1])
        self.assertTrue(report["add_only_preserves_each_original_channel"])

    def test_signed_mode_can_reduce_and_add_channels(self):
        field = np.full((24, 32, 3), [192, 64, 128], dtype=np.uint8)
        path = self.root / "signed.png"
        Image.fromarray(field).save(path)
        final, report, _ = self.render(path, mode="signed")
        self.assertTrue(np.any(final[:, :, 0] > self.original[:, :, 0]))
        self.assertTrue(np.any(final[:, :, 1] < self.original[:, :, 1]))
        np.testing.assert_array_equal(final[:, :, 2], self.original[:, :, 2])
        self.assertIsNone(report["add_only_preserves_each_original_channel"])

    def test_ratio_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            self.render(self.field(size=(30, 30)))
        mask = self.root / "mask.png"
        Image.new("L", (24, 24), 255).save(mask)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            self.render(mask_path=mask)

    def test_exif_orientation_and_prepare_agree(self):
        source = self.root / "oriented.jpg"
        image = Image.fromarray(self.original)
        exif = Image.Exif()
        exif[274] = 6
        exif[34853] = {}  # Input metadata must not be blindly copied to result.
        image.save(source, exif=exif)
        self.source = source
        field = self.field(128, size=(24, 32))
        final, report, out = self.render(field)
        expected = np.array(ImageOps.exif_transpose(Image.open(source)).convert("RGB"))
        np.testing.assert_array_equal(final, expected)
        self.assertEqual(report["source"]["stored_size"], [128, 96])
        self.assertEqual(report["native_size"], [96, 128])
        self.assertNotIn(34853, Image.open(out / "final.png").getexif())
        prepared = self.root / "prepared"
        relight.prepare(source, prepared, 256)
        np.testing.assert_array_equal(np.array(Image.open(prepared / "prepared.png")), expected)

    def test_icc_decode_and_identity_use_the_same_srgb_base(self):
        image = Image.fromarray(self.original)
        source = self.root / "profiled.png"
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        image.save(source, icc_profile=profile.tobytes())
        self.source = source
        final, report, out = self.render(self.field(128))
        expected = np.array(ImageCms.profileToProfile(Image.open(source), profile, relight.SRGB_PROFILE, outputMode="RGB"))
        np.testing.assert_array_equal(final, expected)
        self.assertTrue(report["source"]["embedded_icc"])
        with Image.open(out / "final.png") as saved:
            self.assertIn("icc_profile", saved.info)

    def test_overlay_reconstructs_final_at_full_native_size(self):
        final, report, out = self.render()
        overlay = np.array(Image.open(out / "effect_overlay.png"))
        alpha = overlay[:, :, 3:4].astype(float) / 255
        reconstructed = np.rint(overlay[:, :, :3] * alpha + self.original * (1 - alpha)).astype(np.uint8)
        self.assertLessEqual(np.abs(reconstructed.astype(int) - final.astype(int)).max(), 1)
        self.assertLessEqual(report["overlay"]["reconstruction_max_code_error"], 1)
        self.assertEqual(final.shape, self.original.shape)
        self.assertEqual(report["field"]["size"], [32, 24])
        self.assertTrue(report["source_unchanged"])

    def test_rolloff_neutrality_positive_monotonicity_and_no_hard_white_clip(self):
        original = np.array([[[0.95, 0.7, 0.3], [0.9, 0.9, 0.9], [1.0, 0.5, 0.4]]], dtype=np.float32)
        np.testing.assert_array_equal(relight.soft_rolloff(original, original), original)
        result = relight.soft_rolloff(original, original * 4)
        self.assertTrue(np.all(result >= original))
        self.assertTrue(np.all(result <= 1))
        self.assertLess(result[0, 0, 0], 1)
        self.assertLess(result[0, 1, 0], 1)

    def test_invalid_number_and_source_overwrite_are_rejected(self):
        with self.assertRaises(ValueError):
            self.render(strength=float("nan"))
        with self.assertRaises(ValueError):
            relight.composite(self.source, self.field(name="final.png"), self.root)

    def test_transparent_data_and_source_are_rejected_including_palette_transparency(self):
        transparent = self.root / "transparent.png"
        Image.new("RGBA", (32, 24), (128, 128, 128, 0)).save(transparent)
        with self.assertRaisesRegex(ValueError, "opaque"):
            self.render(transparent)
        palette = Image.new("P", (128, 96), 0)
        palette.info["transparency"] = 0
        palette.save(transparent)
        with self.assertRaisesRegex(ValueError, "opaque"):
            relight.prepare(transparent, self.root / "reference")

    def test_multiple_lamps_add_linear_contributions_instead_of_ev(self):
        self.original.fill(55)
        Image.fromarray(self.original).save(self.source)
        first = self.field(160, name="first.png")
        second = self.field(192, name="second.png")
        weights = [0.4, 1.3]
        final, report, _ = self.render([first, second], field_strengths=weights, smooth=0, edge_radius=0)
        base_linear = relight.srgb_to_linear(self.original)
        gain = 1 + weights[0] * (2 ** ((160 - 128) / 127 * 2) - 1) + weights[1] * (2 ** ((192 - 128) / 127 * 2) - 1)
        expected = relight.linear_to_srgb(base_linear * gain)
        np.testing.assert_array_equal(final, expected)
        self.assertEqual(len(report["fields"]), 2)
        self.assertEqual(report["parameters"]["field_strengths"], weights)
        # The EV-product interpretation would be observably different.
        wrong = relight.linear_to_srgb(base_linear * 2 ** (weights[0] * (160 - 128) / 127 * 2 + weights[1] * (192 - 128) / 127 * 2))
        self.assertFalse(np.array_equal(final, wrong))

    def test_zero_power_second_lamp_preserves_first_lamp_result(self):
        first = self.field(192, name="first.png")
        second = self.field(240, name="second.png")
        expected, _, _ = self.render(first)
        final, _, _ = self.render([first, second], field_strengths=[1, 0])
        np.testing.assert_array_equal(final, expected)

    def test_saved_npz_reuses_native_field_without_filtering_or_regeneration(self):
        field = np.full((24, 32, 3), 128, dtype=np.uint8)
        field[3:22, 2:19] = [210, 150, 131]
        field_path = self.root / "pattern.png"
        Image.fromarray(field).save(field_path)
        final, _, first_out = self.render(field_path, save_arrays=True)
        second_out = self.root / "reused"
        report = relight.composite(self.source, first_out / "relight_arrays.npz", second_out, smooth=12, edge_radius=30)
        np.testing.assert_array_equal(np.array(Image.open(second_out / "final.png")), final)
        self.assertEqual(report["field"]["processing"], "no repeated smoothing or guided filtering")
        self.assertTrue(report["field"]["source_hash_verified"])

    def test_saved_npz_can_adjust_light_power_and_reject_wrong_source(self):
        full, _, first_out = self.render(save_arrays=True)
        weighted_out = self.root / "weighted"
        npz_path = first_out / "relight_arrays.npz"
        relight.composite(self.source, npz_path, weighted_out, field_strengths=[0])
        np.testing.assert_array_equal(np.array(Image.open(weighted_out / "final.png")), self.original)
        relight.composite(self.source, npz_path, weighted_out, field_strengths=[0.5])
        half_power = np.array(Image.open(weighted_out / "final.png"))
        self.assertTrue(np.all(half_power >= self.original))
        self.assertTrue(np.all(half_power <= full))
        self.assertTrue(np.any(half_power < full))
        different = self.root / "different.png"
        Image.new("RGB", (128, 96), (20, 30, 40)).save(different)
        with self.assertRaisesRegex(ValueError, "different original"):
            relight.composite(different, npz_path, self.root / "wrong")

    def test_wrong_lamp_weight_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one field-strength"):
            self.render([self.field(), self.field()], field_strengths=[0.5])

    def test_absolute_addition_lights_black_pixels_without_a_diffuse_field(self):
        self.original.fill(0)
        Image.fromarray(self.original).save(self.source)
        highlights = np.zeros((24, 32, 3), dtype=np.uint8)
        highlights[9:15, 13:19] = 255
        path = self.root / "highlights.png"
        Image.fromarray(highlights).save(path)
        out = self.root / "absolute"
        report = relight.composite(self.source, None, out, additive_paths=[path], additive_scale=0.2, additive_smooth=0)
        final = np.array(Image.open(out / "final.png"))
        self.assertGreater(final[48, 64, 0], 100)
        np.testing.assert_array_equal(final[:28], self.original[:28])
        self.assertEqual(report["additive_fields"][0]["size"], [32, 24])
        self.assertIsNone(report["field"])
        self.assertTrue(report["zero_EV_pixels_exact"])

    def test_additive_coverage_zero_and_zero_strength_are_exact(self):
        additive = self.field(255, name="additive.png")
        mask = np.full((96, 128), 255, dtype=np.uint8)
        mask[:, :64] = 0
        mask_path = self.root / "coverage.png"
        Image.fromarray(mask).save(mask_path)
        out = self.root / "absolute"
        report = relight.composite(self.source, self.field(), out, mask_path=mask_path, additive_paths=[additive])
        result = np.array(Image.open(out / "final.png"))
        np.testing.assert_array_equal(result[:, :64], self.original[:, :64])
        self.assertTrue(report["mask_zero_pixels_exact"])
        self.assertTrue(np.all(result >= self.original))
        relight.composite(self.source, self.field(), out, additive_paths=[additive], strength=0)
        np.testing.assert_array_equal(np.array(Image.open(out / "final.png")), self.original)

    def test_diffuse_and_absolute_addition_are_summed_before_one_rolloff(self):
        self.original[:] = [185, 155, 100]
        Image.fromarray(self.original).save(self.source)
        diffuse = self.field(192)
        specular = self.root / "specular.png"
        Image.new('RGB',(32,24),(255,210,128)).save(specular)
        out = self.root / "mixed"
        report = relight.composite(self.source, diffuse, out, smooth=0, edge_radius=0,
            additive_paths=[specular], additive_strengths=[0.8], additive_scale=0.45, additive_smooth=0)
        linear = relight.srgb_to_linear(self.original)
        gain = np.float32(2 ** ((192 - 128) / 127 * 2))
        addition=np.asarray([255,210,128],dtype=np.float32)/255*np.float32(.8*.45)
        expected = relight.linear_to_srgb(relight.soft_rolloff(linear, linear * gain + addition))
        np.testing.assert_array_equal(np.array(Image.open(out / "final.png")), expected)
        self.assertAlmostEqual(report["additive_linear_max"], 0.36, places=5)
        wrong = relight.linear_to_srgb(relight.soft_rolloff(linear, relight.soft_rolloff(linear, linear * gain) + addition))
        self.assertFalse(np.array_equal(wrong, expected))

    def test_saved_absolute_light_reuses_exactly_including_black_channels(self):
        self.original[:, :64] = 0
        self.original[:, 64:, 1] = 0
        Image.fromarray(self.original).save(self.source)
        highlights = np.zeros((24, 32, 3), dtype=np.uint8)
        highlights[4:20, 5:28] = [240, 190, 90]
        path = self.root / "specular.png"
        Image.fromarray(highlights).save(path)
        first = self.root / "first"
        relight.composite(self.source, self.field(), first, additive_paths=[path], additive_scale=0.3, save_arrays=True)
        expected = np.array(Image.open(first / "final.png"))
        with np.load(first / "relight_arrays.npz") as saved:
            self.assertIn("additive_linear", saved)
            self.assertTrue(np.any(saved["additive_linear"][:, :64] > 0))
        reused = self.root / "reused"
        report = relight.composite(self.source, first / "relight_arrays.npz", reused, smooth=40, edge_radius=40, additive_scale=0)
        np.testing.assert_array_equal(np.array(Image.open(reused / "final.png")), expected)
        self.assertTrue(report["field"]["includes_saved_additive_linear"])
        # The same normal-alpha delivery layer also reproduces newly lit black.
        overlay = np.array(Image.open(reused / "effect_overlay.png"))
        alpha = overlay[:, :, 3:4].astype(float) / 255
        rebuilt = np.rint(overlay[:, :, :3] * alpha + self.original * (1 - alpha)).astype(np.uint8)
        self.assertLessEqual(np.abs(rebuilt.astype(int) - expected.astype(int)).max(), 1)

    def test_additive_and_saved_lights_have_independent_linear_power_controls(self):
        self.original.fill(45)
        Image.fromarray(self.original).save(self.source)
        first = self.field(255, name="white.png")
        second = self.root / "red.png"
        Image.new("RGB", (32, 24), (255, 0, 0)).save(second)
        out = self.root / "both"
        relight.composite(self.source, None, out, additive_paths=[first, second], additive_strengths=[0.7, 0], save_arrays=True)
        single = self.root / "single"
        relight.composite(self.source, None, single, additive_paths=[first], additive_strengths=[0.7])
        np.testing.assert_array_equal(np.array(Image.open(out / "final.png")), np.array(Image.open(single / "final.png")))
        reused = self.root / "half"
        relight.composite(self.source, out / "relight_arrays.npz", reused, field_strengths=[0.5])
        direct = self.root / "directhalf"
        relight.composite(self.source, None, direct, additive_paths=[first], additive_strengths=[0.35])
        np.testing.assert_array_equal(np.array(Image.open(reused / "final.png")), np.array(Image.open(direct / "final.png")))
        relight.composite(self.source, out / "relight_arrays.npz", reused, field_strengths=[0])
        np.testing.assert_array_equal(np.array(Image.open(reused / "final.png")), self.original)

    def test_additive_bad_inputs_and_overwrite_are_rejected(self):
        additive = self.field(255, name="additive.png")
        out = self.root / "bad"
        for kwargs in ({"additive_scale": -0.1}, {"additive_scale": float("nan")}, {"additive_smooth": -1}, {"additive_strengths": [1, 2]}):
            with self.assertRaises(ValueError):
                relight.composite(self.source, None, out, additive_paths=[additive], **kwargs)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            relight.composite(self.source, None, out, additive_paths=[self.field(255, (32, 32), "square.png")])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            relight.composite(self.source, None, self.root, additive_paths=[self.field(255, name="final.png")])
        with self.assertRaisesRegex(ValueError, "At least one"):
            relight.composite(self.source, None, out)
        self.assertFalse(out.exists())

    def test_malformed_saved_absolute_array_is_rejected(self):
        path = self.root / "bad.npz"
        for addition in (np.zeros((4, 5, 3)), np.full((96, 128, 3), -0.2), np.full((96, 128, 3), np.nan)):
            np.savez(path, ev=np.zeros((96, 128, 3)), additive_linear=addition)
            with self.assertRaises(ValueError):
                relight.composite(self.source, path, self.root / "bad")

    def test_cli_accepts_additive_only_without_an_ev_field(self):
        additive = self.field(255, name="additive.png")
        out = self.root / "cli"
        result = subprocess.run([sys.executable, str(Path(relight.__file__)), "composite", "--source", str(self.source),
            "--additive-field", str(additive), "--additive-strength", "0.5", "--additive-scale", "0.3", "--additive-smooth", "0", "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out / "final.png").exists())

    def test_texture_modulation_keeps_detail_and_reuses_without_double_processing(self):
        source = np.full((96, 128, 3), 50, dtype=np.uint8)
        source[:, 60:64] = 120
        Image.fromarray(source).save(self.source)
        additive = self.field(255, name="texture-additive.png")
        first = self.root / "textured"
        relight.composite(self.source, None, first, additive_paths=[additive], additive_scale=.1, additive_texture=1, save_arrays=True)
        with np.load(first / "relight_arrays.npz") as saved:
            self.assertGreater(float(saved["additive_linear"][48, 62, 0]), float(saved["additive_linear"][48, 30, 0]))
        reused = self.root / "reused"
        relight.composite(self.source, first / "relight_arrays.npz", reused, additive_texture=1)
        np.testing.assert_array_equal(np.array(Image.open(first / "final.png")), np.array(Image.open(reused / "final.png")))
        for value in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                relight.composite(self.source, None, self.root/'bad-texture', additive_paths=[additive], additive_texture=value)

    def test_neutral_specular_shoulder_is_local_and_preserves_colored_rims(self):
        base=np.array([[[.8,.4,.3],[.8,.4,.3],[.8,.4,.3]]],dtype=np.float32)
        target=np.array([[[1.4,1.2,1.1],[1.4,1.2,1.1],[.8,.4,.3]]],dtype=np.float32)
        addition=np.array([[[.7,.7,.7],[.7,.4,.1],[0,0,0]]],dtype=np.float32)
        result=relight.neutral_specular_shoulder(base,target,addition)
        np.testing.assert_array_equal(result[0,1:],base[0,1:])
        self.assertTrue(np.all(result>=base))
        self.assertLess(np.ptp(result[0,0]),np.ptp(base[0,0]))

    def test_many_legal_additive_sources_save_and_replay(self):
        path=self.field(255,name='many-additive.png')
        out=self.root/'many'
        relight.composite(self.source,None,out,additive_paths=[path]*9,additive_strengths=[4]*9,additive_scale=4,strength=2,save_arrays=True)
        reused=self.root/'many-reused'
        relight.composite(self.source,out/'relight_arrays.npz',reused)
        np.testing.assert_array_equal(np.array(Image.open(out/'final.png')),np.array(Image.open(reused/'final.png')))


if __name__ == "__main__":
    unittest.main(verbosity=2)
