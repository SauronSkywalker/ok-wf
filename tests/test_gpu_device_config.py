# -*- coding: utf-8 -*-
"""GUI wiring tests for the GPU Device picker (no Qt required)."""
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ok import OkGlobals, og
from ok.util.config import Config
from ok.util.GlobalConfig import (
    GPU_DEVICE_OPTION,
    GlobalConfig,
    create_basic_options,
    gpu_device_options,
    register_basic_options,
)
from ok.util.gpu_devices import AUTO_OPTION, CPU_OPTION, GpuAdapter, auto_dml_enabled

# Config.__setitem__ persists to disk immediately, so isolate the folder or
# tests leak state into each other and into a real user configuration.
_original_config_folder = None
_temp_config_folder = None


def setUpModule():
    global _original_config_folder, _temp_config_folder
    _original_config_folder = Config.config_folder
    _temp_config_folder = tempfile.TemporaryDirectory(prefix="okwf-test-configs-")
    Config.config_folder = _temp_config_folder.name


def tearDownModule():
    global _original_config_folder, _temp_config_folder
    Config.config_folder = _original_config_folder
    if _temp_config_folder is not None:
        _temp_config_folder.cleanup()
        _temp_config_folder = None


FAKE_GPU = GpuAdapter(
    device_id=3, name="Fake GPU", vendor_id=0x1002, pci_device_id=0x1234,
    subsys_id=1, revision=1, luid=(0, 1), ded_vram_mb=418,
    shared_mb=16005, d3d12_levels=("12_0",),
)


def make_globals(choice=AUTO_OPTION, use_dml=True, adapters=None):
    config = GlobalConfig([])
    basic = register_basic_options(config)
    basic[GPU_DEVICE_OPTION] = choice
    globals_ = OkGlobals()
    globals_.global_config = config
    globals_.use_dml = use_dml
    globals_._gpu_adapters = list(adapters if adapters is not None else [FAKE_GPU])
    return globals_


class GpuConfigTestCase(unittest.TestCase):
    """Each test gets its own on-disk config folder.

    Config.__setitem__ writes JSON immediately, so without this a test that
    selects a GPU leaks that choice into the next test.
    """

    def setUp(self):
        self._config_dir = tempfile.TemporaryDirectory(prefix="okwf-test-configs-")
        self._original_config_folder = Config.config_folder
        Config.config_folder = self._config_dir.name

    def tearDown(self):
        Config.config_folder = self._original_config_folder
        self._config_dir.cleanup()


class TestGpuDeviceOptionRegistration(GpuConfigTestCase):
    def test_options_start_with_auto_and_end_with_cpu(self):
        options = gpu_device_options()
        self.assertEqual(options[0], AUTO_OPTION)
        self.assertEqual(options[-1], CPU_OPTION)

    def test_default_value_is_auto(self):
        option = create_basic_options()
        self.assertEqual(option.default_config[GPU_DEVICE_OPTION], AUTO_OPTION)

    def test_declared_as_drop_down_with_scanned_options(self):
        option = create_basic_options()
        declared = option.config_type[GPU_DEVICE_OPTION]
        self.assertEqual(declared["type"], "drop_down")
        self.assertEqual(declared["options"], gpu_device_options())

    def test_option_has_a_description(self):
        option = create_basic_options()
        self.assertIn(GPU_DEVICE_OPTION, option.config_description)
        self.assertTrue(option.config_description[GPU_DEVICE_OPTION])

    def test_registered_global_config_exposes_the_key(self):
        basic = register_basic_options(GlobalConfig([]))
        self.assertIn(GPU_DEVICE_OPTION, basic)
        self.assertEqual(basic[GPU_DEVICE_OPTION], AUTO_OPTION)

    def test_scanner_failure_degrades_to_auto_and_cpu(self):
        with patch("ok.util.GlobalConfig.logger") as _logger:
            # Bypass the cache so the failure path is exercised.
            import ok.util.GlobalConfig as global_config_module
            original = global_config_module._gpu_device_options_cache
            try:
                global_config_module._gpu_device_options_cache = None
                with patch("ok.util.gpu_devices.enumerate_dxgi_adapters",
                           side_effect=RuntimeError("boom")):
                    self.assertEqual(gpu_device_options(), [AUTO_OPTION, CPU_OPTION])
            finally:
                global_config_module._gpu_device_options_cache = original


class TestSetGpuDevice(GpuConfigTestCase):
    def test_auto_leaves_the_legacy_switch_untouched(self):
        for existing in (True, False):
            target = make_globals(AUTO_OPTION, use_dml=existing)
            target.set_gpu_device()
            self.assertEqual(target.use_dml, existing)

    def test_explicit_cpu_disables_directml(self):
        target = make_globals(CPU_OPTION, use_dml=True)
        resolved = target.set_gpu_device()
        self.assertFalse(target.use_dml)
        self.assertTrue(resolved.use_cpu)

    def test_explicit_gpu_enables_directml(self):
        target = make_globals(FAKE_GPU.label, use_dml=False)
        resolved = target.set_gpu_device()
        if target._dml_supported_by_os():
            self.assertTrue(target.use_dml)
            self.assertEqual(resolved.device_id, 3)

    def test_explicit_gpu_resolves_to_that_device_id(self):
        target = make_globals(FAKE_GPU.label, use_dml=True)
        self.assertEqual(target.gpu_device_id, 3)

    def test_stale_choice_falls_back_to_auto_device(self):
        target = make_globals("Unplugged GPU", use_dml=True)
        resolved = target.set_gpu_device()
        self.assertEqual(resolved.device_id, 3)
        self.assertIn("stale choice", resolved.reason)

    def test_cpu_choice_is_live_not_cached(self):
        target = make_globals(AUTO_OPTION, use_dml=True)
        self.assertEqual(target.gpu_device_id, 3)
        target.global_config.get_config("Basic Options")[GPU_DEVICE_OPTION] = CPU_OPTION
        self.assertEqual(target.gpu_choice, CPU_OPTION)
        self.assertIsNone(target.gpu_device_id)

    def test_no_usable_adapter_means_no_device(self):
        target = make_globals(AUTO_OPTION, use_dml=True, adapters=[])
        self.assertIsNone(target.gpu_device_id)


class TestAutoDmlPolicy(GpuConfigTestCase):
    """The old Auto logic only consulted nvidia-smi, so AMD/Intel-only hosts
    never enabled DirectML no matter what hardware they had."""

    def test_healthy_nvidia_gpu_enables_directml(self):
        self.assertTrue(auto_dml_enabled(4000, lambda: []))

    def test_low_but_present_nvidia_vram_stays_off(self):
        self.assertFalse(auto_dml_enabled(2000, lambda: [FAKE_GPU]))

    def test_no_nvidia_reporting_falls_back_to_dxgi(self):
        self.assertTrue(auto_dml_enabled(-1, lambda: [FAKE_GPU]))
        self.assertFalse(auto_dml_enabled(-1, lambda: []))

    def test_dxgi_scan_is_not_run_on_the_nvidia_path(self):
        def boom():
            raise AssertionError("should not scan DXGI when nvidia-smi answered")

        self.assertTrue(auto_dml_enabled(4000, boom))


class TestSetUseDmlIntegration(GpuConfigTestCase):
    """Glue test for set_use_dml; needs psutil to import ok.util.process."""

    def setUp(self):
        super().setUp()
        try:
            import ok.util.process  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"ok.util.process unavailable: {exc}")

    @patch("ok.util.process.get_first_gpu_free_memory_mib", return_value=-1)
    def test_auto_enables_directml_when_dxgi_reports_a_usable_gpu(self, _nv):
        target = make_globals(AUTO_OPTION, use_dml=False)
        target.global_config.get_config("Basic Options")["Use DirectML"] = "Auto"
        target.set_use_dml()
        self.assertTrue(target.use_dml)

    @patch("ok.util.process.get_first_gpu_free_memory_mib", return_value=-1)
    def test_auto_stays_off_when_no_usable_gpu_exists(self, _nv):
        target = make_globals(AUTO_OPTION, use_dml=False, adapters=[])
        target.global_config.get_config("Basic Options")["Use DirectML"] = "Auto"
        target.set_use_dml()
        self.assertFalse(target.use_dml)

    @patch("ok.util.process.get_first_gpu_free_memory_mib", return_value=4000)
    def test_auto_still_prefers_a_healthy_nvidia_gpu(self, _nv):
        target = make_globals(AUTO_OPTION, use_dml=False)
        target.global_config.get_config("Basic Options")["Use DirectML"] = "Auto"
        target.set_use_dml()
        self.assertTrue(target.use_dml)


class TestGlobalsSessionHelper(GpuConfigTestCase):
    MODEL = (pathlib.Path(__file__).resolve().parents[1]
             / "_refs" / "ok-wuthering-waves" / "assets" / "echo_model" / "echo.onnx")

    def setUp(self):
        super().setUp()
        if not self.MODEL.exists():
            self.skipTest("reference model not present")
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            self.skipTest("onnxruntime not installed")

    def test_auto_with_switch_off_forces_cpu_session(self):
        """Auto obeys the Use DirectML switch."""
        target = make_globals(AUTO_OPTION, use_dml=False)
        session, resolved = target.create_onnx_session(self.MODEL)
        self.assertIsNone(resolved.device_id)
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])

    def test_explicit_gpu_choice_wins_over_disabled_switch(self):
        """A live dropdown change must not be ignored just because startup
        computed use_dml=False."""
        target = make_globals(FAKE_GPU.label, use_dml=False)
        _, resolved = target.create_onnx_session(self.MODEL)
        self.assertEqual(resolved.device_id, 3)
        self.assertEqual(resolved.reason, "user selection")

    def test_use_dml_on_builds_directml_session(self):
        target = make_globals(AUTO_OPTION, use_dml=True)
        target._gpu_adapters = None  # real DXGI scan; the fixture id 3 does not exist
        session, resolved = target.create_onnx_session(self.MODEL)
        if resolved.device_id is None:
            self.skipTest("no usable GPU adapter on this machine")
        self.assertIn("DmlExecutionProvider", session.get_providers())

    def test_unknown_device_id_never_raises(self):
        """device_id 3 does not exist on this machine.

        ONNX Runtime silently falls back to CPU instead of raising, which is
        exactly why callers must inspect get_providers() rather than trusting
        the requested device_id.
        """
        target = make_globals(FAKE_GPU.label, use_dml=True)
        session, resolved = target.create_onnx_session(self.MODEL)
        self.assertEqual(resolved.device_id, 3)
        self.assertIn("CPUExecutionProvider", session.get_providers())


class TestGpuDeviceWidget(GpuConfigTestCase):
    """Render the picker through the real Qt widget factory."""

    @classmethod
    def setUpClass(cls):
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as exc:
            raise unittest.SkipTest(f"PySide6 not installed: {exc}")
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_gpu_device_renders_as_combo_box_with_scanned_options(self):
        from ok.ui.qt.tasks.ConfigItemFactory import config_widget

        original_app = og.app
        og.app = SimpleNamespace(tr=lambda text: text)
        try:
            option = create_basic_options()
            config = register_basic_options(GlobalConfig([]))
            widget = config_widget(option.config_type, option.config_description,
                                   config, GPU_DEVICE_OPTION, AUTO_OPTION, None)
            try:
                self.assertTrue(hasattr(widget, "combo_box"),
                                "GPU Device must render as a drop-down")
                items = [widget.combo_box.itemText(i)
                         for i in range(widget.combo_box.count())]
                self.assertEqual(items, gpu_device_options())
                self.assertEqual(items[0], AUTO_OPTION)
                self.assertEqual(items[-1], CPU_OPTION)
            finally:
                widget.deleteLater()
        finally:
            og.app = original_app


class TestGlobalsSingletonHasWiring(GpuConfigTestCase):
    def test_module_singleton_exposes_the_new_api(self):
        for name in ("gpu_adapters", "gpu_choice", "gpu_resolved", "gpu_device_id"):
            self.assertTrue(hasattr(og, name), f"og.{name} missing")
        self.assertTrue(callable(og.set_gpu_device))
        self.assertTrue(callable(og.create_onnx_session))


if __name__ == "__main__":
    unittest.main()
