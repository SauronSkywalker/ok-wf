# -*- coding: utf-8 -*-
import pathlib
import platform
import unittest

from ok.util.gpu_devices import (
    AUTO_OPTION,
    CPU_OPTION,
    GpuAdapter,
    build_options,
    dedupe,
    enumerate_dxgi_adapters,
    create_session,
    filter_usable,
    provider_list,
    resolve,
    select_adapters,
)

AMD = 0x1002
INTEL = 0x8086
NVIDIA = 0x10DE
MS = 0x1414


def adapter(device_id=0, name="AMD Radeon 780M Graphics", vendor_id=AMD,
            pci_device_id=0x1900, subsys_id=0x50081D05, revision=0x00BA,
            luid=(0, 69534), ded_vram_mb=418, dedicated_system_mb=0,
            shared_mb=16005, is_software=False, d3d12_levels=("12_0", "11_0")):
    return GpuAdapter(
        device_id=device_id, name=name, vendor_id=vendor_id,
        pci_device_id=pci_device_id, subsys_id=subsys_id, revision=revision,
        luid=luid, ded_vram_mb=ded_vram_mb, dedicated_system_mb=dedicated_system_mb,
        shared_mb=shared_mb, is_software=is_software, d3d12_levels=d3d12_levels,
    )


def warp(device_id=2):
    """Microsoft Basic Render Driver: D3D12 capable but a CPU rasterizer."""
    return adapter(device_id=device_id, name="Microsoft Basic Render Driver",
                   vendor_id=MS, pci_device_id=0x008C, ded_vram_mb=0, is_software=True)


def discrete_gpu(device_id, vendor_id, name, ded_vram_mb=8192, shared_mb=0):
    return adapter(device_id=device_id, name=name, vendor_id=vendor_id,
                   pci_device_id=0x73BF, ded_vram_mb=ded_vram_mb,
                   shared_mb=shared_mb, luid=(0, 1000 + device_id))


class TestFilteringPolicy(unittest.TestCase):
    def test_software_adapter_is_not_usable(self):
        self.assertFalse(warp().usable)

    def test_adapter_without_d3d12_is_not_usable(self):
        self.assertFalse(adapter(d3d12_levels=()).usable)

    def test_hardware_adapter_with_d3d12_is_usable(self):
        self.assertTrue(adapter().usable)

    def test_select_drops_warp_even_though_it_supports_d3d12(self):
        # Guards the exact failure mode: D3D12 capability alone would let WARP
        # through, and WARP silently runs on the CPU.
        self.assertTrue(warp().d3d12_levels)
        self.assertEqual([a.device_id for a in select_adapters([warp()])], [])

    def test_select_drops_adapter_without_d3d12(self):
        self.assertEqual(select_adapters([adapter(d3d12_levels=())]), [])


class TestDeduplication(unittest.TestCase):
    def test_identity_key_ignores_luid_and_dxgi_index(self):
        first = adapter(device_id=0, luid=(0, 69534))
        second = adapter(device_id=1, luid=(0, 161200))
        self.assertNotEqual(first.device_id, second.device_id)
        self.assertNotEqual(first.luid, second.luid)
        self.assertEqual(first.identity_key, second.identity_key)

    def test_identity_key_differs_for_different_pci_device(self):
        self.assertNotEqual(adapter(pci_device_id=0x1900).identity_key,
                            adapter(pci_device_id=0x73FF).identity_key)

    def test_duplicate_physical_gpu_keeps_first_device_id(self):
        dupes = [adapter(device_id=0, luid=(0, 69534)),
                 adapter(device_id=1, luid=(0, 161200))]
        kept = dedupe(dupes)
        self.assertEqual([a.device_id for a in kept], [0])

    def test_dedupe_preserves_original_device_id_after_filtering(self):
        result = select_adapters([warp(device_id=0), adapter(device_id=1),
                                  adapter(device_id=2, luid=(0, 999))])
        self.assertEqual([a.device_id for a in result], [1])

    def test_different_gpus_are_not_deduped(self):
        amd = adapter(device_id=0, pci_device_id=0x1900, ded_vram_mb=418, shared_mb=16005)
        nvidia = discrete_gpu(1, NVIDIA, "NVIDIA GeForce RTX 4060")
        self.assertEqual([a.device_id for a in dedupe([amd, nvidia])], [0, 1])


class TestPriority(unittest.TestCase):
    def test_integrated_heuristic_uses_dedicated_vram(self):
        self.assertTrue(adapter().is_integrated)
        self.assertFalse(discrete_gpu(0, AMD, "AMD Radeon RX 7900 XTX").is_integrated)
        self.assertEqual(adapter().kind, "iGPU")
        self.assertEqual(discrete_gpu(0, AMD, "AMD Radeon RX 7900 XTX").kind, "dGPU")

    def test_software_adapter_is_never_classified_as_integrated(self):
        self.assertFalse(warp().is_integrated)

    def test_integrated_ranks_before_discrete(self):
        igpu = adapter(device_id=0, luid=(0, 1))
        dgpu = discrete_gpu(1, NVIDIA, "NVIDIA GeForce RTX 4060")
        self.assertEqual([a.device_id for a in select_adapters([dgpu, igpu])], [0, 1])

    def test_amd_discrete_ranks_before_nvidia_discrete(self):
        amd = discrete_gpu(0, AMD, "AMD Radeon RX 7900 XTX")
        nvidia = discrete_gpu(1, NVIDIA, "NVIDIA GeForce RTX 4060")
        self.assertEqual([a.device_id for a in select_adapters([nvidia, amd])], [0, 1])

    def test_intel_integrated_ranks_before_amd_integrated(self):
        intel = adapter(device_id=0, name="Intel(R) Iris(R) Xe Graphics",
                        vendor_id=INTEL, pci_device_id=0x46A6, ded_vram_mb=128)
        amd = adapter(device_id=1, luid=(0, 5))
        self.assertEqual([a.device_id for a in select_adapters([amd, intel])], [0, 1])


class TestDropdownAndResolve(unittest.TestCase):
    def setUp(self):
        self.intel = adapter(device_id=0, name="Intel(R) Iris(R) Xe Graphics",
                             vendor_id=INTEL, pci_device_id=0x46A6,
                             ded_vram_mb=128, luid=(0, 11))
        self.nvidia = discrete_gpu(1, NVIDIA, "NVIDIA GeForce RTX 4060")
        self.all = [warp(device_id=2), self.nvidia, self.intel]

    def test_options_have_auto_first_and_cpu_last(self):
        options = build_options(self.all)
        self.assertEqual(options[0], AUTO_OPTION)
        self.assertEqual(options[-1], CPU_OPTION)

    def test_options_exclude_software_adapter(self):
        self.assertNotIn("Microsoft Basic Render Driver", build_options(self.all))

    def test_duplicate_gpu_instances_collapse_to_one_entry(self):
        options = build_options([adapter(device_id=0, luid=(0, 1)),
                                 adapter(device_id=1, luid=(0, 2)),
                                 warp(device_id=3)])
        self.assertEqual(options, [AUTO_OPTION,
                                   "AMD Radeon 780M Graphics [iGPU]",
                                   CPU_OPTION])

    def test_label_collision_gets_device_id_suffix(self):
        # Two *different* GPUs reporting the same marketing name must stay
        # selectable, so the duplicate label is disambiguated.
        first = adapter(device_id=0, pci_device_id=0x1001, luid=(0, 1))
        second = adapter(device_id=1, name=first.name, pci_device_id=0x1002,
                         luid=(0, 2))
        options = build_options([first, second])
        self.assertEqual(len(options), 4)
        self.assertEqual(len(options), len(set(options)))
        self.assertIn("device_id=1", options[2])

    def test_auto_picks_integrated_first(self):
        resolved = resolve(AUTO_OPTION, self.all)
        self.assertEqual(resolved.device_id, 0)
        self.assertEqual(resolved.resolved_label, "Intel(R) Iris(R) Xe Graphics [iGPU]")
        self.assertEqual(resolved.reason, "Auto")

    def test_cpu_choice_forces_none(self):
        resolved = resolve(CPU_OPTION, self.all)
        self.assertTrue(resolved.use_cpu)
        self.assertEqual(resolved.device_id, None)
        self.assertEqual(resolved.reason, "forced CPU")

    def test_explicit_device_returns_its_device_id(self):
        target = build_options(self.all)[2]  # the discrete NVIDIA entry
        resolved = resolve(target, self.all)
        self.assertEqual(resolved.device_id, 1)
        self.assertEqual(resolved.reason, "user selection")

    def test_stale_choice_falls_back_to_auto(self):
        resolved = resolve("GPU That Was Unplugged", self.all)
        self.assertEqual(resolved.device_id, 0)
        self.assertIn("stale choice", resolved.reason)

    def test_none_choice_falls_back_to_auto(self):
        self.assertEqual(resolve(None, self.all).device_id, 0)

    def test_no_usable_adapter_resolves_to_cpu(self):
        resolved = resolve(AUTO_OPTION, [warp(), adapter(d3d12_levels=())])
        self.assertTrue(resolved.use_cpu)
        self.assertEqual(resolved.reason, "no usable GPU adapter")

    def test_empty_machine_resolves_to_cpu(self):
        resolved = resolve(AUTO_OPTION, [])
        self.assertTrue(resolved.use_cpu)


class TestProviderList(unittest.TestCase):
    def test_gpu_device_requests_directml_with_device_id(self):
        self.assertEqual(provider_list(0),
                         [("DmlExecutionProvider", {"device_id": 0}),
                          "CPUExecutionProvider"])

    def test_cpu_device_requests_cpu_only(self):
        self.assertEqual(provider_list(None), ["CPUExecutionProvider"])


class TestSessionCreation(unittest.TestCase):
    """End-to-end: selection -> session -> active providers.

    Runs only when the reference YOLOv8 model cloned into _refs/ is present.
    """

    MODEL = (pathlib.Path(__file__).resolve().parents[1]
             / "_refs" / "ok-wuthering-waves" / "assets" / "echo_model" / "echo.onnx")

    def setUp(self):
        if not self.MODEL.exists():
            self.skipTest("reference model not present")
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            self.skipTest("onnxruntime not installed")

    def test_auto_choice_lands_on_gpu_or_reports_cpu(self):
        session, resolved = create_session(self.MODEL, AUTO_OPTION)
        active = session.get_providers()
        if resolved.device_id is None:
            self.assertEqual(active, ["CPUExecutionProvider"])
        else:
            self.assertIn("DmlExecutionProvider", active)

    def test_cpu_choice_never_touches_directml(self):
        session, resolved = create_session(self.MODEL, CPU_OPTION)
        self.assertIsNone(resolved.device_id)
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])

    def test_created_session_accepts_the_model_input(self):
        session, _ = create_session(self.MODEL, AUTO_OPTION)
        meta = session.get_inputs()[0]
        self.assertEqual(len(meta.shape), 4)
        self.assertEqual(meta.type, "tensor(float)")


class TestLiveEnumeration(unittest.TestCase):
    """Runs against the real machine; assertions hold on any platform."""

    def test_scanned_adapters_are_usable_and_unique(self):
        selected = select_adapters(enumerate_dxgi_adapters())
        device_ids = [a.device_id for a in selected]
        self.assertEqual(device_ids, sorted(set(device_ids)))
        for found in selected:
            self.assertFalse(found.is_software)
            self.assertTrue(found.d3d12_levels)

    def test_options_always_start_with_auto_and_end_with_cpu(self):
        options = build_options(enumerate_dxgi_adapters())
        self.assertEqual(options[0], AUTO_OPTION)
        self.assertEqual(options[-1], CPU_OPTION)
        self.assertEqual(len(options), len(set(options)))

    def test_selected_device_ids_stay_within_dxgi_range(self):
        all_adapters = enumerate_dxgi_adapters()
        valid = {a.device_id for a in all_adapters}
        resolved = resolve(AUTO_OPTION, all_adapters)
        if resolved.device_id is not None:
            self.assertIn(resolved.device_id, valid)

    @unittest.skipUnless(platform.system() == "Windows", "DXGI is Windows-only")
    def test_windows_reports_at_least_one_dxgi_adapter(self):
        self.assertTrue(enumerate_dxgi_adapters())


if __name__ == "__main__":
    unittest.main()
