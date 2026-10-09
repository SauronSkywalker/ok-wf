# -*- coding: utf-8 -*-
"""DXGI adapter enumeration and GPU selection policy.

Why DXGI instead of WMI
-----------------------
ONNX Runtime's DirectML execution provider resolves ``device_id`` against the
DXGI adapter list.  WMI's ``Win32_VideoController`` reports a *different* set of
devices in a *different* order, so building a picker from WMI silently maps the
user's choice onto the wrong GPU (or onto a device DirectML cannot use).

Measured on one machine (2026-10)::

    DXGI : [0] AMD Radeon 780M, [1] AMD Radeon 780M (2nd instance), [2] Microsoft Basic Render Driver
    WMI  : [0] GameViewer Virtual Display Adapter, [1] AMD Radeon 780M Graphics

The virtual display adapter never appears in DXGI, and the WARP software adapter
never appears in WMI.  This module therefore treats DXGI as the single source of
truth for the picker.

Policy
------
1. Drop ``DXGI_ADAPTER_FLAG_SOFTWARE`` adapters (WARP is D3D12 capable but is a
   pure CPU rasterizer, so "can create a D3D12 device" alone is not enough).
2. Drop adapters that cannot create a D3D12 device (DirectML requirement).
3. De-duplicate identical physical GPUs.  The same GPU can be registered as
   several DXGI adapter instances with distinct LUIDs; the first occurrence wins
   and its DXGI index stays as the ``device_id``.
4. Prefer integrated GPUs first, then Intel > AMD > NVIDIA for discrete ones.
5. Anything the user's stored choice no longer matches falls back to Auto.

Known limitation
----------------
Two physically *identical* discrete GPU models report the same PCI identity and
memory footprint, so they share an identity key and collapse to a single entry.
An inference picker only ever needs one device and Auto still selects the first,
so this is accepted; if two same-named but distinct GPUs do survive filtering,
their labels are disambiguated with the ``device_id`` suffix.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

AUTO_OPTION = "Auto"
CPU_OPTION = "CPU"

DXGI_ADAPTER_FLAG_SOFTWARE = 0x02
IGPU_DEDICATED_VRAM_MB_MAX = 512
IGPU_SHARED_RAM_MB_MIN = 1024

VENDOR_NAMES = {
    0x1002: "AMD",
    0x8086: "Intel",
    0x10DE: "NVIDIA",
    0x1414: "Microsoft",
}
VENDOR_RANK = {"Intel": 0, "AMD": 1, "NVIDIA": 2}


@dataclass(frozen=True)
class GpuAdapter:
    """One DXGI adapter. ``device_id`` is the DXGI index DirectML expects."""

    device_id: int
    name: str
    vendor_id: int = 0
    pci_device_id: int = 0
    subsys_id: int = 0
    revision: int = 0
    luid: Tuple[int, int] = (0, 0)
    ded_vram_mb: int = 0
    dedicated_system_mb: int = 0
    shared_mb: int = 0
    is_software: bool = False
    d3d12_levels: Tuple[str, ...] = ()

    @property
    def vendor(self) -> str:
        return VENDOR_NAMES.get(self.vendor_id, "0x%04X" % self.vendor_id)

    @property
    def usable(self) -> bool:
        """True when DirectML can actually run on this adapter."""
        return not self.is_software and bool(self.d3d12_levels)

    @property
    def is_integrated(self) -> bool:
        """Heuristic label only: tiny carve-out + lots of shared system RAM."""
        if self.is_software:
            return False
        return self.ded_vram_mb < IGPU_DEDICATED_VRAM_MB_MAX and self.shared_mb >= IGPU_SHARED_RAM_MB_MIN

    @property
    def kind(self) -> str:
        return "iGPU" if self.is_integrated else "dGPU"

    @property
    def identity_key(self) -> Tuple:
        """Stable identity of the physical GPU.

        Deliberately excludes the LUID: identical GPUs registered twice expose
        different LUIDs but the same PCI identity and memory footprint.
        """
        return (
            self.vendor_id,
            self.pci_device_id,
            self.subsys_id,
            self.revision,
            self.ded_vram_mb,
            self.dedicated_system_mb,
            self.shared_mb,
        )

    @property
    def priority(self) -> Tuple[int, int, int]:
        return (
            0 if self.is_integrated else 1,
            VENDOR_RANK.get(self.vendor, 9),
            self.device_id,
        )

    @property
    def label(self) -> str:
        return "%s [%s]" % (self.name, self.kind)


def filter_usable(adapters: Sequence[GpuAdapter]) -> List[GpuAdapter]:
    return [a for a in adapters if a.usable]


NVIDIA_FREE_VRAM_MIB_FOR_AUTO = 3000


def auto_dml_enabled(nv_free_gpu_memory_mib: int, adapters_provider) -> bool:
    """Decide the ``Use DirectML: Auto`` outcome.

    A healthy NVIDIA GPU (free VRAM above the historical threshold) enables
    DirectML.  A negative value means nvidia-smi is unavailable, i.e. there is
    no NVIDIA GPU -- but the machine may still have Intel/AMD integrated or
    discrete graphics, so ask DXGI instead of giving up.

    ``adapters_provider`` is a zero-argument callable so the DXGI scan only
    happens on the non-NVIDIA path.
    """
    if nv_free_gpu_memory_mib > NVIDIA_FREE_VRAM_MIB_FOR_AUTO:
        return True
    if nv_free_gpu_memory_mib < 0:
        return bool(filter_usable(adapters_provider()))
    return False


def dedupe(adapters: Iterable[GpuAdapter]) -> List[GpuAdapter]:
    """Keep the first adapter of each physical GPU, preserving device_id."""
    seen = set()
    result = []
    for adapter in adapters:
        key = adapter.identity_key
        if key in seen:
            continue
        seen.add(key)
        result.append(adapter)
    return result


def select_adapters(adapters: Iterable[GpuAdapter]) -> List[GpuAdapter]:
    """Filtered, de-duplicated adapters in user-facing priority order."""
    return sorted(dedupe(filter_usable(adapters)), key=lambda a: a.priority)


def build_options(adapters: Iterable[GpuAdapter]) -> List[str]:
    """Dropdown entries: Auto, one label per usable GPU, then CPU."""
    options = [AUTO_OPTION]
    labels: List[str] = []
    for adapter in select_adapters(adapters):
        label = adapter.label
        if label in labels:
            label = "%s (device_id=%d)" % (label, adapter.device_id)
        labels.append(label)
    options.extend(labels)
    options.append(CPU_OPTION)
    return options


@dataclass(frozen=True)
class ResolvedDevice:
    choice: str
    device_id: Optional[int]
    resolved_label: str
    reason: str

    @property
    def use_cpu(self) -> bool:
        return self.device_id is None


def resolve(choice: Optional[str], adapters: Iterable[GpuAdapter]) -> ResolvedDevice:
    """Map a stored dropdown value onto a DirectML ``device_id``.

    ``None`` means "run on CPU".  Unknown or stale choices deliberately fall
    back to Auto rather than failing, because the device list can change
    between runs (driver updates, virtual displays coming and going).
    """
    selected = select_adapters(adapters)
    by_label: Dict[str, GpuAdapter] = {a.label: a for a in selected}
    wanted = (choice or "").strip()

    if wanted == CPU_OPTION:
        return ResolvedDevice(wanted, None, CPU_OPTION, "forced CPU")

    if wanted in by_label:
        adapter = by_label[wanted]
        return ResolvedDevice(wanted, adapter.device_id, adapter.label, "user selection")

    if not selected:
        return ResolvedDevice(wanted, None, CPU_OPTION, "no usable GPU adapter")

    adapter = selected[0]
    reason = "Auto" if wanted in ("", AUTO_OPTION) else "stale choice %r, fell back to Auto" % wanted
    return ResolvedDevice(wanted, adapter.device_id, adapter.label, reason)


def provider_list(device_id: Optional[int]) -> List[object]:
    """ONNX Runtime provider spec for a resolved device.

    ``None`` means the caller selected CPU (or no usable GPU exists), which is
    a normal outcome rather than an error.
    """
    if device_id is None:
        return ["CPUExecutionProvider"]
    return [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]


def create_session(model_path, choice: Optional[str] = AUTO_OPTION,
                   adapters: Optional[Sequence[GpuAdapter]] = None,
                   session_options=None):
    """Build an inference session on the selected GPU.

    Returns ``(session, resolved)``.  Callers should check
    ``session.get_providers()`` and ``resolved.device_id`` to know whether the
    model really landed on the GPU: ONNX Runtime silently falls back to CPU
    when a provider cannot be initialized.
    """
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is not installed; install onnxruntime-directml to use the GPU"
        ) from exc

    if adapters is None:
        adapters = enumerate_dxgi_adapters()
    resolved = resolve(choice, adapters)

    options = session_options if session_options is not None else ort.SessionOptions()
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=provider_list(resolved.device_id))
    return session, resolved


def onnxruntime_providers() -> Optional[List[str]]:
    """Available ORT providers, or None when onnxruntime is not installed."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    try:
        return list(ort.get_available_providers())
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# DXGI enumeration (Windows only, stdlib only)
# --------------------------------------------------------------------------------------

_HR = ctypes.c_long
_SUCCESS = 0


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]


class _DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


def _guid(value: str) -> _GUID:
    import uuid

    return _GUID.from_buffer_copy(uuid.UUID(value).bytes_le)


def _vtable_call(ptr, index: int, restype, *argtypes):
    """Resolve a COM interface method by vtable index."""
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    func = ctypes.cast(vtable + 8 * index, ctypes.POINTER(ctypes.c_void_p))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(func)


def _mb(size: int) -> int:
    return int(size) // (1024 * 1024)


def enumerate_dxgi_adapters() -> List[GpuAdapter]:
    """Return every DXGI adapter in the order DirectML sees them.

    Returns an empty list when the platform or DLLs do not allow enumeration;
    callers treat that as "no GPU available" and fall back to CPU.
    """
    if sys.platform != "win32":
        return []

    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = [
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create_factory.restype = _HR
    except OSError:
        return []

    factory = ctypes.c_void_p()
    if create_factory(
        ctypes.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")),
        ctypes.byref(factory),
    ) != _SUCCESS:
        return []

    try:
        enum_adapters = _vtable_call(
            factory, 12, _HR, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        release_factory = _vtable_call(factory, 2, ctypes.c_ulong)

        try:
            d3d12 = ctypes.WinDLL("d3d12.dll")
            create_device = d3d12.D3D12CreateDevice
            create_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            create_device.restype = _HR
            iid_device = _guid("189819f1-1db6-4b57-be54-1821339b85f7")
        except OSError:
            create_device = None
            iid_device = None

        adapters: List[GpuAdapter] = []
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, index, ctypes.byref(adapter)) != _SUCCESS:
                break

            desc = _DESC1()
            get_desc = _vtable_call(adapter, 10, _HR, ctypes.POINTER(_DESC1))
            if get_desc(adapter, ctypes.byref(desc)) != _SUCCESS:
                _vtable_call(adapter, 2, ctypes.c_ulong)(adapter)
                index += 1
                continue

            release_adapter = _vtable_call(adapter, 2, ctypes.c_ulong)
            levels: List[str] = []
            if create_device is not None:
                for level, label in ((0xC000, "12_0"), (0xB000, "11_0")):
                    device = ctypes.c_void_p()
                    if (
                        create_device(
                            adapter, level, ctypes.byref(iid_device), ctypes.byref(device)
                        )
                        == _SUCCESS
                    ):
                        levels.append(label)
                        _vtable_call(device, 2, ctypes.c_ulong)(device)

            adapters.append(
                GpuAdapter(
                    device_id=index,
                    name=desc.Description.strip(),
                    vendor_id=int(desc.VendorId),
                    pci_device_id=int(desc.DeviceId),
                    subsys_id=int(desc.SubSysId),
                    revision=int(desc.Revision),
                    luid=(int(desc.AdapterLuid.HighPart), int(desc.AdapterLuid.LowPart)),
                    ded_vram_mb=_mb(desc.DedicatedVideoMemory),
                    dedicated_system_mb=_mb(desc.DedicatedSystemMemory),
                    shared_mb=_mb(desc.SharedSystemMemory),
                    is_software=bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                    d3d12_levels=tuple(levels),
                )
            )
            release_adapter(adapter)
            index += 1

        release_factory(factory)
        return adapters
    except Exception:
        try:
            _vtable_call(factory, 2, ctypes.c_ulong)(factory)
        except Exception:
            pass
        return []


def scan_adapters() -> List[GpuAdapter]:
    """Convenience: enumerate then apply the selection policy."""
    return select_adapters(enumerate_dxgi_adapters())


def main(argv: Optional[Sequence[str]] = None) -> int:
    all_adapters = enumerate_dxgi_adapters()
    selected = select_adapters(all_adapters)

    print("=" * 84)
    print("DXGI adapters (DirectML device_id space)")
    print("=" * 84)
    print("%-4s %-10s %-46s %9s %8s %-8s %s" % ("id", "vendor", "name", "dedVRAM", "shared", "software", "D3D12"))
    print("-" * 84)
    for a in all_adapters:
        print("%-4d %-10s %-46s %6dMB %5dMB %-8s %s" % (
            a.device_id, a.vendor, a.name[:46], a.ded_vram_mb, a.shared_mb,
            a.is_software, ",".join(a.d3d12_levels) or "-"))

    print()
    print("Selectable (filtered + de-duplicated, priority order)")
    if not selected:
        print("  <none>")
    for a in selected:
        print("  device_id=%-2d %s" % (a.device_id, a.label))

    print()
    print("Dropdown options: %s" % " | ".join(build_options(all_adapters)))

    for choice in (AUTO_OPTION, CPU_OPTION, "stale-device-label"):
        resolved = resolve(choice, all_adapters)
        print("  resolve(%-14r) -> device_id=%s (%s) [%s]" % (
            choice, resolved.device_id, resolved.resolved_label, resolved.reason))

    providers = onnxruntime_providers()
    print()
    print("onnxruntime providers: %s" % (providers if providers is not None else "onnxruntime not installed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
