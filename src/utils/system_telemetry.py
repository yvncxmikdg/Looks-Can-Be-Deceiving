"""Host RAM and GPU VRAM telemetry helpers for training.

These are intentionally dependency-light: memory leaks in this pipeline live in the DataLoader
*worker* processes, so the telemetry deliberately reports three separate host-RAM signals that,
read together, localize a leak:

  * ``system_used_gb`` -- node-wide RAM in use (rises if anything on the node grows),
  * ``process_rss_gb`` -- the training (rank) process alone, and
  * ``dataloader_workers_rss_gb`` -- the sum of that process's DataLoader worker children.

A worker-side leak (e.g. copy-on-write refcount amplification across forked workers) shows up as
``dataloader_workers_rss_gb`` and ``system_used_gb`` climbing while ``process_rss_gb`` stays flat
-- which the main-process RSS alone would hide. All of this is read straight from the OS
(``/proc`` on Linux, Win32 APIs on Windows), so it needs no third-party packages; ``psutil`` is
used when present but nothing depends on it.

GPU numbers come from ``torch.cuda`` and, where available, ``torch.cuda.mem_get_info`` (which
also captures allocations made outside PyTorch's caching allocator).
"""
import os
import mmap
import ctypes

import torch

try:
    import psutil  # optional; enriches host telemetry with a process-tree RSS breakdown
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_PSUTIL = False

_BYTES_PER_GB = 1024 ** 3


def _linux_system_memory_bytes():
    """(total, available) host memory in bytes from /proc/meminfo, or None off Linux."""
    try:
        info = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                key, _, rest = line.partition(":")
                info[key.strip()] = int(rest.strip().split()[0]) * 1024  # values are in kB
        total = info.get("MemTotal")
        # MemAvailable is the kernel's own estimate of allocatable memory; fall back to MemFree.
        available = info.get("MemAvailable", info.get("MemFree"))
        if total is None or available is None:
            return None
        return total, available
    except (OSError, ValueError, IndexError):
        return None


def _linux_rss_bytes(pid):
    """Resident set size of ``pid`` in bytes from /proc/<pid>/statm, or None."""
    try:
        with open(f"/proc/{pid}/statm", "r", encoding="utf-8") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * mmap.PAGESIZE  # kernel page size; matches statm's page unit
    except (OSError, ValueError, IndexError):
        return None


def _linux_process_rss_bytes():
    """Resident set size of the current process in bytes, or None off Linux."""
    return _linux_rss_bytes("self")


def _ppid_from_proc_stat(stat):
    """Parse the parent PID out of a /proc/<pid>/stat string.

    comm (field 2) is wrapped in parens and may itself contain spaces and parens, so the fields
    after it are located by splitting past the final ')': they are state, ppid, ...
    """
    after_comm = stat[stat.rindex(")") + 1:].split()
    return int(after_comm[1])


def _linux_children_rss_bytes():
    """(summed RSS bytes, count) of this process's direct children, or None off Linux.

    DataLoader workers are direct children of the process that builds the loader, so this
    isolates worker-side growth from the training process's own RSS.
    """
    try:
        my_pid = os.getpid()
        total = 0
        count = 0
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "r", encoding="utf-8") as stat_file:
                    ppid = _ppid_from_proc_stat(stat_file.read())
            except (OSError, ValueError, IndexError):
                continue
            if ppid != my_pid:
                continue
            child_rss = _linux_rss_bytes(entry)
            if child_rss is not None:
                total += child_rss
                count += 1
        return total, count
    except OSError:
        return None


class _MEMORYSTATUSEX(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_system_memory_bytes():
    """(total, available) host memory in bytes via GlobalMemoryStatusEx, or None off Windows."""
    try:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)  # pylint: disable=attribute-defined-outside-init
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullTotalPhys, status.ullAvailPhys
    except (AttributeError, OSError):
        return None


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_process_rss_bytes():
    """Working set size (RSS-equivalent) of the current process in bytes, or None off Windows."""
    try:
        # Declare prototypes: GetCurrentProcess returns a HANDLE (a pointer-width pseudo-handle,
        # -1). Without an explicit restype ctypes treats it as a 32-bit int and truncates it on
        # 64-bit Windows, so the subsequent call fails.
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        get_process_memory_info.restype = ctypes.c_int

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)  # pylint: disable=attribute-defined-outside-init
        if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize
    except (AttributeError, OSError):
        return None


def get_host_memory_stats():
    """Instantaneous host-RAM gauges (GB), keyed for grouping in TensorBoard.

    Never raises: telemetry must not be able to take down a training run.
    """
    stats = {}
    try:
        total = available = None
        process_rss = None
        workers_rss = None
        num_children = None

        if _HAS_PSUTIL:
            virtual_memory = psutil.virtual_memory()
            total, available = virtual_memory.total, virtual_memory.available
            try:
                proc = psutil.Process()
                process_rss = proc.memory_info().rss
                children = proc.children(recursive=True)
                # DataLoader workers are children; splitting their RSS from the training
                # process's own RSS is what localizes a worker-side leak.
                workers_rss = sum(c.memory_info().rss for c in children)
                num_children = float(len(children))
            except psutil.Error:
                pass
        else:
            sys_mem = _linux_system_memory_bytes() or _windows_system_memory_bytes()
            if sys_mem is not None:
                total, available = sys_mem
            process_rss = _linux_process_rss_bytes() or _windows_process_rss_bytes()
            children = _linux_children_rss_bytes()  # (summed_rss, count); None off Linux
            if children is not None:
                workers_rss, num_children = children[0], float(children[1])

        if total is not None and available is not None:
            stats["system_used_gb"] = (total - available) / _BYTES_PER_GB
            stats["system_available_gb"] = available / _BYTES_PER_GB
            stats["system_total_gb"] = total / _BYTES_PER_GB
        if process_rss is not None:
            stats["process_rss_gb"] = process_rss / _BYTES_PER_GB
        if workers_rss is not None:
            stats["dataloader_workers_rss_gb"] = workers_rss / _BYTES_PER_GB
        if num_children is not None:
            stats["num_dataloader_workers"] = num_children
    except Exception:  # pylint: disable=broad-except
        pass
    return stats


class _Mallinfo2(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


_MALLINFO2 = None  # cached glibc mallinfo2 pointer; False when unavailable (Windows / old glibc)


def _load_mallinfo2():
    global _MALLINFO2  # pylint: disable=global-statement
    if _MALLINFO2 is None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            mallinfo2 = libc.mallinfo2  # AttributeError on glibc < 2.33
            mallinfo2.restype = _Mallinfo2
            _MALLINFO2 = mallinfo2
        except (OSError, AttributeError):
            _MALLINFO2 = False
    return _MALLINFO2


def get_worker_heap_stats():
    """Allocator-level gauges (GB) for the CURRENT process; call from inside a DataLoader worker.

    The point of these is to discriminate a genuine native leak from allocator retention when a
    worker's RSS climbs: ``heap_in_use_gb`` (glibc mallinfo2 uordblks) rising means memory is
    truly not being freed (a real leak); ``heap_retained_free_gb`` (fordblks) rising while in-use
    stays flat means the bytes WERE freed but glibc is holding them (arena fragmentation /
    mmap-threshold retention). ``rss_anon_gb`` is the kernel's view of the process's anonymous
    resident memory for cross-checking. Linux-only; returns {} elsewhere. Never raises.
    """
    stats = {}
    try:
        mallinfo2 = _load_mallinfo2()
        if mallinfo2:
            info = mallinfo2()
            stats["heap_in_use_gb"] = info.uordblks / _BYTES_PER_GB
            stats["heap_retained_free_gb"] = info.fordblks / _BYTES_PER_GB
            stats["heap_arena_gb"] = info.arena / _BYTES_PER_GB
            stats["heap_mmapped_gb"] = info.hblkhd / _BYTES_PER_GB
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as status:
                for line in status:
                    if line.startswith("RssAnon:"):
                        stats["rss_anon_gb"] = int(line.split()[1]) * 1024 / _BYTES_PER_GB
                    elif line.startswith("VmRSS:"):
                        stats["rss_gb"] = int(line.split()[1]) * 1024 / _BYTES_PER_GB
        except (OSError, ValueError, IndexError):
            pass
    except Exception:  # pylint: disable=broad-except
        pass
    return stats


def get_gpu_memory_stats(device=None):
    """Instantaneous GPU-VRAM gauges (GB) for ``device``, keyed for grouping in TensorBoard.

    Never raises. Returns an empty dict when CUDA is unavailable.
    """
    stats = {}
    try:
        if not torch.cuda.is_available():
            return stats
        # Normalize to a concrete device ordinal. A bare torch.device("cuda") (no index) is
        # rejected by mem_get_info, and a non-cuda device has no VRAM to report.
        if isinstance(device, torch.device):
            index = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        elif isinstance(device, int):
            index = device
        else:
            index = torch.cuda.current_device()

        stats["torch_allocated_gb"] = torch.cuda.memory_allocated(index) / _BYTES_PER_GB
        stats["torch_reserved_gb"] = torch.cuda.memory_reserved(index) / _BYTES_PER_GB
        stats["torch_max_reserved_gb"] = torch.cuda.max_memory_reserved(index) / _BYTES_PER_GB

        try:
            # Whole-device view: catches VRAM held outside PyTorch's caching allocator too.
            free, total = torch.cuda.mem_get_info(index)
            stats["device_used_gb"] = (total - free) / _BYTES_PER_GB
            stats["device_free_gb"] = free / _BYTES_PER_GB
        except (RuntimeError, AssertionError):
            pass
    except Exception:  # pylint: disable=broad-except
        pass
    return stats
