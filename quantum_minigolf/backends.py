#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np

# ---------- Make CUDA DLLs discoverable on Windows & preload NVRTC ----------
_DLL_DIR_HANDLES = []
def _add_cuda_dirs_and_preload():
    """On Windows, add CUDA wheel bin dirs to DLL search path and preload NVRTC
    so CuPy JIT works reliably (prevents 'failed to open nvrtc-builtins...').
    No-op on non-Windows.
    """
    if os.name != "nt":
        return

    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"

    # Add the common CUDA wheel bin folders to the DLL search path
    for sub in (
        "cuda_nvrtc", "cuda_runtime", "cufft", "curand", "nvjitlink",
        "cublas", "cusolver", "cusparse", "npp"
    ):
        d = base / sub / "bin"
        if d.is_dir():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(d)))

    # Preload NVRTC and its builtins explicitly.
    # Try known CUDA 12.9 filenames first, then fall back to any matching nvrtc*.dll
    try:
        import ctypes
        bin_dir = base / "cuda_nvrtc" / "bin"
        candidates = ["nvrtc64_129_0.dll", "nvrtc-builtins64_129.dll"]

        # If versions differ, try best-effort discovery
        if bin_dir.is_dir():
            for f in bin_dir.iterdir():
                name = f.name.lower()
                if name.startswith("nvrtc64_") or name.startswith("nvrtc-builtins64_"):
                    if f.name not in candidates:
                        candidates.append(f.name)

        for lib in candidates:
            p = bin_dir / lib
            if p.is_file():
                ctypes.WinDLL(str(p))
    except Exception as e:
        print(f"[warn] NVRTC preload failed: {e}")

try:
    _add_cuda_dirs_and_preload()
except Exception:
    # Never hard-fail if this convenience step has issues
    pass
# ---------------------------------------------------------------------------

# Optional fast CPU blur (used by physics)
try:
    from scipy.ndimage import uniform_filter as _scipy_uniform_filter  # noqa: F401
except Exception:
    _scipy_uniform_filter = None  # noqa: F401


class Backend:
    def __init__(self):
        self.USE_GPU: bool = False
        self.backend_name: str = "NumPy"
        self.xp = np
        self.cp = None  # type: ignore
        self.fft2 = None
        self.ifft2 = None
        self._gpu_fail_reason = None
        self._select_backend()
        print(f"Using GPU: {self.USE_GPU}  |  Backend: {self.backend_name}")
        if not self.USE_GPU and self._gpu_fail_reason:
            print("GPU selection failed because:", self._gpu_fail_reason)

    def _select_backend(self):
        # 1) Try CuPy (must pass alloc + cuFFT + NVRTC/JIT)
        try:
            import cupy as cp  # type: ignore

            # Allocation sanity
            _ = cp.zeros((1,), dtype=cp.float32)

            # cuFFT sanity
            _ = cp.fft.fft2(cp.zeros((8, 8), dtype=cp.float32))

            # NVRTC/JIT sanity (forces nvrtc + builtins usage)
            cp.ElementwiseKernel(
                'float32 x', 'float32 y', 'y = x', name='__jit_probe__'
            )(cp.array([0.0], dtype=cp.float32))

            # Success -> select GPU
            self.USE_GPU = True
            self.backend_name = "CuPy (GPU)"
            self.xp = cp
            self.cp = cp
            self.fft2 = cp.fft.fft2
            self.ifft2 = cp.fft.ifft2
            return
        except Exception as e:
            self._gpu_fail_reason = repr(e)
            self.USE_GPU = False
            self.backend_name = "CPU"
            self.cp = None

        # 2) PyFFTW (CPU, fast)
        try:
            import pyfftw  # type: ignore
            from pyfftw.interfaces.numpy_fft import fft2 as _fftw_fft2, ifft2 as _fftw_ifft2
            pyfftw.interfaces.cache.enable()
            _threads = max(1, (os.cpu_count() or 4) - 1)

            def fft2(a):   return _fftw_fft2(a, threads=_threads)
            def ifft2(a):  return _fftw_ifft2(a, threads=_threads)

            self.fft2, self.ifft2 = fft2, ifft2
            self.backend_name = f"pyFFTW (CPU, threads={_threads})"
            return
        except Exception:
            pass

        # 3) SciPy FFT (CPU)
        try:
            from scipy.fft import fft2 as sp_fft2, ifft2 as sp_ifft2  # type: ignore
            _workers = max(1, (os.cpu_count() or 4) - 1)

            def fft2(a):   return sp_fft2(a, workers=_workers, overwrite_x=True)
            def ifft2(a):  return sp_ifft2(a, workers=_workers, overwrite_x=True)

            self.fft2, self.ifft2 = fft2, ifft2
            self.backend_name = f"SciPy FFT (CPU, workers={_workers})"
            return
        except Exception:
            pass

        # 4) NumPy fallback
        from numpy.fft import fft2 as np_fft2, ifft2 as np_ifft2
        self.fft2, self.ifft2 = np_fft2, np_ifft2
        self.backend_name = "NumPy FFT (CPU)"

    # ----------------------------- helpers -----------------------------
    def to_xp(self, arr, dtype=None):
        if self.USE_GPU:
            return self.cp.asarray(arr, dtype=dtype) if dtype is not None else self.cp.asarray(arr)
        return self.xp.asarray(arr, dtype=dtype) if dtype is not None else self.xp.asarray(arr)

    def to_cpu(self, arr_xp, out=None):
        """Transfer an array to CPU memory, reusing an output buffer when possible."""
        if self.USE_GPU:
            if hasattr(arr_xp, "get"):
                return arr_xp.get(out=out)
            # Fallback: CuPy not exposing get (unexpected) -> force numpy conversion
            result = self.cp.asnumpy(arr_xp)
            if out is not None:
                np.copyto(out, result, casting='unsafe')
                return out
            return result
        if out is not None:
            np.copyto(out, arr_xp, casting='unsafe')
            return out
        return arr_xp

    def build_kgrid(self, Nx, Ny, dx, dy):
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx).astype(np.float32)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy).astype(np.float32)
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        k2 = (KX * KX + KY * KY).astype(np.float32)
        k_max = np.array(0.8 * np.pi / max(dx, dy), dtype=np.float32)
        return self.to_xp(k2, np.float32), self.to_xp(k_max, np.float32)


if __name__ == "__main__":
    backend = Backend()
    print("USE_GPU =", backend.USE_GPU)
