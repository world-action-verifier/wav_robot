"""Bounded CPU batch transport through anonymous RAM fds, not /dev/shm.

The dataset and sampling logic are untouched. Only collation/IPC/pinning change.
Linux workers own one fd per queued batch; the receiver closes it after copying
into normal or pinned torch storage. No batch is stored on the workspace disk.
"""
import mmap
import ctypes
import os
from multiprocessing.reduction import DupFd
import numpy as np
import torch
from torch.utils.data import DataLoader

def create_memfd():
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('mmbench2-batch', getattr(os, 'MFD_CLOEXEC', 1))
    # Some Python builds omit the Python wrapper while the host
    # libc/kernel still provide memfd_create. No elevated capability is needed.
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(b'mmbench2-batch', 1)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd

class MemfdBatch:
    def __init__(self, fd, specs, nbytes):
        self.fd, self.specs, self.nbytes = fd, specs, nbytes
    def __reduce__(self):
        return (_restore, (DupFd(self.fd), self.specs, self.nbytes))
    def close(self):
        fd, self.fd = self.fd, -1
        if fd >= 0:
            os.close(fd)
    def __del__(self):
        if getattr(self, 'fd', -1) >= 0:
            self.close()
    def materialize(self, pin=False):
        result = {}
        try:
            with mmap.mmap(self.fd, self.nbytes, access=mmap.ACCESS_WRITE) as buf:
                for key, shape, dtype, offset in self.specs:
                    array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=buf, offset=offset)
                    view = torch.from_numpy(array)
                    result[key] = view.pin_memory() if pin else view.clone()
                    del view, array
            return result
        finally:
            self.close()
    def pin_memory(self):
        # DataLoader's dedicated pin-memory thread calls this custom-batch API.
        return self.materialize(pin=True)

def _restore(shared_fd, specs, nbytes):
    return MemfdBatch(shared_fd.detach(), specs, nbytes)

def collate_memfd(samples):
    if not samples:
        raise ValueError('Empty batch')
    specs = []
    offset = 0
    for key, value in samples[0].items():
        a = np.asarray(value)
        shape = (len(samples), *a.shape)
        offset = (offset + 63) // 64 * 64
        specs.append((key, shape, a.dtype.str, offset))
        offset += int(np.prod(shape)) * a.dtype.itemsize
    fd = create_memfd()
    try:
        os.ftruncate(fd, offset)
        with mmap.mmap(fd, offset, access=mmap.ACCESS_WRITE) as buf:
            for key, shape, dtype, start in specs:
                target = np.ndarray(shape, dtype=np.dtype(dtype), buffer=buf, offset=start)
                for i, sample in enumerate(samples):
                    a = np.asarray(sample[key])
                    if a.shape != shape[1:] or a.dtype != target.dtype:
                        raise ValueError('Batch shapes/dtypes differ: ' + key)
                    np.copyto(target[i:i+1], a.reshape((1, *a.shape)), casting='no')
                del target, a
        return MemfdBatch(fd, specs, offset)
    except BaseException:
        os.close(fd)
        raise

class MemfdDataLoader:
    def __init__(self, dataset, **kwargs):
        if kwargs.get('num_workers', 0) < 1:
            raise ValueError('Memfd transport is for worker prefetch')
        kwargs['collate_fn'] = collate_memfd
        kwargs['prefetch_factor'] = 1
        kwargs['persistent_workers'] = True
        self.loader = DataLoader(dataset, **kwargs)
    def __iter__(self):
        for batch in self.loader:
            yield batch.materialize(pin=False) if isinstance(batch, MemfdBatch) else batch
    def __len__(self):
        return len(self.loader)
