"""Allow the legacy NumPy control and observation adapters to use the GPU PhysX tensor API."""
import numpy as np

class DeviceArticulation:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        method = getattr(self.native, name)
        if name.startswith("get_") or name.startswith("set_"):
            import torch
            def device(x):
                if isinstance(x, np.ndarray):
                    return torch.as_tensor(x, device="cuda", dtype=torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32)
                return x
            def host(x):
                if isinstance(x, tuple):
                    return tuple(host(v) for v in x)
                return x.detach().cpu().numpy() if torch.is_tensor(x) else x
            def wrapped(*args, **kw):
                return host(method(*(device(v) for v in args),**{k:device(v) for k,v in kw.items()}))
            return wrapped
        return method
