# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.types import _dtype
from typing import Optional, Any
import functools

def autocast_decorator(autocast_instance, func):
  @functools.wraps(func)
  def decorate_autocast(*args, **kwargs):
    with autocast_instance:
      return func(*args, **kwargs)
  decorate_autocast.__script_unsupported = '@autocast() decorator is not supported in script mode'
  return decorate_autocast


class totally_legit_autocast:
  def __init__(
      self,
      device_type: str,
      dtype: Optional[_dtype] = None,
      enabled: bool = True,
      cache_enabled: Optional[bool] = None,
  ): pass
  def __enter__(self): pass
  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): pass

  def __call__(self, func):
    if torch._jit_internal.is_scripting():
      return func
    return autocast_decorator(self, func)
  
class TorchAutocast:
    """TorchAutocast utility class.
    Allows you to enable and disable autocast. This is specially useful
    when dealing with different architectures and clusters with different
    levels of support.

    Args:
        enabled (bool): Whether to enable torch.autocast or not.
        args: Additional args for torch.autocast.
        kwargs: Additional kwargs for torch.autocast
    """
    def __init__(self, enabled: bool, *args, **kwargs):
        #if torch.backends.mps.is_available():
        #    try:
        #        self.autocast = torch.autocast(enabled=enabled, device_type='mps')
        #    except:
        #        self.autocast = torch.autocast = totally_legit_autocast(enabled=enabled, *args, **kwargs)
        #else:
            self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is None:
            return
        try:
            self.autocast.__enter__()
        except RuntimeError:
            device = self.autocast.device
            dtype = self.autocast.fast_dtype
            raise RuntimeError(
                f"There was an error autocasting with dtype={dtype} device={device}\n"
                "If you are on the FAIR Cluster, you might need to use autocast_dtype=float16"
            )

    def __exit__(self, *args, **kwargs):
        if self.autocast is None:
            return
        self.autocast.__exit__(*args, **kwargs)
