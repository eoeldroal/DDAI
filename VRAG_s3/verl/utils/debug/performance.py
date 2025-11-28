# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#추가 gpu 사용량 측정# 수정
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
from typing import Dict, Optional, Union
import contextlib
#######수정

import torch
import torch.distributed as dist
#import logging 제거 gpu 사용량 측정 수정


DeviceSpecifier = Union[torch.device, str, int]
PathSpecifier = Union[str, Path]

__all__ = [
    "GPUMemoryStats",
    "gpu_memory_phase",
    "log_gpu_memory_usage",
    "phase",
]


#수정, 추가 gpu 사용량 측정 
@dataclass
class GPUMemoryStats:
    """Snapshot of GPU memory statistics collected for a context block."""

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    peak_delta_bytes: int

    def to_dict(self) -> Dict[str, int]:
        """Return the raw statistics as a JSON-serializable dictionary."""

        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "allocated_delta_bytes": self.allocated_delta_bytes,
            "reserved_delta_bytes": self.reserved_delta_bytes,
            "peak_delta_bytes": self.peak_delta_bytes,
        }


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def _format_delta(value: int) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value) / (1024 ** 3):.2f} GiB"


# def _resolve_device(device: Optional[DeviceSpecifier]) -> torch.device:
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is required to track GPU memory usage.")

#     if device is None:
#         index = torch.cuda.current_device()
#         return torch.device("cuda", index)

#     if isinstance(device, int):
#         return torch.device("cuda", device)

#     return torch.device(device)

def _resolve_device(device: Optional[DeviceSpecifier]) -> torch.device:
    """
    Best-effort device resolver:
    - CUDA가 없으면 에러 대신 CPU로 폴백
    - device=None이면: cuda 가능 시 current cuda, 아니면 cpu
    - device가 'cuda:*'인데 CUDA 불가면 cpu 폴백
    """
    try:
        if torch.cuda.is_available():
            if device is None:
                index = torch.cuda.current_device()
                return torch.device("cuda", index)
            if isinstance(device, int):
                return torch.device("cuda", device)
            dev = torch.device(device) if device is not None else torch.device("cuda")
            # 'cuda'로 요청했지만 가용 안 되면 cpu 폴백
            if dev.type == "cuda":
                return dev
            return dev
        else:
            return torch.device("cpu")
    except Exception:
        # torch/device 해석 실패 등 어떤 경우든 최후 폴백
        return torch.device("cpu")


class _GpuMemoryPhase:
    """Internal context manager implementation that tracks GPU memory usage."""

    def __init__(
            self,
            name: str,
            device: Optional[DeviceSpecifier] = None,
            logger: Optional[logging.Logger] = None,
            level: int = logging.INFO,
            rank: Optional[int] = 0,
            synchronize: bool = True,
            reset_peak_stats: bool = True,
            json_path: Optional[PathSpecifier] = None,
    ) -> None:
        self._name = name
        self._logger = logger
        self._level = level
        self._rank = rank
        self._synchronize = synchronize
        self._reset_peak_stats = reset_peak_stats
        self._device_spec = device
        self._json_path = Path(json_path).expanduser() if json_path is not None else None
        self._device: Optional[torch.device] = None
        self.stats: Optional[GPUMemoryStats] = None
        self._start_allocated: Optional[int] = None
        self._start_reserved: Optional[int] = None
        self._should_log = True
        self._track_gpu = False

    def __enter__(self) -> "_GpuMemoryPhase":
        # self._device = _resolve_device(self._device_spec)

        # if self._synchronize:
        #     torch.cuda.synchronize(self._device)

        # if self._reset_peak_stats:
        #     torch.cuda.reset_peak_memory_stats(self._device)

        # self._start_allocated = torch.cuda.memory_allocated(self._device)
        # self._start_reserved = torch.cuda.memory_reserved(self._device)
        self._device = _resolve_device(self._device_spec)
        self._track_gpu = (getattr(self._device, "type", None) == "cuda") and torch.cuda.is_available()

        if self._track_gpu:
            if self._synchronize:
                torch.cuda.synchronize(self._device)
            if self._reset_peak_stats:
                torch.cuda.reset_peak_memory_stats(self._device)
            self._start_allocated = torch.cuda.memory_allocated(self._device)
            self._start_reserved = torch.cuda.memory_reserved(self._device)
        else:
            logging.warning("CUDA not available in this process; skipping GPU memory tracking for phase '%s'.", self._name)
            self._start_allocated = 0
            self._start_reserved = 0

        self._should_log = (not dist.is_initialized()) or (self._rank is None) or (dist.get_rank() == self._rank)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # assert self._device is not None

        # if self._synchronize:
        #     torch.cuda.synchronize(self._device)

        # allocated = torch.cuda.memory_allocated(self._device)
        # reserved = torch.cuda.memory_reserved(self._device)
        # peak_allocated = torch.cuda.max_memory_allocated(self._device)

        # assert self._start_allocated is not None
        # assert self._start_reserved is not None

        # self.stats = GPUMemoryStats(
        #     allocated_bytes=allocated,
        #     reserved_bytes=reserved,
        #     peak_allocated_bytes=peak_allocated,
        #     allocated_delta_bytes=allocated - self._start_allocated,
        #     reserved_delta_bytes=reserved - self._start_reserved,
        #     peak_delta_bytes=peak_allocated - self._start_allocated,
        # )

        # if self._should_log:
        #     message = self._format_message()
        #     if self._logger is None:
        #         print(message)
        #     else:
        #         self._logger.log(level=self._level, msg=message)

        # if self._should_log:
        #     self._write_json_record()
        assert self._device is not None

        if self._track_gpu:
            if self._synchronize:
                torch.cuda.synchronize(self._device)
            allocated = torch.cuda.memory_allocated(self._device)
            reserved = torch.cuda.memory_reserved(self._device)
            peak_allocated = torch.cuda.max_memory_allocated(self._device)

            assert self._start_allocated is not None
            assert self._start_reserved is not None

            self.stats = GPUMemoryStats(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                peak_allocated_bytes=peak_allocated,
                allocated_delta_bytes=allocated - self._start_allocated,
                reserved_delta_bytes=reserved - self._start_reserved,
                peak_delta_bytes=peak_allocated - self._start_allocated,
            )

            if self._should_log:
                message = self._format_message()
                if self._logger is None:
                    print(message)
                else:
                    self._logger.log(level=self._level, msg=message)
            if self._should_log:
                self._write_json_record()
        else:
            # GPU 추적 불가 시에도 타임라인 구멍 방지용 스텁 레코드 남김
            if self._should_log and self._json_path is not None:
                try:
                    record = {
                        "phase": self._name,
                        "device": "cpu",
                        "stats": None,
                        "track_gpu": False,
                        "rank": dist.get_rank() if dist.is_initialized() else None,
                    }
                    self._json_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._json_path.open("a", encoding="utf-8") as file:
                        json.dump(record, file, ensure_ascii=False)
                        file.write("\n")
                except Exception:
                    pass

        return False

    def _format_message(self) -> str:
        assert self._device is not None
        assert self.stats is not None

        return (
            f"[phase: {self._name}] {self._device} | "
            f"allocated={_format_bytes(self.stats.allocated_bytes)} ({_format_delta(self.stats.allocated_delta_bytes)}) | "
            f"reserved={_format_bytes(self.stats.reserved_bytes)} ({_format_delta(self.stats.reserved_delta_bytes)}) | "
            f"peak_allocated={_format_bytes(self.stats.peak_allocated_bytes)} ({_format_delta(self.stats.peak_delta_bytes)})"
        )

    def _write_json_record(self) -> None:
        if self._json_path is None:
            return

        assert self._device is not None
        assert self.stats is not None

        record = {
            "phase": self._name,
            "device": str(self._device),
            "stats": {
                **self.stats.to_dict(),
                "allocated_gib": self.stats.allocated_bytes / (1024 ** 3),
                "reserved_gib": self.stats.reserved_bytes / (1024 ** 3),
                "peak_allocated_gib": self.stats.peak_allocated_bytes / (1024 ** 3),
                "allocated_delta_gib": self.stats.allocated_delta_bytes / (1024 ** 3),
                "reserved_delta_gib": self.stats.reserved_delta_bytes / (1024 ** 3),
                "peak_delta_gib": self.stats.peak_delta_bytes / (1024 ** 3),
            },
            "rank": dist.get_rank() if dist.is_initialized() else None,
        }

        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        with self._json_path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def gpu_memory_phase(
        name: str,
        *,
        device: Optional[DeviceSpecifier] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        rank: Optional[int] = 0,
        synchronize: bool = True,
        reset_peak_stats: bool = True,
        json_path: Optional[PathSpecifier] = None,
) -> _GpuMemoryPhase:
    """Context manager that measures GPU memory usage for a specific training phase.

    Args:
        name: Identifier for the profiled phase that will be included in logs and JSON records.
        device: Target CUDA device. Defaults to the current device.
        logger: Optional logger instance. When omitted, messages are printed to stdout.
        level: Logging level used when ``logger`` is provided.
        rank: Only emit logs/JSON records on this rank when :mod:`torch.distributed` is initialized.
        synchronize: Whether to synchronize the device before collecting metrics.
        reset_peak_stats: Reset peak stats on entry so ``peak_allocated_bytes`` reflect the block.
        json_path: Optional path to append JSONL records with the collected statistics.

    Returns:
        A context manager that records the memory statistics for the wrapped block.
    """

    return _GpuMemoryPhase(
        name=name,
        device=device,
        logger=logger,
        level=level,
        rank=rank,
        synchronize=synchronize,
        reset_peak_stats=reset_peak_stats,
        json_path=json_path,
    )


def phase(
        name: str,
        *,
        device: Optional[DeviceSpecifier] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        rank: Optional[int] = 0,
        synchronize: bool = True,
        reset_peak_stats: bool = True,
        json_path: Optional[PathSpecifier] = None,
) -> _GpuMemoryPhase:
    """Alias for :func:`gpu_memory_phase` that supports ``with phase("rollout")`` syntax.

    All keyword arguments accepted by :func:`gpu_memory_phase` are forwarded transparently.

    Example:
        >>> from verl.utils.debug import phase
        >>> with phase("rollout", device="cuda:0", json_path="metrics.jsonl"):
        ...     perform_rollout()
        >>> with phase("backprop", device="cuda:0", json_path="metrics.jsonl"):
        ...     loss.backward()
    """

    return gpu_memory_phase(
        name=name,
        device=device,
        logger=logger,
        level=level,
        rank=rank,
        synchronize=synchronize,
        reset_peak_stats=reset_peak_stats,
        json_path=json_path,
    )
#gpu 사용량 측정 추가 수정

def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    # if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
    #     memory_allocated = torch.cuda.memory_allocated() / 1024**3
    #     memory_reserved = torch.cuda.memory_reserved() / 1024**3

    #     message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

    #     if logger is None:
    #         print(message)
    #     else:
    #         logger.log(msg=message, level=level)
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'
        else:
            message = f'{head}, CUDA not available; skipping GPU memory usage logging.'
        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
