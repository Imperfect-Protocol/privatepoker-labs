"""Handle-based asynchronous tensor engine with a single device-owning worker.

The public API is deliberately tiny:

    h = engine.create_tensor()
    engine.store(h, (2, 2), [[1, 2], [3, 4]])   # enqueue upload; does not fence GPU
    doubled = engine.queue_op(multiply, h, h)  # enqueue operation; returns handle
    result = engine.load(doubled, (2, 2))      # fence: wait and copy to host

Only ``load`` explicitly waits for queued work and synchronizes an accelerator.
``store`` snapshots the caller's host data and queues the upload; it never waits
for earlier GPU work. One background thread owns backend tensor objects, and all
commands execute in FIFO order. This avoids exposing GPU tensors to the event loop
and keeps intermediate tensors resident on the selected device.

PyTorch is optional for the NumPy CPU backend. MPS/CUDA backends require PyTorch.
"""
from __future__ import annotations

import operator
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional unless an accelerator is requested.
    import torch
except Exception:  # pragma: no cover - exercised on minimal CPU installations
    torch = None  # type: ignore[assignment]


Shape = Optional[Tuple[int, ...]]
Operation = Union[str, Callable[..., Any]]


class TensorEngineError(RuntimeError):
    """Raised when an asynchronous tensor command fails."""


class TensorHandleError(TensorEngineError):
    """Raised for invalid or unknown tensor handles."""


# Callable symbols for the ergonomic API shown in the design proposal.
# The operation itself is evaluated only by the engine thread.
multiply = operator.mul
add = operator.add
subtract = operator.sub
divide = operator.truediv
matmul = operator.matmul


@dataclass(frozen=True)
class _Command:
    kind: str
    handle: Optional[int] = None
    handles: Tuple[int, ...] = ()
    shape: Shape = None
    dtype: str = "float32"
    data: Any = None
    operation: Any = None
    kwargs: Optional[dict] = None
    event: Optional[threading.Event] = None
    result_box: Optional[dict] = None


class TensorEngine:
    """Asynchronous tensor storage and operation queue.

    Parameters
    ----------
    device:
        ``"auto"`` prefers CUDA, then Apple MPS, then NumPy CPU. Explicit
        ``"mps"``/``"cuda"`` requests fail clearly if unavailable.
    default_dtype:
        Default dtype for newly created tensors and uploaded data.

    Notes
    -----
    * Handles are monotonically increasing integers; they are not pointers.
    * ``create_tensor``, ``store`` and ``queue_op`` never wait for queued GPU
      operations. ``store`` does make a host-side snapshot of its input so callers
      may safely reuse or mutate their input after it returns.
    * ``load`` is a fence: all commands queued before it have been dispatched,
      accelerator work is synchronized, and a NumPy array is returned.
    * A custom callable must operate on its tensor arguments without converting
      them to CPU (for example, don't call ``.cpu()``, ``.numpy()`` or ``.item()``).
    * ``close`` drains queued work before releasing the backend resources.
    """

    def __init__(self, device: str = "auto", default_dtype: str = "float32") -> None:
        self.device = self._select_device(device)
        self.default_dtype = self._dtype_name(default_dtype)
        self._commands: "queue.Queue[Optional[_Command]]" = queue.Queue()
        self._next_handle = 1
        self._handle_lock = threading.Lock()
        self._metadata: Dict[int, dict] = {}
        self._closed = False
        self._closed_lock = threading.Lock()
        self._fatal_error: Optional[BaseException] = None
        self._fatal_traceback: Optional[str] = None
        self._tensors: Dict[int, Any] = {}  # Accessed only by the worker thread.
        self._worker = threading.Thread(
            target=self._worker_main,
            name=f"TensorEngine-{self.device}",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        if torch is not None and isinstance(dtype, torch.dtype):
            # torch.float32 -> "float32"
            return str(dtype).removeprefix("torch.")
        try:
            return np.dtype(dtype).name
        except (TypeError, ValueError):
            text = str(dtype)
            return text.removeprefix("torch.")

    @staticmethod
    def _select_device(requested: str) -> str:
        requested = str(requested).lower()
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")

        cuda_ok = bool(torch is not None and torch.cuda.is_available())
        mps_ok = bool(
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if requested == "auto":
            if cuda_ok:
                return "cuda"
            if mps_ok:
                return "mps"
            return "cpu"
        if requested == "cuda":
            if not cuda_ok:
                raise RuntimeError("device='cuda' requested, but CUDA is unavailable")
            return "cuda"
        if requested == "mps":
            if not mps_ok:
                raise RuntimeError("device='mps' requested, but PyTorch MPS is unavailable")
            return "mps"
        return "cpu"

    @property
    def backend(self) -> str:
        if self.device == "mps":
            return "PyTorch MPS / Apple Metal"
        if self.device == "cuda":
            return "PyTorch CUDA"
        return "NumPy CPU" if torch is None else "PyTorch/NumPy CPU"

    def _new_handle(self, shape: Shape = None, dtype: Optional[str] = None) -> int:
        with self._handle_lock:
            handle = self._next_handle
            self._next_handle += 1
            self._metadata[handle] = {
                "shape": tuple(shape) if shape is not None else None,
                "dtype": dtype or self.default_dtype,
            }
            return handle

    def _ensure_open(self) -> None:
        if self._closed:
            raise TensorEngineError("TensorEngine is closed")

    def create_tensor(
        self,
        shape: Optional[Sequence[int]] = None,
        *,
        dtype: Optional[Any] = None,
        fill: Optional[Union[int, float, bool]] = None,
    ) -> int:
        """Return an integer handle immediately and enqueue tensor allocation.

        ``shape`` can be omitted when the first operation that defines the tensor
        is ``store``. For a shaped tensor, an optional scalar ``fill`` initializes
        its contents; without fill, the allocation is uninitialized.
        """
        self._ensure_open()
        normalized_shape = self._normalize_shape(shape) if shape is not None else None
        dtype_name = self._dtype_name(dtype or self.default_dtype)
        handle = self._new_handle(normalized_shape, dtype_name)
        self._commands.put(_Command(
            kind="create", handle=handle, shape=normalized_shape,
            dtype=dtype_name, data=fill,
        ))
        return handle

    def store(self, tensor_handle: int, shape: Sequence[int], tensor_data: Any) -> None:
        """Queue host data for a tensor handle without an accelerator fence.

        The shape must contain exactly as many elements as ``tensor_data``. For a
        CPU PyTorch tensor the data is detached and snapshotted on the host. A
        non-CPU torch tensor is rejected to avoid an implicit device-to-host wait;
        use ``load`` to cross that boundary explicitly.
        """
        self._ensure_open()
        handle = self._check_handle(tensor_handle)
        normalized_shape = self._normalize_shape(shape)
        host = self._host_snapshot(tensor_data)
        expected = int(np.prod(normalized_shape, dtype=np.int64))
        if host.size != expected:
            raise ValueError(
                f"store shape {normalized_shape} requires {expected} values, "
                f"but received {host.size}"
            )
        dtype_name = self._metadata[handle]["dtype"]
        # Snapshot now; the worker will enqueue the backend upload in command order.
        host = np.asarray(host).reshape(normalized_shape).copy()
        self._metadata[handle]["shape"] = normalized_shape
        self._commands.put(_Command(
            kind="store", handle=handle, shape=normalized_shape,
            dtype=dtype_name, data=host,
        ))

    def queue_op(
        self,
        operation: Operation,
        *tensor_handles: int,
        out_handle: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Queue an operation and immediately return its output handle.

        Example::

            product = engine.queue_op(multiply, left, right)
            total = engine.queue_op("sum", product, axis=0)

        String operations include multiply, add, subtract, divide, matmul, sum,
        mean, reshape, transpose, relu, sigmoid, tanh, exp, log, sqrt, abs,
        maximum, minimum and clamp. A callable can also be supplied; it receives
        backend tensor objects and must return a tensor-like result.
        """
        self._ensure_open()
        if not tensor_handles:
            raise ValueError("queue_op requires at least one input tensor handle")
        checked = tuple(self._check_handle(h) for h in tensor_handles)
        if out_handle is None:
            dtype_name = self._metadata[checked[0]]["dtype"]
            result_handle = self._new_handle(None, dtype_name)
        else:
            result_handle = self._check_handle(out_handle)
        self._commands.put(_Command(
            kind="op", handle=result_handle, handles=checked,
            operation=operation, kwargs=dict(kwargs),
        ))
        return result_handle

    def load(self, tensor_handle: int, shape: Optional[Sequence[int]] = None) -> np.ndarray:
        """Fence prior queued work, then return the tensor as a host NumPy array."""
        self._ensure_open()
        handle = self._check_handle(tensor_handle)
        normalized_shape = self._normalize_shape(shape) if shape is not None else None
        event = threading.Event()
        result_box: dict = {}
        self._commands.put(_Command(
            kind="load", handle=handle, shape=normalized_shape,
            event=event, result_box=result_box,
        ))
        event.wait()
        if "error" in result_box:
            raise TensorEngineError(
                f"load({handle}) failed because an earlier tensor command failed: "
                f"{result_box['error']}\n{result_box.get('traceback', '')}"
            ) from result_box["error"]
        return result_box["result"]

    def release(self, tensor_handle: int) -> None:
        """Queue release of a tensor handle without synchronizing device work.

        Call this after the handle's final ``load`` or after all queued consumers
        have completed. The backend object is dropped by the owner thread.
        """
        self._ensure_open()
        handle = self._check_handle(tensor_handle)
        with self._handle_lock:
            self._metadata.pop(handle, None)
        self._commands.put(_Command(kind="release", handle=handle))

    def close(self) -> None:
        """Drain pending commands and stop the backend thread."""
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
            event = threading.Event()
            result_box: dict = {}
            self._commands.put(_Command(kind="close", event=event, result_box=result_box))
        event.wait()
        self._worker.join()
        if "error" in result_box:
            raise TensorEngineError(f"TensorEngine close failed: {result_box['error']}") from result_box["error"]

    def __enter__(self) -> "TensorEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _check_handle(self, handle: int) -> int:
        if not isinstance(handle, (int, np.integer)):
            raise TensorHandleError(f"tensor handle must be an integer, got {type(handle).__name__}")
        handle = int(handle)
        if handle not in self._metadata:
            raise TensorHandleError(f"unknown tensor handle {handle}")
        return handle

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
        result = tuple(int(n) for n in shape)
        if any(n < 0 for n in result):
            raise ValueError(f"tensor dimensions must be non-negative: {result}")
        return result

    @staticmethod
    def _host_snapshot(data: Any) -> np.ndarray:
        if torch is not None and isinstance(data, torch.Tensor):
            if data.device.type != "cpu":
                raise ValueError(
                    "store() accepts host/CPU data only; use load() to transfer "
                    "device data to the CPU without hiding a synchronization fence"
                )
            data = data.detach().numpy()
        try:
            arr = np.asarray(data)
        except Exception as exc:
            raise TypeError(f"tensor data cannot be represented as a NumPy array: {exc}") from exc
        if arr.dtype.kind == "O":
            raise TypeError("object-dtype tensor data is not supported")
        return np.ascontiguousarray(arr).copy()

    def _to_backend_dtype(self, dtype_name: str) -> Any:
        if self.device == "cpu":
            return np.dtype(dtype_name)
        if torch is None:  # Device selection should already have prevented this.
            raise RuntimeError("PyTorch is required for accelerator devices")
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported PyTorch dtype: {dtype_name}") from exc

    def _make_backend_tensor(self, array: np.ndarray, dtype_name: str) -> Any:
        dtype = self._to_backend_dtype(dtype_name)
        if self.device == "cpu":
            return np.asarray(array, dtype=dtype).copy()
        # The host snapshot is isolated from the caller. The device upload happens
        # on the worker thread, never synchronizing the caller's event loop.
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def _empty_backend_tensor(self, shape: Sequence[int], dtype_name: str, fill: Any = None) -> Any:
        dtype = self._to_backend_dtype(dtype_name)
        if self.device == "cpu":
            if fill is None:
                return np.empty(tuple(shape), dtype=dtype)
            return np.full(tuple(shape), fill, dtype=dtype)
        if fill is None:
            return torch.empty(tuple(shape), dtype=dtype, device=self.device)
        return torch.full(tuple(shape), fill, dtype=dtype, device=self.device)

    def _synchronize_device(self) -> None:
        # Explicit synchronization is only called at a load/close fence.
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def _to_host_result(self, tensor: Any) -> np.ndarray:
        self._synchronize_device()
        if self.device == "cpu":
            return np.asarray(tensor).copy()
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().to(device="cpu").numpy().copy()
        # A user callable may return a scalar or NumPy array; normalize it.
        return np.asarray(tensor).copy()

    @staticmethod
    def _resolve_operation(operation: Operation) -> Tuple[Callable[..., Any], str]:
        if callable(operation):
            return operation, getattr(operation, "__name__", repr(operation))
        name = str(operation).lower()

        def unary(name_: str, numpy_fn: Callable[..., Any], torch_fn: Optional[Callable[..., Any]] = None):
            def call(x: Any, **kw: Any) -> Any:
                if torch is not None and isinstance(x, torch.Tensor):
                    fn = torch_fn or getattr(torch, name_)
                    return fn(x, **kw)
                return numpy_fn(x, **kw)
            return call

        def reduction(fn_name: str):
            def call(x: Any, axis: Any = None, keepdims: bool = False, **kw: Any) -> Any:
                if torch is not None and isinstance(x, torch.Tensor):
                    if "dim" in kw:
                        axis = kw.pop("dim")
                    if "keepdim" in kw:
                        keepdims = kw.pop("keepdim")
                    return getattr(x, fn_name)(dim=axis, keepdim=keepdims, **kw)
                return getattr(np, fn_name)(x, axis=axis, keepdims=keepdims, **kw)
            return call

        def reshape(x: Any, shape: Sequence[int]) -> Any:
            return x.reshape(tuple(shape))

        def transpose(x: Any, axes: Optional[Sequence[int]] = None, dim0: Optional[int] = None, dim1: Optional[int] = None) -> Any:
            if torch is not None and isinstance(x, torch.Tensor):
                if dim0 is not None and dim1 is not None:
                    return x.transpose(dim0, dim1)
                if axes is None:
                    return x.transpose(-2, -1) if x.ndim >= 2 else x
                # torch.permute and numpy.transpose both consume an axis sequence.
                return x.permute(tuple(axes))
            if dim0 is not None and dim1 is not None:
                axes2 = list(range(x.ndim))
                axes2[dim0], axes2[dim1] = axes2[dim1], axes2[dim0]
                return np.transpose(x, axes2)
            return np.transpose(x, axes=axes)

        operations: Dict[str, Callable[..., Any]] = {
            "multiply": operator.mul,
            "mul": operator.mul,
            "add": operator.add,
            "subtract": operator.sub,
            "sub": operator.sub,
            "divide": operator.truediv,
            "div": operator.truediv,
            "matmul": operator.matmul,
            "sum": reduction("sum"),
            "mean": reduction("mean"),
            "reshape": reshape,
            "transpose": transpose,
            "relu": unary("relu", lambda x, **kw: np.maximum(x, 0), lambda x, **kw: torch.relu(x)),
            "sigmoid": unary("sigmoid", lambda x, **kw: 1.0 / (1.0 + np.exp(-x)), lambda x, **kw: torch.sigmoid(x)),
            "tanh": unary("tanh", np.tanh, lambda x, **kw: torch.tanh(x)),
            "exp": unary("exp", np.exp),
            "log": unary("log", np.log),
            "sqrt": unary("sqrt", np.sqrt),
            "abs": unary("abs", np.abs, lambda x, **kw: torch.abs(x)),
            "maximum": np.maximum if torch is None else (lambda a, b: torch.maximum(a, b) if isinstance(a, torch.Tensor) else np.maximum(a, b)),
            "minimum": np.minimum if torch is None else (lambda a, b: torch.minimum(a, b) if isinstance(a, torch.Tensor) else np.minimum(a, b)),
            "clamp": (lambda x, min=None, max=None: torch.clamp(x, min=min, max=max) if torch is not None and isinstance(x, torch.Tensor) else np.clip(x, min, max)),
        }
        if name not in operations:
            raise ValueError(f"unknown operation {operation!r}; supported names: {', '.join(sorted(operations))}")
        return operations[name], name

    def _worker_main(self) -> None:
        while True:
            command = self._commands.get()
            if command is None:
                return
            if command.kind == "close":
                try:
                    if self._fatal_error is None:
                        self._synchronize_device()
                except BaseException as exc:
                    command.result_box["error"] = exc  # type: ignore[index]
                if command.event is not None:
                    command.event.set()
                return
            try:
                if self._fatal_error is not None:
                    raise TensorEngineError(
                        f"previous asynchronous command failed: {self._fatal_error}\n"
                        f"{self._fatal_traceback or ''}"
                    ) from self._fatal_error
                self._execute_command(command)
            except BaseException as exc:
                if self._fatal_error is None:
                    self._fatal_error = exc
                    self._fatal_traceback = traceback.format_exc()
                if command.kind == "load" and command.result_box is not None:
                    command.result_box["error"] = exc
                    command.result_box["traceback"] = traceback.format_exc()
                elif command.event is not None and command.result_box is not None:
                    command.result_box["error"] = exc
                    command.result_box["traceback"] = traceback.format_exc()
                # Asynchronous create/store/op failures are deliberately surfaced
                # at the next load() fence instead of blocking queue_op/store.
            finally:
                if command.event is not None:
                    command.event.set()

    def _execute_command(self, command: _Command) -> None:
        if command.kind == "release":
            if command.handle is not None:
                self._tensors.pop(command.handle, None)
            return

        if command.kind == "create":
            if command.shape is not None:
                self._tensors[command.handle] = self._empty_backend_tensor(
                    command.shape, command.dtype, command.data
                )
            # Shape-less handles are virtual until store() or queue_op defines them.
            return

        if command.kind == "store":
            tensor = self._make_backend_tensor(command.data, command.dtype)
            self._tensors[command.handle] = tensor
            return

        if command.kind == "op":
            missing = [h for h in command.handles if h not in self._tensors]
            if missing:
                raise TensorHandleError(
                    f"operation references tensor(s) not initialized by create/store/op: {missing}"
                )
            fn, op_name = self._resolve_operation(command.operation)
            args = [self._tensors[h] for h in command.handles]
            result = fn(*args, **(command.kwargs or {}))
            if isinstance(result, (int, float, bool, np.generic)):
                result = np.asarray(result)
                if self.device != "cpu":
                    result = torch.as_tensor(result, device=self.device)
            if not (isinstance(result, np.ndarray) or (torch is not None and isinstance(result, torch.Tensor))):
                raise TypeError(
                    f"operation {op_name!r} returned unsupported {type(result).__name__}; "
                    "expected a tensor/array result"
                )
            self._tensors[command.handle] = result
            self._metadata[command.handle]["shape"] = tuple(result.shape)
            return

        if command.kind == "load":
            if command.handle not in self._tensors:
                raise TensorHandleError(f"tensor {command.handle} has no data; store to it first")
            result = self._to_host_result(self._tensors[command.handle])
            if command.shape is not None:
                expected = int(np.prod(command.shape, dtype=np.int64))
                if result.size != expected:
                    raise ValueError(
                        f"load requested shape {command.shape} ({expected} values), "
                        f"but tensor {command.handle} has {result.size} values"
                    )
                result = result.reshape(command.shape)
            command.result_box["result"] = result  # type: ignore[index]
            return

        raise RuntimeError(f"unknown tensor command kind: {command.kind}")


__all__ = [
    "TensorEngine", "TensorEngineError", "TensorHandleError",
    "multiply", "add", "subtract", "divide", "matmul",
]
