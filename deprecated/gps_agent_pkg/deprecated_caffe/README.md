# Deprecated Caffe Files

These files implement the original Caffe-based neural network controller from the
GPS codebase. Caffe is no longer actively maintained and has been superseded by
PyTorch in this codebase.

## Migration Status

| Original file | Replacement |
|---|---|
| `caffenncontroller.h/.cpp` | `include/gps_agent_pkg/pytorchcontroller.h` + `src/pytorchcontroller.cpp` |
| `neuralnetworkcaffe.h/.cpp` | Uses `torch::jit::load()` (TorchScript) via `pytorchcontroller` |

## How the original worked

`CaffeNNController` loaded a serialised Caffe net from a protobuf string,
applied input scale/bias normalisation, and ran a forward pass to obtain
the action mean `u_t`. It then added pre-computed Gaussian noise to produce
the final action.

## Replacement design

`PyTorchController` (see `src/pytorchcontroller.cpp`) reproduces the same
GPS interface using a TorchScript model loaded with `torch::jit::load()`.
This allows the Python-side `PolicyOptPyTorch` to export trained models via
`torch.jit.script()` and deploy them in C++ without any Python runtime.

Do NOT restore these files to the build system. They are kept here for
historical reference only.
