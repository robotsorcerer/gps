# PyTorch (TorchScript) controller parameters.
# The Python side serialises the trained model with torch.jit.script() and
# sends the raw bytes here; the C++ side calls torch::jit::load() from a
# std::istringstream.
string model_bytes     # TorchScript model serialised to a byte string
float32[] bias         # Observation normalisation bias  (length dO)
float32[] scale        # Observation normalisation scale (length dO*dO, row-major)
float32[] noise        # Pre-sampled Gaussian noise, shape [T * dU], row-major
int32 dim_bias         # Length of bias / scale row (= dO)
uint32 dU              # Action dimension
