import torch
import time

# Check for CUDA (NVIDIA GPU) availability
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for GPU on Windows
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Fallback to CPU if CUDA is unavailable
    print("⚡ Using CPU as CUDA is not available")

# Define tensor size (reduce for CPU if needed)
size = 1500

# Create large random tensors
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Number of iterations for performance test
iterations = 100

start_time = time.time()
print("Starting heavy GPU computation..." if device.type == "cuda" else "Starting CPU computation...")

# Perform heavy matrix multiplication and non-linear operation
for i in range(iterations):
    c = torch.matmul(a, b)  # Matrix multiplication
    c = torch.sin(c)  # Apply non-linear function
    if i % 10 == 0:
        print(f"Iteration {i}/{iterations} completed...")

end_time = time.time()

print(f"Computation completed in {end_time - start_time:.2f} seconds.")
