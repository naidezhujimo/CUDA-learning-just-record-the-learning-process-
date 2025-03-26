import torch
import time 
import math

N = 1 << 19
warmup_runs = 10
benchmark_runs = 100
batch_size = 256

def custom_tanh(x):
    return (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)

def benchmark_custom_tanh(input_tensor):
    for _ in range(warmup_runs):
        _ = custom_tanh(input_tensor)
    
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = custom_tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / benchmark_runs
    print(f"Custom Tanh: Avg time per run: {avg_time:.3f} ms")
    
    return custom_tanh(input_tensor)

def benchmark_builtin_tanh(input_tensor):
    for _ in range(warmup_runs):
        _ = torch.tanh(input_tensor)
    
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) * 1000 / benchmark_runs
    print(f"Built-in Tanh: Avg time per run: {avg_time:.3f} ms")
    
    return torch.tanh(input_tensor)

def verify_outputs(custom_output, builtin_output):
    max_diff = torch.max(torch.abs(custom_output - builtin_output)).item()
    print(f"Max difference between custom and built-in outputs: {max_diff:.6e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.rand((128, 32, 224, 224), device=device) * 2 - 1

    _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()

    custom_output = benchmark_custom_tanh(input_tensor)

    builtin_output = benchmark_builtin_tanh(input_tensor)

    verify_outputs(custom_output, builtin_output)

if __name__ == '__main__':
    main()