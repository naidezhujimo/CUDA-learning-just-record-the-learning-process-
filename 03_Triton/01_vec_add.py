import torch
import triton
import triton.language as tl
import time

@triton.jit # 将Python函数编译为GPU内核
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0) # 获取当前内核的 ID
    block_start = pid * BLOCK_SIZE # 当前块的起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # 当前块中每个元素的偏移量
    mask = offsets < n_elements # 防止数组越界访问
    x = tl.load(x_ptr + offsets, mask=mask) # 内存加载
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask) # 存储数据

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x) # 创建一个与 x 形状相同但未初始化
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel() # 获取输出张量的元素
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), ) # 定义内核的启动网格
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024) # 调用编译后的内核函数

torch.manual_seed(0)
size = 2**25
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

@triton.testing.perf_report( # 性能测试和绘图
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['triton', 'torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector_add-performance',
        args={}
    )
)

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(min_ms), gbps(max_ms)

benchmark.run(print_data=True, show_plots=True)