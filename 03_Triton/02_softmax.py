import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0) # 获取当前内核的id
    # 计算当前行的输入和输出指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE),
                  mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))
    
    row_max = tl.max(row, axis=0) # 计算当前行的最大值，用于数值稳定性
    numerator = tl.exp(row - row_max) # 从每行的元素中减去最大值，然后计算指数
    denominator = tl.sum(numerator, axis=0) # 计算指数部分的和，用于归一化
    softmax_output = numerator / denominator

    tl.store(output_row_start_ptr + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    grid = (n_rows, )
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


torch.manual_seed(620)
x = torch.randn(256, 1024, device='cuda')
torch_result = torch.softmax(x, dim=1)

triton_result = triton_softmax(x)

max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"Maximum difference between PyTorch and Triton results: {max_diff:.2e}")

is_close = torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)
print(f"Results are close: {is_close}")