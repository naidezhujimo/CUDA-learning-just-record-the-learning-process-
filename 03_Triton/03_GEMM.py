import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # 输入输出矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长（内存布局）
    stride_am, stride_ak,  # A的步长（行主序）
    stride_bk, stride_bn,  # B的步长（列主序）
    stride_cm, stride_cn,  # C的步长（行主序）
    # 分块大小
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1. 确定当前线程块负责计算 C 的哪一部分
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M) # 行方向分块
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # 2. 计算子块的起始位置
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 3. 初始化累加器(寄存器优化)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 4. 分块加载A和B，并进行乘累加
    for k in range(0, K, BLOCK_K):
        # 加载A的一个分块（BLOCK_M * BLOCK_K）
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # 加载B的一个分块（BLOCK_K * BLOCK_N）
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        # 矩阵乘累加
        accumulator += tl.dot(a, b)

    # 5. 将结果写回到全局内存
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0], "维度不匹配"
    M, K = a.shape
    K, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c

# 测试与PyTorch对比
torch.manual_seed(0)
M, N, K = 1024, 1024, 1024
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

# Triton结果
triton_output = matmul(a, b)
# PyTorch结果
torch_output = torch.matmul(a, b)

# 验证精度
print(f"最大误差: {torch.max(torch.abs(triton_output - torch_output)):.4e}")
print(f"结果是否接近: {torch.allclose(triton_output, torch_output, atol=1e-2)}")
