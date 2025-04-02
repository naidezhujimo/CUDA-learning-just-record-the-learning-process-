import torch
import triton
import triton.language as tl

# 定义自动调优配置
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_warps=8),
    ],
    key=['M', 'N', 'D']
)
@triton.jit
def qk_kernel(
    q_ptr, k_ptr, qk_ptr,  # 输入输出指针
    B, H, M, N, D,          # Batch, Heads, SeqLen_Q, SeqLen_K, Dim
    stride_qb, stride_qh, stride_qm, stride_qd,  # Q的步长
    stride_kb, stride_kh, stride_kn, stride_kd,  # K的步长
    stride_qkb, stride_qkh, stride_qkm, stride_qkn,  # QK的步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,     # 缩放因子 1/sqrt(D)
):
    # 分块索引计算
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_combined = tl.program_id(2)
    num_N_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_combined // num_N_blocks
    pid_n = pid_combined % num_N_blocks
    
    # 定义分块偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 循环处理 D 的每个分块
    for d in range(0, D, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # 加载Q和K的分块
        q = tl.load(q_ptr + pid_batch*stride_qb + pid_head*stride_qh + 
                    offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd,
                    mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
        
        # 转置
        k = tl.load(k_ptr + pid_batch*stride_kb + pid_head*stride_kh + 
                    offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd,
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D) , other=0.0)
        
        # 计算分块Q @ K^T
        k = tl.trans(k)
        acc += tl.dot(q, k)
    
    acc *= SCALE #缩放
    # 存储到全局内存
    tl.store(qk_ptr + pid_batch*stride_qkb + pid_head*stride_qkh + 
             offs_m[:, None]*stride_qkm + offs_n[None, :]*stride_qkn,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    
# 定义自动调优配置
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_warps=8),
    ],
    key=['M', 'N', 'D']
)
@triton.jit
def attn_v_kernel(
    qk_ptr, v_ptr, out_ptr,  # 输入输出指针
    B, H, M, N, D,           # 维度参数
    stride_qkb, stride_qkh, stride_qkm, stride_qkn,  # QK的步长
    stride_vb, stride_vh, stride_vn, stride_vd,      # V的步长
    stride_outb, stride_outh, stride_outm, stride_outd,  # 输出的步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_combined = tl.program_id(2)
    num_D_blocks = tl.cdiv(D, BLOCK_D)
    pid_m = pid_combined // num_D_blocks  # 修正分块逻辑
    pid_d = pid_combined % num_D_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # 分块D的偏移

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # 初始化累加器和统计量
    row_max = tl.zeros((BLOCK_M, ), dtype=tl.float32) - float('inf')
    row_sum = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    # 分块计算Softmax统计量
    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        
        # 加载QK分块并计算Softmax
        qk = tl.load(qk_ptr + pid_batch*stride_qkb + pid_head*stride_qkh + 
                    offs_m[:, None]*stride_qkm + offs_n[None, :]*stride_qkn,
                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        # 更新行最大值和求和
        current_max = tl.maximum(row_max, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - current_max[:, None])
        row_sum = row_sum * tl.exp(row_max - current_max) + tl.sum(exp_qk, axis=1)
        row_max = current_max

    # 计算加权和，处理D的分块
    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        qk = tl.load(qk_ptr + pid_batch*stride_qkb + pid_head*stride_qkh + 
                    offs_m[:, None]*stride_qkm + offs_n[None, :]*stride_qkn,
                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        exp_qk = tl.exp(qk - row_max[:, None])
        softmax = exp_qk / row_sum[:, None]
    
        # 加载V分块
        v = tl.load(v_ptr + pid_batch*stride_vb + pid_head*stride_vh + 
                    offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd,
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        acc += tl.dot(softmax, v)
    
    # 存储结果（按D分块写入）
    tl.store(
        out_ptr + pid_batch*stride_outb + pid_head*stride_outh +
        offs_m[:, None]*stride_outm + offs_d[None, :]*stride_outd,
        acc,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
    )

    

def triton_attention(q, k, v):
    B, H, M, D = q.shape    # Batch, Heads, SeqLen_Q, Dim
    _, _, N, _ = k.shape    # SeqLen_K
    assert k.shape == (B, H, N, D)
    
    # 初始化中间结果QK和输出
    qk = torch.empty((B, H, M, N), device=q.device, dtype=q.dtype)
    output = torch.empty_like(q)
    
    # 分块参数（可自动调优）
    scale = 1.0 / (D ** 0.5)
    
    # 计算QK^T
    grid_qk = (B, H, triton.cdiv(M, 64) * triton.cdiv(N, 64))
    qk_kernel[grid_qk](
        q, k, qk,
        B, H, M, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
        SCALE=scale
    )
    
    # 计算Softmax和V相乘
    grid_attn = (B, H, triton.cdiv(M, 64) * triton.cdiv(D, 32))
    attn_v_kernel[grid_attn](
        qk, v, output,
        B, H, M, N, D,
        qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output

# 生成测试数据
B, H, M, N, D = 2, 8, 1024, 1024, 64
q = torch.randn((B, H, M, D), device='cuda')
k = torch.randn((B, H, N, D), device='cuda')
v = torch.randn((B, H, N, D), device='cuda')

# Triton实现
triton_output = triton_attention(q, k, v)

# PyTorch原生实现
scale = 1.0 / (D ** 0.5)
qk = torch.matmul(q, k.transpose(-2, -1)) * scale
softmax = torch.softmax(qk, dim=-1)
torch_output = torch.matmul(softmax, v)

# 精度检查
print(f"最大误差: {torch.max(torch.abs(triton_output - torch_output)):.4e}")
print(f"结果是否一致: {torch.allclose(triton_output, torch_output, atol=1e-3)}")