import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=8),
    ],
    key=['M', 'N', 'D']
)
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B, H, M, N, D,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_outb, stride_outh, stride_outm, stride_outd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr
):
    # 计算分块索引
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # 初始化累加器和统计量
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32) # 用于存储中间结果的累加器
    row_max = tl.full((BLOCK_M, ), -float('inf'), dtype=tl.float32) # 用于存储每行的最大值
    row_sum = tl.zeros((BLOCK_M, ), dtype=tl.float32) # 用于存储每行的指数和

    # 分块加载Q
    mask_q = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(
        q_ptr + pid_batch*stride_qb + pid_head*stride_qh +
        offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd,
        mask=mask_q,
        other=0.0
    ).to(tl.float16)

    # 分别遍历 K 和 V
    for n in range(0, N, BLOCK_N):
        # 计算当前块的偏移
        offs_n = n + tl.arange(0, BLOCK_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        # 加载 K 和 V的分块
        k = tl.load(
            k_ptr + pid_batch*stride_kb + pid_head*stride_kh +
            offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd,
            mask=mask_kv,
            other=0.0
        ).to(tl.float16)
        v = tl.load(
            v_ptr + pid_batch*stride_vb + pid_head*stride_vh +
            offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd,
            mask=mask_kv,
            other=0.0
        ).to(tl.float32)

        # 计算 QK^T（提升到 float32 计算）
        k_trans = tl.trans(k.to(tl.float32))  # [BLOCK_D, BLOCK_N]
        qk = tl.dot(q.to(tl.float32), k_trans.to(tl.float32)) * SCALE
        qk = tl.where(qk > 50.0, 50.0, qk)  # 限制最大值防止exp爆炸
        qk = tl.where(qk < -50.0, -50.0, qk)  # 限制最小值
        # 在线 Softmax 操作
        # 更新行最大值和指数和
        current_max = tl.maximum(row_max, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - current_max[:, None]).to(tl.float32)
        exp_qk = tl.where(exp_qk > 1e5, 1e5, exp_qk)  # 限制指数最大值
        old_row_max = row_max  # 保存旧的最大值
        row_sum = row_sum * tl.exp(row_max - current_max) + tl.sum(exp_qk, axis=1)
        row_max = current_max

        # 更新累加器
        acc *= tl.exp(old_row_max - row_max)[:, None] # 将 acc 调整到新的最大值的尺度上
        acc += tl.dot(exp_qk, v)

    # 归一化并存储结果
    acc = acc / (row_sum[:, None] + 1e-5)  # 防止除零
    tl.store(
        out_ptr + pid_batch*stride_outb + pid_head*stride_outh +
        offs_m[:, None]*stride_outm + offs_d[None, :]*stride_outd,
        acc.to(tl.float16),
        mask=((offs_m[:, None] < M) & (offs_d[None, :] < D)),
    )

def flash_attention(q, k, v):
    B, H, M, D = q.shape
    N = k.shape[2] # [B, H, N, D]
    output = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)

    grid = (B, H, triton.cdiv(M, 64))
    flash_attention_kernel[grid](
        q, k, v, output,
        B, H, M, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        SCALE=scale
    )
    return output

# 测试数据
B, H, M, N, D = 2, 8, 1024, 1024, 64
q = torch.randn((B, H, M, D), device='cuda', dtype=torch.float16)
k = torch.randn((B, H, N, D), device='cuda', dtype=torch.float16)
v = torch.randn((B, H, N, D), device='cuda', dtype=torch.float16)

# Flash Attention结果
flash_output = flash_attention(q, k, v)

# PyTorch原生结果
scale = 1.0 / (D ** 0.5)
qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
softmax = torch.softmax(qk, dim=-1)
torch_output = torch.matmul(softmax, v.float()).half()

# 精度检查
print(f"最大误差: {torch.max(torch.abs(flash_output - torch_output)):.4e}")
print(f"结果是否一致: {torch.allclose(flash_output, torch_output, atol=1e-3)}")