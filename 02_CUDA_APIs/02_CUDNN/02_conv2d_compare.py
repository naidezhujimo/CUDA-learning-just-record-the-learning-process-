import torch
import torch.nn.functional as F

width = 4
height = 4
kernel_size = 3
in_channels = 1
out_channels = 1
batch_size = 1

input_values = torch.tensor(
    [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ],
    dtype=torch.float32,
).reshape(batch_size, in_channels, height, width)

kernel_values = torch.tensor(
    [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ],
    dtype=torch.float32,
).reshape(out_channels, in_channels, kernel_size, kernel_size)

output = F.conv2d(input_values, kernel_values, padding=kernel_size // 2)

print("Input:")
print(input_values)
print("\nKernel:")
print(kernel_values)
print("\nOutput:")
print(output)

print("\nFlattened output:")
print(output.flatten().tolist())
print(len(output.flatten().tolist()))