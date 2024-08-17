import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import io
import os

torch.ops.load_library("../build/lib/libdietgpu.so")
torch.ops.load_library("../build/lib/libgpu_float_compress.so")

dev = torch.device("cuda:0")

# Given a 1D array, and reshape to a 2D array, pad with values
def reshape_pad(tx, col, value = 0):
    tx = torch.reshape(tx, (-1,))
    row = (tx.shape[0]+col-1) // col
    padsize = row * col - tx.shape[0]

    # Create another vector containing zeroes to pad `a` to (2 * 3) elements.
    m = torch.nn.ConstantPad1d((0,padsize), 0)
    tx = m(tx)
    tx = torch.reshape(tx, (row, col))
    return tx

def normalize_data(ts):
    print(ts.shape)
    ts = reshape_pad(ts, 512)
    print(ts.shape)
    ts = F.normalize(ts, p=2.0, dim = -2)
    ts = torch.reshape(ts, (-1,))
    return ts


def convert_data(input_path, input_dtype, output_path, output_dtype):
    """
    Convert data from one dtype to another.

    Parameters:
    - input_path: Path to input binary file.
    - output_path: Path to save converted binary file.
    - input_dtype: Input data type (e.g., torch.float16, torch.float32, torch.bfloat16).
    - output_dtype: Output data type (e.g., torch.bfloat16, torch.float32).
    """

    data_tensor = read_data_to_tensor(input_path, input_dtype)

    # Convert tensor to target dtype
    data_converted_tensor = data_tensor.to(dtype=output_dtype)

    # Save converted tensor to disk
    if output_dtype == torch.bfloat16:
        torch.save(data_converted_tensor, output_path)
    else:
        data_converted_np = data_converted_tensor.numpy()
        data_converted_np.tofile(output_path)


def torch_to_numpy_dtype(torch_dtype):
    """Convert torch dtype to numpy dtype."""
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")


def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0

    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s

    return total_input_size, total_comp_size,  total_input_size / total_comp_size


def get_any_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([1123303296], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            False, ts, True, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        comp_time = start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            False, comp_ts, out_ts, True, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        decomp_time = start.elapsed_time(end)

        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    return comp_time, decomp_time, total_size, comp_size



def get_float_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([500 * 500 * 100 * 6], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_float_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            True, ts, True, tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        if i > 0:
            comp_time += start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            True, comp_ts, out_ts, True, tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        if i > 0:
            decomp_time += start.elapsed_time(end)

        # validate
        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    comp_time /= num_runs
    decomp_time /= num_runs

    return comp_time, decomp_time, total_size, comp_size

def read_data_to_tensor(filepath: str, 
                       read_dtype: Union[np.dtype, torch.dtype, type] = np.float32, 
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Read binary data from a file and convert it to a PyTorch tensor.

    Parameters:
    - filepath (str): Path to the binary file.
    - read_dtype (Union[np.dtype, torch.dtype, type]): Data type of the binary data in the file.
    - device (torch.device): Device to which the tensor should be moved.

    Returns:
    - torch.Tensor: Tensor containing the data from the file.
    """
    
    if read_dtype == torch.bfloat16:
        int16_array = np.fromfile(filepath, dtype=np.int16)
        int32_array = (int16_array.astype(np.int32) << 16)
        float32_array = int32_array.view(np.float32)
        data_tensor = torch.from_numpy(float32_array)
        data_tensor = data_tensor.to(torch.bfloat16)
    else:
        with open(filepath, 'rb') as f:
            if isinstance(read_dtype, torch.dtype):
                np_dtype = torch_to_numpy_dtype(read_dtype)
            else:
                np_dtype = read_dtype
            np_array = np.frombuffer(f.read(), dtype=np_dtype).copy()
        data_tensor = torch.from_numpy(np_array)
    
    ts = data_tensor.to(device)
    
    return ts

def small_test():
    abspath = '/ocean/projects/asc200010p/jjia1/data/jinda/tensor_parallel/iteration_00040/layer_012/SelfAttention/'
    files = ['tensor_rank_0.bin']
    data_descs = ['1554*1*768']
    read_dtype = torch.bfloat16
    k = len(files)

    print('\nStart Test for Raw ANS\n')
    for i in range(k):
        print('test case {} of {}, test data name: {}' .format(i+1, k, files[i]))

        # Non-batched
        ts = []
        ts.append(read_data_to_tensor(abspath + files[i], read_dtype, dev))

        # Normalize the original data
        ts [0] = normalize_data(ts[0])

        c, dc, total_size, comp_size = get_any_comp_timings(ts)
        ratio = comp_size / total_size
        c_bw = (total_size / 1e9) / (c * 1e-3)
        dc_bw = (total_size / 1e9) / (dc * 1e-3)

        print("Raw ANS byte-wise non-batched perf  {} {}" .format(data_descs[i], read_dtype))
        print(
            "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                c, c_bw, total_size, comp_size, ratio
            )
        )
        print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

if __name__ == "__main__":
    small_test()
