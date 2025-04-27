import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math
import os
from tqdm import tqdm # Keep tqdm for user feedback if needed later

# Keep the original CUDA kernel as it's functionally correct
CUDA_KERNEL = """
#include <cuda_runtime.h>
#include <stdio.h> // Include for printf if debug needed

// Structure to hold output pairs
struct Pair {
    int id1_idx;
    int id2_idx;
    float similarity;
};

extern "C"
__global__ void cosine_similarity_filter(
    const float *matrix,        // Input normalized embeddings (flattened N*D)
    Pair *output_pairs,         // Output buffer for pairs meeting threshold
    int *result_count,          // Atomic counter for number of results found
    int max_pairs,              // Size of the output_pairs buffer
    float threshold,            // Similarity threshold
    int n,                      // Number of vectors
    int d,                      // Dimension of vectors
    int debug_mode              // Flag for debug prints (0 or 1)
) {
    // Calculate global thread indices for pair (i, j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // --- Boundary and Symmetry Check ---
    // Ensure indices are within bounds (0 to n-1)
    // Ensure j > i to compute only the upper triangle (avoid duplicates and self-compare)
    if (i < n && j < n && j > i) {
        float dot = 0.0f;

        // --- Compute Dot Product (Cosine Similarity on normalized vectors) ---
        const float* vec_i = matrix + (size_t)i * d; // Pointer arithmetic for row i
        const float* vec_j = matrix + (size_t)j * d; // Pointer arithmetic for row j
        for (int k = 0; k < d; ++k) {
            // dot += vec_i[k] * vec_j[k];
            // Faster version: 
            dot += matrix[i * d + k] * matrix[j * d + k];
        }

        // --- Debug Print (Optional) ---
        // Be careful: printf in kernels with many threads can drastically slow things down
        // and produce overwhelming output. Use sparingly for small N or specific indices.
        // if (debug_mode && i < 5 && j < 10) {
        //     printf("DEBUG KERNEL (%d,%d): sim=%.4f\\n", i, j, dot);
        // }

        // --- Threshold Check and Output ---
        // Check for valid float and if similarity meets threshold
        if (!isnan(dot) && !isinf(dot) && dot >= threshold) {
            // Atomically increment result counter to get a unique index for writing
            int idx = atomicAdd(result_count, 1);

            // Check if the allocated output buffer is large enough
            if (idx < max_pairs) {
                // Write the pair information to the output buffer
                output_pairs[idx].id1_idx = i;
                output_pairs[idx].id2_idx = j;
                output_pairs[idx].similarity = dot;
            }
            // else: If idx >= max_pairs, the buffer is full.
            //       This pair is discarded. (We aim to prevent this by allocating enough space)
        }
    }
}
"""

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes embeddings using L2 norm.
    Zero vectors remain zero vectors after normalization (norm becomes 1 to avoid division by zero).
    """
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Prevent division by zero for zero-norm vectors
    norms[norms == 0] = 1.0
    return embeddings / norms

def compute_similarity_cuda_filtered(embeddings: np.ndarray, threshold: float = 0.7, debug_mode: bool = False):
    """
    Computes cosine similarities above a threshold using CUDA.
    This version uses an *estimated* size for the output buffer.
    WARNING: May truncate results if the estimate is too low.
    """
    n, d = embeddings.shape
    pair_dtype = np.dtype([('id1_idx', np.int32), ('id2_idx', np.int32), ('similarity', np.float32)])
    if n == 0:
        return np.empty(0, dtype=pair_dtype)

    print(f"Starting compute_similarity_cuda_filtered (N={n}, D={d})")
    # Ensure embeddings are contiguous in memory (C order)
    embeddings_flat = np.ascontiguousarray(embeddings.flatten(), dtype=np.float32)
    embeddings_gpu = drv.mem_alloc(embeddings_flat.nbytes)
    drv.memcpy_htod(embeddings_gpu, embeddings_flat)
    del embeddings_flat # Free CPU memory early

    # --- Output Buffer Allocation (Estimate Based) ---
    # Calculate theoretical max, but use a smaller estimate to save memory,
    # accepting the risk of truncation.
    # max_possible_pairs = n * (n - 1) // 2 if n > 1 else 0
    estimated_max_pairs = max(1000, int(n * n * 0.01)) # Estimate 1% fill rate, min 1000
    print(f"Allocating estimated output buffer for {estimated_max_pairs} pairs.")
    # Add a warning if estimate seems low compared to theoretical max
    # if n > 1000 and estimated_max_pairs < max_possible_pairs * 0.001:
    #     print("Warning: Estimated max pairs seems very low. Results might be truncated.")

    output_pairs_gpu = drv.mem_alloc(estimated_max_pairs * pair_dtype.itemsize)

    result_count_gpu = drv.mem_alloc(4) # Allocate 4 bytes for one int32
    drv.memcpy_htod(result_count_gpu, np.int32(0)) # Initialize counter to 0

    # --- Compile and Launch Kernel ---
    try:
        mod = SourceModule(CUDA_KERNEL)
        # mod = SourceModule(CUDA_KERNEL) # Or let PyCUDA detect
        func = mod.get_function("cosine_similarity_filter")
    except drv.CompileError as e:
        print("CUDA Kernel Compilation Failed:")
        print(e)
        # Clean up GPU memory before raising
        embeddings_gpu.free()
        output_pairs_gpu.free()
        result_count_gpu.free()
        raise

    block_size = 16 # 16x16 = 256 threads per block (common choice)
    grid_x = math.ceil(n / block_size)
    grid_y = math.ceil(n / block_size)
    grid = (grid_x, grid_y, 1)
    block = (block_size, block_size, 1)

    print(f"Launching CUDA kernel (filtered mode): Grid={grid}, Block={block}, N={n}, Threshold={threshold}")
    try:
        func(
            embeddings_gpu, output_pairs_gpu, result_count_gpu,
            np.int32(estimated_max_pairs), np.float32(threshold),
            np.int32(n), np.int32(d), np.int32(debug_mode),
            block=block, grid=grid
        )
        drv.Context.synchronize() # Wait for kernel completion
    except drv.Error as e:
        print(f"CUDA Kernel Launch/Execution Error: {e}")
        embeddings_gpu.free()
        output_pairs_gpu.free()
        result_count_gpu.free()
        raise

    # --- Retrieve Results ---
    actual_count = np.empty(1, dtype=np.int32)
    drv.memcpy_dtoh(actual_count, result_count_gpu)
    actual_count = int(actual_count[0])

    print(f"Kernel finished. Found {actual_count} pairs (estimated max was {estimated_max_pairs}).")

    # Check if the buffer was overflowed (actual count reached the max allocated size)
    if actual_count >= estimated_max_pairs and estimated_max_pairs > 0 :
         print(f"WARNING: Output buffer might have been truncated. Found count ({actual_count}) reached the estimated limit ({estimated_max_pairs}). Consider using 'allpairs' mode or increasing the estimate.")
         # Only retrieve up to the allocated buffer size to avoid reading invalid memory
         num_to_retrieve = estimated_max_pairs
    else:
         num_to_retrieve = actual_count

    result_pairs = np.empty(num_to_retrieve, dtype=pair_dtype)
    if num_to_retrieve > 0:
        drv.memcpy_dtoh(result_pairs, output_pairs_gpu)

    # --- Clean Up ---
    embeddings_gpu.free()
    output_pairs_gpu.free()
    result_count_gpu.free()

    print("Finished compute_similarity_cuda_filtered.")
    return result_pairs


def compute_all_pairs_batched_gpu(embeddings: np.ndarray, threshold: float = 0.53, batch_size: int = 256):
    """
    Computes all pairs cosine similarities above a threshold using CUDA.
    Optimized version: transfers data once, launches one kernel, allocates
    sufficient buffer space to avoid truncation (memory permitting).
    The 'batch_size' argument is no longer used for computation logic but kept for signature compatibility.
    """
    n, d = embeddings.shape
    pair_dtype = np.dtype([('id1_idx', np.int32), ('id2_idx', np.int32), ('similarity', np.float32)])
    if n < 2: # Need at least 2 vectors to form a pair
        return np.empty(0, dtype=pair_dtype)

    print(f"Starting compute_all_pairs_batched_gpu (Optimized Single Launch) (N={n}, D={d})")
    print(f"Note: 'batch_size' argument ({batch_size}) is ignored in this optimized version.")

    # --- Allocate GPU Memory & Transfer Embeddings ONCE ---
    # Ensure embeddings are contiguous in memory (C order)
    embeddings_flat = np.ascontiguousarray(embeddings.flatten(), dtype=np.float32)
    try:
        embeddings_gpu = drv.mem_alloc(embeddings_flat.nbytes)
        drv.memcpy_htod(embeddings_gpu, embeddings_flat)
        del embeddings_flat # Free CPU memory early
    except drv.Error as e:
        print(f"Error allocating or copying embeddings to GPU: {e}")
        print(f"Required memory: {embeddings_flat.nbytes / (1024**3):.2f} GB")
        raise

    # --- Output Buffer Allocation (Allocate for MAX possible pairs) ---
    # Calculate the theoretical maximum number of pairs (upper triangle)
    max_possible_pairs = n * (n - 1) // 2
    output_buffer_size = max_possible_pairs * pair_dtype.itemsize
    print(f"Attempting to allocate output buffer for {max_possible_pairs:,} potential pairs ({output_buffer_size / (1024**3):.2f} GB).")

    try:
        output_pairs_gpu = drv.mem_alloc(output_buffer_size)
    except drv.Error as e:
        # Handle memory allocation error - suggest alternatives if allocation fails
        print(f"FATAL: Failed to allocate GPU memory for {max_possible_pairs:,} output pairs ({output_buffer_size / (1024**3):.2f} GB).")
        print(f"Error: {e}")
        print("Possible Solutions:")
        print("  - Use a GPU with more memory.")
        print("  - Increase the similarity threshold to reduce the number of expected results.")
        print("  - Implement a multi-pass approach (e.g., count first, then retrieve) or revert to a (slower) truly batched retrieval if necessary.")
        embeddings_gpu.free() # Clean up already allocated memory
        raise # Re-raise the exception

    # Allocate and initialize the result counter on GPU
    result_count_gpu = drv.mem_alloc(4) # int32 size
    drv.memcpy_htod(result_count_gpu, np.int32(0))

    # --- Compile and Launch Kernel ONCE ---
    try:
        mod = SourceModule(CUDA_KERNEL) # Specify compute capability if needed
        # mod = SourceModule(CUDA_KERNEL) # Or let PyCUDA detect
        func = mod.get_function("cosine_similarity_filter")
    except drv.CompileError as e:
        print("CUDA Kernel Compilation Failed:")
        print(e)
        embeddings_gpu.free()
        output_pairs_gpu.free()
        result_count_gpu.free()
        raise

    block_size = 16 # 16x16 = 256 threads per block
    grid_x = math.ceil(n / block_size)
    grid_y = math.ceil(n / block_size)
    grid = (grid_x, grid_y, 1)
    block = (block_size, block_size, 1)

    print(f"Launching Optimized CUDA kernel (all pairs): Grid={grid}, Block={block}, N={n}, Threshold={threshold}")
    print("This may take a significant amount of time depending on N and the GPU...")
    # Add tqdm progress bar for visual feedback during kernel execution?
    # Note: A simple tqdm wrapper won't work directly for kernel time.
    # It would require periodic synchronization or a more complex monitoring setup.
    # For now, just print start/end messages.

    try:
        # Launch the kernel covering the full N x N space
        start_time = drv.Event()
        end_time = drv.Event()
        start_time.record() # Record time before kernel launch

        func(
            embeddings_gpu, output_pairs_gpu, result_count_gpu,
            np.int32(max_possible_pairs), # Pass the actual allocated buffer size
            np.float32(threshold),
            np.int32(n), np.int32(d),
            np.int32(0), # debug_mode set to 0 (can be made a parameter)
            block=block, grid=grid
        )

        end_time.record() # Record time after kernel launch (async)
        end_time.synchronize() # Wait for kernel and timing to complete
        kernel_time_ms = start_time.time_till(end_time)
        print(f"Kernel execution finished in {kernel_time_ms / 1000:.2f} seconds.")

    except drv.Error as e:
        print(f"CUDA Kernel Launch/Execution Error: {e}")
        embeddings_gpu.free()
        output_pairs_gpu.free()
        result_count_gpu.free()
        raise

    # --- Retrieve Results ---
    actual_count = np.empty(1, dtype=np.int32)
    drv.memcpy_dtoh(actual_count, result_count_gpu)
    actual_count = int(actual_count[0])

    print(f"Found {actual_count:,} pairs meeting the threshold.")

    # Check for overflow (shouldn't happen with max allocation, but good sanity check)
    if actual_count > max_possible_pairs:
         print(f"ERROR: Found count ({actual_count}) exceeds theoretical maximum ({max_possible_pairs}). This indicates a potential bug.")
         # Retrieve based on max_possible_pairs to prevent reading garbage
         actual_count = max_possible_pairs

    # Allocate host memory and copy results back
    print(f"Allocating host memory for {actual_count:,} pairs...")
    result_pairs = np.empty(actual_count, dtype=pair_dtype)

    if actual_count > 0:
        print("Copying results from GPU to CPU...")
        try:
            copy_start_time = drv.Event()
            copy_end_time = drv.Event()
            copy_start_time.record()

            drv.memcpy_dtoh(result_pairs, output_pairs_gpu)

            copy_end_time.record()
            copy_end_time.synchronize()
            copy_time_ms = copy_start_time.time_till(copy_end_time)
            print(f"Finished copying results in {copy_time_ms / 1000:.2f} seconds.")
        except drv.Error as e:
            print(f"Error copying results from GPU: {e}")
            # Clean up before raising
            embeddings_gpu.free()
            output_pairs_gpu.free()
            result_count_gpu.free()
            raise

    # --- Clean Up ---
    print("Freeing GPU memory...")
    embeddings_gpu.free()
    output_pairs_gpu.free()
    result_count_gpu.free()

    print("Finished compute_all_pairs_batched_gpu (Optimized).")
    return result_pairs

# Example usage (for testing within this module if needed)
if __name__ == '__main__':
    print("Testing whisper_gpu module...")

    # Create dummy data
    num_vectors = 1000 # Small test case
    dimensions = 768 # Example dimension
    print(f"Generating {num_vectors} dummy vectors with {dimensions} dimensions...")
    # Ensure reproducibility
    np.random.seed(42)
    dummy_embeddings = np.random.rand(num_vectors, dimensions).astype(np.float32)

    print("Normalizing embeddings...")
    normalized_embeddings = normalize_embeddings(dummy_embeddings)

    test_threshold = 0.8 # Use a high threshold for testing to get fewer results

    # --- Test the optimized 'allpairs' function ---
    print("\n--- Testing compute_all_pairs_batched_gpu (Optimized) ---")
    try:
        results_allpairs = compute_all_pairs_batched_gpu(normalized_embeddings, threshold=test_threshold)
        print(f"Found {len(results_allpairs)} pairs using optimized allpairs function.")
        if len(results_allpairs) > 0:
             print("Sample results (optimized):")
             # Sort by similarity descending for preview
             results_allpairs[::-1].sort(order='similarity')
             print(results_allpairs[:5])
    except drv.Error as e:
         print(f"GPU Memory Error during test: {e}")
         print("Test might require more GPU RAM or a smaller num_vectors.")
    except Exception as e:
        print(f"An error occurred during optimized allpairs test: {e}")

    # --- Test the original 'filtered' function ---
    print("\n--- Testing compute_similarity_cuda_filtered ---")
    # Note: This might truncate results if the estimate is low
    try:
        results_filtered = compute_similarity_cuda_filtered(normalized_embeddings, threshold=test_threshold, debug_mode=False)
        print(f"Found {len(results_filtered)} pairs using filtered function (may be truncated).")
        if len(results_filtered) > 0:
            print("Sample results (filtered):")
            # Sort by similarity descending for preview
            results_filtered[::-1].sort(order='similarity')
            print(results_filtered[:5])

        # Compare lengths (if no truncation occurred in filtered, they should be same)
        if 'results_allpairs' in locals() and len(results_allpairs) != len(results_filtered):
             print(f"\nWarning: Length mismatch between optimized ({len(results_allpairs)}) and filtered ({len(results_filtered)}). Filtered mode likely truncated results.")

    except drv.Error as e:
         print(f"GPU Memory Error during test: {e}")
    except Exception as e:
        print(f"An error occurred during filtered test: {e}")