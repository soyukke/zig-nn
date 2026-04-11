# MPSGraph Auto-Diff Notes

Pitfalls when using `gradientForPrimaryTensor:withTensors:name:` in Apple's MPSGraph.
None of these are documented — they only surface as runtime crashes.

## Ops That Do NOT Support Auto-Diff (5)

| Op | Selector | Error | Workaround |
|---|---|---|---|
| **SDPA** | `scaledDotProductAttentionWithQueryTensor:...` | `Couldn't get gradient Tensor for tensor of op` | Manual attention: `softmax(scale * Q @ K^T + mask) @ V` |
| **Normalization** | `normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:` | `Op gradient not implemented, file a radar` | Manual LayerNorm: `(x - mean) / sqrt(var + eps) * gamma + beta` (compute mean/var via `meanOfTensor:axes:name:`) |
| **Erf** | `erfWithTensor:name:` | `Op gradient not implemented` | Tanh GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` |
| **Tile** | `tileTensor:withMultiplier:name:` | `Op gradient not implemented` | Broadcast add: MPSGraph `add` auto-broadcasts shapes. Reshape to a broadcast-compatible shape first |
| **Concat** | `concatTensor:withTensor:dimension:name:` | `Op gradient not implemented` | Pre-concat on CPU side and feed as a placeholder |

## Ops Confirmed to Support Auto-Diff

The following ops pass gradients correctly:

- `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:` (matmul)
- `additionWithPrimaryTensor:secondaryTensor:name:` (add, with broadcasting)
- `multiplicationWithPrimaryTensor:secondaryTensor:name:` (mul)
- `subtractionWithPrimaryTensor:secondaryTensor:name:` (sub)
- `divisionWithPrimaryTensor:secondaryTensor:name:` (div)
- `tanhWithTensor:name:` (tanh)
- `sigmoidWithTensor:name:` (sigmoid)
- `squareRootWithTensor:name:` (sqrt)
- `squareWithTensor:name:` (square)
- `softMaxWithTensor:axis:name:` (softmax)
- `logarithmWithTensor:name:` (log)
- `negativeWithTensor:name:` (negative)
- `reshapeTensor:withShape:name:` (reshape)
- `transposeTensor:dimension:withDimension:name:` (transpose)
- `meanOfTensor:axes:name:` (mean)
- `reductionSumWithTensor:axis:name:` (reduction sum)
- `constantWithScalar:dataType:` (constant)
- `constantWithData:shape:dataType:` (constant data)

## Integer Placeholders and Auto-Diff

MPSGraph auto-diff attempts to compute gradients through **all** placeholders.
If any placeholder has an integer type (e.g., embedding indices, class labels):

```
Couldn't get gradient Tensor for tensor of op : mps_placeholder_N
```

**Solution**: pre-process integer inputs on the CPU side and feed them as float32 placeholders.

- `gather(emb_table, indices)` — look up `emb_table[token_id]` on CPU, feed result as a float placeholder
- `oneHot(labels, depth)` — build the one-hot vector on CPU, feed as a float placeholder

## Xavier Initialization Caveat

```zig
// Wrong: size = total element count (262144 for 512x512)
const scale = @sqrt(2.0 / @as(f32, @floatFromInt(size)));
// -> scale = 0.0028 (too small -> gradient vanishing)

// Correct: fan_in = shape[0]
const fan_in: f32 = @floatFromInt(param_shape[0]);
const scale = @sqrt(1.0 / fan_in);
// -> scale = 0.044 (for 512x512)
```

Using total element count makes the scale 10x+ too small, causing gradients to vanish at ~1e-12.
The loss numerically decreases but the change is invisible at display precision.

## Reading Gradients via readTensorData

The result of MPSGraph `run` is `NSDictionary<MPSGraphTensor, MPSGraphTensorData>`.
To read f32 values from MPSGraphTensorData:

```zig
const ndarray = send0(tensor_data, sel("mpsndarray"));  // MPSNDArray
// readBytes:strideBytes: for contiguous read (strideBytes=nil)
send(ndarray, sel("readBytes:strideBytes:"), out_ptr, null);
```

## Parameter Updates and Graph Re-execution

MPSGraph `run` **references** (not copies) the MTLBuffer from feeds.
On UMA, directly modifying the MTLBuffer's `contents` on CPU reflects in the next `run` automatically.
This means you can run Adam on CPU, update MTLBuffer contents, and feed them as-is in the next step.

## Performance Reference (M4 Max, batch=128)

| Implementation | ms/step | Notes |
|---|---|---|
| Per-dispatch Metal (~840 custom kernels) | 448 | Individual forward+backward dispatches |
| MPSGraph (compiled graph) | 165 | First run ~860ms (graph compilation) |
| PyTorch MPS | 186 | MPS Graph + auto-diff |

The first MPSGraph execution incurs 800-900ms for graph compilation. Subsequent runs are stable.

## Notes on Using MPSGraph from Zig

- All APIs are called via ObjC runtime (`objc_msgSend`)
- Function pointer casts for `objc_msgSend` must match argument types exactly
  - `NSInteger` (i64) vs `NSUInteger` (u64) distinction matters
  - `id` (nullable pointer) maps to `?*anyopaque`
- MPSDataTypeFloat32 = `0x10000020` (268435488)
- MPSDataTypeInt32 = `0x20000020`
- NSArray/NSDictionary construction also goes through the ObjC runtime
- MetalPerformanceShadersGraph must be linked as a separate framework from Metal/MetalPerformanceShaders
