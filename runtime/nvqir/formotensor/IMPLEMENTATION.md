# FormoTensor Batch Simulator Implementation

## 概述

FormoTensor 是一個為 Quantum ML 批次運算優化的量子模擬器後端，基於 cuTensorNet 實作。本實作的核心理念是利用 **Direct Sum** 的數學架構，將 batch 維度作為「外掛維度」而非物理量子維度，從而避免維度爆炸問題。

## 核心設計原則

### 1. Direct Sum 架構

**數學原理：**

對於 `n` 個 qubit 和 `B` 個批次樣本：
- **狀態張量形狀**：`(B, 2, 2, ..., 2)` 其中有 `n` 個 2
- **批次維度 B** 是最外層維度，永不參與量子門的收縮（contraction）
- **等價於**：B 個獨立的量子系統的 Direct Sum

$$
\mathcal{H}_{\text{total}} = \bigoplus_{b=1}^{B} \mathcal{H}_Q^{(b)}
$$

**錯誤做法（會導致維度爆炸）：**
- 把 batch 當作可糾纏的物理維度：$\mathcal{H}_{total} = \mathcal{H}_B \otimes \mathcal{H}_Q$
- 這會導致總維度變成 $2^{n + \log_2 B}$，指數成長

**正確做法：**
- Batch 維度僅作為標籤/索引，不參與量子收縮
- 總記憶體：$B \times 2^n$（線性成長）

### 2. Gate 廣播機制

**單量子閘（如 Ry(θ)）：**

```cpp
// 原始 gate 形狀：[2, 2]
// Batch gate 形狀：[1, 2, 2]  (batch dim = 1，可廣播)
```

在 batch mode 中：
1. Gate 添加一個 size=1 的 batch 維度
2. 量子 qubit 索引 +1（因為 index 0 被 batch 占用）
3. cuTensorNet 自動廣播到所有 B 個樣本

**控制門處理：**
- 小於閾值（default=2）：展開成完整張量
- 大於閾值：使用 `cutensornetStateApplyControlledTensorOperator`

### 3. 資料結構

**FormoTensorState：**
```cpp
template <typename ScalarType>
class FormoTensorState {
protected:
  std::size_t m_numQubitsPerSample;  // n: qubits per sample
  std::size_t m_batchSize;           // B: number of samples  
  bool m_isBatched;                  // batch mode flag
  cutensornetState_t m_quantumState; // cuTensorNet state
  // ...
};
```

**SimulatorFormoTensor：**
```cpp
template <typename ScalarType>
class SimulatorFormoTensor : public CircuitSimulatorBase<ScalarType> {
protected:
  std::unique_ptr<FormoTensorState<ScalarType>> m_state;
  bool m_isBatchMode;
  std::size_t m_batchSize;
  std::unordered_map<std::string, void *> m_gateDeviceMemCache;
  // ...
};
```

## 關鍵實作細節

### 1. Batch State 初始化

```cpp
// Create batch state with dims = [B, 2, 2, ..., 2]
std::vector<int64_t> qubitDims(numQubitsPerSample + 1);
qubitDims[0] = batchSize;  // batch dimension
for (std::size_t i = 1; i <= numQubitsPerSample; i++) {
  qubitDims[i] = 2;  // qubit dimensions
}

cutensornetCreateState(
    m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, 
    numQubitsPerSample + 1, qubitDims.data(), 
    cudaDataType, &m_quantumState);
```

### 2. Gate Application

**關鍵：qubit 索引調整**

```cpp
if (m_isBatched) {
  // Adjust target qubit indices (+1 for batch dim)
  std::vector<int32_t> adjustedTargets;
  for (auto target : targetQubits) {
    adjustedTargets.push_back(target + 1);  // Skip index 0 (batch)
  }
  
  // Create batch-aware gate tensor
  void *batchGateMem = createBatchGateTensor(gateDeviceMem, targetQubits.size());
  
  // Apply to adjusted indices
  cutensornetStateApplyTensorOperator(...);
}
```

### 3. State Vector 提取

**單一樣本（batch 中的第 i 個）：**

```cpp
// Project to specific batch index
std::vector<int32_t> projectedModes = {0};  // batch mode
std::vector<int64_t> projectedValues = {static_cast<int64_t>(batchIdx)};

// Contract with projection
cutensornetAccessor_t accessor;
cutensornetCreateAccessor(m_cutnHandle, m_quantumState, 
                         projectedModes.size(), 
                         projectedModes.data(), 
                         projectedValues.data(), 
                         &accessor);
```

### 4. Gate 快取機制

類似 cutensornet 的實作，使用雜湊快取避免重複配置：

```cpp
// Cache key: gateName_params_matrixHash
std::string gateKey = task.operationName + "_" + paramString + "__" + matrixHash;

// Check cache
if (m_gateDeviceMemCache.find(gateKey) != m_gateDeviceMemCache.end()) {
  // Use cached gate
  m_state->applyGate(..., m_gateDeviceMemCache[gateKey]);
} else {
  // Allocate new gate and cache it
  void *dMem = allocateGateMatrix(task.matrix);
  m_gateDeviceMemCache[gateKey] = dMem;
  m_state->applyGate(..., dMem);
}
```

## 效能優化策略

### 1. 為什麼比 for-loop 快？

| 開銷類型 | For-loop (重複 B 次) | Batch (一次) |
|---------|---------------------|-------------|
| Kernel 啟動 | B 次 | 1 次 |
| 記憶體配置 | B 次 | 1 次 |
| 路徑規劃 | B 次 | 1 次 |
| 資料傳輸 | B 次小傳輸 | 1 次大傳輸 |

### 2. 實際優化技巧

**Memory Layout (SoA):**
```cpp
// Structure of Arrays (更好的記憶體合併訪問)
real_parts[B, 2^n]
imag_parts[B, 2^n]

// 而非 Array of Structures:
complex_data[B, 2^n]  // stride 不連續
```

**Contraction Path 重用：**
```cpp
// 使用不含 batch 維度的結構簽名做 key
std::string pathKey = gateSequence + topology;
// 同一個 path 可服務整個 batch
```

**CUDA Graph Capture（進階）：**
```cpp
// 捕捉一輪 gate 序列
cudaStreamBeginCapture(stream);
// ... apply gates ...
cudaStreamEndCapture(stream, &graph);

// 對每個 batch slice 重播
for (auto slice : batch_slices) {
  cudaGraphLaunch(graph, stream);
}
```

### 3. 避免維度爆炸的關鍵

**永不收縮 batch 維度：**
```cpp
// cuTensorNet contraction 配置
// batch index (mode 0) 標記為 "free index"
// 路徑優化器只在 quantum indices 上工作
```

## 使用範例

### C++ API

```cpp
#include <cudaq.h>

// Initialize batch state
cudaq::set_target("formotensor");

auto simulator = cudaq::get_platform().get_simulator();
auto *formoSim = dynamic_cast<SimulatorFormoTensor<double>*>(simulator);

// Prepare batch data: B samples, each 2^n amplitudes
std::size_t numQubits = 4;
std::size_t batchSize = 128;
std::vector<std::complex<double>> batchData(batchSize * (1 << numQubits));

// Initialize batch state
formoSim->initializeBatchState(numQubits, batchSize, batchData.data());

// Apply gates (automatically batched)
kernel(q);  // Your quantum circuit

// Get all batch state vectors
auto batchResults = formoSim->getBatchStateVectors();
// batchResults[i] = state vector for sample i
```

### Python API (Future)

```python
import cudaq

cudaq.set_target("formotensor")

# Create kernel
@cudaq.kernel
def circuit(theta: float):
    q = cudaq.qubit(4)
    ry(theta[0], q[0])
    cx(q[0], q[1])
    # ...

# Batch execution context
context = cudaq.BatchExecutionContext(batch_size=128)

# Batch parameters: [B, num_params]
thetas = [[theta_b_0, theta_b_1, ...] for b in range(128)]

result = cudaq.evaluate(context, circuit, thetas)
# result.expectations shape: (128,)
```

## 檔案結構

```
runtime/nvqir/formotensor/
├── formotensor_utils.h              # 共用工具（錯誤處理、常數）
├── formotensor_state.h              # State 管理類別定義
├── formotensor_state.inc            # State 實作
├── FormoTensorCircuitSimulator.h    # Simulator 類別定義
├── FormoTensorCircuitSimulator.inc  # Simulator 實作
├── FormoTensorCircuitSimulator.cpp  # 註冊入口
├── simulator_formotensor_fp64_register.cpp
├── simulator_formotensor_fp32_register.cpp
├── formotensor.yml                  # 後端配置
├── CMakeLists.txt                   # 建置配置
└── IMPLEMENTATION.md                # 本文件
```

## 批次採樣實作

### CPU-based Sampling

當前實作使用 CPU-based 採樣策略：

```cpp
// 對每個 batch 樣本
for (size_t b = 0; b < batchSize; ++b) {
  // 1. 取得該樣本的 state vector
  auto stateVec = batchStates[b];
  
  // 2. 計算測量機率分布
  std::unordered_map<std::string, double> probMap;
  for (size_t idx = 0; idx < stateDim; ++idx) {
    std::string bitstring = extractBits(idx, measuredQubits);
    probMap[bitstring] += std::norm(stateVec[idx]);
  }
  
  // 3. 使用 discrete_distribution 採樣
  std::discrete_distribution<int> dist(probs);
  for (int shot = 0; shot < shots; ++shot) {
    results[b][dist(randomEngine)]++;
  }
}
```

**優點：**
- 實作簡單、數學正確
- 適合中小規模 qubit 數（< 20）

**限制：**
- 需要完整 state vector（記憶體 = `B × 2^n × 16 bytes`）
- 對於大 qubit 數（> 25）會遇到記憶體瓶頸

### 使用範例

```cpp
// Batch 採樣 - 每個樣本獨立結果
auto batchResults = simulator->sampleBatch({0, 1, 2}, 1024);
// batchResults[i] = 第 i 個樣本的計數表

// 或聚合採樣 - 所有樣本混合
auto aggregated = simulator->sample({0, 1, 2}, 1024);
// 總共 batchSize × 1024 次採樣
```

## 噪聲通道實作

### 支援的噪聲類型

FormoTensor 支援以下噪聲通道（透過 cuTensorNet API）：

1. **Amplitude Damping** - 能量耗散
2. **Phase Flip** - 相位翻轉
3. **Bit Flip** - 位元翻轉  
4. **Depolarization** - 去極化

### Batch 模式下的噪聲

```cpp
// 噪聲應用於所有 batch 樣本
// Kraus operators: [K_0, K_1, ...]
// 在 batch mode: 每個變成 [1, 2^k, 2^k] 的張量

void applyUnitaryChannel(qubits, krausOps, probabilities) {
  if (m_isBatched) {
    // 調整 qubit indices (+1 for batch dim)
    adjustedQubits = qubits + 1;
    
    // 為每個 Kraus operator 添加 batch 維度
    for (auto K : krausOps) {
      batchK = createBatchGateTensor(K, numQubits);
      // batchK shape: [1, 2^k, 2^k]
    }
    
    // 應用到 state (廣播到所有 batch)
    cutensornetStateApplyNetworkOperator(...);
  }
}
```

**重點：**
- 所有 batch 樣本使用**相同**的噪聲通道
- 但每個樣本的噪聲**實現**（stochastic outcome）是獨立的
- 這符合 QML batch training 的常見需求

### 使用範例

```cpp
// 定義噪聲模型
cudaq::noise_model noise;
noise.add_channel<cudaq::types::amplitude_damping>({0}, 0.1);
noise.add_channel<cudaq::types::depolarization>({1}, 0.05);

// 應用噪聲
simulator->setNoiseModel(noise);

// 所有 batch 樣本都受到相同定義的噪聲影響
// 但每個樣本的隨機實現是獨立的
```

## 限制與未來工作

### 當前限制

1. ✅ **Sampling 已實作**：batch mode 使用 CPU-based 採樣
2. ✅ **Noise Channel 已實作**：支援常見噪聲類型
3. ❌ **動態 Qubit 分配**：batch state 不支援動態添加 qubit
4. ❌ **GPU-based Sampling**：大 qubit 數時採樣效率較低

### 未來擴展

1. **Batch Sampling**
   - 對每個樣本獨立採樣
   - 或使用 `cutensornetStateSampler_t` 批次化

2. **MPS Factorization**
   - 在 batch 維度上共享 MPS 結構
   - 減少深電路的記憶體需求

3. **Heisenberg Picture**
   - 預編譯 $U^\dagger O U$ 為 MPO
   - 對固定門序列的期望值計算極速優化

4. **分散式批次**
   - 使用 MPI 將 batch 分散到多 GPU
   - 結合 cuTensorNet 的分散式支援

## 參考文獻

1. cuTensorNet 官方文檔：https://docs.nvidia.com/cuda/cutensornet/
2. CUDA-Q 原始碼：`runtime/nvqir/cutensornet/`
3. 與 Gemini/GPT 的討論記錄（見專案根目錄）

## 聯絡

專案開發者：Charless
Email: [Your Email]
