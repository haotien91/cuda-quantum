# FormoTensor Batch Simulator - API 快速參考

## 目錄

1. [初始化 Batch State](#1-初始化-batch-state)
2. [量子門操作](#2-量子門操作)
3. [批次採樣](#3-批次採樣)
4. [噪聲通道](#4-噪聲通道)
5. [State Vector 提取](#5-state-vector-提取)
6. [狀態管理](#6-狀態管理)

---

## 1. 初始化 Batch State

### 1.1 從零態初始化

```cpp
#include "FormoTensorCircuitSimulator.h"

auto *sim = new SimulatorFormoTensor<double>();

std::size_t numQubits = 4;      // 每個樣本的 qubit 數
std::size_t batchSize = 128;    // batch 大小

// 初始化為 |0...0⟩ 狀態
sim->initializeBatchState(numQubits, batchSize, nullptr);
```

### 1.2 從自訂 State Vector 初始化

```cpp
// 準備 batch data: shape = [batchSize, 2^numQubits]
std::size_t stateSize = 1ULL << numQubits;
std::vector<std::complex<double>> batchData(batchSize * stateSize);

// 填充 batch data
for (std::size_t b = 0; b < batchSize; ++b) {
  for (std::size_t i = 0; i < stateSize; ++i) {
    batchData[b * stateSize + i] = /* your state */;
  }
}

// 初始化
sim->initializeBatchState(numQubits, batchSize, batchData.data());
```

### 1.3 檢查 Batch 狀態

```cpp
bool isBatch = sim->isBatchMode();
std::size_t B = sim->getBatchSize();
std::size_t n = sim->getNumQubitsPerSample();

std::cout << "Batch mode: " << (isBatch ? "Yes" : "No") << "\n";
std::cout << "Batch size: " << B << "\n";
std::cout << "Qubits per sample: " << n << "\n";
```

---

## 2. 量子門操作

### 2.1 單 Qubit Gates

```cpp
// H, X, Y, Z, S, T gates
sim->h(0);           // Hadamard on qubit 0
sim->x(1);           // Pauli X on qubit 1
sim->y(2);           // Pauli Y on qubit 2
sim->z(3);           // Pauli Z on qubit 3

// Rotation gates
sim->rx(M_PI / 4, 0);   // RX(π/4) on qubit 0
sim->ry(M_PI / 2, 1);   // RY(π/2) on qubit 1
sim->rz(M_PI, 2);       // RZ(π) on qubit 2
```

### 2.2 Controlled Gates

```cpp
// CNOT (CX)
sim->x({0}, 1);  // CNOT: control=0, target=1

// Controlled-Y
sim->y({0}, 1);

// Toffoli (CCX)
sim->x({0, 1}, 2);  // Control=0,1, Target=2

// Multi-controlled rotation
sim->ry(M_PI / 3, {0, 1, 2}, 3);  // CCC-RY
```

### 2.3 自訂 Gate

```cpp
// 定義 2x2 unitary matrix
std::vector<std::complex<double>> gateMatrix = {
  {1.0, 0.0}, {0.0, 0.0},
  {0.0, 0.0}, {0.0, 1.0}
};

// 應用到 qubit 0
sim->applyCustomOperation(gateMatrix, {}, {0});

// 應用受控自訂閘：control=0, target=1
sim->applyCustomOperation(gateMatrix, {0}, {1});
```

**重要：** 所有 gate 自動應用到所有 batch 樣本（廣播）

---

## 3. 批次採樣

### 3.1 獨立採樣（每個樣本單獨結果）

```cpp
// 測量 qubits {0, 1, 2}，每個樣本 1024 shots
auto batchResults = sim->sampleBatch({0, 1, 2}, 1024);

// batchResults[b] = 第 b 個樣本的計數表
for (std::size_t b = 0; b < batchSize; ++b) {
  std::cout << "Sample " << b << ":\n";
  for (const auto &[bitstring, count] : batchResults[b]) {
    std::cout << "  " << bitstring << ": " << count << "\n";
  }
}
```

**輸出範例：**
```
Sample 0:
  000: 512
  111: 512
Sample 1:
  001: 1024
...
```

### 3.2 聚合採樣（所有樣本混合）

```cpp
// 測量並聚合所有 batch 的結果
auto aggregated = sim->sample({0, 1, 2}, 1024);

// aggregated 包含 batchSize × 1024 次測量的總計數
for (const auto &[bitstring, count] : aggregated) {
  std::cout << bitstring << ": " << count << "\n";
}
```

**輸出範例：**
```
000: 32768    // 來自所有 128 個樣本
001: 16384
...
總計: 128 × 1024 = 131072 次測量
```

### 3.3 部分 Qubit 測量

```cpp
// 只測量 qubits {0, 2}（跳過 qubit 1）
auto results = sim->sampleBatch({0, 2}, 2048);

// Bitstring 長度 = 2 (對應 qubits 0 和 2)
// results[b] = {"00": count1, "01": count2, "10": count3, "11": count4}
```

---

## 4. 噪聲通道

### 4.1 定義噪聲模型

```cpp
#include <cudaq/noise_model.h>

cudaq::noise_model noise;

// Amplitude damping on qubit 0 (T1 decay)
noise.add_channel<cudaq::types::amplitude_damping>({0}, 0.1);

// Phase flip on qubit 1
noise.add_channel<cudaq::types::phase_flip>({1}, 0.05);

// Depolarization on all 2-qubit gates
noise.add_all_qubit_channel<cudaq::types::depolarization>(2, 0.01);

// 應用噪聲模型
sim->setNoiseModel(noise);
```

### 4.2 支援的噪聲類型

| 類型 | 參數 | 物理意義 |
|------|------|---------|
| `amplitude_damping` | γ ∈ [0,1] | 能量耗散（T1 decay） |
| `phase_flip` | p ∈ [0,1] | 隨機相位翻轉 |
| `bit_flip` | p ∈ [0,1] | 隨機位元翻轉 |
| `depolarization` | p ∈ [0,1] | 完全去極化 |

### 4.3 Batch 模式下的噪聲行為

```cpp
// 所有 batch 樣本受到相同定義的噪聲
// 但隨機實現對每個樣本是獨立的

// 範例：amplitude damping with γ = 0.3
noise.add_channel<cudaq::types::amplitude_damping>({0}, 0.3);

// 結果：
// - Sample 0: 可能發生衰減（機率 30%）
// - Sample 1: 可能不發生衰減（機率 70%）
// - Sample 2: 可能發生衰減（機率 30%）
// ... (每個樣本獨立隨機)
```

### 4.4 檢查噪聲支援

```cpp
bool supported = sim->isValidNoiseChannel(
    cudaq::noise_model_type::amplitude_damping);

if (supported) {
  std::cout << "Amplitude damping is supported!\n";
}
```

---

## 5. State Vector 提取

### 5.1 提取所有 Batch State Vectors

```cpp
// 返回 vector<vector<complex<double>>>
// batchStates[b][i] = 第 b 個樣本的第 i 個 basis state 振幅
auto batchStates = sim->getBatchStateVectors();

for (std::size_t b = 0; b < batchSize; ++b) {
  std::cout << "Sample " << b << ":\n";
  for (std::size_t i = 0; i < batchStates[b].size(); ++i) {
    std::cout << "  |" << i << "⟩: " << batchStates[b][i] << "\n";
  }
}
```

### 5.2 計算期望值

```cpp
// 對每個樣本計算 <Z_0>
auto batchStates = sim->getBatchStateVectors();
std::vector<double> expectations(batchSize);

for (std::size_t b = 0; b < batchSize; ++b) {
  double expZ = 0.0;
  std::size_t halfDim = batchStates[b].size() / 2;
  
  // <Z> = P(0) - P(1) for qubit 0
  for (std::size_t i = 0; i < halfDim; ++i) {
    expZ += std::norm(batchStates[b][i]);  // |0⟩ 子空間
  }
  for (std::size_t i = halfDim; i < batchStates[b].size(); ++i) {
    expZ -= std::norm(batchStates[b][i]);  // |1⟩ 子空間
  }
  
  expectations[b] = expZ;
}
```

### 5.3 單一樣本模式

```cpp
// 如果只有 1 個樣本（非 batch mode）
if (!sim->isBatchMode()) {
  auto stateVec = sim->getStateVector();
  // stateVec[i] = amplitude of |i⟩
}
```

---

## 6. 狀態管理

### 6.1 重置狀態

```cpp
// 清除當前狀態，釋放記憶體
sim->resetState();

// 之後可以重新初始化
sim->initializeBatchState(numQubits, batchSize, nullptr);
```

### 6.2 查詢 Simulator 資訊

```cpp
// Simulator 名稱
std::string name = sim->name();
std::cout << "Using simulator: " << name << "\n";
// Output: "formotensor" (fp64) or "formotensor-fp32" (fp32)

// 精度查詢
bool isSinglePrecision = sim->isSinglePrecision();
bool isDoublePrecision = sim->isDoublePrecision();
```

### 6.3 記憶體考量

```cpp
// 估算記憶體使用
std::size_t memoryBytes = batchSize * (1ULL << numQubits) * sizeof(std::complex<double>);
double memoryGB = memoryBytes / (1024.0 * 1024.0 * 1024.0);

std::cout << "Estimated memory: " << memoryGB << " GB\n";

// 範例：
// B=128, n=20: 128 × 2^20 × 16 bytes = 2 GB
// B=128, n=25: 128 × 2^25 × 16 bytes = 64 GB (可能超過記憶體限制)
```

---

## 完整範例：QML Batch Training

```cpp
#include "FormoTensorCircuitSimulator.h"
#include <cudaq/noise_model.h>
#include <vector>
#include <complex>

int main() {
  // === 1. 初始化 ===
  const std::size_t numQubits = 4;
  const std::size_t batchSize = 32;
  
  auto *sim = new SimulatorFormoTensor<double>();
  sim->initializeBatchState(numQubits, batchSize, nullptr);
  
  std::cout << "Initialized batch state: " << batchSize 
            << " samples, " << numQubits << " qubits\n";
  
  // === 2. 定義參數化電路 ===
  std::vector<std::vector<double>> batchParams(batchSize);
  for (std::size_t b = 0; b < batchSize; ++b) {
    batchParams[b] = {
      /* theta_0 */ M_PI * (b / static_cast<double>(batchSize)),
      /* theta_1 */ M_PI / 4,
      /* theta_2 */ M_PI / 2
    };
  }
  
  // === 3. 應用電路（所有 batch 共享結構，參數不同） ===
  // 注意：這裡簡化了，實際需要為每個 batch 設置不同參數
  sim->h(0);
  sim->ry(batchParams[0][0], 1);  // 簡化範例
  sim->x({0}, 1);
  sim->ry(batchParams[0][1], 2);
  sim->x({1}, 2);
  sim->ry(batchParams[0][2], 3);
  
  // === 4. 添加噪聲（可選） ===
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::depolarization>({0, 1, 2, 3}, 0.01);
  sim->setNoiseModel(noise);
  
  // === 5. 批次採樣 ===
  auto batchResults = sim->sampleBatch({0, 1, 2, 3}, 1024);
  
  // === 6. 計算損失函數 ===
  std::vector<double> losses(batchSize);
  for (std::size_t b = 0; b < batchSize; ++b) {
    // 假設標籤為 "0000"
    std::string targetLabel = "0000";
    double correct = batchResults[b][targetLabel];
    double total = 1024.0;
    losses[b] = 1.0 - (correct / total);  // 簡化的損失函數
    
    std::cout << "Sample " << b << " loss: " << losses[b] << "\n";
  }
  
  // === 7. 清理 ===
  sim->resetState();
  delete sim;
  
  return 0;
}
```

**編譯：**
```bash
g++ -std=c++20 example.cpp \
    -I/path/to/cuda-quantum/include \
    -L/path/to/cuda-quantum/lib \
    -lnvqir_formotensor -lcutensornet -lcudart \
    -o batch_qml
```

---

## 效能最佳化建議

### 1. 選擇合適的 Batch Size

```cpp
// 記憶體限制：B × 2^n × 16 bytes < GPU memory
// GPU 利用率：B ≥ 32 for good parallelism
// 建議範圍：
std::size_t optimalBatchSize = [numQubits]() {
  if (numQubits <= 10) return 512;
  if (numQubits <= 15) return 128;
  if (numQubits <= 20) return 32;
  return 8;  // Large qubit count
}();
```

### 2. Gate 重用（自動快取）

```cpp
// Simulator 自動快取 gate 矩陣
// 重複使用的 gate 只配置一次記憶體

for (int layer = 0; layer < 10; ++layer) {
  sim->ry(M_PI / 4, 0);  // 同參數 gate 會被快取
  sim->x({0}, 1);        // CNOT 也會被快取
}
```

### 3. 避免頻繁 State Vector 提取

```cpp
// ❌ 低效：
for (int i = 0; i < 100; ++i) {
  sim->applyGate(...);
  auto states = sim->getBatchStateVectors();  // 每次都提取
}

// ✅ 高效：
for (int i = 0; i < 100; ++i) {
  sim->applyGate(...);
}
auto states = sim->getBatchStateVectors();  // 只提取一次
```

### 4. 精度選擇

```cpp
// Float32 (faster, 2x less memory)
auto *simFP32 = new SimulatorFormoTensor<float>();

// Float64 (default, higher precision)
auto *simFP64 = new SimulatorFormoTensor<double>();
```

---

## 錯誤處理

### 常見錯誤訊息

| 錯誤訊息 | 原因 | 解決方法 |
|---------|------|---------|
| `"State not initialized"` | 未初始化狀態就呼叫操作 | 先呼叫 `initializeBatchState()` |
| `"sampleBatch only valid in batch mode"` | 在單一樣本模式呼叫 | 檢查 `isBatchMode()` |
| `"Cannot get single state vector from batch state"` | 在 batch 模式呼叫 `getStateVector()` | 改用 `getBatchStateVectors()` |
| `"CUDA error: out of memory"` | Batch size 太大 | 減少 batch size 或 qubit 數 |

### 防禦性編程

```cpp
try {
  sim->initializeBatchState(numQubits, batchSize, data);
  
  if (!sim->isBatchMode()) {
    throw std::runtime_error("Failed to enter batch mode");
  }
  
  // Apply circuit...
  
  auto results = sim->sampleBatch({0, 1}, 1024);
  
} catch (const std::exception &e) {
  std::cerr << "Error: " << e.what() << "\n";
  sim->resetState();
  return -1;
}
```

---

## 版本資訊

- **當前版本**: v1.1
- **支援精度**: FP32, FP64
- **支援平台**: CUDA 11.0+, cuTensorNet 2.0+
- **最後更新**: 2025-01-15

---

## 快速查找表

| 需求 | API | 檔案 |
|------|-----|------|
| 初始化 batch | `initializeBatchState()` | `FormoTensorCircuitSimulator.inc` |
| 應用量子門 | `h()`, `x()`, `ry()`, ... | `CircuitSimulator.h` |
| 批次採樣 | `sampleBatch()` | `formotensor_state.inc` |
| 聚合採樣 | `sample()` | `FormoTensorCircuitSimulator.inc` |
| 噪聲通道 | `setNoiseModel()` | `CircuitSimulator.h` |
| 提取 state | `getBatchStateVectors()` | `formotensor_state.inc` |
| 重置狀態 | `resetState()` | `FormoTensorCircuitSimulator.inc` |

**詳細文件：**
- 設計原理：`IMPLEMENTATION.md`
- 批次採樣與噪聲：`BATCH_SAMPLING_NOISE.md`
- 修改總結：`CHANGES_SUMMARY.md`

---

**維護者**: Charless  
**最後編輯**: 2025-01-15
