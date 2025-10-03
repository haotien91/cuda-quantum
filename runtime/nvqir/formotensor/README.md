# FormoTensor Backend

FormoTensor 是一個支援 batch 處理的 CUDA-Q 模擬器後端，專為 Quantum-ML 應用設計。它基於 cuTensorNet 並擴展了對 `d × 2^n` 狀態空間的支援，實現真正的 tensor 並行處理。

## 核心特性

### 1. Batch 處理支援
- **狀態空間**：`d × 2^n` (d 個樣本，每個 n 個 qubits)
- **Tensor 形狀**：`(d, 2, 2, ..., 2)` (1 個 batch 維度 + n 個 qubit 維度)
- **並行計算**：Gate 在 batch 維度上自動 broadcast

### 2. 效能優勢
- **減少 kernel 啟動次數**：從 d 次減少到 1 次
- **更好的 GPU 利用率**：batch 維度提供更多並行性
- **共享優化成本**：contraction path 優化只做一次
- **記憶體存取優化**：連續的 batch 資料存取更有效率

## API 使用方式

### 基本 Batch 初始化

```cpp
#include "FormoTensorCircuitSimulator.h"

// 建立 simulator
SimulatorFormoTensor<double> simulator;

// 初始化 batch 狀態
const int numQubits = 4;        // 每個樣本的 qubit 數量
const int batchSize = 8;        // batch 大小
const int stateSize = 1 << numQubits;  // 每個狀態的大小

// 準備 batch 資料 (shape: [batchSize, stateSize])
std::vector<std::complex<double>> batchData(batchSize * stateSize);
// ... 填入資料 ...

simulator.initializeBatchState(numQubits, batchSize, batchData.data());

// 驗證 batch 模式
assert(simulator.isBatchMode());
assert(simulator.getBatchSize() == batchSize);
assert(simulator.getNumQubitsPerSample() == numQubits);
```

### Gate 應用

```cpp
// Gate 會自動在 batch 維度上 broadcast
// 例如：H gate 會同時作用於所有 batch 樣本

// 建立 Hadamard gate
std::vector<std::complex<double>> hGate = {
    1.0/std::sqrt(2.0),  1.0/std::sqrt(2.0),
    1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0)
};

// 應用 gate (會同時作用於所有 batch 樣本)
// 注意：實際使用時需要透過 CUDA-Q 的 gate 系統
```

### 狀態查詢

```cpp
// 取得所有 batch 樣本的狀態
auto batchStates = simulator.getBatchStateVectors();
// batchStates[i] 是第 i 個樣本的狀態向量

// 注意：無法直接取得單一狀態向量，因為這是 batch 模式
```

## 數學模型

### 狀態表示
```
單一樣本：|ψ⟩ ∈ C^(2^n)
Batch 狀態：|Ψ⟩ ∈ C^d ⊗ C^(2^n) = C^(d × 2^n)
```

### Gate 作用
```
Gate 作用：I_d ⊗ U
其中 U 是原本的量子門，I_d 是 batch 維度的單位矩陣
```

### Tensor 結構
```
原始狀態：(2, 2, ..., 2) - n 個 qubit 維度
Batch 狀態：(d, 2, 2, ..., 2) - 1 個 batch 維度 + n 個 qubit 維度
```

## 建置和安裝

### 1. 建置
```bash
cd runtime/nvqir/formotensor
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. 安裝
```bash
make install
```

### 3. 使用
```bash
# 設定環境變數
export CUDAQ_TARGET=formotensor

# 或使用 Python
import cudaq
cudaq.set_target("formotensor")
```

## 測試

執行測試程式：
```bash
./test_batch
```

測試包含：
- 單一狀態初始化
- Batch 狀態初始化
- 效能比較 (batch vs sequential)
- 錯誤處理

## 限制和注意事項

### 目前限制
1. **Noise channels**：尚未支援 batch 模式下的 noise channels
2. **Sampling**：尚未支援 batch 模式下的 sampling
3. **Gate 應用**：需要透過 CUDA-Q 的 gate 系統，不能直接呼叫

### 使用注意事項
1. **記憶體需求**：batch 模式需要更多 GPU 記憶體
2. **Batch 大小**：建議 batch 大小為 2 的冪以獲得最佳效能
3. **狀態查詢**：batch 模式下無法直接取得單一狀態向量

## 效能預期

基於 cuTensorNet 的優化能力，預期效能提升：

- **小 batch (2-8)**：2-4x 加速
- **中等 batch (16-64)**：4-8x 加速  
- **大 batch (128+)**：8-16x 加速

實際效能取決於：
- GPU 型號和記憶體
- 電路複雜度
- Batch 大小
- Qubit 數量

## 未來改進

1. **Noise 支援**：實作 batch 模式下的 noise channels
2. **Sampling 支援**：實作 batch 模式下的 sampling
3. **Python 綁定**：提供 Python API 直接支援
4. **自動批次化**：自動將多個獨立計算合併為 batch
5. **記憶體優化**：更智能的記憶體管理策略

## 貢獻

歡迎貢獻代碼和建議！請確保：
1. 遵循現有的代碼風格
2. 添加適當的測試
3. 更新文檔
4. 通過所有測試
