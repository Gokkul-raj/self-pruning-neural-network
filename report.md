# 📄 Case Study Report – Self-Pruning Neural Network

---

## 🧠 1. Introduction

In real-world deployments, neural networks are often constrained by memory and computational resources. To address this, model pruning techniques are used to remove unnecessary parameters and improve efficiency.

In this project, we implement a **self-pruning neural network** that learns to remove its own weights during training using a learnable gating mechanism. Unlike traditional pruning methods that operate after training, this approach integrates pruning directly into the training process.

---

## ⚙️ 2. Methodology

### 2.1 Prunable Linear Layer

A custom layer, `PrunableLinear`, was implemented to replace the standard fully connected layer.

Each weight in the layer is associated with a learnable parameter called **gate score**. These gate scores are passed through a sigmoid function to obtain values between 0 and 1:


- If gate → 1 → weight remains active  
- If gate → 0 → weight is effectively removed  

This enables the model to dynamically decide which connections are important.

---

### 2.2 Sparsity Regularization

To encourage pruning, an additional loss term is introduced:


Where:
- Classification Loss = CrossEntropyLoss  
- Sparsity Loss = L1 norm of gate values  

The L1 regularization penalizes non-zero gate values, pushing many gates toward zero and promoting sparsity.

---

## 📊 3. Experiments

The model was trained on the **CIFAR-10 dataset** using different values of λ to analyze the trade-off between accuracy and sparsity.

### Results:

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 1e-5   | ~55%    | ~5%      |
| 1e-4   | ~50%    | ~30%     |
| 1e-3   | **44.84%** | **58.25%** |

---

## 📈 4. Analysis

### 4.1 Effect of Lambda (λ)

- **Low λ (1e-5):**
  - Minimal sparsity
  - High accuracy
  - Model retains most connections

- **Medium λ (1e-4):**
  - Balanced sparsity and accuracy
  - Moderate pruning observed

- **High λ (1e-3):**
  - High sparsity (~58%)
  - Reduced accuracy (~45%)
  - Strong pruning effect

---

### 4.2 Gate Distribution

The distribution of gate values shows a **large spike near zero**, indicating that many weights have been effectively pruned.

This confirms that the model successfully learns to suppress unimportant connections during training.

---

## 🧠 5. Why L1 Regularization Encourages Sparsity

L1 regularization adds a penalty proportional to the absolute value of parameters:


This has the effect of:
- Driving many values exactly to zero  
- Encouraging sparse representations  

Unlike L2 regularization, which only reduces magnitude, L1 promotes actual elimination of parameters.

---

## 🎯 6. Key Observations

- The model successfully learns to prune itself during training.
- Increasing λ increases sparsity but reduces accuracy.
- There exists a clear trade-off between model efficiency and performance.
- Optimal performance is achieved at intermediate λ values.

---

## 🏆 7. Conclusion

This project demonstrates that neural networks can be made more efficient by incorporating pruning directly into the training process.

The use of learnable gates and L1 regularization enables the model to automatically identify and remove unnecessary connections, resulting in a compact and efficient architecture.

At λ = 1e-3, the model achieved **~58% sparsity** while maintaining acceptable accuracy, validating the effectiveness of the approach.

---

## 🚀 8. Future Work

- Extend to structured pruning (neurons/channels)
- Apply to deeper architectures (CNNs)
- Optimize for real-time deployment
- Explore adaptive λ tuning

---