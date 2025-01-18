# Influence of Codebook Perplexity in One-Shot Voice Conversion Based on Vector Quantization Method

---

### **Overview**
This project investigates the impact of codebook perplexity on one-shot voice conversion using Vector Quantization (VQ). By integrating the CVQ-VAE (Clustered Vector Quantization Variational Autoencoder) update mechanism into the VQVC (Vector Quantization Voice Conversion) model, we explore its effects on:
- Codebook utility (perplexity),
- Training performance (reconstruction accuracy),
- Test performance (voice conversion accuracy).

---

### **Motivation**
- **Voice Conversion Task**:
  - Transforms the source speaker’s voice into a target speaker’s voice while preserving the original content.
  - VQVC utilizes vector quantization to discretize speech features and enables one-shot voice conversion.
- **Challenges**:
  - **Codebook Collapse**: Low perplexity values can hinder codebook utilization.
  - Previous solutions in vision tasks inspired this study to tackle similar issues in voice conversion.

---

### **Proposed Method**
1. **CVQ-VAE Integration**:
   - The VQ codebook update rule in VQVC is replaced with the CVQ-VAE update mechanism, which averages sampled vectors from the activated codebook to enhance perplexity.
2. **Evaluation Metrics**:
   - **Codebook Utility**: Measured via perplexity.
   - **Training Performance**: Evaluated through reconstruction accuracy.
   - **Test Performance**: Assessed by similarity between the source and converted voices.

| **CVQ-VAE Codebook Update** |
|------------------------------|
| <img src="images/cvq.png" alt="CVQ-VAE Codebook Update" width="80%"> |

---

### **Key Results**
1. **Codebook Utility**:
   - CVQ significantly improves codebook perplexity, enhancing utilization of the VQ codebook.
   
   | **Perplexity Comparisons** |
   |----------------------------|
   | <img src="images/perplexity" alt="Perplexity Comparisons" width="75%"> |

2. **Training Performance**:
   - Reconstruction performance increases with the integration of CVQ-VAE.

   | **Reconstruction Comparisons** |
   |----------------------------|
   | <img src="images/reconstructioin" alt="Reconstructioin Comparisons" width="75%"> |

3. **Test Performance**:
   - Conversion performance shows no improvement and sometimes declines, indicating a trade-off between reconstruction accuracy and conversion effectiveness.

   | **Similarity Scores** |
   |------------------------|
   | <img src="images/sim_comparison" alt="Similarity Scores" width="60%"> |

4. **Result Summary**:
   - Enhanced perplexity leads to better reconstruction performance but does not necessarily improve voice conversion accuracy. This is consistent with findings in vision tasks, where higher perplexity benefits reconstruction but may disrupt speaker-content separation.

   | **Performance Summary** |
   |--------------------------|
   | <img src="images/concolusion" alt="Performance Summary" width="85%"> |

---

### **Conclusion**
- Integrating CVQ-VAE improves codebook utilization and reconstruction performance.
- Enhanced perplexity interrupts the regularization role of VQ, hindering speaker-content separation and affecting conversion performance.
- The study highlights the trade-off between reconstruction quality and conversion accuracy, providing insights for future research.
