# G$^2$N$^2$: Weisfeiler and Lehman Go Grammatical

This repository contains the code for the experiments and models presented in our paper *G$^2$N$^2$: Weisfeiler and Lehman Go Grammatical*, accepted at ICLR 2024. The implementation uses Python, PyTorch, and CUDA.

## Requirements

- **Python version**: 3.8.10  
- **PyTorch version**: 1.12.0  

Ensure that PyTorch and CUDA are installed correctly before running the experiments.

---

## Reproducing Experiments

### **QM9 Experiment**
1. Open `qm9_dataset.py`.
2. Set the `ntask` variable to the target you want to learn (values range from 0 to 11).
3. Run the script:
   ```bash
   python qm9_dataset.py
   ```

### **QM9 12 Targets**
To evaluate all 12 targets in QM9:
```bash
python qm9_dataset_12_labels.py
```

For experiments using the GNN derived from the exhaustive CFG:
```bash
python qm9_dataset_12_labels_exhaust_GNN.py
```

---

### **TUD Experiment**
1. Open `TU_dataset.py`.
2. Configure the G$^2$N$^2$ settings according to the configurations provided in the supplementary material of the paper.  
   - The variable `Name` corresponds to the dataset you want to use.
3. Run the script:
   ```bash
   python TU_dataset.py
   ```

To view the results, run:
```bash
python TUD_results.py
```

---

### **Filtering Experiment**
1. Open `G2N2_filtering.py`.
2. Select the type of filter by setting the `ntask` variable:
   - `0`: Low-pass  
   - `1`: High-pass  
   - `2`: Band-pass  
3. Run the script:
   ```bash
   python G2N2_filtering.py
   ```

---

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{piquenot2024grammatical,
  title={G$^2$N$^2$: Weisfeiler and Lehman Go Grammatical},
  author={Piquenot, Jason and Moscatelli, Aldo and Berar, Maxime and Héroux, Pierre and Raveaux, Romain and RAMEL, Jean-Yves and Adam, Sébastien},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}


