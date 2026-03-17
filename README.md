# LSH-DP

Locality-Sensitive Hashing-Based Dataset Reduction for Deep Potential Training.

> Anmol, Anuj Kumar Sirohi, Neha, Jayadeva, Sandeep Kumar, and Tarak Karmakar.
> *J. Chem. Theory Comput.* **2025**, 21 (12), 6113–6120.

---

## Installation

### CPU

```bash
conda env create -f environment.yml
conda activate lshdp
pip install -e .
```

### GPU

Pre-built environment files with PyTorch CUDA via pip (avoids conda channel conflicts):

```bash
# CUDA 12.1 (driver >= 530.30.02)
conda env create -f environment-gpu-cu121.yml

# CUDA 11.8 (driver >= 520.61.05)
conda env create -f environment-gpu-cu118.yml

conda activate lshdp-gpu
pip install -e .
```

Check your driver version with `nvidia-smi`. The CUDA version in the top-right corner is the maximum supported — pick an environment file at or below that version.

### Verify

```bash
lshdp info
```

---

## Usage

```bash
# Generate example config
lshdp init-config -o config.yaml

# Run full pipeline
lshdp run --config config.yaml

# CLI overrides
lshdp run --config config.yaml --bin-width 0.01 --device cuda --n-jobs 16

# Run specific steps
lshdp run --config config.yaml --start-step 1 --end-step 4

# Resume from step 5
lshdp run --config config.yaml --start-step 5

# Validate config
lshdp validate config.yaml
```

---

## Configuration

```yaml
soap:
  r_cut: 6.0
  n_max: 4
  l_max: 4
  sigma: 1.0
  rbf: gto
  periodic: true
  n_jobs: -1          # parallel SOAP workers (-1 = all cores)

hashing:
  n_components: 100
  n_hash: 100
  bin_width: 0.004
  random_seed: 42

io:
  input_file: simulation.xyz
  output_dir: results
  format: auto
  # cell: [15.82, 15.82, 30.76]
  # pbc: true

split:
  frames_per_file: 120

device: auto          # cpu, cuda, or auto
deterministic: true
```

`n_jobs` controls parallel SOAP computation: `1` = serial, `8`/`16` = fixed worker count, `-1` = all cores. For large trajectories (>50k frames), `n_jobs: 16` often outperforms `-1` due to lower serialization overhead.

---

## Pipeline

| Step | Description |
|------|-------------|
| 1 | Compute SOAP descriptors (parallel via DScribe) |
| 2 | Flatten → PCA → LSH hashing → buckets |
| 3 | Organise frames into bins |
| 4 | Extract representative frame numbers |
| 5 | Update XYZ metadata |
| 6 | Extract selected frames |
| 7 | Split trajectory into parts |

GPU acceleration (`device: cuda`) applies to Step 2 (normalization, random projection, hashing). SOAP computation (Step 1) is CPU-parallelized via joblib.

---

## Supported Formats

Any format readable by ASE: XYZ, extended XYZ, LAMMPS (`.lammpstrj`), CIF, VASP (`POSCAR`/`CONTCAR`), PDB, and others. Set `format: auto` for detection or specify explicitly.

---

## Citation

```bibtex
@article{anmol2025lshdp,
  author  = {Anmol and Sirohi, Anuj Kumar and Neha and Jayadeva and Kumar, Sandeep and Karmakar, Tarak},
  title   = {Locality-Sensitive Hashing-Based Data Set Reduction for Deep Potential Training},
  journal = {J. Chem. Theory Comput.},
  year    = {2025},
  volume  = {21},
  number  = {12},
  pages   = {6113--6120}
}
```

```bash
lshdp cite
```

---

## License

MIT
