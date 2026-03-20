# LSH-DP

Locality-Sensitive Hashing-Based Dataset Reduction for Deep Potential Training.

> Anmol, Anuj Kumar Sirohi, Neha, Jayadeva, Sandeep Kumar, and Tarak Karmakar.
> *J. Chem. Theory Comput.* **2025**, 21 (12), 6113–6120.

---

## Installation

```bash
# CPU
conda env create -f environment.yml

# GPU — pick your CUDA version (check with nvidia-smi)
conda env create -f environment-gpu-cu121.yml   # CUDA 12.1
conda env create -f environment-gpu-cu118.yml   # CUDA 11.8

conda activate lshdp      # or lshdp-gpu
pip install -e .
lshdp info                 # verify
```

---

## Usage

```bash
lshdp init-config -o config.yaml    # generate example config
lshdp run --config config.yaml      # run full pipeline
lshdp validate config.yaml          # check config
```

CLI overrides:

```bash
lshdp run -c config.yaml --bin-width 0.01 --device cuda --n-jobs 16 --selection medoid --output-format extxyz
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
  n_jobs: -1              # -1 = all cores, 16 recommended for >50k frames

hashing:
  n_components: 100
  n_hash: 100
  bin_width: 0.004
  random_seed: 42

io:
  input_file: simulation.xyz
  output_dir: results
  format: auto            # auto, extxyz, gro, lammps-dump-text, cif, vasp, ...
  output_format: extxyz   # output format for extracted frames
  # cell: [15.82, 15.82, 30.76]  # fallback for formats without cell info
  # pbc: true

selection:
  method: medoid          # first, random, or medoid

split:
  frames_per_file: 120

device: auto
deterministic: true
```

---

## Supported Formats

Any format readable by ASE: extXYZ, GRO, LAMMPS dump, CIF, VASP, PDB, plain XYZ, and others. Use `extxyz` as output format — it preserves per-frame cell, PBC, and species.

For plain XYZ (no cell in file), set `io.cell` and `io.pbc` in config.

---

## NPT Simulations

Per-frame cell information is preserved automatically. The `io.cell` override is a fallback — it only applies to frames missing cell data, never overwrites existing cells. If `soap.periodic: true` is set but frames have no cell, the pipeline errors out before computing.

---

## Frame Selection

`selection.method` controls how representative frames are picked from each LSH bin: `first` (first frame), `random` (random pick), or `medoid` (frame closest to bin centroid in SOAP descriptor space — default, best quality).

---

## Pipeline

| Step | Description |
|------|-------------|
| 1 | SOAP descriptors (parallel via DScribe) |
| 2 | Flatten → PCA → LSH hashing → buckets |
| 3 | Organise frames into bins |
| 4 | Select representative frames per bin |
| 5 | Extract selected frames (via ASE) |
| 6 | Split trajectory into parts |

GPU (`device: cuda`) applies to Step 2. SOAP (Step 1) is CPU-parallelized via `n_jobs`.

---

## Citation

```bibtex
@article{anmol2025lshdp,
  author  = {Anmol and Sirohi, Anuj Kumar and Neha and Jayadeva
             and Kumar, Sandeep and Karmakar, Tarak},
  title   = {Locality-Sensitive Hashing-Based Data Set Reduction
             for Deep Potential Training},
  journal = {J. Chem. Theory Comput.},
  year    = {2025},
  volume  = {21},
  number  = {12},
  pages   = {6113--6120}
}
```

## License

MIT
