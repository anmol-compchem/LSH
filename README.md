# LSH-DP: Locality-Sensitive Hashing Dataset Reduction for Deep Potential Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready Python package implementing the LSH-based dataset reduction algorithm for training deep potentials from molecular dynamics trajectories.

## Reference

> **Locality-Sensitive Hashing-Based Data Set Reduction for Deep Potential Training**
> Anmol, Anuj Kumar Sirohi, Neha, Jayadeva, Sandeep Kumar, and Tarak Karmakar
> *J. Chem. Theory Comput.* **2025**, 21, 12, 6113–6120

This software is a faithful implementation of the algorithm described in the above publication. The scientific logic and mathematical formulation are preserved exactly.

---

## Features

- **SOAP descriptors** via DScribe with fully configurable parameters
- **Locality-Sensitive Hashing** with PCA dimensionality reduction
- **GPU acceleration** (optional) for tensor operations via PyTorch
- **Universal trajectory support** — any format readable by ASE (XYZ, extXYZ, LAMMPS, CIF, POSCAR, …)
- **YAML configuration** — no hard-coded parameters
- **Professional CLI** with `lshdp run`, `lshdp info`, `lshdp cite`
- **Structured logging** with per-step timing and hardware metadata
- **Reproducible** — deterministic seeding and optional strict mode
- **Step-wise execution** — start/resume from any pipeline step

---

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate lshdp
```

**For GPU support**, edit `environment.yml` before creating the environment:
- Remove the `- cpuonly` line
- Uncomment the appropriate `pytorch-cuda` line for your CUDA version

### 2. Install the package

```bash
pip install -e .
```

### 3. Verify

```bash
lshdp info
```

---

## Quick Start

### Generate an example config

```bash
lshdp init-config -o config.yaml
```

### Edit `config.yaml`

Set your input file, SOAP parameters, and output directory.

### Run the full pipeline

```bash
lshdp run --config config.yaml
```

### With CLI overrides

```bash
lshdp run --config config.yaml --bin-width 0.01 --device cuda
```

### Run specific steps

```bash
# Only steps 1-4
lshdp run --config config.yaml --start-step 1 --end-step 4

# Resume from step 5
lshdp run --config config.yaml --start-step 5
```

---

## Configuration

All parameters are specified in a YAML file:

```yaml
soap:
  r_cut: 6.0
  n_max: 4
  l_max: 4
  sigma: 1.0
  rbf: gto
  periodic: true

hashing:
  n_components: 100
  n_hash: 100
  bin_width: 0.004
  random_seed: 42

io:
  input_file: simulation.xyz
  output_dir: results
  format: auto

split:
  frames_per_file: 120

device: auto
deterministic: true
```

Validate your config:

```bash
lshdp validate config.yaml
```

---

## GPU Support

Set `device: cuda` in config or use the CLI flag:

```bash
lshdp run --config config.yaml --device cuda
```

Use `device: auto` (default) to automatically use CUDA when available.

GPU is used for:
- Tensor normalization
- Random projection matrix generation
- Hash value computation
- Matrix multiplications

---

## Supported Trajectory Formats

Any format supported by ASE's `ase.io.read()`:

| Format | Extension | Notes |
|--------|-----------|-------|
| XYZ | `.xyz` | Standard and extended |
| Extended XYZ | `.extxyz` | With properties |
| LAMMPS | `.lammpstrj` | Trajectory dumps |
| CIF | `.cif` | Crystallographic |
| VASP | `POSCAR`, `CONTCAR` | VASP structures |
| Protein Data Bank | `.pdb` | Biomolecular |

Set `format: auto` for automatic detection, or specify explicitly.

Cell and PBC can be overridden in the config:

```yaml
io:
  cell: [15.82, 15.82, 30.76]
  pbc: true
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `descriptors` | Compute SOAP descriptors |
| 2 | `hashing` | Flatten → PCA → LSH → Buckets |
| 3 | `hashing` | Organise frames into bins |
| 4 | `io` | Extract representative frame numbers |
| 5 | `io` | Update XYZ metadata |
| 6 | `io` | Extract selected frames |
| 7 | `io` | Split trajectory into parts |

---

## Logging

Every run produces a detailed log file in the output directory:

```
results/lshdp_run_20250227_143000.log
```

Includes: hardware info, configuration, per-step timings, PCA variance retained, bin statistics.

---

## CLI Commands

```
lshdp run          Run the pipeline
lshdp validate     Validate a config file
lshdp info         Show version and environment
lshdp cite         Print citation info
lshdp init-config  Generate example config
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Project Structure

```
lshdp/
├── lsh/
│   ├── __init__.py          # Package metadata
│   ├── cli.py               # Click CLI
│   ├── config.py            # YAML config management
│   ├── logging_utils.py     # Structured logging
│   ├── banner.py            # Startup banner
│   ├── io/
│   │   └── __init__.py      # Trajectory I/O, XYZ utils
│   ├── descriptors/
│   │   └── __init__.py      # SOAP calculation
│   ├── hashing/
│   │   └── __init__.py      # LSH, PCA, binning
│   ├── pipeline/
│   │   └── __init__.py      # Pipeline orchestrator
│   └── utils/
│       └── __init__.py      # Device & seed helpers
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_hashing.py
│   └── test_io.py
├── environment.yml
├── pyproject.toml
├── config.yaml              # Example config
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Citation

If you use this software, please cite:

```
Anmol, Anuj Kumar Sirohi, Neha, Jayadeva, Sandeep Kumar, Tarak Karmakar,
"Locality-Sensitive Hashing-Based Data Set Reduction for Deep Potential Training",
J. Chem. Theory Comput. 2025, 21, 12, 6113–6120.
```

```bash
lshdp cite
```

---

## Reproducibility Statement

This implementation preserves the exact algorithmic workflow and mathematical formulation described in the original publication. The refactoring applies only to software engineering aspects: packaging, configuration, interfaces, logging, and optional GPU acceleration. Scientific results are reproducible with `deterministic: true` and a fixed `random_seed`.

---

## License

MIT License. See [LICENSE](LICENSE).
