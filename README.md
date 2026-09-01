# 🏔️ Elasto-Plastic FEM Tunnel Support & Excavation Engine (v37)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Engine: Finite Element Method](https://img.shields.io/badge/Engine-FEM%202D-green.svg)]()
[![Theory: Mohr--Coulomb Elasto--Plastic](https://img.shields.io/badge/Theory-Mohr--Coulomb-orange.svg)]()
[![Compilable: Nuitka C++ Binary](https://img.shields.io/badge/Nuitka-C%2B%2B%20Stand-purple.svg)]()

A lightweight, high-performance, and mathematically rigorous **2D Elasto-Plastic Finite Element Method (FEM) solver** and preprocessor for tunnel excavation, support-structure interaction, and NATM progressive stress relaxation analysis. 

This educational and research-grade engine is designed to bridge the gap between expensive closed-source commercial geotech software (e.g., Rocscience, Plaxis) and academic transparency. Users can fully customize geometries, support elements, and soil profiles using an external JSON file, run single-stage or progressive sequential excavations, and execute multi-model parametric sensitivity sweeps.

---

## 🚀 Core Capabilities

1. **Generalized Auto-Mesh Preprocessor**
   - Automatically generates structured 2D meshes for three classic tunnel geometries: **Circular**, **Horseshoe**, and **Horseshoe with Curved Invert**.
   - Supports three symmetry configurations: **Quarter Symmetry** ($90^\circ$), **Half Symmetry** ($180^\circ$), and **Full Geometry** ($360^\circ$).
   - Dynamically scales rendering boundaries and coordinates with an *Auto-Zoom* algorithm to ensure zero white margins in all symmetry profiles.

2. **Perfect Loop Seam Topology (v37 closed-loop)**
   - Resolves the polar boundary "slit" singularity on full $360^\circ$ meshes by mapping boundary index nodes onto a seamless topological cylinder.
   - Eliminates numerical drift and displacement discontinuities along the seam (negative x-axis), yielding $100\%$ symmetric contour stress and deformation fields.

3. **Incremental Elasto-Plastic Solver Core**
   - Incorporates a plane strain formulation solved via an incremental active initial stress (AIS) scheme.
   - Applies the **Mohr-Coulomb yield criterion** in principal stress space with radial return mapping for plastic material correction.
   - Models progressive stress relaxation ($\lambda = 40\%$) simulating three-dimensional NATM face advance.

4. **Dynamic Composite Structural Models**
   - **Shotcrete / Concrete Liner:** Formulated as curved composite section beam-spring boundaries with dynamic stiffness assembly.
   - **Steel Sets (Steel Ribs):** Composite cross-section integration (TH380 / I-Profile) mapped to the liner stiffness.
   - **Friction Rockbolts:** Models passive friction anchoring truss elements utilizing our custom **Anti-Double Bolting Filter** to prevent overlapping bolt nodes on full-geometry seams.

5. **Parametric Sensitivity Batch Sweeps**
   - Execute **24 distinct elasto-plastic models** in seconds to perform multi-parameter sweep analyses.
   - Fully controllable via an external JSON block: Cohesion ($c$) vs. Settlement, Liner Thickness ($t_{shot}$) vs. Stress, and Rockbolt Length ($L_{bolt}$) vs. Anchorage Tension.

---

## 🗺️ Software Architecture

```
/
├── tunnel_fem_optimized_v37.py   <-- Main Interactive Simulation & Solvers
├── tunnel_fem_batch_sweep_v37.py <-- Shape & Symmetry-Aware Parametric Sweeps
├── parameters.json               <-- External Geotechnical Config (Auto-Generates)
└── tunnel-reference-manual-v39.pdf <-- 11-Page Academic Handbook & Student Labs
```

---

## 🎨 Visualization Dashboards

### 1. 12-Panel Comparative Dashboard (`tunnel_results_sequential_v37.png`)
Provides a comprehensive overview of the simulation stages (Heading & Bench excavation) showing:
- **Geometry & Mesh:** Real-time support state, structural components, and boundary conditions.
- **Displacement Contours:** Millimetric settlement gradients.
- **Stress Fields:** Maximum principal stress ($\sigma_1$) and Max shear stress  \(\tau _{\text{max}}\) .
- **Plastic Yielded Zones:** Red-highlighted Gauss points exceeding Mohr-Coulomb limits.
- **GRC / SRC Equilibrium:** Ground Reaction Curve vs. Support Reaction Curve balance calculation.
- **Lining Capacity Utilization:** Dynamic bar charts showcasing safety factor margins for Shotcrete, Steel Ribs, and Rockbolts.

### 2. 3-Panel Parametric Sweep (`tunnel_sensitivity_sweep_v37.png`)
Plots the three distinct parametric studies side-by-side:
- **A. Rock Cohesion Effect:** Evaluates settlement control and plastic zone shrink with vs. without supports.
- **B. Shotcrete Thickness Effect:** Visualizes lining stress mobilization against lining capacity boundaries.
- **C. Rockbolt Optimization:** Locates the optimal bolt anchorage length beyond the active plastic envelope.

---

## ⚙️ Configuration (`parameters.json`)

The program is **self-healing**—if `parameters.json` is missing, it auto-generates a pristine baseline. Below is the configuration file format, including our new dynamic **Batch JSON Sweep** parameters:

```json
{
    "geometry": {
        "R_tunnel": 4.0,
        "H_wall": 3.0,
        "R_outer": 20.0,
        "H_depth": 100.0
    },
    "rock_mass": {
        "E_rock_init": 5000.0,
        "nu_rock": 0.25,
        "c_rock_init": 1.2,
        "phi_deg": 30.0,
        "gamma_rock": 0.025
    },
    "support_limits": {
        "E_shotcrete": 25000.0,
        "fc_shotcrete_limit": 16.7,
        "E_bolt": 210000.0,
        "d_bolt": 0.025,
        "T_bolt_limit": 200.0,
        "E_steel": 210000.0,
        "A_steel_per_m": 0.005,
        "I_steel_per_m": 4e-05,
        "fy_steel_limit": 235.0
    },
    "active_simulation": {
        "use_shotcrete": true,
        "t_shot": 0.2,
        "use_rockbolts": true,
        "L_bolt_val": 3.0,
        "use_steel": true,
        "degradation_factor": 0.0
    },
    "preprocessing": {
        "tunnel_type": "horseshoe_with_invert",
        "symmetry_type": "full",
        "R_invert": 8.0,
        "n_r": 16,
        "n_theta": 32,
        "stages": [
            {
                "stage_idx": 1,
                "name": "Heading Excavation (Kalota)",
                "excavate_above_y": 0.0,
                "active_supports": ["shotcrete", "rockbolts"]
            },
            {
                "stage_idx": 2,
                "name": "Bench Excavation (Stros)",
                "excavate_above_y": -99.0,
                "active_supports": ["shotcrete", "rockbolts", "steel_ribs"]
            }
        ]
    },
    "batch_sweeps": {
        "cohesion_sweep": {
            "min_MPa": 0.2,
            "max_MPa": 2.0,
            "steps": 8
        },
        "shotcrete_sweep": {
            "min_cm": 0.0,
            "max_cm": 35.0,
            "steps": 8
        },
        "rockbolt_sweep": {
            "min_m": 0.0,
            "max_m": 6.0,
            "steps": 7
        }
    }
}
```

---

## 🛠️ Installation & Execution

### Running the Python Source
Ensure you have `scipy`, `numpy`, and `matplotlib` installed:
```bash
pip install numpy scipy matplotlib
```
To run the interactive sequential simulation:
```bash
python tunnel_fem_optimized_v37.py
```
To run the parametric batch sweeps:
```bash
python tunnel_fem_batch_sweep_v37.py
```

### Compiling to a Standalone C++ Binary (Nuitka)
To secure the source code, prevent accidental student edit errors, and bypass the python environment installation bottleneck, compile the script into a single-file executable utilizing **Nuitka**:
```bash
pip install nuitka
nuitka --standalone --onefile --enable-plugin=matplotlib --enable-plugin=numpy tunnel_fem_optimized_v37.py
```
This generates `tunnel_fem_optimized_v37.exe` (on Windows). Students can simply modify parameters in `parameters.json` and double-click the `.exe` to run the model natively.

---

## 👥 Contributors & Collaboration DNA

This software was engineered through a unique, state-of-the-art **Vibe-Coding Partnership** between:
- **Lead Geotechnical Engineer & System Architect:** (The Principal Investigator / "The Professor" behind the math, scheduling, and structural filters).
- **Gemini Notebook (formerly known as NotebookLM):** An advanced AI agent serving as the dedicated co-developer, translating complex elasto-plastic return mapping algorithms and polar finite element topologies into clean, zero-bloat, production-grade Python/C++ code.

*This project stands as a testament to the power of agentic collaboration: transforming abstract soil mechanics into a perfectly tuned numerical engine.*

---

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
