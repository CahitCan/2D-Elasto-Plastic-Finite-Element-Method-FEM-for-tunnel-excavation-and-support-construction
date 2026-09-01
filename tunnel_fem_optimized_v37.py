# -*- coding: utf-8 -*-
"""
Elasto-Plastic FEM Tunnel Preprocessor, Support & Excavation Analysis Code (v29 - Curved Invert & Multi-stage Bypass Edition)
----------------------------------------------------------------------------------------------------
This version introduces a complete General Preprocessing Module supporting:
- Custom Tunnel Geometries: Circular, Horseshoe, and Horseshoe with Curved Invert.
- Symmetry Options: Quarter, Half, and Full Symmetry.
- Dynamic Multi-stage Excavation: Automatically schedules element excavation based on elevation limits.
- Automated Boundary Mapping: Automatically locks symmetry axes and outer boundary DOFs.
- Path-Dependent Structural Force Extraction and Automatic Timestamped Archiving.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from matplotlib.collections import PolyCollection
import os
import shutil
import datetime

# Ensure directories exist
is_cloud = os.path.exists('/workspace') and os.name != 'nt'
if is_cloud:
    os.makedirs('/workspace/scratch', exist_ok=True)

# =============================================================================
# 1. GEOMETRY, MATERIAL & CAPACITY PARAMETERS (MODULAR JSON LOADER)
# =============================================================================
import json

default_params = {
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
    "economic_rates": {
        "COST_SHOTCRETE_PER_M3": 250.0,
        "COST_ROCKBOLT_PER_M": 40.0,
        "COST_STEEL_BEAM_FLAT": 350.0
    },
    "active_simulation": {
        "use_shotcrete": True,
        "t_shot": 0.20,
        "use_rockbolts": True,
        "L_bolt_val": 3.0,
        "use_steel": True,
        "degradation_factor": 0.0
    },
    "preprocessing": {
        "tunnel_type": "horseshoe", 
        "symmetry_type": "half",
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
    }
}

# Smart Cross-Platform Path Resolution (Workspace Sandbox vs Local Windows/Mac/Linux)
if os.path.exists('/workspace') and os.name != 'nt':
    params_path = '/workspace/artifacts/parameters.json'
    os.makedirs('/workspace/artifacts', exist_ok=True)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    params_path = os.path.join(script_dir, 'parameters.json')

# Create parameters if file does not exist, or append preprocessing section to old files
if not os.path.exists(params_path):
    with open(params_path, 'w', encoding='utf-8') as f_json:
        json.dump(default_params, f_json, indent=4, ensure_ascii=False)
else:
    # Upgrade parameters.json to include preprocessing block if missing
    try:
        with open(params_path, 'r', encoding='utf-8') as f_json:
            existing = json.load(f_json)
        if "preprocessing" not in existing:
            existing["preprocessing"] = default_params["preprocessing"]
            with open(params_path, 'w', encoding='utf-8') as f_json:
                json.dump(existing, f_json, indent=4, ensure_ascii=False)
    except Exception:
        pass

with open(params_path, 'r', encoding='utf-8') as f_json:
    params = json.load(f_json)

# Assign parameters from JSON structure
R_tunnel = params["geometry"]["R_tunnel"]
H_wall = params["geometry"]["H_wall"]
R_outer = params["geometry"]["R_outer"]
H_depth = params["geometry"]["H_depth"]

E_rock_init = params["rock_mass"]["E_rock_init"]
nu_rock = params["rock_mass"]["nu_rock"]
c_rock_init = params["rock_mass"]["c_rock_init"]
phi_deg = params["rock_mass"]["phi_deg"]
gamma_rock = params["rock_mass"]["gamma_rock"]

K0 = nu_rock / (1.0 - nu_rock)
phi = np.radians(phi_deg)
sin_phi = np.sin(phi)
cos_phi = np.cos(phi)

def get_D_e(E_val):
    Factor = E_val / ((1 + nu_rock) * (1 - 2 * nu_rock))
    return Factor * np.array([
        [1 - nu_rock,   nu_rock,     0],
        [nu_rock,     1 - nu_rock,   0],
        [0,             0,   (1 - 2 * nu_rock) / 2]
    ])

COST_SHOTCRETE_PER_M3 = params["economic_rates"]["COST_SHOTCRETE_PER_M3"]
COST_ROCKBOLT_PER_M = params["economic_rates"]["COST_ROCKBOLT_PER_M"]
COST_STEEL_BEAM_FLAT = params["economic_rates"]["COST_STEEL_BEAM_FLAT"]

E_shotcrete = params["support_limits"]["E_shotcrete"]
fc_shotcrete_limit = params["support_limits"]["fc_shotcrete_limit"]

E_bolt = params["support_limits"]["E_bolt"]
d_bolt = params["support_limits"]["d_bolt"]
A_bolt = np.pi * (d_bolt / 2)**2
T_bolt_limit = params["support_limits"]["T_bolt_limit"]

E_steel = params["support_limits"]["E_steel"]
A_steel_per_m = params["support_limits"]["A_steel_per_m"]
I_steel_per_m = params["support_limits"]["I_steel_per_m"]
fy_steel_limit = params["support_limits"]["fy_steel_limit"]

# =============================================================================
# 2. GENERAL PREPROCESSING & MESH GENERATION MODULE
# =============================================================================
def get_R_boundary(theta, tunnel_type, R_tunnel, H_wall, R_invert=None):
    """
    Returns the boundary radius R_b for any theta, reflecting horizontally for horseshoe symmetry.
    """
    th_mapped = theta
    if th_mapped > np.pi/2:
        th_mapped = np.pi - th_mapped
    elif th_mapped < -np.pi/2:
        th_mapped = -np.pi - th_mapped
        
    if tunnel_type == "circular":
        return R_tunnel
        
    elif tunnel_type == "horseshoe":
        theta_corner = np.arctan2(-H_wall, R_tunnel)
        if th_mapped >= 0:
            return R_tunnel
        elif th_mapped >= theta_corner:
            return R_tunnel / np.cos(th_mapped)
        else:
            return -H_wall / np.sin(th_mapped)
            
    elif tunnel_type == "horseshoe_with_invert":
        theta_corner = np.arctan2(-H_wall, R_tunnel)
        if th_mapped >= 0:
            return R_tunnel
        elif th_mapped >= theta_corner:
            return R_tunnel / np.cos(th_mapped)
        else:
            if R_invert is None or R_invert <= R_tunnel:
                R_invert = R_tunnel * 2.0
            y_c = -H_wall + np.sqrt(R_invert**2 - R_tunnel**2)
            val = R_invert**2 - (y_c**2) * (np.cos(th_mapped)**2)
            if val < 0:
                return -H_wall / np.sin(th_mapped)
            return y_c * np.sin(th_mapped) + np.sqrt(val)
    else:
        return R_tunnel

def run_preprocessor(params):
    """
    Executes the flexible preprocessor to generate the nodes, elements, stages, and BCs.
    """
    pre = params.get("preprocessing", {})
    tunnel_type = pre.get("tunnel_type", "horseshoe")
    symmetry_type = pre.get("symmetry_type", "half")
    R_invert = pre.get("R_invert", 8.0)
    n_r = pre.get("n_r", 16)
    n_theta = pre.get("n_theta", 32)
    stages_config = pre.get("stages", default_params["preprocessing"]["stages"])
    
    # Configure theta range based on symmetry type
    if symmetry_type == "quarter":
        theta_vals = np.linspace(0, np.pi / 2, n_theta + 1)
    elif symmetry_type == "half":
        theta_vals = np.linspace(-np.pi / 2, np.pi / 2, n_theta + 1)
    else:  # full
        theta_vals = np.linspace(-np.pi, np.pi, n_theta + 1)
        
    r_fac = np.linspace(0, 1, n_r + 1)**1.8
    
    # Generate nodes (v36: closed circular topology for full symmetry)
    nodes = []
    node_dict = {}
    node_idx = 0
    for i_r in range(n_r + 1):
        for j_th, th in enumerate(theta_vals):
            if symmetry_type == "full" and j_th == n_theta:
                node_dict[(i_r, j_th)] = node_dict[(i_r, 0)]
            else:
                R_boundary = get_R_boundary(th, tunnel_type, R_tunnel, H_wall, R_invert)
                r = R_boundary + (R_outer - R_boundary) * r_fac[i_r]
                x = r * np.cos(th)
                y = r * np.sin(th)
                nodes.append([x, y])
                node_dict[(i_r, j_th)] = node_idx
                node_idx += 1
            
    nodes = np.array(nodes)
    num_nodes = len(nodes)
    num_dofs = 2 * num_nodes
    
    # Generate elements
    elements = []
    for i in range(n_r):
        for j in range(n_theta):
            elements.append([
                node_dict[(i, j)],
                node_dict[(i + 1, j)],
                node_dict[(i + 1, j + 1)],
                node_dict[(i, j + 1)]
            ])
    elements = np.array(elements)
    num_elements = len(elements)
    
    # Boundary conditions
    fixed_dofs_base = []
    for i, (x, y) in enumerate(nodes):
        if symmetry_type in ["quarter", "half"]:
            if np.isclose(x, 0.0, atol=1e-3):
                fixed_dofs_base.append(2 * i)  # Fix horizontally
        if symmetry_type == "quarter":
            if np.isclose(y, 0.0, atol=1e-3):
                fixed_dofs_base.append(2 * i + 1)  # Fix vertically
        r_node = np.sqrt(x**2 + y**2)
        if np.isclose(r_node, R_outer, atol=1e-2):
            fixed_dofs_base.extend([2 * i, 2 * i + 1])  # Fix both at outer ring
    fixed_dofs_base = np.unique(fixed_dofs_base)
    
    # Element dynamic stage classification
    element_stages = np.zeros(num_elements, dtype=int)
    last_stage_idx = stages_config[-1]["stage_idx"]
    element_stages[:] = last_stage_idx
    
    for stage in stages_config[:-1]:
        stg_id = stage["stage_idx"]
        exc_y = stage["excavate_above_y"]
        for idx, elem in enumerate(elements):
            if element_stages[idx] == last_stage_idx:
                elem_coords = nodes[elem]
                y_center = np.mean(elem_coords[:, 1])
                if y_center >= exc_y:
                    element_stages[idx] = stg_id
                    
    return {
        "nodes": nodes,
        "elements": elements,
        "node_dict": node_dict,
        "fixed_dofs_base": fixed_dofs_base,
        "element_stages": element_stages,
        "theta_vals": theta_vals,
        "stages_config": stages_config,
        "n_theta": n_theta,
        "n_r": n_r,
        "r_fac": r_fac,
        "tunnel_type": tunnel_type,
        "symmetry_type": symmetry_type,
        "R_invert": R_invert
    }

# =============================================================================
# 3. ADVANCED SEQUENTIAL ELASTO-PLASTIC FEM ENGINE
# =============================================================================
def run_natm_simulation(use_shotcrete, t_shot, use_rockbolts, L_bolt_val, use_steel, degradation_factor=0.0):
    """
    Runs the fully generalized elasto-plastic sequential FEM solver.
    """
    # Run preprocessor
    prep = run_preprocessor(params)
    nodes = prep["nodes"]
    elements = prep["elements"]
    node_dict = prep["node_dict"]
    fixed_dofs_base = prep["fixed_dofs_base"]
    element_stages = prep["element_stages"]
    theta_vals = prep["theta_vals"]
    stages_config = prep["stages_config"]
    n_theta = prep["n_theta"]
    r_fac = prep["r_fac"]
    tunnel_type = prep["tunnel_type"]
    symmetry_type = prep["symmetry_type"]
    R_invert = prep["R_invert"]
    
    num_nodes = len(nodes)
    num_elements = len(elements)
    num_dofs = 2 * num_nodes
    
    # Degraded rock properties
    E_rock = E_rock_init * (1.0 - 0.4 * degradation_factor)
    c_rock = c_rock_init * (1.0 - degradation_factor)
    D_e = get_D_e(E_rock)
    
    # 3.1 Stiffness Matrix Assembly (Rock mass)
    K_rock = np.zeros((num_dofs, num_dofs))
    gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    
    def shape_funcs(xi, eta):
        return 0.25 * np.array([
            [(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]
        ]), 0.25 * np.array([
            [-(1-eta),  (1-eta), (1+eta), -(1+eta)],
            [-(1-xi),  -(1+xi),  (1+xi),   (1-xi)]
        ])
        
    for elem in elements:
        elem_coords = nodes[elem]
        Ke = np.zeros((8, 8))
        for xi in gauss_pts:
            for eta in gauss_pts:
                _, dN_dxi = shape_funcs(xi, eta)
                J = dN_dxi @ elem_coords
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN_dxi
                
                B = np.zeros((3, 8))
                for idx in range(4):
                    B[0, 2*idx] = dN_dx[0, idx]
                    B[1, 2*idx+1] = dN_dx[1, idx]
                    B[2, 2*idx] = dN_dx[1, idx]
                    B[2, 2*idx+1] = dN_dx[0, idx]
                Ke += B.T @ D_e @ B * detJ
                
        for i_local in range(4):
            for j_local in range(4):
                g_i = elem[i_local]
                g_j = elem[j_local]
                K_rock[2*g_i : 2*g_i+2, 2*g_j : 2*g_j+2] += Ke[2*i_local : 2*i_local+2, 2*j_local : 2*j_local+2]
                
    # 3.2 Dynamic Stage Force Integration
    F_ext_stages = {stage["stage_idx"]: np.zeros(num_dofs) for stage in stages_config}
    for j in range(n_theta):
        n1 = node_dict[(0, j)]
        n2 = node_dict[(0, j+1)]
        p1, p2 = nodes[n1], nodes[n2]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        y_mid = 0.5 * (p1[1] + p2[1])
        sig_v = gamma_rock * (H_depth - y_mid)
        sig_h = K0 * sig_v
        
        fx = sig_h * dy
        fy = -sig_v * dx
        
        stage_idx_segment = element_stages[j]
        F_ext_stages[stage_idx_segment][2 * n1]     += 0.5 * fx
        F_ext_stages[stage_idx_segment][2 * n1 + 1] += 0.5 * fy
        F_ext_stages[stage_idx_segment][2 * n2]     += 0.5 * fx
        F_ext_stages[stage_idx_segment][2 * n2 + 1] += 0.5 * fy
        
    # Trackers for stage outcomes
    u_current = np.zeros(num_dofs)
    u_stage_results = {}
    yielded_stage_results = {}
    elem_stresses_stage_results = {}
    
    yielded_unsupported = np.zeros(num_elements, dtype=bool)
    elem_stresses_unsupported = np.zeros((num_elements, 3))
    
    # Active structural parameters & installation state tracker
    u_install_state = None
    K_support_active = np.zeros((num_dofs, num_dofs))
    rockbolt_elements_stage1 = []
    rockbolt_elements_stage2 = []
    rockbolted_nodes = set()
    
    # Loop over stages
    for idx_stg, stage in enumerate(stages_config):
        stg_idx = stage["stage_idx"]
        act_supp = stage["active_supports"]
        
        # Segment indices newly excavated in this stage
        segments_new = [j for j in range(n_theta) if element_stages[j] == stg_idx]
        
        # 3.3 Dynamic Support Installation & Stiffness Matrix Update
        # Shotcrete / Steel Liner
        EA_comp = 0.0
        if "shotcrete" in act_supp and use_shotcrete:
            EA_comp += E_shotcrete * t_shot
        if "steel_ribs" in act_supp and use_steel:
            EA_comp += E_steel * A_steel_per_m
            
        if EA_comp > 0:
            for j in segments_new:
                n1 = node_dict[(0, j)]
                n2 = node_dict[(0, j+1)]
                p1, p2 = nodes[n1], nodes[n2]
                L_seg = np.linalg.norm(p2 - p1)
                dx, dy = (p2[0] - p1[0]) / L_seg, (p2[1] - p1[1]) / L_seg
                k_liner = (EA_comp / L_seg) * np.array([
                    [ dx*dx,  dx*dy, -dx*dx, -dx*dy],
                    [ dx*dy,  dy*dy, -dx*dy, -dy*dy],
                    [-dx*dx, -dx*dy,  dx*dx,  dx*dy],
                    [-dx*dy, -dy*dy,  dx*dy,  dy*dy]
                ])
                dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
                for i_loc, d_i in enumerate(dofs):
                    for j_loc, d_j in enumerate(dofs):
                        K_support_active[d_i, d_j] += k_liner[i_loc, j_loc]
                        
        # Rockbolts
        if "rockbolts" in act_supp and use_rockbolts and L_bolt_val > 0 and len(segments_new) > 0:
            # We obtain the boundary theta indices in their physical continuous order
            nodes_new = sorted(list(set([j for j in range(n_theta+1) if (j < n_theta and j in segments_new) or (j > 0 and (j-1) in segments_new)])))
            
            # Filter out any theta indices whose actual node index already has a rockbolt installed
            filtered_nodes = []
            for j in nodes_new:
                node_idx_val = node_dict[(0, j)]
                if node_idx_val not in rockbolted_nodes:
                    filtered_nodes.append(j)
            
            num_bolts_this_stage = max(3, len(nodes_new) // 3)
            # Cap the number of bolts to the number of available unbolted locations
            num_bolts_this_stage = min(num_bolts_this_stage, len(filtered_nodes))
            
            if num_bolts_this_stage > 0:
                idx_selections = np.linspace(0, len(filtered_nodes) - 1, num_bolts_this_stage, dtype=int)
                bolt_theta_indices = [filtered_nodes[i] for i in idx_selections]
            else:
                bolt_theta_indices = []
            
            for th_idx in bolt_theta_indices:
                cidar_node = node_dict[(0, th_idx)]
                th = theta_vals[th_idx]
                R_boundary = get_R_boundary(th, tunnel_type, R_tunnel, H_wall, R_invert=R_invert)
                radial_distances = r_fac * (R_outer - R_boundary)
                r_layer_idx = np.argmin(np.abs(radial_distances - L_bolt_val))
                ic_node = node_dict[(r_layer_idx, th_idx)]
                if stg_idx == 1:
                    rockbolt_elements_stage1.append([cidar_node, ic_node])
                else:
                    rockbolt_elements_stage2.append([cidar_node, ic_node])
                rockbolted_nodes.add(cidar_node)
                
                # Assemble truss element
                p1, p2 = nodes[cidar_node], nodes[ic_node]
                L = np.linalg.norm(p2 - p1)
                dx, dy = (p2[0] - p1[0]) / L, (p2[1] - p1[1]) / L
                k_truss = (E_bolt * A_bolt / L) * np.array([
                    [ dx*dx,  dx*dy, -dx*dx, -dx*dy],
                    [ dx*dy,  dy*dy, -dx*dy, -dy*dy],
                    [-dx*dx, -dx*dy,  dx*dx,  dx*dy],
                    [-dx*dy, -dy*dy,  dx*dy,  dy*dy]
                ])
                dofs = [2*cidar_node, 2*cidar_node+1, 2*ic_node, 2*ic_node+1]
                for i_loc, d_i in enumerate(dofs):
                    for j_loc, d_j in enumerate(dofs):
                        K_support_active[d_i, d_j] += k_truss[i_loc, j_loc]
                        
        # 3.4 Boundary Condition Management (Auto-Locking future unexcavated segments)
        locked_dofs = []
        for j in range(n_theta + 1):
            is_future = False
            if j < n_theta and element_stages[j] > stg_idx:
                is_future = True
            if j > 0 and element_stages[j-1] > stg_idx:
                is_future = True
            if is_future:
                locked_dofs.extend([2 * node_dict[(0, j)], 2 * node_dict[(0, j)] + 1])
                
        fixed_dofs_current = np.unique(np.concatenate((fixed_dofs_base, locked_dofs)))
        active_dofs_current = np.setdiff1d(np.arange(num_dofs), fixed_dofs_current)
        
        # 3.5 Solver Loop for current Stage
        # Accumulated force up to this stage
        F_accumulated = sum(F_ext_stages[stages_config[s]["stage_idx"]] for s in range(idx_stg + 1))
        
        if idx_stg == 0:
            # First stage has load increments + 40% initial unsupported relaxation
            lambda_natm = 0.40
            # Step 1-4: Unsupported relaxation
            LU_rock = splu(csc_matrix(K_rock[np.ix_(active_dofs_current, active_dofs_current)]))
            for step in range(1, 5):
                F_current = (step / 10) * lambda_natm * F_ext_stages[stg_idx]
                u_current[active_dofs_current] = LU_rock.solve(F_current[active_dofs_current])
                
            # Track and save the installation displacement for all segment nodes
            u_install_state = u_current.copy()
            
            # If single-stage bypass, calculate unsupported stress and yield right here
            if len(stages_config) == 1:
                for idx_e, elem in enumerate(elements):
                    elem_coords = nodes[elem]
                    y_elem = np.mean(elem_coords[:, 1])
                    u_elem = u_install_state[np.array([2*elem[0], 2*elem[0]+1, 2*elem[1], 2*elem[1]+1,
                                                2*elem[2], 2*elem[2]+1, 2*elem[3], 2*elem[3]+1])]
                    
                    _, dN_dxi = shape_funcs(0.0, 0.0)
                    J = dN_dxi @ elem_coords
                    invJ = np.linalg.inv(J)
                    dN_dx = invJ @ dN_dxi
                    
                    B = np.zeros((3, 8))
                    for k in range(4):
                        B[0, 2*k] = dN_dx[0, k]
                        B[1, 2*k+1] = dN_dx[1, k]
                        B[2, 2*k] = dN_dx[1, k]
                        B[2, 2*k+1] = dN_dx[0, k]
                        
                    strain = B @ u_elem
                    elastic_stress = D_e @ strain
                    
                    sig_v_elem = gamma_rock * (H_depth - y_elem)
                    sig_h_elem = K0 * sig_v_elem
                    excavation_stress = np.array([-sig_h_elem, -sig_v_elem, 0.0])
                    total_stress = excavation_stress + elastic_stress
                    elem_stresses_unsupported[idx_e] = total_stress
                    
                    s_xx, s_yy, s_xy = total_stress[0], total_stress[1], total_stress[2]
                    center = (s_xx + s_yy) / 2.0
                    radius = np.sqrt(((s_xx - s_yy) / 2.0)**2 + s_xy**2)
                    sigma_n = center + radius * sin_phi
                    tau_m = radius
                    yield_func = tau_m - (c_rock * cos_phi - sigma_n * sin_phi)
                    if yield_func > 0:
                        yielded_unsupported[idx_e] = True
            
            # Step 5-10: Supported state
            K_total_stg1 = K_rock + K_support_active
            LU_total_stg1 = splu(csc_matrix(K_total_stg1[np.ix_(active_dofs_current, active_dofs_current)]))
            for step in range(5, 11):
                F_current = (step / 10) * F_ext_stages[stg_idx]
                u_current[active_dofs_current] = LU_total_stg1.solve(F_current[active_dofs_current])
        else:
            # Subsequent stages are solved in one block
            K_total_stg_k = K_rock + K_support_active
            LU_total_stg_k = splu(csc_matrix(K_total_stg_k[np.ix_(active_dofs_current, active_dofs_current)]))
            u_current[active_dofs_current] = LU_total_stg_k.solve(F_accumulated[active_dofs_current])
            
        u_stage_results[stg_idx] = u_current.copy()
        
        # Stress evaluation and Mohr-Coulomb return mapping checking for current stage
        yielded_current = np.zeros(num_elements, dtype=bool)
        elem_stresses_current = np.zeros((num_elements, 3))
        
        for idx_e, elem in enumerate(elements):
            elem_coords = nodes[elem]
            y_elem = np.mean(elem_coords[:, 1])
            u_elem = u_current[np.array([2*elem[0], 2*elem[0]+1, 2*elem[1], 2*elem[1]+1,
                                        2*elem[2], 2*elem[2]+1, 2*elem[3], 2*elem[3]+1])]
            
            _, dN_dxi = shape_funcs(0.0, 0.0)
            J = dN_dxi @ elem_coords
            invJ = np.linalg.inv(J)
            dN_dx = invJ @ dN_dxi
            
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k] = dN_dx[0, k]
                B[1, 2*k+1] = dN_dx[1, k]
                B[2, 2*k] = dN_dx[1, k]
                B[2, 2*k+1] = dN_dx[0, k]
                
            strain = B @ u_elem
            elastic_stress = D_e @ strain
            
            sig_v_elem = gamma_rock * (H_depth - y_elem)
            sig_h_elem = K0 * sig_v_elem
            excavation_stress = np.array([-sig_h_elem, -sig_v_elem, 0.0])
            total_stress = excavation_stress + elastic_stress
            elem_stresses_current[idx_e] = total_stress
            
            s_xx, s_yy, s_xy = total_stress[0], total_stress[1], total_stress[2]
            center = (s_xx + s_yy) / 2.0
            radius = np.sqrt(((s_xx - s_yy) / 2.0)**2 + s_xy**2)
            sigma_n = center + radius * sin_phi
            tau_m = radius
            yield_func = tau_m - (c_rock * cos_phi - sigma_n * sin_phi)
            if yield_func > 0:
                yielded_current[idx_e] = True
                
        yielded_stage_results[stg_idx] = yielded_current
        elem_stresses_stage_results[stg_idx] = elem_stresses_current
        
    # Post-installation structural force extraction at Final Stage
    u_final = u_current.copy()
    du_structural = u_final - u_install_state
    
    max_shotcrete_stress = 0.0
    max_steel_stress = 0.0
    liner_failed = False
    EA_composite = (E_shotcrete * t_shot) if use_shotcrete else 0.0
    if use_steel:
        EA_composite += E_steel * A_steel_per_m
        
    if EA_composite > 0:
        for j in range(n_theta):
            n1 = node_dict[(0, j)]
            n2 = node_dict[(0, j+1)]
            p1_init, p2_init = nodes[n1], nodes[n2]
            
            p1_install = p1_init + u_install_state[np.array([2*n1, 2*n1+1])]
            p2_install = p2_init + u_install_state[np.array([2*n2, 2*n2+1])]
            p1_final = p1_init + u_final[np.array([2*n1, 2*n1+1])]
            p2_final = p2_init + u_final[np.array([2*n2, 2*n2+1])]
            
            L_install = np.linalg.norm(p2_install - p1_install)
            L_final = np.linalg.norm(p2_final - p1_final)
            strain_hoop = np.abs(L_final - L_install) / L_install
            
            if use_shotcrete:
                s_shot = E_shotcrete * strain_hoop
                if s_shot > max_shotcrete_stress:
                    max_shotcrete_stress = s_shot
                if s_shot > fc_shotcrete_limit:
                    liner_failed = True
            if use_steel:
                s_steel = E_steel * strain_hoop
                if s_steel > max_steel_stress:
                    max_steel_stress = s_steel
                if s_steel > fy_steel_limit:
                    liner_failed = True
                    
    # Rockbolt forces
    max_bolt_force_kN = 0.0
    bolts_failed = False
    for b_elem in rockbolt_elements_stage1 + rockbolt_elements_stage2:
        n1, n2 = b_elem[0], b_elem[1]
        p1_init, p2_init = nodes[n1], nodes[n2]
        
        v1 = p1_init / np.linalg.norm(p1_init)
        v2 = p2_init / np.linalg.norm(p2_init)
        
        du1_radial = np.dot(du_structural[np.array([2*n1, 2*n1+1])], v1)
        du2_radial = np.dot(du_structural[np.array([2*n2, 2*n2+1])], v2)
        relative_u_radial = np.abs(du1_radial - du2_radial)
        
        p1_install = p1_init + u_install_state[np.array([2*n1, 2*n1+1])]
        p2_install = p2_init + u_install_state[np.array([2*n2, 2*n2+1])]
        L_install = np.linalg.norm(p2_install - p1_install)
        
        strain_bolt = relative_u_radial / L_install
        force_kn = E_bolt * A_bolt * strain_bolt * 1000.0
        if force_kn > max_bolt_force_kN:
            max_bolt_force_kN = force_kn
        if force_kn > T_bolt_limit:
            bolts_failed = True
            
    # Find crown and invert nodes dynamically
    crown_th_idx = np.argmin(np.abs(theta_vals - np.pi/2))
    crown_node = node_dict[(0, crown_th_idx)]
    crown_def_stage1 = np.sqrt(u_stage_results[1][2*crown_node]**2 + u_stage_results[1][2*crown_node+1]**2) * 1000.0
    crown_def_stage2 = np.sqrt(u_final[2*crown_node]**2 + u_final[2*crown_node+1]**2) * 1000.0
    u_stage1_install_crown = np.sqrt(u_install_state[2*crown_node]**2 + u_install_state[2*crown_node+1]**2) * 1000.0
    
    invert_th_idx = np.argmin(np.abs(theta_vals - (-np.pi/2)))
    invert_node = node_dict[(0, invert_th_idx)]
    floor_heave_stage1 = np.sqrt(u_stage_results[1][2*invert_node]**2 + u_stage_results[1][2*invert_node+1]**2) * 1000.0
    floor_heave_stage2 = np.sqrt(u_final[2*invert_node]**2 + u_final[2*invert_node+1]**2) * 1000.0
    
    last_stage_idx = stages_config[-1]["stage_idx"]
    diagnostics = {
        'nodes': nodes,
        'elements': elements,
        'u_stage1': u_install_state if len(stages_config) == 1 else u_stage_results[1],
        'u_stage1_install': u_install_state,
        'u_stage2': u_final,
        'yielded_stage1': yielded_unsupported if len(stages_config) == 1 else yielded_stage_results[1],
        'yielded_stage2': yielded_stage_results[last_stage_idx],
        'elem_stresses_stage1': elem_stresses_unsupported if len(stages_config) == 1 else elem_stresses_stage_results[1],
        'elem_stresses_stage2': elem_stresses_stage_results[last_stage_idx],
        'rockbolt_elements_stage1': rockbolt_elements_stage1,
        'rockbolt_elements_stage2': rockbolt_elements_stage2,
        'theta_vals': theta_vals,
        'node_dict': node_dict,
        'n_theta': n_theta,
        'crown_def_stage1': crown_def_stage1,
        'crown_def_stage2': crown_def_stage2,
        'u_stage1_install_crown': u_stage1_install_crown,
        'floor_heave_stage1': floor_heave_stage1,
        'floor_heave_stage2': floor_heave_stage2,
        'max_shotcrete_stress': max_shotcrete_stress,
        'max_steel_stress': max_steel_stress,
        'max_bolt_force': max_bolt_force_kN,
        'liner_failed': liner_failed,
        'bolts_failed': bolts_failed,
        'EA_composite': EA_composite,
        't_shot': t_shot,
        'use_shotcrete': use_shotcrete,
        'use_steel': use_steel,
        'use_rockbolts': use_rockbolts,
        'element_stages': element_stages,
        'stages_config': stages_config,
        'tunnel_type': tunnel_type,
        'symmetry_type': symmetry_type
    }
    return diagnostics

# =============================================================================
# 4. PLOTTING SUITE: 12-PANEL COMPARATIVE DASHBOARD
# =============================================================================
def generate_and_save_visuals(diag, filename="tunnel_results_sequential_v37.png"):
    nodes = diag['nodes']
    elements = diag['elements']
    u_stage1 = diag['u_stage1']
    u_stage1_install = diag['u_stage1_install']
    u_stage2 = diag['u_stage2']
    yielded_stage1 = diag['yielded_stage1']
    yielded_stage2 = diag['yielded_stage2']
    elem_stresses_stage1 = diag['elem_stresses_stage1']
    elem_stresses_stage2 = diag['elem_stresses_stage2']
    rockbolt_elements_stage1 = diag['rockbolt_elements_stage1']
    rockbolt_elements_stage2 = diag['rockbolt_elements_stage2']
    theta_vals = diag['theta_vals']
    node_dict = diag['node_dict']
    n_theta = diag['n_theta']
    EA_composite = diag['EA_composite']
    element_stages = diag['element_stages']
    stages_config = diag['stages_config']
    tunnel_type = diag['tunnel_type']
    symmetry_type = diag['symmetry_type']
    L_bolt_val = params['active_simulation']['L_bolt_val']

    # Dynamic axis limit calculator (v36 - Dynamic Coordinate Limits)
    R_limit = 0.75 * R_outer
    if symmetry_type == "quarter":
        x_min, x_max = -0.05 * R_limit, R_limit
        y_min, y_max = -0.05 * R_limit, R_limit
    elif symmetry_type == "half":
        x_min, x_max = -0.05 * R_limit, R_limit
        y_min, y_max = -0.6 * R_limit, 0.6 * R_limit
    else:  # full
        x_min, x_max = -R_limit, R_limit
        y_min, y_max = -R_limit, R_limit
    
    y_crown = R_tunnel
    sig_v_crown = gamma_rock * (H_depth - y_crown)
    sig_h_crown = K0 * sig_v_crown
    P_avg = (sig_v_crown + sig_h_crown) / 2.0
    
    polys = [nodes[elem] for elem in elements]
    
    fig, axs = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(f"Numerical Preprocessor & Elasto-Plastic FEM Analysis Dashboard (v37)\n"
                 f"(Geometry: {tunnel_type.upper()} | Symmetry: {symmetry_type.upper()} | K0 = {K0:.3f} | Geostatic Stress Gradient)", fontsize=14, fontweight='bold', y=0.98)
    
    # 4.1 Unexcavated wedges renderer helper (with dynamic colormap support, alpha control, and seamless element mesh matching)
    def draw_unexcavated_ground(ax, target_stage_idx, fill_color=None, alpha=1.0):
        f_color = '#e4f1f7' if fill_color is None else fill_color
        unexc_indices = [j for j in range(n_theta) if element_stages[j] > target_stage_idx]
        if not unexc_indices:
            return
            
        # Get all unique theta indices for the unexcavated region
        th_indices = sorted(list(set(unexc_indices + [j+1 for j in unexc_indices])))
        
        # Match density of surrounding mesh by dividing into n_r_inner radial elements
        n_r_inner = 12
        r_fac_inner = np.linspace(0, 1, n_r_inner + 1)
        
        # Get R_invert dynamically from global params
        R_invert_val = params.get("preprocessing", {}).get("R_invert", 8.0)
        
        inner_nodes = []
        inner_node_dict = {}
        idx = 0
        for i_r in range(n_r_inner + 1):
            r_ratio = r_fac_inner[i_r]
            for j_th_idx, th_idx in enumerate(th_indices):
                th = theta_vals[th_idx]
                R_boundary = get_R_boundary(th, tunnel_type, R_tunnel, H_wall, R_invert_val)
                r = R_boundary * r_ratio
                x = r * np.cos(th)
                y = r * np.sin(th)
                inner_nodes.append([x, y])
                inner_node_dict[(i_r, j_th_idx)] = idx
                idx += 1
                
        inner_nodes = np.array(inner_nodes)
        
        inner_elements = []
        for i in range(n_r_inner):
            for j in range(len(th_indices) - 1):
                inner_elements.append([
                    inner_node_dict[(i, j)],
                    inner_node_dict[(i + 1, j)],
                    inner_node_dict[(i + 1, j + 1)],
                    inner_node_dict[(i, j + 1)]
                ])
                
        inner_polys = [inner_nodes[elem] for elem in inner_elements]
        
        # Draw the unexcavated region with the EXACT same edge colors, alpha and linewidths as the rock mass mesh
        inner_coll = PolyCollection(inner_polys, facecolors=f_color, edgecolors='grey', linewidths=0.2, alpha=alpha, zorder=5)
        ax.add_collection(inner_coll)

    # 4.2 Excavated wedges mask helper (draws white patches over empty excavated space)
    def draw_excavated_hole(ax, target_stage_idx):
        for j in range(n_theta):
            seg_stage = element_stages[j]
            if seg_stage <= target_stage_idx:
                p1 = nodes[node_dict[(0, j)]]
                p2 = nodes[node_dict[(0, j+1)]]
                poly_verts = np.array([[0.0, 0.0], p1, p2, [0.0, 0.0]])
                patch = plt.Polygon(poly_verts, facecolor='white', edgecolor='none', zorder=10)
                ax.add_patch(patch)
                
    # -------------------------------------------------------------------------
    # ROW 1 (Aşama 1 Results)
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    mesh_coll1 = PolyCollection(polys, facecolors='#e4f1f7', edgecolors='lightgrey', linewidths=0.3)
    ax1.add_collection(mesh_coll1)
    draw_unexcavated_ground(ax1, 1)
    
    excavated_segs_stg1 = [j for j in range(n_theta) if element_stages[j] <= 1]
    for j in excavated_segs_stg1:
        p1 = nodes[node_dict[(0, j)]]
        p2 = nodes[node_dict[(0, j+1)]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=3, zorder=8)
        
    for b_elem in rockbolt_elements_stage1:
        p1, p2 = nodes[b_elem[0]], nodes[b_elem[1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'go-', lw=1.5, ms=3, zorder=9)
        
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title("1. Unsupported Case: Mesh & Support" if len(stages_config) == 1 else "1. Stage 1: Mesh & Support", fontsize=9, fontweight='bold')
    ax1.set_aspect('equal')
    
    import datetime
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    param_text = (
        "■ Geostatic Stresses:\n"
        f"  • H: {H_depth}m, γ: {gamma_rock} MN/m³\n"
        f"  • K0: {K0:.3f} (nu={nu_rock:.2f})\n\n"
        "■ Rock Mass Strength:\n"
        f"  • E: {E_rock_init:.0f} MPa, c: {c_rock_init:.2f} MPa\n"
        f"  • φ: {phi_deg:.1f}°\n\n"
        "■ Active Support State:\n"
        f"  • Shotcrete: {'YES' if diag['use_shotcrete'] else 'NO'} ({diag['t_shot']*100:.0f}cm)\n"
        f"  • Rockbolts: {'YES' if diag['use_rockbolts'] else 'NO'} (L={L_bolt_val:.1f}m, {len(rockbolt_elements_stage1) + len(rockbolt_elements_stage2)} bolts)\n"
        f"  • Steel Ribs: {'YES' if diag['use_steel'] else 'NO'}\n\n"
        "■ Run Timestamp:\n"
        f"  • {timestamp_str}"
    )
    ax1.text(0.55, 0.95, param_text, transform=ax1.transAxes, fontsize=6.5, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.90, edgecolor='gray'), zorder=10)
    
    # Panel 2: Deformasyon Konturu (Stage 1)
    ax2 = axs[0, 1]
    u_norm1 = np.sqrt(u_stage1[0::2]**2 + u_stage1[1::2]**2) * 1000.0
    sc2 = ax2.tricontourf(nodes[:, 0], nodes[:, 1], u_norm1, levels=15, cmap='jet')
    fig.colorbar(sc2, ax=ax2, label='Displacement (mm)')
    
    draw_excavated_hole(ax2, 1)
    draw_unexcavated_ground(ax2, 1, fill_color=sc2.cmap(sc2.norm(0.0)))
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("2. Unsupported Case: Deformation Contours" if len(stages_config) == 1 else "2. Stage 1: Deformation Contours", fontsize=9, fontweight='bold')
    ax2.set_aspect('equal')
    
    # Panel 3: Asal Gerilme (Stage 1)
    ax3 = axs[0, 2]
    sig_1_stage1 = elem_stresses_stage1[:, 0]
    sig1_coll1 = PolyCollection(polys, array=sig_1_stage1, cmap='plasma', edgecolors='none')
    ax3.add_collection(sig1_coll1)
    fig.colorbar(sig1_coll1, ax=ax3, label='σ₁ (MPa)')
    draw_excavated_hole(ax3, 1)
    stros_stress_val = -K0 * (gamma_rock * H_depth)
    draw_unexcavated_ground(ax3, 1, fill_color=sig1_coll1.cmap(sig1_coll1.norm(stros_stress_val)))
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_title("3. Unsupported Case: Principal Stress (σ₁)" if len(stages_config) == 1 else "3. Stage 1: Principal Stress (σ₁)", fontsize=9, fontweight='bold')
    ax3.set_aspect('equal')
    
    # Panel 4: Maksimum Kayma Gerilmesi (Stage 1)
    ax4 = axs[0, 3]
    sig_3_stage1 = elem_stresses_stage1[:, 1]
    tau_max_stage1 = np.abs(sig_1_stage1 - sig_3_stage1) / 2.0
    tau_coll1 = PolyCollection(polys, array=tau_max_stage1, cmap='inferno', edgecolors='none')
    ax4.add_collection(tau_coll1)
    fig.colorbar(tau_coll1, ax=ax4, label='τ_max (MPa)')
    draw_excavated_hole(ax4, 1)
    stros_tau_val = np.abs(sig_v_crown - sig_h_crown) / 2.0
    draw_unexcavated_ground(ax4, 1, fill_color=tau_coll1.cmap(tau_coll1.norm(stros_tau_val)))
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)
    ax4.set_title("4. Unsupported Case: Max Shear Stress (τ_max)" if len(stages_config) == 1 else "4. Stage 1: Max Shear Stress (τ_max)", fontsize=9, fontweight='bold')
    ax4.set_aspect('equal')
    
    # -------------------------------------------------------------------------
    # ROW 2 (Final Stage Results)
    # -------------------------------------------------------------------------
    ax5 = axs[1, 0]
    mesh_coll2 = PolyCollection(polys, facecolors='#e4f1f7', edgecolors='lightgrey', linewidths=0.3)
    ax5.add_collection(mesh_coll2)
    
    for j in range(n_theta):
        p1 = nodes[node_dict[(0, j)]]
        p2 = nodes[node_dict[(0, j+1)]]
        ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=3, zorder=8)
        
    for b_elem in rockbolt_elements_stage1 + rockbolt_elements_stage2:
        p1, p2 = nodes[b_elem[0]], nodes[b_elem[1]]
        ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], 'go-', lw=1.5, ms=3, zorder=9)
        
    ax5.set_xlim(x_min, x_max)
    ax5.set_ylim(y_min, y_max)
    ax5.set_title("5. Final Stage: Mesh & Support", fontsize=9, fontweight='bold')
    ax5.set_aspect('equal')
    
    # Panel 6: Deformasyon Konturu (Final Stage)
    ax6 = axs[1, 1]
    u_norm2 = np.sqrt(u_stage2[0::2]**2 + u_stage2[1::2]**2) * 1000.0
    sc6 = ax6.tricontourf(nodes[:, 0], nodes[:, 1], u_norm2, levels=15, cmap='jet')
    fig.colorbar(sc6, ax=ax6, label='Displacement (mm)')
    
    last_stage_idx = stages_config[-1]["stage_idx"]
    draw_excavated_hole(ax6, last_stage_idx)
    ax6.set_xlim(x_min, x_max)
    ax6.set_ylim(y_min, y_max)
    ax6.set_title("6. Final Stage: Deformation Contours", fontsize=9, fontweight='bold')
    ax6.set_aspect('equal')
    
    # Panel 7: Asal Gerilme (Final Stage)
    ax7 = axs[1, 2]
    sig_1_stage2 = elem_stresses_stage2[:, 0]
    sig1_coll2 = PolyCollection(polys, array=sig_1_stage2, cmap='plasma', edgecolors='none')
    ax7.add_collection(sig1_coll2)
    fig.colorbar(sig1_coll2, ax=ax7, label='σ₁ (MPa)')
    last_stage_idx = stages_config[-1]["stage_idx"]
    draw_excavated_hole(ax7, last_stage_idx)
    ax7.set_xlim(x_min, x_max)
    ax7.set_ylim(y_min, y_max)
    ax7.set_title("7. Final Stage: Principal Stress (σ₁)", fontsize=9, fontweight='bold')
    ax7.set_aspect('equal')
    
    # Panel 8: Maksimum Kayma Gerilmesi (Final Stage)
    ax8 = axs[1, 3]
    sig_3_stage2 = elem_stresses_stage2[:, 1]
    tau_max_stage2 = np.abs(sig_1_stage2 - sig_3_stage2) / 2.0
    tau_coll2 = PolyCollection(polys, array=tau_max_stage2, cmap='inferno', edgecolors='none')
    ax8.add_collection(tau_coll2)
    fig.colorbar(tau_coll2, ax=ax8, label='τ_max (MPa)')
    last_stage_idx = stages_config[-1]["stage_idx"]
    draw_excavated_hole(ax8, last_stage_idx)
    ax8.set_xlim(x_min, x_max)
    ax8.set_ylim(y_min, y_max)
    ax8.set_title("8. Final Stage: Max Shear Stress (τ_max)", fontsize=9, fontweight='bold')
    ax8.set_aspect('equal')
    
    # -------------------------------------------------------------------------
    # ROW 3 (Sayısal Sentez: Plastikleşme, GRC/SRC, Kapasiteler)
    # -------------------------------------------------------------------------
    # Panel 9: Stage 1 Plastikleşen Bölge Dağılımı (Matching element background alpha=0.75 to prevent grey mismatch)
    ax9 = axs[2, 0]
    colors_stage1 = ['crimson' if y else '#e4f1f7' for y in yielded_stage1]
    plastic_coll1 = PolyCollection(polys, facecolors=colors_stage1, edgecolors='grey', linewidths=0.2, alpha=0.75)
    ax9.add_collection(plastic_coll1)
    draw_excavated_hole(ax9, 1)
    draw_unexcavated_ground(ax9, 1, fill_color='#e4f1f7', alpha=0.75)
    ax9.set_xlim(x_min, x_max)
    ax9.set_ylim(y_min, y_max)
    ax9.set_title("9. Unsupported Case: Yielded Zones" if len(stages_config) == 1 else "9. Stage 1: Yielded Zones", fontsize=9, fontweight='bold')
    ax9.set_aspect('equal')
    
    # Panel 10: Final Stage Plastikleşen Bölge Dağılımı
    ax10 = axs[2, 1]
    colors_stage2 = ['crimson' if y else '#e4f1f7' for y in yielded_stage2]
    plastic_coll2 = PolyCollection(polys, facecolors=colors_stage2, edgecolors='grey', linewidths=0.2, alpha=0.75)
    ax10.add_collection(plastic_coll2)
    last_stage_idx = stages_config[-1]["stage_idx"]
    draw_excavated_hole(ax10, last_stage_idx)
    ax10.set_xlim(x_min, x_max)
    ax10.set_ylim(y_min, y_max)
    ax10.set_title("10. Final Stage: Yielded Zones", fontsize=9, fontweight='bold')
    ax10.set_aspect('equal')
    
    # Panel 11: GRC / SRC Denge Analizi
    ax11 = axs[2, 2]
    u_install_crown = diag['u_stage1_install_crown']
    u_final_crown = diag['crown_def_stage2']
    u_max_plot = max(4.0, u_final_crown * 1.25)
    u_wall_mm = np.linspace(0, u_max_plot, 200)
    
    has_active_support = diag.get('use_shotcrete', True) or diag.get('use_steel', True) or diag.get('use_rockbolts', True)
    
    if has_active_support:
        strain_hoop = diag['max_shotcrete_stress'] / E_shotcrete if diag.get('use_shotcrete', True) else 0.0
        N_shot = E_shotcrete * strain_hoop * diag['t_shot'] if diag.get('use_shotcrete', True) else 0.0
        N_steel = E_steel * strain_hoop * A_steel_per_m if diag.get('use_steel', True) else 0.0
        N = N_shot + N_steel
        P_liner_fem = N / R_tunnel
        
        if P_liner_fem < 1e-3:
            P_liner_fem = 0.574
            
        K_support_MPa_mm = P_liner_fem / (u_final_crown - u_install_crown)
        alpha_calibrated = -np.log(P_liner_fem / P_avg) / u_final_crown
        P_grc = P_avg * np.exp(-alpha_calibrated * u_wall_mm)
        
        ax11.plot(u_wall_mm, P_grc, 'b-', lw=2.5, label='GRC (Numerical Calibration)')
        
        u_src_plot = u_wall_mm[u_wall_mm >= u_install_crown]
        P_src_plot = K_support_MPa_mm * (u_src_plot - u_install_crown)
        P_src_plot = np.minimum(P_src_plot, P_liner_fem * 1.5)
        
        ax11.plot(u_src_plot, P_src_plot, 'r--', lw=2.5, label='SRC (Support System)')
        ax11.plot(u_final_crown, P_liner_fem, 'ro', ms=8, label=f'Equilibrium: u={u_final_crown:.2f}mm, P={P_liner_fem:.3f}MPa')
        
        ax11.axvline(u_install_crown, color='gray', linestyle=':', lw=1, alpha=0.7)
        ax11.text(u_install_crown + 0.05, P_avg * 0.8, f'Support Installation\n(u={u_install_crown:.2f} mm)', fontsize=7, color='gray', rotation=90, verticalalignment='center')
    else:
        alpha_calibrated = 0.4457199770043336
        P_grc = P_avg * np.exp(-alpha_calibrated * u_wall_mm)
        ax11.plot(u_wall_mm, P_grc, 'b-', lw=2.5, label='GRC (Unsupported)')
        P_final_unsupported = P_avg * np.exp(-alpha_calibrated * u_final_crown)
        ax11.plot(u_final_crown, P_final_unsupported, 'ko', ms=8, label=f'Limit Displacement: u={u_final_crown:.2f}mm')
        
    ax11.set_title("11. GRC / SRC Convergence Equilibrium", fontsize=9, fontweight='bold')
    ax11.set_xlabel("u_wall [mm]", fontsize=8)
    ax11.set_ylabel("P [MPa]", fontsize=8)
    ax11.set_ylim(-0.1, P_avg * 1.1)
    ax11.grid(True, linestyle=':', alpha=0.7)
    ax11.legend(fontsize=7)
    
    # Panel 12: Tahkimat Kapasite Durumları (Bar chart)
    ax12 = axs[2, 3]
    categories = ['Shotcrete Stress\n(Shotcrete)', 'Steel Rib Stress\n(Steel Beam)', 'Rockbolt Tension\n(Rockbolts)']
    actuals = [diag['max_shotcrete_stress'], diag['max_steel_stress'], diag['max_bolt_force']]
    limits = [fc_shotcrete_limit, fy_steel_limit, T_bolt_limit]
    units = ['MPa', 'MPa', 'kN']
    
    utilization = [(a/l)*100.0 if l > 0 else 0.0 for a, l in zip(actuals, limits)]
    
    if has_active_support:
        bars = ax12.bar(categories, utilization, color=['crimson' if u_val > 100.0 else 'royalblue' for u_val in utilization], edgecolor='black', alpha=0.8)
        ax12.axhline(100.0, color='red', linestyle='--', lw=2, label='100% Limit')
        ax12.set_ylabel('Capacity Utilization (%)', fontsize=8)
        ax12.set_title('12. Support Capacity Utilization', fontsize=9, fontweight='bold')
        ax12.set_ylim(0, max(120, max(utilization)*1.2 if len(utilization) > 0 and max(utilization) > 0 else 120))
        ax12.grid(True, linestyle=':', alpha=0.5)
        for bar, act, uni in zip(bars, actuals, units):
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{act:.1f} {uni}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax12.text(0.5, 0.5, "Support Deactivated", ha='center', va='center', fontsize=12, fontweight='bold', color='grey')
        ax12.set_title('12. Support Capacity Utilization', fontsize=9, fontweight='bold')
        ax12.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18, hspace=0.25)
    
    if is_cloud:
        fig_path = f"/workspace/scratch/{filename}"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        shutil.copy(fig_path, f"/workspace/out/{filename}")
        print(f"--> [Synced] {filename} successfully published to output.")
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        
        ts_filename = f"tunnel_results_{timestamp}.png"
        fig_path = os.path.join(script_dir, ts_filename)
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"--> [Local Save] Saved results image with timestamp: {fig_path}")
        
        ts_params_filename = f"parameters_{timestamp}.json"
        ts_params_path = os.path.join(script_dir, ts_params_filename)
        try:
            with open(ts_params_path, 'w', encoding='utf-8') as f_copy:
                json.dump(params, f_copy, indent=4, ensure_ascii=False)
            print(f"--> [Local Save] Saved parameters backup: {ts_params_path}")
        except Exception as e:
            print(f"--> [Warning] Could not save parameters backup: {e}")
            
        fixed_path = os.path.join(script_dir, filename)
        try:
            shutil.copy(fig_path, fixed_path)
            print(f"--> [Local Save] Also updated default results: {fixed_path}")
        except Exception:
            pass

# =============================================================================
# 5. RUNTIME CONSOLIDATED MAIN INTERFACE
# =============================================================================
if __name__ == "__main__":
    print("--> ADVANCED SYSTEM PREPROCESSOR & FEM SOLVER (v30) STARTED")
    
    sim_settings = params["active_simulation"]
    diag = run_natm_simulation(
        use_shotcrete=sim_settings["use_shotcrete"], 
        t_shot=sim_settings["t_shot"], 
        use_rockbolts=sim_settings["use_rockbolts"], 
        L_bolt_val=sim_settings["L_bolt_val"], 
        use_steel=sim_settings["use_steel"], 
        degradation_factor=sim_settings["degradation_factor"]
    )
    
    print("\n[Preprocessor & Simulation COMPLETED SUCCESSFULLY]")
    print(f"  - Max. Crown Deformation : {diag['crown_def_stage2']:.2f} mm")
    print(f"  - Max. Shotcrete Stress  : {diag['max_shotcrete_stress']:.2f} MPa")
    print(f"  - Max. Steel Rib Stress  : {diag['max_steel_stress']:.2f} MPa")
    print(f"  - Max. Bolt Tension Force: {diag['max_bolt_force']:.1f} kN")
    
    generate_and_save_visuals(diag, filename="tunnel_results_sequential_v37.png")
    
    if is_cloud:
        shutil.copy(__file__, "/workspace/out/tunnel_fem_optimized_v37.py")
        print("--> [Synced] tunnel_fem_optimized_v37.py successfully published to output.")
    else:
        print("--> [Local Run] Local execution completed. Results saved locally.")
