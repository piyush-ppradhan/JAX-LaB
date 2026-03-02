import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
# บังคับให้ JAX ทำงานด้วยความละเอียด float64 เพื่อความเสถียรของ MCMP และ MRT
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.tree import map as jax_map
import csv
import os
import sys
import scipy.ndimage as ndimage
import pyvista as pv

sys.path.append(os.path.abspath("../../"))

from src.lattice import LatticeD3Q19
from src.physics.crystallization import compute_heterogeneous_precipitation, calculate_equilibrium_concentration
from src.physics.porous_media import compute_permeability, compute_supersaturation
from src.physics.wettability import compute_virtual_density

# -------------------------------------------------------------------
# [JAX-LaB Core Imports]
# -------------------------------------------------------------------
from src.multiphase import MultiphaseBGK
from src.eos import Peng_Robinson

def parse_ui_args():
    parser = argparse.ArgumentParser(description="JAX-LaB CuSO4 (v5 PR-EOS MCMP)")
    parser.add_argument("--geom", type=str, default="geometry_mask.npy")
    parser.add_argument("--axis", type=str, choices=['X', 'Y', 'Z'], default='X')
    parser.add_argument("--flow_rate", type=float, default=1.0, help="Flow rate in mL/hr")
    parser.add_argument("--dx_um", type=float, default=20.0, help="Voxel size in micrometers (um)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--inject_size", type=int, default=60)
    return parser.parse_args()

def save_vti_file(filename, array, name, is_vector=False):
    if is_vector:
        dimensions = array.shape[:-1]
        grid = pv.ImageData(dimensions=dimensions)
        grid.point_data[name] = array.reshape(-1, 3, order="F")
    else:
        dimensions = array.shape
        grid = pv.ImageData(dimensions=dimensions)
        grid.point_data[name] = array.flatten(order="F")
    grid.save(filename)

@jit
def calculate_tau_f(T_celsius, tau_ref=1.0):
    temp_points = jnp.array([25.0, 35.0, 45.0, 55.0, 65.0, 75.0], dtype=jnp.float64)
    viscosity_points = jnp.array([1.35, 1.08, 0.89, 0.74, 0.63, 0.55], dtype=jnp.float64)
    mu_T = jnp.interp(T_celsius, temp_points, viscosity_points)
    mu_ref = 1.35 
    tau_f = 0.5 + (tau_ref - 0.5) * (mu_T / mu_ref)
    return tau_f

# =========================================================================
# คลาสจำลอง Reactive MCMP Simulator (สืบทอดจาก Multiphase Core)
# =========================================================================
class ReactiveMCMP_Simulator(MultiphaseBGK):
    def __init__(self, mask, kappa_mrt=0.15, **kwargs):
        super().__init__(**kwargs)
        self.solid_mask = ~mask
        self.fluid_mask = mask
        self.kappa_mrt = kappa_mrt
        
    def macroscopic_velocity(self, fin_tree, rho_tree):
        # 1. แทรก Wettability: ปรับความหนาแน่นจำลองที่ขอบของแข็ง (มุม 45 องศา)
        rho_tree_wet = jax_map(
            lambda rho: compute_virtual_density(rho, self.solid_mask, self.fluid_mask, theta=jnp.pi/4, phi=0.8, delta_rho=0.05),
            rho_tree
        )

        # 2. ให้ JAX-LaB คำนวณความเร็วต่อ component ด้วย PR-EOS และ Shan-Chen Forces
        u_tree = super().macroscopic_velocity(fin_tree, rho_tree_wet)

        # 3. คำนวณความเร็วรวม (mass-averaged) เพื่อให้เป็น single array
        u_eq = self.compute_total_velocity(rho_tree_wet, u_tree)

        # 4. [STABILITY] จํากัดความเร็ว (Clipping) ป้องกันโค้ดระเบิดช่วงแรก
        u_eq = jnp.clip(u_eq, -0.1, 0.1)
        return u_eq

    def collision(self, fin_tree, T_field):
        # 1. ให้ JAX-LaB ทำการชนแบบ MRT / Cascaded ให้เสร็จสมบูรณ์
        fout_tree = super().collision(fin_tree)
        
        # 2. คํานวณความหนืดเฉพาะจุดจากอุณหภูมิ
        tau_f_local = calculate_tau_f(T_field, tau_ref=1.0)
        omega_f_local = 1.0 / tau_f_local
        
        # 3. ดึงความหนาแน่นและความเร็วมาคำนวณสมดุล (f_eq) เพื่อใช้ในเทอม Source
        rho_tree, _ = self.update_macroscopic(fin_tree)
        u_eq = self.macroscopic_velocity(fin_tree, rho_tree)
        feq_tree = jax_map(lambda rho: self.equilibrium(rho, u_eq), rho_tree)
        
        # 4. บวกเทอม Surface Tension (kappa) ทับลงไปบนผลลัพธ์ของคลาสแม่
        fout_tree_modified = jax_map(
            lambda fout, f, feq: fout + self.kappa_mrt * (feq - f) * (1.0 - 0.5 * omega_f_local[..., None]),
            fout_tree, fin_tree, feq_tree
        )
        return fout_tree_modified

# =========================================================================
# ฟังก์ชันคำนวณ LBM หลัก
# =========================================================================
def run_simulation():
    args = parse_ui_args()
    
    print(f"Loading geometry and cropping to {args.inject_size}^3...")
    mask_np_full = np.load(args.geom).astype(bool)
    
    c_size = args.inject_size
    half_c = c_size // 2
    cx_o, cy_o, cz_o = mask_np_full.shape[0]//2, mask_np_full.shape[1]//2, mask_np_full.shape[2]//2
    xs, xe = cx_o - half_c, cx_o + half_c
    ys, ye = cy_o - half_c, cy_o + half_c
    zs, ze = cz_o - half_c, cz_o + half_c
    
    mask = jnp.array(mask_np_full[xs:xe, ys:ye, zs:ze])
    
    ny, nz = mask.shape[1], mask.shape[2]
    Y, Z = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    radius = 30.0 
    cy, cz = ny / 2.0, radius 
    r_sq = (Y - cy)**2 + (Z - cz)**2
    circular_mask_np = r_sq <= radius**2
    circular_mask = jnp.array(circular_mask_np) 
    
    lattice = LatticeD3Q19()
    c_int = np.array(lattice.c, dtype=int).T.tolist()   
    c = jnp.array(lattice.c, dtype=jnp.float64).T       
    w = jnp.array(lattice.w, dtype=jnp.float64)
    c_np = np.array(lattice.c, dtype=np.float64).T 
    
    dx_m = args.dx_um * 1e-6    
    dx_mm = args.dx_um * 1e-3   
    nu_phys = 1e-6 
    
    tau_f_ref = 1.0   
    tau_t = 0.55  
    tau_c = 0.95  
    k_r = 0.15
    omega_t, omega_c = 1.0/tau_t, 1.0/tau_c
    
    nu_lb = (tau_f_ref - 0.5) / 3.0
    dt_s = (nu_lb * (dx_m ** 2)) / nu_phys 
    
    Q_m3s = args.flow_rate / 3.6e9  
    cross_section_area_m2 = float(np.sum(circular_mask_np)) * (dx_m ** 2)
    u_phys_inlet = Q_m3s / cross_section_area_m2 
    u_lb = u_phys_inlet * (dt_s / dx_m)          
    
    vol_scale_mm3 = dx_mm ** 3
    k_scale_m2 = dx_m ** 2
    k_scale_darcy = k_scale_m2 / 0.9869233e-12  
    u_scale_mms = (dx_mm / dt_s)                
    
    print("\n--- Physical Scales Confirmed (MCMP PR-EOS Mode) ---")
    print(f"  Voxel Size: {args.dx_um} um")
    print(f"  Time Step (dt): {dt_s:.2e} s")
    print(f"  Inlet Velocity (Target): {u_phys_inlet*1000:.2f} mm/s (LBM: {u_lb:.4f})")
    print("----------------------------------------------------\n")

    # ---------------------------------------------------------
    # ตั้งค่า Native JAX-LaB Core สำหรับ MCMP PR-EOS
    # ---------------------------------------------------------
    # 1. พารามิเตอร์ Peng-Robinson แบบ Safe-mode สำหรับ 2 Components
    # ---------------------------------------------------------
    # ตั้งค่า Native JAX-LaB Core สำหรับ MCMP PR-EOS (Bullet-proof)
    # ---------------------------------------------------------
    
    # 1. พารามิเตอร์ Peng-Robinson (ใส่ให้ครบทุกตัวที่ eos.py ต้องการ)
    pr_eos = Peng_Robinson(
        a=[0.02, 0.02],           # Cohesion parameter 
        b=[0.05, 0.05],           # Co-volume parameter
        R=[1.0, 1.0],             # Gas constant
        Tc=[1.0, 1.0],            # Critical temperature
        pr_omega=[0.344, 0.344],  # Acentric factor (ใช้ชื่อ pr_omega ตามไลบรารี)
        T=0.85                    # Isothermal temperature
    )
    
    # 2. ปฏิสัมพันธ์ Shan-Chen
    g_kk_val = jnp.array([-1.0, -1.0], dtype=jnp.float64) 
    g_kkprime_val = jnp.array([
        [0.0, 0.57], 
        [0.57, 0.0]
    ], dtype=jnp.float64)
    
    # 3. เตรียมขนาดโดเมน (Grid Size) เผื่อ LBMBase ต้องการ
    nx, ny, nz = mask.shape
    
    # 4. สร้าง Object ของ Simulator โดยอัด kwargs ให้ครบทุกระดับ!
    sim = ReactiveMCMP_Simulator(
        # --- Custom Physics Kwargs ---
        mask=mask,
        kappa_mrt=0.10,           # ลดแรงตึงผิวช่วงต้นเพื่อความเสถียร

        # --- LBMBase Kwargs (Core) ---
        lattice=lattice,
        precision="f64",
        nx=nx,
        ny=ny,
        nz=nz,
        omega=[1.0, 1.0],         # ความถี่การผ่อนคลายของ 2 Components

        # --- Multiphase Kwargs ---
        EOS=pr_eos,               # ต้องใช้ชื่อ EOS (ตัวพิมพ์ใหญ่) ตาม Multiphase.__init__
        n_components=2,           # บังคับระบุจำนวน Component อย่างชัดเจน
        k=[1.0, 1.0],             # Modification coefficient สำหรับ EOS potential
        A=np.zeros((2, 2)),       # Zhang-Chen weighting (0 = pure Shan-Chen)
        g_kkprime=g_kkprime_val
    )

    # ---------------------------------------------------------
    # Initialize State Variables
    # ---------------------------------------------------------
    T_hot, T_cold, C_inlet = 75.0, 25.0, 1.0
    
    rho1 = jnp.zeros(mask.shape, dtype=jnp.float64) 
    rho2 = jnp.ones(mask.shape, dtype=jnp.float64) * 0.5 # Native Air
    u_init = jnp.zeros(mask.shape + (3,), dtype=jnp.float64)
    
    f1 = sim.equilibrium(rho1[..., None], u_init)
    f2 = sim.equilibrium(rho2[..., None], u_init)
    f_tree = [f1, f2] # โครงสร้าง PyTree สำหรับ MCMP
    
    T_field = jnp.ones(mask.shape, dtype=jnp.float64) * T_cold
    C_field = jnp.zeros(mask.shape, dtype=jnp.float64)
    solid_fraction = jnp.zeros(mask.shape, dtype=jnp.float64)

    @jit
    def calc_equilibrium_single(phi, u_eq):
        cu = jnp.dot(u_eq, c.T)
        usqr = jnp.sum(u_eq**2, axis=-1, keepdims=True)
        return phi[..., None] * w * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usqr)

    def shift_no_wrap(a, sx, sy, sz):
        out = jnp.zeros_like(a)
        xs_src = slice(max(-sx, 0), a.shape[0] - max(sx, 0))
        ys_src = slice(max(-sy, 0), a.shape[1] - max(sy, 0))
        zs_src = slice(max(-sz, 0), a.shape[2] - max(sz, 0))
        xs_dst = slice(max(sx, 0), a.shape[0] - max(-sx, 0))
        ys_dst = slice(max(sy, 0), a.shape[1] - max(-sy, 0))
        zs_dst = slice(max(sz, 0), a.shape[2] - max(-sz, 0))
        return out.at[xs_dst, ys_dst, zs_dst].set(a[xs_src, ys_src, zs_src])

    @jit
    def lbm_step(state, step_idx):
        f_tree, g, h, solid_frac = state
        
        # 1. Update Macroscopic ของไหล
        rho_tree, _ = sim.update_macroscopic(f_tree)
        u_eq = sim.macroscopic_velocity(f_tree, rho_tree)
        rho1, rho2 = rho_tree
        
        T_curr = jnp.sum(g, axis=-1)
        C_curr = jnp.sum(h, axis=-1)
        
        # 2. [COLLISION] ใช้ Override Method ที่มี PR-EOS, Wettability และ Kappa
        f_post_tree = sim.collision(f_tree, T_curr)
        f1_post, f2_post = f_post_tree
        
        # Collision สำหรับความร้อนและสารละลาย (T, C)
        g_post = g - omega_t * (g - calc_equilibrium_single(T_curr, u_eq))
        h_post = h - omega_c * (h - calc_equilibrium_single(C_curr, u_eq))
        
        # 3. [PRECIPITATION KINETICS]
        effective_fluid_mask = mask & (solid_frac < 0.5)
        wall_mask = jnp.zeros_like(effective_fluid_mask, dtype=bool)
        if args.axis == 'X':
            wall_mask = wall_mask.at[:, 0, :].set(True)
            wall_mask = wall_mask.at[:, -1, :].set(True)
            wall_mask = wall_mask.at[:, :, 0].set(True)
            wall_mask = wall_mask.at[:, :, -1].set(True)
            wall_mask = wall_mask.at[0, :, :].set(wall_mask[0, :, :] | ~circular_mask)
            wall_mask = wall_mask.at[-1, :, :].set(wall_mask[-1, :, :] | ~circular_mask)
            
        effective_fluid_mask_bc = effective_fluid_mask & (~wall_mask)
        delta_C = compute_heterogeneous_precipitation(C_curr, T_curr, k_r, effective_fluid_mask_bc, c_int)
        h_post = h_post - w * delta_C[..., None]
        solid_frac = solid_frac + delta_C
        
        # 4. [STREAMING & BOUNCE-BACK]
        f1_str, f2_str = jnp.zeros_like(f1_post), jnp.zeros_like(f2_post)
        g_str, h_str = jnp.zeros_like(g), jnp.zeros_like(h)
        solid_mask_bc = ~effective_fluid_mask_bc
        domain_ones = jnp.ones_like(mask, dtype=bool)
        opp = jnp.array(lattice.opp_indices)
        
        for i in range(19):
            sx, sy, sz = c_int[i][0], c_int[i][1], c_int[i][2]
            f1_str_i = shift_no_wrap(f1_post[..., i], sx, sy, sz)
            f2_str_i = shift_no_wrap(f2_post[..., i], sx, sy, sz)
            g_str_i = shift_no_wrap(g_post[..., i], sx, sy, sz)
            h_str_i = shift_no_wrap(h_post[..., i], sx, sy, sz)
            
            is_invalid = shift_no_wrap(solid_mask_bc, sx, sy, sz) | ~shift_no_wrap(domain_ones, sx, sy, sz)
            
            f1_str = f1_str.at[..., i].set(jnp.where(is_invalid, f1_post[..., opp[i]], f1_str_i))
            f2_str = f2_str.at[..., i].set(jnp.where(is_invalid, f2_post[..., opp[i]], f2_str_i))
            g_str = g_str.at[..., i].set(jnp.where(is_invalid, g_post[..., opp[i]], g_str_i))
            h_str = h_str.at[..., i].set(jnp.where(is_invalid, h_post[..., opp[i]], h_str_i))
        
        # 5. [INLET/OUTLET BOUNDARY CONDITIONS] - Soft Start
        ramp_factor = jnp.clip(step_idx / 500.0, 0.0, 1.0)
        current_u_lb = u_lb * ramp_factor
        target_u_dynamic = jnp.zeros(3).at[0].set(current_u_lb)
        
        if args.axis == 'X':
            u_in = jnp.zeros_like(u_eq[0]).at[..., 0].set(current_u_lb)
            f1_eq_in = calc_equilibrium_single(jnp.ones_like(rho1[0, ..., 0]), u_in)
            f2_eq_in = calc_equilibrium_single(jnp.zeros_like(rho2[0, ..., 0]), u_in)
            g_eq_in = calc_equilibrium_single(jnp.ones_like(T_curr[0]) * T_hot, u_in)
            h_eq_in = calc_equilibrium_single(jnp.ones_like(C_curr[0]) * C_inlet, u_in)
            
            f1_str = f1_str.at[0].set(jnp.where(circular_mask[..., None], f1_eq_in, f1_str[0]))
            f2_str = f2_str.at[0].set(jnp.where(circular_mask[..., None], f2_eq_in, f2_str[0]))
            g_str = g_str.at[0].set(jnp.where(circular_mask[..., None], g_eq_in, g_str[0]))
            h_str = h_str.at[0].set(jnp.where(circular_mask[..., None], h_eq_in, h_str[0]))
            
            f1_str = f1_str.at[-1].set(jnp.where(circular_mask[..., None], f1_str[-2], f1_str[-1]))
            f2_str = f2_str.at[-1].set(jnp.where(circular_mask[..., None], f2_str[-2], f2_str[-1]))
            g_str = g_str.at[-1].set(jnp.where(circular_mask[..., None], g_str[-2], g_str[-1]))
            h_str = h_str.at[-1].set(jnp.where(circular_mask[..., None], h_str[-2], h_str[-1]))

        f_tree_out = [f1_str, f2_str]
        return (f_tree_out, g_str, h_str, solid_frac), None

    chunk_size = 500
    num_chunks = args.steps // chunk_size
    
    @jit(static_argnums=(1,))
    def run_chunk(state_in, steps, chunk_start):
        step_indices = jnp.arange(steps) + chunk_start
        state_out, _ = jax.lax.scan(lbm_step, state_in, step_indices)
        return state_out

    os.makedirs("outputs/vti", exist_ok=True)
    os.makedirs("outputs/analytics", exist_ok=True)
    
    with open("outputs/global_kinetics.csv", "w", newline="") as f_csv1, \
         open("outputs/object_analysis.csv", "w", newline="") as f_csv2, \
         open("outputs/pore_clogging_stats.csv", "w", newline="") as f_csv3:
        csv.writer(f_csv1).writerow(["Step", "Time_s", "Total_Solid_Volume_mm3", "Porosity", "Global_Permeability_Darcy", "Avg_Temperature"])
        csv.writer(f_csv2).writerow(["Step", "Number_of_Crystals", "Avg_Crystal_Size", "Max_Crystal_Size", "Surface_Area"])
        csv.writer(f_csv3).writerow(["Step", "Min_Throat_Size", "Tortuosity_Index"])

    domain_length = float(args.inject_size)
    D_solute = (1.0/3.0) * (tau_c - 0.5)

    print(f"Running Reactive MCMP (PR-EOS) {args.steps} LBM steps with comprehensive I/O...")
    state = (f_tree, T_field, C_field, solid_fraction)
    mask_cpu = np.array(mask)
    mid_x, mid_y, mid_z = mask_cpu.shape[0]//2, mask_cpu.shape[1]//2, mask_cpu.shape[2]//2
    
    vel_mag_t0 = None
    vel_mag_tfinal = None
    pe_da_data = [] 
    maps_data = {} 
    
    for i in range(num_chunks + 1):
        if i > 0:
            chunk_start_step = (i - 1) * chunk_size
            state = run_chunk(state, chunk_size, chunk_start_step)
            
        state[0][0].block_until_ready()
        current_step = i * chunk_size
        
        if current_step == 0 or current_step % 500 == 0:
            f_tree_np, g_np, h_np, solid_np = state
            f1_np, f2_np = [np.array(x) for x in f_tree_np]
            g_np, h_np, solid_np = np.array(g_np), np.array(h_np), np.array(solid_np)
            
            rho1_np = np.sum(f1_np, axis=-1)
            rho2_np = np.sum(f2_np, axis=-1)
            rho_tot_np = rho1_np + rho2_np
            
            safe_rho1 = np.where(rho1_np == 0, 1e-8, rho1_np)
            safe_rho2 = np.where(rho2_np == 0, 1e-8, rho2_np)
            safe_rho_tot = np.where(rho_tot_np == 0, 1e-8, rho_tot_np)
            
            u1_np = np.dot(f1_np, c_np) / safe_rho1[..., None]
            u2_np = np.dot(f2_np, c_np) / safe_rho2[..., None]
            u_np = (rho1_np[..., None] * u1_np + rho2_np[..., None] * u2_np) / safe_rho_tot[..., None]
            
            T_np = np.sum(g_np, axis=-1)
            C_np = np.sum(h_np, axis=-1)
            
            binary_precipitate = np.where(solid_np > 0.1, 1.0, 0.0).astype(np.float32)
            fluid_mask_current = mask_cpu & (solid_np < 0.5)
            supersat_map = np.array(compute_supersaturation(jnp.array(C_np), jnp.array(T_np), calculate_equilibrium_concentration))
            u_mag = np.linalg.norm(u_np, axis=-1)
            
            if current_step == 0:
                vel_mag_t0 = u_mag[fluid_mask_current]

            maps_data[current_step] = {
                'XY': {'T': T_np[:, :, mid_z].copy(), 'C': C_np[:, :, mid_z].copy(), 'solid': solid_np[:, :, mid_z].copy()},
                'XZ': {'T': T_np[:, mid_y, :].copy(), 'C': C_np[:, mid_y, :].copy(), 'solid': solid_np[:, mid_y, :].copy()},
                'YZ': {'T': T_np[mid_x, :, :].copy(), 'C': C_np[mid_x, :, :].copy(), 'solid': solid_np[mid_x, :, :].copy()},
                'Z_proj': {'solid_sum': np.sum(solid_np, axis=2).copy()}
            }

            save_vti_file(f"outputs/vti/precipitate_growth_t{current_step}.vti", binary_precipitate, "CuSO4_Solid")
            save_vti_file(f"outputs/vti/velocity_evolution_t{current_step}.vti", u_np, "Velocity", is_vector=True)
            save_vti_file(f"outputs/vti/supersaturation_map_t{current_step}.vti", supersat_map, "Supersaturation")
            save_vti_file(f"outputs/vti/cuso4_phase_t{current_step}.vti", rho1_np / safe_rho_tot, "CuSO4_Phase")

            P_in = np.mean(rho_tot_np[0][circular_mask_np]) / 3.0
            P_out = np.mean(rho_tot_np[-1][circular_mask_np]) / 3.0
            delta_P = P_in - P_out
            mean_u = np.mean(u_np[..., 0]) 
            
            avg_T = np.mean(T_np)
            tau_f_avg = float(calculate_tau_f(jnp.array(avg_T), tau_ref=tau_f_ref))
            mu_fluid_avg = (tau_f_avg - 0.5) / 3.0
            
            k_raw = float(np.array(compute_permeability(mean_u, mu_fluid_avg, domain_length, delta_P)))
            k_perm_darcy = abs(k_raw) * k_scale_darcy
            
            current_solid_vol_mm3 = np.sum(solid_np) * vol_scale_mm3
            time_s = current_step * dt_s 
            
            total_voxels = mask_cpu.size         
            initial_fluid_voxels = np.sum(mask_cpu) 
            porosity = (initial_fluid_voxels - np.sum(solid_np)) / total_voxels

            labeled_array, num_features = ndimage.label(binary_precipitate > 0)
            avg_size, max_size = 0, 0
            if num_features > 0:
                sizes = np.bincount(labeled_array.ravel())[1:]
                avg_size = np.mean(sizes)
                max_size = np.max(sizes)
            
            solid_dilated = ndimage.binary_dilation(binary_precipitate > 0)
            surface_area = np.sum(solid_dilated & fluid_mask_current)

            u_x = u_np[..., 0]
            tortuosity = np.sum(u_mag[fluid_mask_current]) / (np.sum(u_x[fluid_mask_current]) + 1e-8)
            
            dt = ndimage.distance_transform_edt(fluid_mask_current)
            max_r_per_slice = [np.max(dt[x, :, :]) for x in range(dt.shape[0])]
            min_throat = np.min(max_r_per_slice)

            max_supersat = np.max(supersat_map[fluid_mask_current]) if np.any(fluid_mask_current) else 0.0

            with open("outputs/global_kinetics.csv", "a", newline="") as f_csv1, \
                 open("outputs/object_analysis.csv", "a", newline="") as f_csv2, \
                 open("outputs/pore_clogging_stats.csv", "a", newline="") as f_csv3:
                csv.writer(f_csv1).writerow([current_step, time_s, current_solid_vol_mm3, porosity, k_perm_darcy, avg_T])
                csv.writer(f_csv2).writerow([current_step, num_features, avg_size, max_size, surface_area])
                csv.writer(f_csv3).writerow([current_step, min_throat, tortuosity])

            print(f"Step {current_step}/{args.steps} | Time: {time_s:.2f} s | Porosity: {porosity:.4f} | Crystals: {num_features} | k: {k_perm_darcy:.2e} Darcy | Max Supersat: {max_supersat:.4f} | Crystal Vol: {current_solid_vol_mm3:.2e} mm3")

    vel_mag_tfinal = u_mag[fluid_mask_current]
    
    u_mag_safe = np.where(u_mag == 0, 1e-8, u_mag)
    L_ref = 1.0 
    Pe_map = (u_mag * L_ref) / D_solute
    Da_map = (k_r * L_ref) / u_mag_safe
    pe_da_data = (Pe_map[fluid_mask_current], Da_map[fluid_mask_current])

    return (vel_mag_t0, vel_mag_tfinal, pe_da_data, maps_data, mask_cpu, u_scale_mms)

# --- ละโค้ด generate_reaction_maps และ generate_analytical_plots ไว้ด้านล่าง (ใช้โค้ดชุด v3 เดิมได้เลย) ---
def generate_reaction_maps(maps_data, mask_np):
    print("\nGenerating Spatial Reaction Maps (XY, XZ, YZ, and Z-Projection)...")
    steps_saved = sorted(list(maps_data.keys()))
    
    if len(steps_saved) >= 3:
        steps_to_plot = [steps_saved[0], steps_saved[len(steps_saved)//2], steps_saved[-1]]
    else:
        steps_to_plot = steps_saved
        
    mid_x, mid_y, mid_z = mask_np.shape[0]//2, mask_np.shape[1]//2, mask_np.shape[2]//2
    
    planes = {
        'XY': {'mask': mask_np[:, :, mid_z], 'title': 'X-Y Cross Section (Mid-Z)'},
        'XZ': {'mask': mask_np[:, mid_y, :], 'title': 'X-Z Cross Section (Mid-Y)'},
        'YZ': {'mask': mask_np[mid_x, :, :], 'title': 'Y-Z Cross Section (Mid-X)'}
    }
    
    for plane_name, plane_info in planes.items():
        fig, axes = plt.subplots(len(steps_to_plot), 3, figsize=(18, 5 * len(steps_to_plot)))
        if len(steps_to_plot) == 1: axes = np.expand_dims(axes, axis=0) 
        mask_slice = plane_info['mask']
        
        for row_idx, step in enumerate(steps_to_plot):
            T_slice = maps_data[step][plane_name]['T'].astype(float)
            C_slice = maps_data[step][plane_name]['C'].astype(float)
            solid_slice = maps_data[step][plane_name]['solid'].astype(float)
            
            T_slice[~mask_slice] = np.nan
            C_slice[~mask_slice] = np.nan
            solid_slice[~mask_slice] = np.nan
            
            im0 = axes[row_idx, 0].imshow(T_slice.T, cmap='inferno', origin='lower', vmin=25, vmax=75)
            axes[row_idx, 0].set_title(f'Step {step}: Temp (°C)')
            fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)
            
            im1 = axes[row_idx, 1].imshow(C_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=1.0)
            axes[row_idx, 1].set_title(f'Step {step}: CuSO4 Conc.')
            fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[row_idx, 2].imshow(solid_slice.T, cmap='cool', origin='lower')
            axes[row_idx, 2].set_title(f'Step {step}: Crystal Vol')
            fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
            
        fig.suptitle(f"Reactive Transport Evolution: {plane_info['title']}", fontsize=20)
        fig.tight_layout()
        fig.savefig(f"outputs/analytics/cuso4_reaction_maps_{plane_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    fig_proj, axes_proj = plt.subplots(1, len(steps_to_plot), figsize=(6 * len(steps_to_plot), 5))
    if len(steps_to_plot) == 1: axes_proj = [axes_proj]

    pore_depth = np.sum(mask_np, axis=2).astype(float)
    pore_depth[pore_depth == 0] = np.nan 

    for col_idx, step in enumerate(steps_to_plot):
        solid_sum = maps_data[step]['Z_proj']['solid_sum'].astype(float)
        solid_sum[np.isnan(pore_depth)] = np.nan 

        im = axes_proj[col_idx].imshow(solid_sum.T, cmap='magma', origin='lower')
        axes_proj[col_idx].set_title(f'Step {step}: Total Crystal Depth')
        fig_proj.colorbar(im, ax=axes_proj[col_idx], fraction=0.046, pad=0.04)

    fig_proj.suptitle("Z-Projection (Top-down Sum of Crystal Volume)", fontsize=20)
    fig_proj.tight_layout()
    fig_proj.savefig("outputs/analytics/cuso4_reaction_maps_Z_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved Z-Projection Map")

def generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms):
    print("Generating Analytical PNG Plots (Physical Units)...")
    os.makedirs("outputs/analytics", exist_ok=True)
    
    kinetics = np.genfromtxt("outputs/global_kinetics.csv", delimiter=',', skip_header=1)
    objects = np.genfromtxt("outputs/object_analysis.csv", delimiter=',', skip_header=1)
    
    if kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        permeability = kinetics[:, 4]
        
        plt.figure(figsize=(8, 6))
        k_mag = np.abs(permeability) 
        valid_idx = k_mag > 1e-10  
        
        if np.any(valid_idx):
            plt.plot(time_s[valid_idx], k_mag[valid_idx], 'b-o', linewidth=2)
        else:
            plt.plot(time_s, k_mag, 'b-o', linewidth=2)
            
        plt.title("Absolute Permeability Reduction")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Permeability (Darcy)")
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig("outputs/analytics/permeability_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(8, 6))
    v0_valid = (vel_t0[vel_t0 > 1e-6] * u_scale_mms) if vel_t0 is not None else []
    vf_valid = (vel_tfinal[vel_tfinal > 1e-6] * u_scale_mms) if vel_tfinal is not None else []
    
    if len(v0_valid) > 0:
        plt.hist(v0_valid, bins=50, alpha=0.5, label='Initial', density=True, color='blue')
    if len(vf_valid) > 0:
        plt.hist(vf_valid, bins=50, alpha=0.5, label='Clogged', density=True, color='red')
        
    plt.title("Pore Velocity Distribution Shift")
    plt.xlabel("Local Velocity Magnitude (mm/s)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/analytics/velocity_distribution_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        surface_area = objects[:, 4]
        vol_mm3 = kinetics[:, 2] 
        
        sa_v_ratio = surface_area / (vol_mm3 + 1e-8)
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'tab:red'
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color1)
        ax1.plot(time_s, vol_mm3, color=color1, linewidth=2, marker='s', label='Volume')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'tab:blue'
        ax2.set_ylabel('Surface Area / Volume Ratio (SA/V)', color=color2)  
        ax2.plot(time_s, sa_v_ratio, color=color2, linewidth=2, marker='o', label='SA/V Ratio')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Morphology Trajectory: Patchy vs Layer-like Growth")
        fig.tight_layout()  
        plt.savefig("outputs/analytics/morphology_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close()

    if pe_da_data is not None:
        Pe_vals, Da_vals = pe_da_data
        valid_mask = (Pe_vals > 0) & (Da_vals > 0)
        Pe_valid = Pe_vals[valid_mask]
        Da_valid = Da_vals[valid_mask]
        
        plt.figure(figsize=(8, 6))
        num_points = min(5000, len(Pe_valid))
        if num_points > 0:
            idx = np.random.choice(len(Pe_valid), num_points, replace=False)
            plt.scatter(Da_valid[idx], Pe_valid[idx], alpha=0.4, c='purple', s=15, edgecolors='none')
            
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Local Transport Regime ($Pe$ vs $Da$)")
        plt.xlabel("Damköhler Number ($Da$) - Reaction Dominance")
        plt.ylabel("Péclet Number ($Pe$) - Advection Dominance")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.8)
        plt.axvline(x=1, color='k', linestyle='-', alpha=0.8)
        plt.text(0.01, 10, 'Advection-Limited', fontsize=10, color='darkgreen')
        plt.text(10, 0.01, 'Reaction-Limited', fontsize=10, color='darkred')
        plt.savefig("outputs/analytics/transport_regime_da_pe.png", dpi=300, bbox_inches='tight')
        plt.close()

    if objects.ndim > 1 and kinetics.ndim > 1:
        time_s = kinetics[:, 1]
        num_crystals = objects[:, 1] 
        vol_mm3 = kinetics[:, 2] 
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color1 = 'tab:green'
        ax1.set_xlabel('Time (Seconds)')
        ax1.set_ylabel('Number of Crystals (Nucleation Sites)', color=color1)
        ax1.plot(time_s, num_crystals, color=color1, linewidth=2, marker='^', label='Crystal Count')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Total Precipitation Volume ($mm^3$)', color=color2)  
        ax2.plot(time_s, vol_mm3, color=color2, linewidth=2, marker='s', label='Volume')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Nucleation Saturation: Crystal Count & Volume vs Time")
        fig.tight_layout()  
        plt.savefig("outputs/analytics/nucleation_saturation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All analytical PNGs (Physical Units) exported successfully!")

if __name__ == "__main__":
    vel_t0, vel_tfinal, pe_da_data, maps_data, mask_np, u_scale_mms = run_simulation()
    generate_reaction_maps(maps_data, mask_np)
    generate_analytical_plots(vel_t0, vel_tfinal, pe_da_data, u_scale_mms)