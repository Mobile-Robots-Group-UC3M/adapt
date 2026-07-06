#!/usr/bin/env python3
# ============================================================================
#  gello_ADAM_collision_sdf_apf_haptics_optimized.py 
# ----------------------------------------------------------------------------
#  Teleoperacion GELLO -> ADAMSim con BLOQUEO DIRECCIONAL (SDF).
#  - Evasión Suave y Firme (APF): Deslizamiento mediante gradientes con Clipping.
#  - MULTIPROCESSING: SDF y Hardware en un núcleo libre de GIL, PyBullet en otro.
#  - SIN LAG: Renderizado fluido a 60 FPS fijos.
# ============================================================================

import math
import time
import threading
import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import torch
import multiprocessing as mp

torch.set_num_threads(1) 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import trimesh
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite

# ============================================================================
#  1. PANEL DE CONFIGURACIÓN MAESTRO
# ============================================================================
class CONFIG:
    ADAMSIM_DIR = "/home/alumnos/tfg-laura/AdamSim"
    SDF_DIR     = "/home/alumnos/tfg-laura/SignedDistanceFields"
    ROBOT_URDF  = f"{ADAMSIM_DIR}/models/robot/rb1_base_description/robots/robotDummy.urdf"
    SDF_URDF    = f"{SDF_DIR}/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf"
    SDF_MODEL   = f"{SDF_DIR}/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt"

    PORT_R      = "/dev/ttyUSB1"
    MOTORS_R    = [8, 9, 10, 11, 12, 13]
    SIGNS_R     = [1, 1, -1, 1, 1, 1]
    OFFSETS_R = [-6.2262, -3.3671, 3.1002, -1.6222, -3.2817, -4.6493]
    ACTIVO_R    = [1, 1, 1, 1, 1, 1] 

    PORT_L      = "/dev/ttyUSB0"
    MOTORS_L    = [1, 2, 3, 4, 5, 6]
    SIGNS_L     = [1, 1, -1, 1, 1, 1]
    OFFSETS_L = [-4.4477, -4.4879, 3.0840, 3.2282, -3.0217, 1.5567]
    ACTIVO_L    = [1, 1, 1, 1, 1, 1]

    BAUDRATE    = 57600
    PROTOCOL    = 2.0
    TICKS_REV   = 4096
    MODE_POS    = 5 
    CURRENT_HOLD= [350, 500, 350, 220, 160, 160] 

    ADDR_MODE        = 11
    ADDR_TORQUE      = 64
    ADDR_GOAL_CURR   = 102
    ADDR_GOAL_POS    = 116
    ADDR_PRESENT_POS = 132

    OBSTACLE_RADIUS       = 0.08   
    SELF_COLLISION_RADIUS = 0.00   
    
    # DOBLE DENSIDAD OPTIMIZADA PARA RENDIMIENTO
    NUM_PTS_PER_LINK_MATH = 5      
    
    # --- MÁRGENES Y AJUSTES HÁPTICOS ---
    MARGIN_EXT   = 0.08   
    AVISO_EXT    = 0.13   

    MARGIN_SELF  = 0.12   
    AVISO_SELF   = 0.17   
    
    HAPTIC_MODE  = "WALL" 
    K_REPULSION  = 500.0          
    MAX_STEP     = 0.12           

    # Obstáculos corregidos a altura de mesa (Z = 0.82)
    OBSTACLES = [
        [0.50, -0.15, 0.82],
        [0.58,  0.18, 0.85]
    ]

    SIM_FPS = 60             

# ============================================================================
#  2. INICIALIZACIÓN DE RUTAS Y SIMULADOR
# ============================================================================
sys.path.append(CONFIG.ADAMSIM_DIR)
sys.path.append(CONFIG.SDF_DIR)

from scripts.adam import ADAM
from robot_sdf.robot_sdf import RobotSDF

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

adam = ADAM(CONFIG.ROBOT_URDF, useRealTimeSimulation=False, used_fixed_base=True, use_ros=False)
ADAM_BODY_ID = adam.robot_id

num_joints = p.getNumJoints(ADAM_BODY_ID)
pb_joint_names = []
q_all_static = {}
link_idx_side = {}

for i in range(num_joints):
    info = p.getJointInfo(ADAM_BODY_ID, i)
    jname = info[1].decode()
    lname = info[12].decode()
    pb_joint_names.append(jname)
    q_all_static[jname] = p.getJointState(ADAM_BODY_ID, i)[0]
    
    if "left_arm" in lname or lname.startswith("L_"):
        link_idx_side[lname] = "left"
    elif "right_arm" in lname or lname.startswith("R_"):
        link_idx_side[lname] = "right"

print("Cargando modelo SDF...")
robot_sdf = RobotSDF(CONFIG.SDF_URDF, mesh_type="visual", n_func=4, n_seg=3)
robot_sdf.load(CONFIG.SDF_MODEL)

pk_joint_names = robot_sdf.chain.get_joint_parameter_names()

left_link_names = [name for name, side in link_idx_side.items() if side == "left" and name in robot_sdf.model]
right_link_names = [name for name, side in link_idx_side.items() if side == "right" and name in robot_sdf.model]

GELLO_KEYWORDS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
joint_name_to_idx = {name: idx for idx, name in enumerate(pb_joint_names)}

def _arm_joints(side):
    return [next((v for k, v in joint_name_to_idx.items() if side in k and kw in k), None) for kw in GELLO_KEYWORDS]

LEFT_ARM_JOINTS = _arm_joints("left_arm")
RIGHT_ARM_JOINTS = _arm_joints("right_arm")

def get_ordered_links(joint_indices):
    ordered = []
    for idx in joint_indices:
        if idx is not None:
            info = p.getJointInfo(ADAM_BODY_ID, idx)
            lname = info[12].decode()
            if lname in robot_sdf.model:
                ordered.append(lname)
    return ordered

left_ordered_links = get_ordered_links(LEFT_ARM_JOINTS)
right_ordered_links = get_ordered_links(RIGHT_ARM_JOINTS)

left_links_self = [n for n in left_ordered_links if any(x in n.lower() for x in ["elbow", "wrist"])]
right_links_self = [n for n in right_ordered_links if any(x in n.lower() for x in ["elbow", "wrist"])]

# ============================================================================
#  3. PRE-MUESTREO OPTIMIZADO
# ============================================================================
print("Pre-muestreando superficies locales de la malla 3D...")
PRE_SAMPLED_MATH = {}

links_to_sample = list(set(left_links_self + right_links_self))
for name in links_to_sample:
    if name in robot_sdf.link_meshes:
        mesh = robot_sdf.link_meshes[name]
        pts_math, _ = trimesh.sample.sample_surface(mesh, CONFIG.NUM_PTS_PER_LINK_MATH)
        pts_math_hom = np.hstack((pts_math, np.ones((len(pts_math), 1))))
        PRE_SAMPLED_MATH[name] = torch.tensor(pts_math_hom, dtype=torch.float32, device=device).T

CACHED_LOCAL_TRANSFORMS = {}
for link_name, (parent_name, local_np) in robot_sdf.joint_tree.items():
    CACHED_LOCAL_TRANSFORMS[link_name] = torch.tensor(local_np, dtype=torch.float32, device=device)

# ============================================================================
#  4. ENTORNO VISUAL Y TENSORES EXTERNOS
# ============================================================================
def add_sphere(center, radius, color):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=center)

for obs in CONFIG.OBSTACLES:
    add_sphere(obs, CONFIG.OBSTACLE_RADIUS, [0.2, 0.5, 0.9, 1.0])

obst_tensor = torch.tensor(CONFIG.OBSTACLES, dtype=torch.float32, device=device)

def get_theta_tensor(q_left, q_right):
    q_dict = q_all_static.copy()
    for i, idx in enumerate(LEFT_ARM_JOINTS):
        if idx is not None: q_dict[pb_joint_names[idx]] = q_left[i]
    for i, idx in enumerate(RIGHT_ARM_JOINTS):
        if idx is not None: q_dict[pb_joint_names[idx]] = q_right[i]
    return torch.tensor([[q_dict.get(name, 0.0) for name in pk_joint_names]], dtype=torch.float32, device=device)

# ============================================================================
#  5. FUNCIONES GLOBALES DE LOS GELLO Y HARDWARE
# ============================================================================
def ticks_a_rad(ticks): return ((ticks % CONFIG.TICKS_REV) / CONFIG.TICKS_REV) * 2 * math.pi

def aplicar_transformacion(t_list, signs, offsets): 
    return [signs[i] * ticks_a_rad(t) + offsets[i] for i, t in enumerate(t_list)]

def rad_to_tick_delta(delta_q, signs):
    delta_ticks = []
    for i, dq in enumerate(delta_q):
        d_rad = dq / signs[i]
        dt = (d_rad / (2 * math.pi)) * CONFIG.TICKS_REV
        delta_ticks.append(dt)
    return np.array(delta_ticks)

q_brazo_r, q_brazo_l = [0.0] * 6, [0.0] * 6
ticks_actuales_r, ticks_actuales_l = [0] * 6, [0] * 6
lock_qr, lock_ql = threading.Lock(), threading.Lock()
running = True

def init_puerto(port):
    ph = PortHandler(port); pk = PacketHandler(CONFIG.PROTOCOL)
    if not ph.openPort() or not ph.setBaudRate(CONFIG.BAUDRATE): raise IOError(f"Error en {port}")
    return ph, pk

# Dejamos las variables preparadas para que se asignen dentro del proceso aislado
ph_r, pk_r, ph_l, pk_l = None, None, None, None
sync_read_r, sync_read_l = None, None
lock_port_r, lock_port_l = threading.Lock(), threading.Lock()

def _port_lock(port): return lock_port_r if port == ph_r else lock_port_l

def hilo_lectura(m_ids, signs, offsets, sync_read, lock_p, lock_q, q_ref, t_ref):
    while running:
        try:
            with lock_p: res = sync_read.txRxPacket()
            if res == 0:
                t_list = []
                for mid in m_ids:
                    if sync_read.isAvailable(mid, CONFIG.ADDR_PRESENT_POS, 4):
                        pos = sync_read.getData(mid, CONFIG.ADDR_PRESENT_POS, 4)
                        if pos > 0x7FFFFFFF: pos -= 0x100000000
                        t_list.append(pos)
                if len(t_list) == 6:
                    q_new = aplicar_transformacion(t_list, signs, offsets)
                    with lock_q: q_ref[:], t_ref[:] = q_new, t_list
        except Exception: pass
        time.sleep(0.002)

def set_modo5(port, pk, motor_ids):
    for mid in motor_ids:
        pk.write1ByteTxRx(port, mid, CONFIG.ADDR_TORQUE, 0); time.sleep(0.02)
        pk.write1ByteTxRx(port, mid, CONFIG.ADDR_MODE, CONFIG.MODE_POS); time.sleep(0.03)
        pk.write2ByteTxRx(port, mid, CONFIG.ADDR_GOAL_CURR, 0)

def engage_hold(port, pk, motor_ids, activo):
    for i, mid in enumerate(motor_ids):
        if not activo[i]: continue
        with _port_lock(port):
            pk.write2ByteTxRx(port, mid, CONFIG.ADDR_GOAL_CURR, CONFIG.CURRENT_HOLD[i])
            pk.write1ByteTxRx(port, mid, CONFIG.ADDR_TORQUE, 1)

def update_lock(port, pk, motor_ids, activo, goal_ticks):
    sync = GroupSyncWrite(port, pk, CONFIG.ADDR_GOAL_POS, 4)
    for i, mid in enumerate(motor_ids):
        if not activo[i]: continue
        v = int(goal_ticks[i]) & 0xFFFFFFFF
        sync.addParam(mid, [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
    with _port_lock(port): sync.txPacket()
    sync.clearParam()

def release_hold(port, pk, motor_ids):
    with _port_lock(port):
        for mid in motor_ids: pk.write1ByteTxRx(port, mid, CONFIG.ADDR_TORQUE, 0)

# ============================================================================
#  6. EVALUACIÓN DIFERENCIABLE Y LÓGICA DE MODOS HÁPTICOS (WALL vs REPULSION)
# ============================================================================
def get_exact_surface_points(theta_tensor, link_names, cache_dict):
    transforms = robot_sdf._get_link_transforms(theta_tensor, link_names)
    
    def get_transform(link_name):
        if link_name in transforms:
            return transforms[link_name][0]
        if link_name not in robot_sdf.joint_tree:
            fallback = list(transforms.keys())[0] if transforms else 'rb1_base_footprint'
            return transforms.get(fallback, torch.eye(4, device=device, dtype=torch.float32))[0]
        
        parent_name, _ = robot_sdf.joint_tree[link_name]
        parent_T = get_transform(parent_name)
        local_T = CACHED_LOCAL_TRANSFORMS[link_name]
        return torch.matmul(parent_T, local_T)

    world_points = []
    for name in link_names:
        if name in cache_dict:
            T_matrix = get_transform(name)                
            pts_local_hom = cache_dict[name]      
            pts_world_hom = torch.matmul(T_matrix, pts_local_hom)
            world_points.append(pts_world_hom[:3, :].T)
            
    if world_points:
        return torch.cat(world_points, dim=0)
    return torch.empty((0, 3), device=device)

def evaluate_distances(theta_tensor):
    ext_l, ext_r = torch.tensor(9.9, device=device), torch.tensor(9.9, device=device)
    self_l, self_r = torch.tensor(9.9, device=device), torch.tensor(9.9, device=device)
    
    if left_link_names:
        sdf_ext_l, _ = robot_sdf.query(obst_tensor, theta_tensor, return_grad=False, used_links=left_link_names)
        ext_l = sdf_ext_l.min() - CONFIG.OBSTACLE_RADIUS
        
    if right_link_names:
        sdf_ext_r, _ = robot_sdf.query(obst_tensor, theta_tensor, return_grad=False, used_links=right_link_names)
        ext_r = sdf_ext_r.min() - CONFIG.OBSTACLE_RADIUS

    left_pts_mesh = get_exact_surface_points(theta_tensor, left_links_self, PRE_SAMPLED_MATH)
    right_pts_mesh = get_exact_surface_points(theta_tensor, right_links_self, PRE_SAMPLED_MATH)
    
    if len(left_pts_mesh) > 0 and right_link_names:
        sdf_rarm, _ = robot_sdf.query(left_pts_mesh, theta_tensor, return_grad=False, used_links=right_link_names)
        self_l = sdf_rarm.min() - CONFIG.SELF_COLLISION_RADIUS
            
    if len(right_pts_mesh) > 0 and left_link_names:
        sdf_larm, _ = robot_sdf.query(right_pts_mesh, theta_tensor, return_grad=False, used_links=left_link_names)
        self_r = sdf_larm.min() - CONFIG.SELF_COLLISION_RADIUS
        
    return ext_l, ext_r, self_l, self_r

last_safe_q_l = np.zeros(6, dtype=np.float32)
last_safe_q_r = np.zeros(6, dtype=np.float32)
last_safe_init = False

def calculate_push_out_poses(q_l, q_r):
    global last_safe_q_l, last_safe_q_r, last_safe_init

    if not last_safe_init:
        last_safe_q_l[:] = q_l
        last_safe_q_r[:] = q_r
        last_safe_init = True

    theta_tensor = get_theta_tensor(q_l, q_r)
    
    with torch.inference_mode():
        ext_l, ext_r, self_l, self_r = evaluate_distances(theta_tensor)
        
    pen_ext_l = CONFIG.MARGIN_EXT - ext_l
    pen_self_l = CONFIG.MARGIN_SELF - self_l
    pen_l = torch.max(pen_ext_l, pen_self_l)

    pen_ext_r = CONFIG.MARGIN_EXT - ext_r
    pen_self_r = CONFIG.MARGIN_SELF - self_r
    pen_r = torch.max(pen_ext_r, pen_self_r)

    q_safe_l = np.array(q_l, dtype=np.float32)
    q_safe_r = np.array(q_r, dtype=np.float32)
    
    if pen_l <= 0:
        last_safe_q_l[:] = q_l
    if pen_r <= 0:
        last_safe_q_r[:] = q_r

    if pen_l > 0 or pen_r > 0:
        if CONFIG.HAPTIC_MODE == "WALL":
            if pen_l > 0: q_safe_l[:] = last_safe_q_l
            if pen_r > 0: q_safe_r[:] = last_safe_q_r

        elif CONFIG.HAPTIC_MODE == "REPULSION":
            with torch.enable_grad():
                theta_grad = get_theta_tensor(q_l, q_r)
                theta_grad.requires_grad_(True)
                
                ext_l_g, ext_r_g, self_l_g, self_r_g = evaluate_distances(theta_grad)
                
                pl = torch.clamp(CONFIG.MARGIN_EXT - ext_l_g, min=0.0)**2 + torch.clamp(CONFIG.MARGIN_SELF - self_l_g, min=0.0)**2
                pr = torch.clamp(CONFIG.MARGIN_EXT - ext_r_g, min=0.0)**2 + torch.clamp(CONFIG.MARGIN_SELF - self_r_g, min=0.0)**2
                
                loss = (pl + pr) * CONFIG.K_REPULSION
                loss.backward() 
                
                grad_theta = theta_grad.grad[0].cpu().numpy()
                
                idx_l = [pk_joint_names.index(pb_joint_names[i]) for i in LEFT_ARM_JOINTS]
                idx_r = [pk_joint_names.index(pb_joint_names[i]) for i in RIGHT_ARM_JOINTS]
                
                step_l = np.clip(grad_theta[idx_l], -CONFIG.MAX_STEP, CONFIG.MAX_STEP)
                step_r = np.clip(grad_theta[idx_r], -CONFIG.MAX_STEP, CONFIG.MAX_STEP)
                
                q_safe_l -= step_l
                q_safe_r -= step_r

    return q_safe_l, q_safe_r, ext_l.item(), ext_r.item(), self_l.item(), self_r.item(), pen_l.item(), pen_r.item()

# ============================================================================
#  7. PROCESO DE HARDWARE MULTIPROCESSING AISLADO
# ============================================================================
def proceso_haptico_sdf(shared_q_l, shared_q_r, shared_metrics, run_flag):
    """
    Se ejecuta en su propio núcleo aislando la carga de PyTorch y lectura USB.
    """
    global ph_r, pk_r, ph_l, pk_l, sync_read_r, sync_read_l, running
    
    running = True
    torch.set_num_threads(1)
    
    # 1. Iniciamos puertos desde dentro del proceso
    try:
        ph_r, pk_r = init_puerto(CONFIG.PORT_R)
        ph_l, pk_l = init_puerto(CONFIG.PORT_L)
        
        sync_read_r = GroupSyncRead(ph_r, pk_r, CONFIG.ADDR_PRESENT_POS, 4)
        sync_read_l = GroupSyncRead(ph_l, pk_l, CONFIG.ADDR_PRESENT_POS, 4)
        for mid in CONFIG.MOTORS_R: sync_read_r.addParam(mid)
        for mid in CONFIG.MOTORS_L: sync_read_l.addParam(mid)

        set_modo5(ph_r, pk_r, CONFIG.MOTORS_R)
        set_modo5(ph_l, pk_l, CONFIG.MOTORS_L)

        threading.Thread(target=hilo_lectura, args=(CONFIG.MOTORS_R, CONFIG.SIGNS_R, CONFIG.OFFSETS_R, sync_read_r, lock_port_r, lock_qr, q_brazo_r, ticks_actuales_r), daemon=True).start()
        threading.Thread(target=hilo_lectura, args=(CONFIG.MOTORS_L, CONFIG.SIGNS_L, CONFIG.OFFSETS_L, sync_read_l, lock_port_l, lock_ql, q_brazo_l, ticks_actuales_l), daemon=True).start()
    except Exception as e:
        print(f"Error de Hardware: {e}")
        return

    state_r, state_l = 0, 0   

    # 2. Bucle pesado (>500Hz) de SDF
    while run_flag.value:
        with lock_ql: q_l, t_l = list(q_brazo_l), list(ticks_actuales_l)
        with lock_qr: q_r, t_r = list(q_brazo_r), list(ticks_actuales_r)
        
        q_safe_l, q_safe_r, ext_l, ext_r, self_l, self_r, pen_l, pen_r = calculate_push_out_poses(q_l, q_r)
                
        # Empujar variables a la memoria compartida para PyBullet
        shared_q_l[:] = q_safe_l.tolist()
        shared_q_r[:] = q_safe_r.tolist()
        shared_metrics[0] = min(ext_l, ext_r)
        shared_metrics[1] = min(self_l, self_r)
        shared_metrics[2] = min(self_l, self_r)

        # --- FORCE FEEDBACK IZQUIERDO REAL ---
        if pen_l > 0:
            if state_l == 0:
                state_l = 1
                engage_hold(ph_l, pk_l, CONFIG.MOTORS_L, CONFIG.ACTIVO_L)
            
            delta_ticks_l = rad_to_tick_delta(q_safe_l - q_l, CONFIG.SIGNS_L)
            ticks_target_l = np.array(t_l) + delta_ticks_l
            update_lock(ph_l, pk_l, CONFIG.MOTORS_L, CONFIG.ACTIVO_L, ticks_target_l)
        else:
            if state_l == 1:
                state_l = 0
                release_hold(ph_l, pk_l, CONFIG.MOTORS_L)

        # --- FORCE FEEDBACK DERECHO REAL ---
        if pen_r > 0:
            if state_r == 0:
                state_r = 1
                engage_hold(ph_r, pk_r, CONFIG.MOTORS_R, CONFIG.ACTIVO_R)
                
            delta_ticks_r = rad_to_tick_delta(q_safe_r - q_r, CONFIG.SIGNS_R)
            ticks_target_r = np.array(t_r) + delta_ticks_r
            update_lock(ph_r, pk_r, CONFIG.MOTORS_R, CONFIG.ACTIVO_R, ticks_target_r)
        else:
            if state_r == 1:
                state_r = 0
                release_hold(ph_r, pk_r, CONFIG.MOTORS_R)

        time.sleep(0.001)

    # 3. Cierre Seguro
    running = False
    time.sleep(0.1)
    release_hold(ph_r, pk_r, CONFIG.MOTORS_R)
    release_hold(ph_l, pk_l, CONFIG.MOTORS_L)
    ph_r.closePort()
    ph_l.closePort()


# ============================================================================
#  8. MAIN LOOP PYBULLET 
# ============================================================================
if __name__ == '__main__':
    # Preparación de Arrays para Memoria Compartida
    shared_q_l = mp.Array('f', [0.0] * 6)
    shared_q_r = mp.Array('f', [0.0] * 6)
    shared_metrics = mp.Array('f', [9.9, 9.9, 9.9]) # d_ext, d_self, dist_brazos
    run_flag = mp.Value('b', True)

    # Iniciar núcleo de colisiones
    p_worker = mp.Process(target=proceso_haptico_sdf, args=(shared_q_l, shared_q_r, shared_metrics, run_flag))
    p_worker.start()

    banner_id = None
    BANNER_POS = [0.5, 0.0, 1.5]

    def update_banner(min_ext, min_self):
        global banner_id
        if min_ext < CONFIG.MARGIN_EXT or min_self < CONFIG.MARGIN_SELF: 
            txt, col = f"SDF BLOCKED [{CONFIG.HAPTIC_MODE}]", [1.0, 0.0, 0.0]
        elif min_ext < (CONFIG.MARGIN_EXT + CONFIG.AVISO_EXT) or min_self < (CONFIG.MARGIN_SELF + CONFIG.AVISO_SELF): 
            txt, col = "SDF APPROACHING", [1.0, 0.85, 0.0]
        else: 
            txt, col = "SDF FREE TO MOVE", [0.0, 0.8, 0.0]
        
        if banner_id is None: banner_id = p.addUserDebugText(txt, BANNER_POS, textColorRGB=col, textSize=2.2)
        else: banner_id = p.addUserDebugText(txt, BANNER_POS, textColorRGB=col, textSize=2.2, replaceItemUniqueId=banner_id)

    print(f"\n[+] Teleop + Haptics ACTIVOS + MULTIPROCESSING. Ctrl+C para salir.\n")

    print_counter = 0

    try:
        while p.isConnected():
            # Extraer variables procesadas instantáneamente
            qsl = list(shared_q_l)
            qsr = list(shared_q_r)
            d_ext = shared_metrics[0]
            d_self = shared_metrics[1]
            dist_brazos = shared_metrics[2]
            
            # Sincronizamos simulador visual
            adam.arm_kinematics.set_arm_pose('left',  'joint', qsl)
            adam.arm_kinematics.set_arm_pose('right', 'joint', qsr)
            
            p.stepSimulation()
            
            print_counter += 1
            
            if print_counter % (CONFIG.SIM_FPS // 5) == 0: 
                update_banner(d_ext, d_self)
                print(f"Distancia entre pieles de los brazos: {dist_brazos * 100:.2f} cm | Entorno: {d_ext*100:.2f} cm      ", end="\r")
                
            time.sleep(1.0 / CONFIG.SIM_FPS)

    except KeyboardInterrupt:
        print("\nDeteniendo simulación...")
    except Exception as e:
        print(f"\nError de Ejecución: {e}")
    finally:
        # Cierre ordenado
        run_flag.value = False
        p_worker.join()
        print("Procesos y Motores cerrados de forma segura.")