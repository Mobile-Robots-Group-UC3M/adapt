#!/usr/bin/env python3
# ============================================================================
#  gello_ADAM_collision_sdf.py (v6.0 - ARQUITECTURA ASÍNCRONA REAL-TIME)
# ----------------------------------------------------------------------------
#  Teleoperacion GELLO -> ADAMSim con BLOQUEO DIRECCIONAL usando RobotSDF.
#  Separación total de hilos para garantizar latencia cero en PyBullet.
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

# Configurar rutas del proyecto
ADAMSIM_DIR = "/home/alumnos/tfg-laura/AdamSim"
SDF_DIR = "/home/alumnos/tfg-laura/SignedDistanceFields"
sys.path.append(ADAMSIM_DIR)
sys.path.append(SDF_DIR)

from scripts.adam import ADAM
from robot_sdf.robot_sdf import RobotSDF
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite

ROBOT_URDF = "/home/alumnos/tfg-laura/AdamSim/models/robot/rb1_base_description/robots/robotDummy.urdf"

# Fuerza el uso de CPU optimizado para evitar latencias de memoria
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# ════════════════════════════════════════════════════════════════════════════
#  1. ARRANCAR ADAMSIM Y CARGAR MODELO SDF
# ════════════════════════════════════════════════════════════════════════════
adam = ADAM(ROBOT_URDF, useRealTimeSimulation=True, used_fixed_base=True, use_ros=False)
ADAM_BODY_ID = adam.robot_id

# Obtener articulaciones revolutas para el SDF
num_joints = p.getNumJoints(ADAM_BODY_ID)
revolute_joint_indices = []
for i in range(num_joints):
    info = p.getJointInfo(ADAM_BODY_ID, i)
    if info[2] == p.JOINT_REVOLUTE:
        revolute_joint_indices.append(i)

# Cachear la configuración estática de articulaciones en memoria local
q_all_static = [p.getJointState(ADAM_BODY_ID, i)[0] for i in range(num_joints)]

# Cargar modelo neuronal (.pt)
urdf_sdf_path = f"{SDF_DIR}/urdf/ADAM/models/robot/rb1_base_description/robots/robotDummyOriginal.urdf"
model_path = f"{SDF_DIR}/models/ADAM/robotDummyOriginal_bf_4_seg3_visual.pt"

robot_sdf = RobotSDF(urdf_sdf_path, mesh_type="visual", n_func=4, n_seg=3)
robot_sdf.load(model_path)

# ════════════════════════════════════════════════════════════════════════════
#  2. CONFIGURACION DE LOS DOS BRAZOS (GELLO MAPPING ORIGINAL)
# ════════════════════════════════════════════════════════════════════════════
# Brazo 2 (DERECHA en hardware -> Mapea a 'left' en ADAMSim)
PORT_2      = "/dev/ttyUS1"
MOTOR_IDS_2 = [1, 2, 3, 4, 5, 6]
SIGNS_2     = [1, 1, -1, 1, 1, 1]
OFFSETS_2   = [-4.4477, -4.4879, 3.0840, 3.2282, -3.0217, 1.5567]

# Brazo 1 (IZQUIERDA en hardware -> Mapea a 'right' en ADAMSim)
PORT_1      = "/dev/ttyUSB0"
MOTOR_IDS_1 = [8, 9, 10, 11, 12, 13]
SIGNS_1     = [1, 1, -1, 1, 1, 1]
OFFSETS_1   = [-6.2262, -3.3671, 3.1002, -1.6222, -3.2817, -4.6493]

BAUDRATE  = 57600
PROTOCOL  = 2.0

ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_CURRENT     = 102
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132

TICKS_POR_VUELTA            = 4096
MODE_CURRENT_BASED_POSITION = 5

D_BLOCK   = 0.02    # Distancia de bloqueo (2 cm)
D_RELEASE = 0.04    # Distancia de liberación (4 cm)
CURRENT_HOLD  = [350, 500, 350, 220, 160, 160]
ACTIVO_1 = [1, 1, 1, 1, 1, 1]
ACTIVO_2 = [1, 1, 1, 1, 1, 1]

GELLO_KEYWORDS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

joint_name_to_idx = {}
for j in range(p.getNumJoints(ADAM_BODY_ID)):
    info  = p.getJointInfo(ADAM_BODY_ID, j)
    joint_name_to_idx[info[1].decode()] = j

def _arm_joints(side):
    js = []
    for kw in GELLO_KEYWORDS:
        idx = next((v for k, v in joint_name_to_idx.items() if (side in k) and (kw in k)), None)
        js.append(idx)
    return js

LEFT_ARM_JOINTS  = _arm_joints("left_arm")
RIGHT_ARM_JOINTS = _arm_joints("right_arm")

# ════════════════════════════════════════════════════════════════════════════
#  3. ENTORNO EN PYBULLET Y GENERACIÓN DE OBSTÁCULOS OPTIMIZADOS
# ════════════════════════════════════════════════════════════════════════════
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

def add_box(center, half, color):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=center)

add_box([0.50, -0.15, 0.82], [0.04, 0.04, 0.05], [0.2, 0.5, 0.9, 1.0])
add_box([0.58,  0.18, 0.85], [0.05, 0.05, 0.08], [0.9, 0.7, 0.1, 1.0])

def generate_box_points(center, half, num_per_dim=3):
    x = np.linspace(center[0] - half[0], center[0] + half[0], num_per_dim)
    y = np.linspace(center[1] - half[1], center[1] + half[1], num_per_dim)
    z = np.linspace(center[2] - half[2], center[2] + half[2], num_per_dim)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

pts1 = generate_box_points([0.50, -0.15, 0.82], [0.04, 0.04, 0.05], num_per_dim=3)
pts2 = generate_box_points([0.58,  0.18, 0.85], [0.05, 0.05, 0.08], num_per_dim=3)
all_obst_points = np.vstack([pts1, pts2])
obst_points_tensor = torch.tensor(all_obst_points, dtype=torch.float32, device=device)

# ════════════════════════════════════════════════════════════════════════════
#  4. RECONSTRUCCIÓN EXCLUSIVA EN MEMORIA (INFERENCE_MODE)
# ════════════════════════════════════════════════════════════════════════════
def build_q_revolute(q_left, q_right):
    """Combina cinemática en memoria local de Python a velocidad extrema"""
    q_all = q_all_static[:]
    for i, idx in enumerate(LEFT_ARM_JOINTS):
        if idx is not None: q_all[idx] = q_left[i]
    for i, idx in enumerate(RIGHT_ARM_JOINTS):
        if idx is not None: q_all[idx] = q_right[i]
    return [q_all[i] for i in revolute_joint_indices]

def evaluate_sdf_distance(q_revolute_list):
    """Inferencia asíncrona de la red neuronal con PyTorch optimizado"""
    theta_tensor = torch.as_tensor(q_revolute_list, dtype=torch.float32, device=device)
    with torch.inference_mode():
        sdf_values, _ = robot_sdf.query(obst_points_tensor, theta_tensor)
        min_dist = torch.min(sdf_values).item()
    return min_dist

# ════════════════════════════════════════════════════════════════════════════
#  5. HILOS DE LECTURA NATIVOS (IDÉNTICOS A TU SCRIPT ORIGINAL)
# ════════════════════════════════════════════════════════════════════════════
def ticks_a_rad(ticks):
    return ((ticks % TICKS_POR_VUELTA) / TICKS_POR_VUELTA) * 2 * math.pi

def aplicar_transformacion(ticks_list, signs, offsets):
    return [signs[i] * ticks_a_rad(t) + offsets[i] for i, t in enumerate(ticks_list)]

q_brazo1 = [0.0] * 6; q_brazo2 = [0.0] * 6
ticks_actuales_1 = [0] * 6; ticks_actuales_2 = [0] * 6
lock_q1 = threading.Lock(); lock_q2 = threading.Lock()
running = True

def init_puerto(port):
    ph = PortHandler(port); pk = PacketHandler(PROTOCOL)
    if not ph.openPort():            raise IOError(f"No se pudo abrir {port}")
    if not ph.setBaudRate(BAUDRATE): raise IOError(f"Baudrate falló en {port}")
    print(f"✓ Conectado a {port}")
    return ph, pk

ph1, pk1 = init_puerto(PORT_1)
ph2, pk2 = init_puerto(PORT_2)

lock_port1 = threading.Lock(); lock_port2 = threading.Lock()
_PORT_LOCKS = {ph1: lock_port1, ph2: lock_port2}
def _port_lock(port): return _PORT_LOCKS[port]

def leer_posicion(motor_id, port_handler, packet_handler, lock_port):
    try:
        with lock_port:
            pos, result, error = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result != 0 or error != 0: return None
        return pos
    except Exception:
        return None

def hilo_lectura(motor_ids, signs, offsets, port_handler, packet_handler, lock_port, lock_q, q_ref, ticks_ref):
    """Hilo continuo sin sleep artificial para latencia de hardware mínima"""
    while running:
        try:
            ticks_list = []
            hay_error  = False
            for motor_id in motor_ids:
                pos = leer_posicion(motor_id, port_handler, packet_handler, lock_port)
                if pos is None:
                    hay_error = True
                    break
                ticks_list.append(pos)
            if not hay_error:
                q_nuevo = aplicar_transformacion(ticks_list, signs, offsets)
                with lock_q:
                    q_ref[:] = q_nuevo
                    ticks_ref[:] = ticks_list
        except Exception:
            pass

# ════════════════════════════════════════════════════════════════════════════
#  6. HARDWARE CONTROL (EMBRAGUE MODO 5)
# ════════════════════════════════════════════════════════════════════════════
def set_modo5(port, pk, motor_ids):
    for mid in motor_ids:
        pk.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0); time.sleep(0.02)
        pk.write1ByteTxRx(port, mid, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION); time.sleep(0.03)
        pk.write2ByteTxRx(port, mid, ADDR_GOAL_CURRENT, 0)

def engage_hold(port, pk, motor_ids, activo, locked_ticks):
    for i, mid in enumerate(motor_ids):
        if not activo[i]: continue
        pos_int = int(locked_ticks[i]) & 0xFFFFFFFF
        with _port_lock(port):
            pk.write4ByteTxRx(port, mid, ADDR_GOAL_POSITION, pos_int)
            pk.write2ByteTxRx(port, mid, ADDR_GOAL_CURRENT, CURRENT_HOLD[i])
            pk.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 1)

def update_lock(port, pk, motor_ids, activo, goal_ticks):
    sync = GroupSyncWrite(port, pk, ADDR_GOAL_POSITION, 4)
    for i, mid in enumerate(motor_ids):
        if not activo[i]: continue
        v = int(goal_ticks[i]) & 0xFFFFFFFF
        sync.addParam(mid, [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
    with _port_lock(port):
        sync.txPacket()
    sync.clearParam()

def release_hold(port, pk, motor_ids):
    with _port_lock(port):
        for mid in motor_ids:
            pk.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0)

set_modo5(ph1, pk1, MOTOR_IDS_1)
set_modo5(ph2, pk2, MOTOR_IDS_2)

# Arrancar hilos de hardware nativos
threading.Thread(target=hilo_lectura, args=(MOTOR_IDS_1, SIGNS_1, OFFSETS_1, ph1, pk1, lock_port1, lock_q1, q_brazo1, ticks_actuales_1), daemon=True).start()
threading.Thread(target=hilo_lectura, args=(MOTOR_IDS_2, SIGNS_2, OFFSETS_2, ph2, pk2, lock_port2, lock_q2, q_brazo2, ticks_actuales_2), daemon=True).start()

# ════════════════════════════════════════════════════════════════════════════
#  7. ESTADOS SEGUROS Y HILO ASÍNCRONO DE COLISIONES (SDF)
# ════════════════════════════════════════════════════════════════════════════
state1 = 0; state2 = 0  # 0 = libre, 1 = bloqueado
q_safe1 = [0.0] * 6; q_safe2 = [0.0] * 6
safe_ticks1 = [0] * 6; safe_ticks2 = [0] * 6
lbl1 = lbl2 = "LIBRE"

lock_safety = threading.Lock()
banner_id = None
BANNER_POS = [0.5, 0.0, 1.5]

def update_banner(dmin):
    global banner_id
    if dmin < D_BLOCK:   txt, col = "SDF: COLLISION/BLOCK", [1.0, 0.0, 0.0]
    elif dmin < D_RELEASE: txt, col = "SDF: WARNING NEAR OBJECT", [1.0, 0.85, 0.0]
    else:                  txt, col = "SDF: FREE TO MOVE", [0.0, 0.8, 0.0]
    if banner_id is None:  banner_id = p.addUserDebugText(txt, BANNER_POS, textColorRGB=col, textSize=2.2)
    else:                  banner_id = p.addUserDebugText(txt, BANNER_POS, textColorRGB=col, textSize=2.2, replaceItemUniqueId=banner_id)

def hilo_colisiones():
    """Hilo dedicado exclusivamente a evaluar el SDF e inyectar fuerzas en Dynamixel"""
    global state1, state2, q_safe1, q_safe2, safe_ticks1, safe_ticks2, lbl1, lbl2
    
    time.sleep(0.5)
    with lock_q1: q_safe1[:] = list(q_brazo1); safe_ticks1[:] = list(ticks_actuales_1)
    with lock_q2: q_safe2[:] = list(q_brazo2); safe_ticks2[:] = list(ticks_actuales_2)
    
    last_col = 0.0
    COLLISION_PERIOD = 0.02 # Frecuencia fija para el cómputo del lazo de seguridad (50 Hz)
    print_counter = 0
    
    while running:
        if time.time() - last_col >= COLLISION_PERIOD:
            last_col = time.time()
            
            with lock_q1: t1_local = list(ticks_actuales_1); q1_local = list(q_brazo1)
            with lock_q2: t2_local = list(ticks_actuales_2); q2_local = list(q_brazo2)
            if len(t1_local) < 6 or len(t2_local) < 6: continue
            
            # Evaluar distancias candidatas con SDF asíncronas
            q_rev_L = build_q_revolute(q2_local, q_safe1) # Izq (B2) candidata vs Der segura
            dL_cand = evaluate_sdf_distance(q_rev_L)

            q_rev_R = build_q_revolute(q_safe2, q1_local) # Der (B1) candidata vs Izq segura
            dR_cand = evaluate_sdf_distance(q_rev_R)

            d_frozen = 0.0
            if state2 == 1 or state1 == 1:
                q_rev_frozen = build_q_revolute(q_safe2, q_safe1)
                d_frozen = evaluate_sdf_distance(q_rev_frozen)

            with lock_safety:
                # --- CONTROL ASÍNCRONO: BRAZO IZQUIERDO (B2) ---
                if state2 == 0:
                    if dL_cand < D_BLOCK:
                        state2 = 1; engage_hold(ph2, pk2, MOTOR_IDS_2, ACTIVO_2, safe_ticks2); lbl2 = "BLOQUEA"
                    else:
                        q_safe2[:] = q2_local; safe_ticks2[:] = t2_local; lbl2 = "LIBRE"
                else:
                    if dL_cand > D_RELEASE:
                        state2 = 0; release_hold(ph2, pk2, MOTOR_IDS_2); q_safe2[:] = q2_local; safe_ticks2[:] = t2_local; lbl2 = "LIBRE"
                    elif dL_cand > d_frozen + 1e-4:
                        q_safe2[:] = q2_local; safe_ticks2[:] = t2_local; update_lock(ph2, pk2, MOTOR_IDS_2, ACTIVO_2, t2_local); lbl2 = "RETIRA"
                    else:
                        update_lock(ph2, pk2, MOTOR_IDS_2, ACTIVO_2, safe_ticks2); lbl2 = "MURO"

                # --- CONTROL ASÍNCRONO: BRAZO DERECHO (B1) ---
                if state1 == 0:
                    if dR_cand < D_BLOCK:
                        state1 = 1; engage_hold(ph1, pk1, MOTOR_IDS_1, ACTIVO_1, safe_ticks1); lbl1 = "BLOQUEA"
                    else:
                        q_safe1[:] = q1_local; safe_ticks1[:] = t1_local; lbl1 = "LIBRE"
                else:
                    if dR_cand > D_RELEASE:
                        state1 = 0; release_hold(ph1, pk1, MOTOR_IDS_1); q_safe1[:] = q1_local; safe_ticks1[:] = t1_local; lbl1 = "LIBRE"
                    elif dR_cand > d_frozen + 1e-4:
                        q_safe1[:] = q1_local; safe_ticks1[:] = t1_local; update_lock(ph1, pk1, MOTOR_IDS_1, ACTIVO_1, t1_local); lbl1 = "RETIRA"
                    else:
                        update_lock(ph1, pk1, MOTOR_IDS_1, ACTIVO_1, safe_ticks1); lbl1 = "MURO"

            print_counter += 1
            if print_counter % 5 == 0: # Telemetría limpia en consola a 10Hz
                print(f"SDF -> Izq(B2): {dL_cand*100:5.1f}cm [{lbl2:7s}] | Der(B1): {dR_cand*100:5.1f}cm [{lbl1:7s}]")
                update_banner(min(dL_cand, dR_cand))
                
        time.sleep(0.002)

# Lanzar el hilo asíncrono de seguridad
threading.Thread(target=hilo_colisiones, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════════
#  8. BUCLE PRINCIPAL (STREAMING CINEMÁTICO INMEDIATO - LATENCIA CERO)
# ════════════════════════════════════════════════════════════════════════════
print("\n[+] Teleop + Bloqueo Direccional Asíncrono REAL-TIME Activo. Ctrl+C para salir.\n")
try:
    while p.isConnected():
        # Captura instantánea de estados cinemáticos desde memoria local
        with lock_q1: q1_now = list(q_brazo1)
        with lock_q2: q2_now = list(q_brazo2)
        
        with lock_safety:
            s1, s2 = state1, state2
            qs1, qs2 = list(q_safe1), list(q_safe2)

        # Inyección inmediata al simulador PyBullet sin esperas matematicas
        adam.arm_kinematics.set_arm_pose('left',  'joint', qs2 if s2 else q2_now)
        adam.arm_kinematics.set_arm_pose('right', 'joint', qs1 if s1 else q1_now)
        
        # Muestreo de refresco visual ultra-fluido (~200Hz)
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nDeteniendo...")
except Exception as e:
    print(f"\nError en ejecución: {e}")
finally:
    running = False
    time.sleep(0.2)
    release_hold(ph1, pk1, MOTOR_IDS_1)
    release_hold(ph2, pk2, MOTOR_IDS_2)
    ph1.closePort(); ph2.closePort()
    print("Puertos cerrados y motores liberados correctamente.")