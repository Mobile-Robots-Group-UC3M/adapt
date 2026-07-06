#!/usr/bin/env python3
"""
Home-in (visualización + gate visual) CON compensación de gravedad
+ control de manos por gatillo on/off (close_hand).

VERSIÓN SIN TELEOPERACIÓN: muestra brazos reales (sólidos), GELLO (fantasma),
letreros de alineación y compensación de gravedad, pero NO envía posiciones al
robot real (no hay cuenta atrás ni leader-follower).

Manos: el último motor de cada GELLO (gatillo) funciona como interruptor:
  - apretado  -> close_hand (agarra hasta contacto/tope).
  - suelto    -> abrir mano.
Se procesa por flanco con histéresis. OJO: close_hand es BLOQUEANTE (PyBullet no
es thread-safe, así que se llama desde el bucle principal; los brazos se congelan
mientras cierra).

Órdenes de junta:
  - Topic ROS publica:  [elbow, lift, pan, w1, w2, w3]  (orden robot real)
  - set_arm_pose / GELLO usan: [pan, lift, elbow, w1, w2, w3]  (SIM_ORDER)
Mapeo GELLO->ADAM:
  - brazo2 (PORT_2 /ttyUSB0, IDs 1-6)  -> lado 'left'   (gripper ID 7)
  - brazo1 (PORT_1 /ttyUSB1, IDs 8-13) -> lado 'right'  (gripper ID 14)
"""

import os
import sys
import math
import time
import threading

import numpy as np
import rospy
import pybullet as p
from sensor_msgs.msg import JointState
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite
import roboticstoolbox as rtb
from roboticstoolbox import RevoluteDH

sys.path.append(os.path.join(os.path.dirname(__file__), "AdamSim"))
from scripts.adam import ADAM

ROBOT_URDF_PATH = "/home/alumnos/tfg-laura/AdamSim/models/robot/rb1_base_description/robots/robotDummy.urdf"

# ════════════════════════════════════════════════════════════════════════════
#  SUBSISTEMA GELLO + COMPENSACIÓN DE GRAVEDAD  (verbatim de con_gravity)
# ════════════════════════════════════════════════════════════════════════════
PORT_2      = "/dev/ttyUSB0"
MOTOR_IDS_2 = [1, 2, 3, 4, 5, 6]
SIGNS_2     = [1, 1, -1, 1, 1, 1]
OFFSETS_2   = [-4.4477, -4.4879, 3.0840, 3.2282, -3.0217, 1.5567]

PORT_1      = "/dev/ttyUSB1"
MOTOR_IDS_1 = [8, 9, 10, 11, 12, 13]
SIGNS_1     = [1, 1, -1, 1, 1, 1]
OFFSETS_1   = [-6.2262, -2.8435, 3.1002, -1.6222, -3.2817, -4.6493]   # J2=-2.8435 (ver NOTA)

BAUDRATE  = 57600
PROTOCOL  = 2.0

ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_CURRENT     = 102
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132

TICKS_POR_VUELTA            = 4096
MODE_CURRENT_BASED_POSITION = 5

# ── Grippers: gatillo como ON/OFF de close_hand ───────────────────────────────
GRIPPER_ID_1 = 14          # bus PORT_1 -> mano 'right'
GRIPPER_ID_2 = 7           # bus PORT_2 -> mano 'left'

GRIP_CLOSED_DEG = 236.0    # gatillo apretado
GRIP_OPEN_DEG   = 289.0    # gatillo suelto
THUMB_ABD       = 0        # abducción fija del pulgar

UMBRAL_CERRAR = 0.35       # norm por debajo (apretando) -> close_hand
UMBRAL_ABRIR  = 0.65       # norm por encima (soltando)  -> abrir
CLOSE_SPEED     = 10
FORCE_THRESHOLD = 2
DOFS_ABIERTO = [1000, 1000, 1000, 1000, 1000, THUMB_ABD]   # dof 1000 = abierto
DOFS_CERRADO = [600, 600, 600, 600, 600, THUMB_ABD]

# ── Modelo físico y parámetros DH ────────────────────────────────────────────
M     = [0.041, 0.063, 0.065, 0.034, 0.030, 0.074]
A     = [0.0,   -0.135, -0.105, 0.0,    0.0,    0.0]
D     = [0.01,   0.0,    0.0,    0.04,   0.03,   0.02]
ALPHA = [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0]
R = [
    [0.0,  -0.010, 0.0],
    [0.072, 0.0,   0.064],
    [0.025, 0.0,   0.012],
    [0.0,   0.0,   0.005],
    [0.0,   0.0,   0.005],
    [0.0,   0.0,  -0.010],
]
_links = [RevoluteDH(a=A[i], d=D[i], alpha=ALPHA[i], m=M[i], r=R[i], I=[1e-5]*3+[0]*3) for i in range(6)]
robot = rtb.DHRobot(_links, name="gello_leader")

# ── Parámetros del embrague ──────────────────────────────────────────────────
GAIN        = 1.2
K_CURRENT   = 1500.0
MAX_CURRENT = 700
MAX_CURRENT_MOV  = 180
#                  J1   J2   J3   J4   J5   J6
MAX_CURRENT_HOLD = [350, 500, 350, 220, 160, 160]
MARGEN_AGARRE    = [120, 220, 120,  40,  20,  20]
UMBRAL_ENTRAR    = [ 35,  35,  35,  20,  12,  12]
UMBRAL_SALIR     = [ 12,  12,  12,   8,   5,   5]
FILTRO_RUIDO     = 0.9

ACTIVO_1 = [1, 1, 1, 1, 0, 0]
ACTIVO_2 = [1, 1, 1, 1, 0, 0]
CURRENT_SIGN_1 = [-1, -1, 1, 1, 1, 1]
CURRENT_SIGN_2 = [-1, -1, 1, 1, 1, 1]
GRAVEDAD_1 = [6.936, 0.0, 6.936]
GRAVEDAD_2 = [6.936, 0.0, 6.936]
ESCALA_1 = [4.5, 6.5, 0.0, 0.1, 0.1, 0.1]
ESCALA_2 = [4.5, 6.5, 0.0, 0.1, 0.1, 0.1]

GRAV_PERIODO = 0.02


# ── Conversión / dinámica ────────────────────────────────────────────────────
def ticks_a_rad(ticks):
    return ((ticks % TICKS_POR_VUELTA) / TICKS_POR_VUELTA) * 2 * math.pi

def aplicar_transformacion(ticks_list, signs, offsets):
    return [signs[i] * ticks_a_rad(t) + offsets[i] for i, t in enumerate(ticks_list)]

def derivar_y_filtrar(actual, previo, derivada_previa, dt, alpha=FILTRO_RUIDO):
    derivada_bruta = [(a - pr) / dt for a, pr in zip(actual, previo)]
    return [alpha * bruta + (1 - alpha) * prev for bruta, prev in zip(derivada_bruta, derivada_previa)]

def corriente_dinamica(q, qd, qdd, signs, current_sign, activo, gravedad, escala):
    tau = robot.rne(q, qd, qdd, gravity=gravedad)
    corrientes = []
    for i in range(6):
        c = activo[i] * escala[i] * current_sign[i] * signs[i] * GAIN * K_CURRENT * tau[i]
        corrientes.append(int(np.clip(c, -MAX_CURRENT, MAX_CURRENT)))
    return corrientes


# ── Envoltura de ángulos (para el home-in) ───────────────────────────────────
def wrap_centrado(q, centro):
    return centro + (q - centro + math.pi) % (2 * math.pi) - math.pi

def wrap_a_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ── Gripper: lectura del gatillo ──────────────────────────────────────────────
def leer_gripper_deg(port, packet, gid, lock_port):
    with lock_port:
        pos, res, _ = packet.read4ByteTxRx(port, gid, ADDR_PRESENT_POSITION)
    if res != 0:
        return None
    if pos > 0x7FFFFFFF:
        pos -= 0x100000000
    return (pos % TICKS_POR_VUELTA) / TICKS_POR_VUELTA * 360.0

def gatillo_norm(angle_deg):
    """0.0 = gatillo apretado (cerrado), 1.0 = gatillo suelto (abierto)."""
    if angle_deg is None:
        return None
    norm = (angle_deg - GRIP_CLOSED_DEG) / (GRIP_OPEN_DEG - GRIP_CLOSED_DEG)
    return float(np.clip(norm, 0.0, 1.0))


# ── Dynamixel: setup y apagado ───────────────────────────────────────────────
def init_puerto(port):
    ph = PortHandler(port)
    pk = PacketHandler(PROTOCOL)
    if not ph.openPort():
        raise IOError(f"No se pudo abrir {port}")
    if not ph.setBaudRate(BAUDRATE):
        raise IOError(f"No se pudo configurar baudrate en {port}")
    return ph, pk

def configurar_modo5(port, packet, motor_ids, etiqueta):
    print(f"Configurando MODO 5 en {etiqueta}:")
    for mid in motor_ids:
        packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.02)
        packet.write1ByteTxRx(port, mid, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
        time.sleep(0.03)
        packet.write2ByteTxRx(port, mid, ADDR_GOAL_CURRENT, 0)
        packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 1)

def apagar(motor_ids, port, packet, lock_port):
    with lock_port:
        for mid in motor_ids:
            packet.write2ByteTxRx(port, mid, ADDR_GOAL_CURRENT, 0)
            packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0)


# ── Estado compartido de los GELLO ───────────────────────────────────────────
q_brazo1 = [0.0] * 6
q_brazo2 = [0.0] * 6
ticks_actuales_1 = [0] * 6
ticks_actuales_2 = [0] * 6
lock_q1  = threading.Lock()
lock_q2  = threading.Lock()
running  = True

port_handler_1 = packet_handler_1 = None
port_handler_2 = packet_handler_2 = None
sync_read_1 = sync_read_2 = None
sync_write_cur_1 = sync_write_pos_1 = None
sync_write_cur_2 = sync_write_pos_2 = None
lock_port1 = threading.Lock()
lock_port2 = threading.Lock()


def hilo_lectura_sync(motor_ids, signs, offsets, sync_read, lock_port, lock_q, q_ref, ticks_ref):
    while running:
        try:
            with lock_port:
                res = sync_read.txRxPacket()
            if res == 0:
                t_list = []
                for mid in motor_ids:
                    if sync_read.isAvailable(mid, ADDR_PRESENT_POSITION, 4):
                        pos = sync_read.getData(mid, ADDR_PRESENT_POSITION, 4)
                        if pos > 0x7FFFFFFF:
                            pos -= 0x100000000
                        t_list.append(pos)
                if len(t_list) == 6:
                    q_nuevo = aplicar_transformacion(t_list, signs, offsets)
                    with lock_q:
                        q_ref[:] = q_nuevo
                        ticks_ref[:] = t_list
        except Exception:
            pass
        time.sleep(0.01)


def procesar_embrague(motor_ids, ticks_actuales, ticks_anteriores, locked_ticks, mov_state,
                      corrientes_dinamicas, sync_cur, sync_pos, lock_port):
    sync_cur.clearParam()
    sync_pos.clearParam()

    for i, mid in enumerate(motor_ids):
        vel = abs(ticks_actuales[i] - ticks_anteriores[i])

        if mov_state[i]:
            if vel < UMBRAL_SALIR[i]:
                mov_state[i] = 0
        else:
            if vel > UMBRAL_ENTRAR[i]:
                mov_state[i] = 1

        if mov_state[i]:
            locked_ticks[i] = ticks_actuales[i]
            goal_current = min(abs(int(corrientes_dinamicas[i])), MAX_CURRENT_MOV)
        else:
            goal_current = min(abs(int(corrientes_dinamicas[i])) + MARGEN_AGARRE[i],
                               MAX_CURRENT_HOLD[i])

        param_cur = [goal_current & 0xFF, (goal_current >> 8) & 0xFF]
        pos_int = int(locked_ticks[i]) & 0xFFFFFFFF
        param_pos = [pos_int & 0xFF, (pos_int >> 8) & 0xFF, (pos_int >> 16) & 0xFF, (pos_int >> 24) & 0xFF]

        sync_cur.addParam(mid, param_cur)
        sync_pos.addParam(mid, param_pos)

        ticks_anteriores[i] = ticks_actuales[i]

    with lock_port:
        sync_pos.txPacket()
        sync_cur.txPacket()


def hilo_dinamica_embrague():
    last_t1 = [0]*6; locked_t1 = [0]*6; mov_state_1 = [0]*6
    last_t2 = [0]*6; locked_t2 = [0]*6; mov_state_2 = [0]*6

    q1_prev = [0]*6; qd1_prev = [0]*6; qdd1_prev = [0]*6
    q2_prev = [0]*6; qd2_prev = [0]*6; qdd2_prev = [0]*6

    time.sleep(0.5)
    with lock_q1:
        last_t1[:] = ticks_actuales_1; locked_t1[:] = ticks_actuales_1
        q1_prev[:] = q_brazo1
    with lock_q2:
        last_t2[:] = ticks_actuales_2; locked_t2[:] = ticks_actuales_2
        q2_prev[:] = q_brazo2

    while running:
        try:
            with lock_q1:
                q1 = list(q_brazo1); t1 = list(ticks_actuales_1)
            with lock_q2:
                q2 = list(q_brazo2); t2 = list(ticks_actuales_2)

            qd1 = derivar_y_filtrar(q1, q1_prev, qd1_prev, GRAV_PERIODO)
            qdd1 = derivar_y_filtrar(qd1, qd1_prev, qdd1_prev, GRAV_PERIODO)
            qd2 = derivar_y_filtrar(q2, q2_prev, qd2_prev, GRAV_PERIODO)
            qdd2 = derivar_y_filtrar(qd2, qd2_prev, qdd2_prev, GRAV_PERIODO)

            c1 = corriente_dinamica(np.array(q1), np.array(qd1), np.array(qdd1),
                                    SIGNS_1, CURRENT_SIGN_1, ACTIVO_1, GRAVEDAD_1, ESCALA_1)
            c2 = corriente_dinamica(np.array(q2), np.array(qd2), np.array(qdd2),
                                    SIGNS_2, CURRENT_SIGN_2, ACTIVO_2, GRAVEDAD_2, ESCALA_2)

            procesar_embrague(MOTOR_IDS_1, t1, last_t1, locked_t1, mov_state_1, c1,
                              sync_write_cur_1, sync_write_pos_1, lock_port1)
            procesar_embrague(MOTOR_IDS_2, t2, last_t2, locked_t2, mov_state_2, c2,
                              sync_write_cur_2, sync_write_pos_2, lock_port2)

            q1_prev[:] = q1; qd1_prev[:] = qd1; qdd1_prev[:] = qdd1
            q2_prev[:] = q2; qd2_prev[:] = qd2; qdd2_prev[:] = qdd2
        except Exception:
            pass
        time.sleep(GRAV_PERIODO)


def iniciar_gravedad():
    global port_handler_1, packet_handler_1, port_handler_2, packet_handler_2
    global sync_read_1, sync_read_2
    global sync_write_cur_1, sync_write_pos_1, sync_write_cur_2, sync_write_pos_2

    port_handler_1, packet_handler_1 = init_puerto(PORT_1)
    port_handler_2, packet_handler_2 = init_puerto(PORT_2)

    sync_read_1 = GroupSyncRead(port_handler_1, packet_handler_1, ADDR_PRESENT_POSITION, 4)
    sync_read_2 = GroupSyncRead(port_handler_2, packet_handler_2, ADDR_PRESENT_POSITION, 4)
    sync_write_cur_1 = GroupSyncWrite(port_handler_1, packet_handler_1, ADDR_GOAL_CURRENT, 2)
    sync_write_pos_1 = GroupSyncWrite(port_handler_1, packet_handler_1, ADDR_GOAL_POSITION, 4)
    sync_write_cur_2 = GroupSyncWrite(port_handler_2, packet_handler_2, ADDR_GOAL_CURRENT, 2)
    sync_write_pos_2 = GroupSyncWrite(port_handler_2, packet_handler_2, ADDR_GOAL_POSITION, 4)

    for mid in MOTOR_IDS_1: sync_read_1.addParam(mid)
    for mid in MOTOR_IDS_2: sync_read_2.addParam(mid)

    configurar_modo5(port_handler_1, packet_handler_1, MOTOR_IDS_1, "brazo 1 (PORT_1)")
    configurar_modo5(port_handler_2, packet_handler_2, MOTOR_IDS_2, "brazo 2 (PORT_2)")

    # Grippers: el operador los mueve a mano -> torque OFF, solo leemos el gatillo
    packet_handler_1.write1ByteTxRx(port_handler_1, GRIPPER_ID_1, ADDR_TORQUE_ENABLE, 0)
    packet_handler_2.write1ByteTxRx(port_handler_2, GRIPPER_ID_2, ADDR_TORQUE_ENABLE, 0)

    threading.Thread(target=hilo_lectura_sync,
                     args=(MOTOR_IDS_1, SIGNS_1, OFFSETS_1, sync_read_1,
                           lock_port1, lock_q1, q_brazo1, ticks_actuales_1),
                     daemon=True).start()
    threading.Thread(target=hilo_lectura_sync,
                     args=(MOTOR_IDS_2, SIGNS_2, OFFSETS_2, sync_read_2,
                           lock_port2, lock_q2, q_brazo2, ticks_actuales_2),
                     daemon=True).start()
    threading.Thread(target=hilo_dinamica_embrague, daemon=True).start()
    print("[+] Compensación de gravedad activa (embrague reforzado).")


def apagar_gravedad():
    global running
    running = False
    time.sleep(0.2)
    try: apagar(MOTOR_IDS_1, port_handler_1, packet_handler_1, lock_port1)
    except Exception: pass
    try: apagar(MOTOR_IDS_2, port_handler_2, packet_handler_2, lock_port2)
    except Exception: pass
    try: port_handler_1.closePort()
    except Exception: pass
    try: port_handler_2.closePort()
    except Exception: pass
    print("\nMotores GELLO liberados.")


# ════════════════════════════════════════════════════════════════════════════
#  HOME-IN VISUAL
# ════════════════════════════════════════════════════════════════════════════
SIM_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
LABELS = ['pan', 'lift', 'elbow', 'w1', 'w2', 'w3']

TOL_DEG     = 5.0
GHOST_COLOR = [0.35, 0.75, 1.0, 0.45]
GHOST_FORCE = 200.0
COLOR_OK    = [0.10, 0.90, 0.10]
COLOR_NO    = [0.95, 0.45, 0.10]
COLOR_WAIT  = [0.60, 0.60, 0.60]


def detectar_brazos(body):
    grupos = {}
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[1].decode()
        for suf in SIM_ORDER:
            if name.endswith(suf):
                prefijo = name[:-len(suf)]
                grupos.setdefault(prefijo, {})[suf] = (j, name)
                break

    completos = {pre: d for pre, d in grupos.items() if len(d) == len(SIM_ORDER)}
    if len(completos) < 2:
        raise RuntimeError(
            f"No se detectaron 2 brazos UR completos. Prefijos: "
            f"{ {pre: list(d) for pre, d in grupos.items()} }")

    asign = {}
    for pre in completos:
        low = pre.lower()
        if 'left' in low:
            asign['left'] = pre
        elif 'right' in low:
            asign['right'] = pre
    if set(asign) != {'left', 'right'}:
        pres = list(completos)
        ys = {pre: p.getLinkState(body, completos[pre]['shoulder_pan_joint'][0])[0][1]
              for pre in pres}
        asign = {'left': max(pres, key=lambda pr: ys[pr]),
                 'right': min(pres, key=lambda pr: ys[pr])}
        print("[AVISO] left/right por geometría (y). Verifícalo visualmente.")

    idx, nombres = {}, {}
    for arm, pre in asign.items():
        idx[arm]     = [completos[pre][suf][0] for suf in SIM_ORDER]
        nombres[arm] = [completos[pre][suf][1] for suf in SIM_ORDER]
    return idx, nombres


class Fantasma:
    def __init__(self, urdf_path, base_pos, base_orn):
        self.body = p.loadURDF(urdf_path, base_pos, base_orn, useFixedBase=True)
        p.changeVisualShape(self.body, -1, rgbaColor=GHOST_COLOR)
        p.setCollisionFilterGroupMask(self.body, -1, 0, 0)
        for j in range(p.getNumJoints(self.body)):
            p.changeVisualShape(self.body, j, rgbaColor=GHOST_COLOR)
            p.setCollisionFilterGroupMask(self.body, j, 0, 0)
            q = p.getJointState(self.body, j)[0]
            p.setJointMotorControl2(self.body, j, p.POSITION_CONTROL,
                                    targetPosition=q, force=GHOST_FORCE)
        self.arm_joint_idx, nombres = detectar_brazos(self.body)
        print("[Fantasma] Juntas de brazo detectadas:")
        for arm in ('left', 'right'):
            print(f"    {arm:>5}: {nombres[arm]}")

    def set_arm(self, arm, q_sim):
        p.setJointMotorControlArray(
            self.body, self.arm_joint_idx[arm], p.POSITION_CONTROL,
            targetPositions=q_sim, forces=[GHOST_FORCE] * 6)


class VisualizadorFantasma:
    def __init__(self, urdf_path=ROBOT_URDF_PATH):
        rospy.init_node('home_in', anonymous=True)

        self.adam = ADAM(urdf_path, useRealTimeSimulation=True,
                         used_fixed_base=True, use_ros=False)
        p.setGravity(0, 0, 0)   # sim sin física; nada se desploma

        base_pos, base_orn = self._base_de_adam()
        self.fantasma = Fantasma(urdf_path, base_pos, base_orn)

        self._real = {'left': None, 'right': None}
        self._lock_real = threading.Lock()
        self._gello_prev = {'left': None, 'right': None}
        self._snap = {'left': (None, None), 'right': (None, None)}
        self._mano_cerrada = {'right': False, 'left': False}

        rospy.Subscriber('/robot/left_arm/joint_states', JointState,
                         self._cb_real, callback_args='left', queue_size=1)
        rospy.Subscriber('/robot/right_arm/joint_states', JointState,
                         self._cb_real, callback_args='right', queue_size=1)

        iniciar_gravedad()

        # Estado inicial de manos: abiertas
        self.adam.hand_kinematics.move_hand_to_dofs('both', DOFS_ABIERTO, DOFS_ABIERTO)

        self._crear_letreros()
        rospy.loginfo("Visualizador listo. Esperando datos de topic y GELLO...")

    # ── Base de ADAM ───────────────────────────────────────────────────────────
    def _base_de_adam(self):
        adam_id = None
        for attr in ('robot_id', 'robotId', 'robot', 'id', 'uid', 'body_id'):
            val = getattr(self.adam, attr, None)
            if isinstance(val, int):
                adam_id = val
                break
        if adam_id is None:
            mejor, njmax = None, -1
            for b in range(p.getNumBodies()):
                uid = p.getBodyUniqueId(b)
                nj = p.getNumJoints(uid)
                if nj > njmax:
                    njmax, mejor = nj, uid
            adam_id = mejor
        self.adam_id = adam_id
        return p.getBasePositionAndOrientation(adam_id)

    # ── Brazos reales (topic) ──────────────────────────────────────────────────
    def _cb_real(self, msg, arm):
        with self._lock_real:
            self._real[arm] = dict(zip(msg.name, msg.position))

    def _q_real_sim(self, arm):
        with self._lock_real:
            estado = self._real[arm]
            estado = dict(estado) if estado is not None else None
        if estado is None:
            return None
        pref = f'robot_{arm}_arm_'
        try:
            return [estado[pref + n] for n in SIM_ORDER]
        except KeyError:
            return None

    # ── GELLO (consume el subsistema de gravedad) ──────────────────────────────
    def _gello_raw(self, arm):
        if arm == 'left':
            with lock_q2:
                return list(q_brazo2)
        with lock_q1:
            return list(q_brazo1)

    def _gello_listo(self, arm):
        t = ticks_actuales_2 if arm == 'left' else ticks_actuales_1
        return any(v != 0 for v in t)

    # ── Snapshot por ciclo (real + gello desenrollado) ─────────────────────────
    def _tomar_snapshot(self):
        for arm in ('left', 'right'):
            q_real = self._q_real_sim(arm)
            if self._gello_listo(arm):
                raw = self._gello_raw(arm)
                prev = self._gello_prev[arm]
                if prev is None:
                    q_gello = list(raw)
                else:
                    q_gello = [wrap_centrado(raw[i], prev[i]) for i in range(6)]
                self._gello_prev[arm] = q_gello
            else:
                q_gello = None
            self._snap[arm] = (q_real, q_gello)

    def _estado_arm(self, arm):
        q_real, q_gello = self._snap[arm]
        if q_real is None or q_gello is None:
            return None
        difs = [math.degrees(wrap_a_pi(g - r)) for g, r in zip(q_gello, q_real)]
        max_abs = max(abs(d) for d in difs)
        return dict(q_real=q_real, q_gello=q_gello, difs=difs,
                    max_abs=max_abs, aligned=max_abs <= TOL_DEG)

    # ── Manos: gatillo on/off de close_hand ─────────────────────────────────────
    def _procesar_manos(self):
        n1 = gatillo_norm(leer_gripper_deg(port_handler_1, packet_handler_1, GRIPPER_ID_1, lock_port1))  # right
        n2 = gatillo_norm(leer_gripper_deg(port_handler_2, packet_handler_2, GRIPPER_ID_2, lock_port2))  # left
        self._procesar_gatillo('right', n1)
        self._procesar_gatillo('left',  n2)

    def _procesar_gatillo(self, hand, norm):
        """Por flanco con histéresis: apretar -> close_hand (bloqueante), soltar -> abrir."""
        if norm is None:
            return
        if not self._mano_cerrada[hand] and norm < UMBRAL_CERRAR:
            self._mano_cerrada[hand] = True
            print(f"[mano {hand}] cerrando (close_hand)...")
            ok = self.adam.hand_kinematics.close_hand(hand_side=hand,
                                                      thumb_abd_value=THUMB_ABD,
                                                      close_speed=CLOSE_SPEED,
                                                      force_threshold=FORCE_THRESHOLD)
            print(f"[mano {hand}] agarre {'OK' if ok else 'sin contacto'}.")
        elif self._mano_cerrada[hand] and norm > UMBRAL_ABRIR:
            self._mano_cerrada[hand] = False
            print(f"[mano {hand}] abriendo.")
            self.adam.hand_kinematics.move_hand_to_dofs(hand, DOFS_ABIERTO)

    # ── Letreros ───────────────────────────────────────────────────────────────
    def _crear_letreros(self):
        self._label_id = {}
        for arm in ('left', 'right'):
            self._label_id[arm] = p.addUserDebugText(
                f"{arm.upper()}: ...", self._pos_letrero(arm),
                textColorRGB=COLOR_WAIT, textSize=1.5)

    def _pos_letrero(self, arm):
        pan_idx = self.fantasma.arm_joint_idx[arm][0]
        try:
            wp = p.getLinkState(self.adam_id, pan_idx)[0]
            return [wp[0], wp[1], wp[2] + 0.40]
        except Exception:
            return [0, 0.3 if arm == 'left' else -0.3, 1.6]

    def _actualizar_letreros(self):
        for arm in ('left', 'right'):
            est = self._estado_arm(arm)
            if est is None:
                texto, color = f"{arm.upper()}: sin datos", COLOR_WAIT
            elif est['aligned']:
                texto, color = f"{arm.upper()}: LISTO", COLOR_OK
            else:
                texto, color = f"{arm.upper()}: alinea  dmax={est['max_abs']:.0f} deg", COLOR_NO
            p.addUserDebugText(texto, self._pos_letrero(arm),
                               textColorRGB=color, textSize=1.5,
                               replaceItemUniqueId=self._label_id[arm])

    # ── Actualización ──────────────────────────────────────────────────────────
    def _actualizar(self):
        self._tomar_snapshot()
        for arm in ('left', 'right'):
            q_real, q_gello = self._snap[arm]
            if q_real is not None:
                self.adam.arm_kinematics.set_arm_pose(arm, 'joint', q_real)
            if q_gello is not None:
                self.fantasma.set_arm(arm, q_gello)
        self._actualizar_letreros()
        self._procesar_manos()   # gatillo -> close_hand / abrir (bloqueante al cerrar)

    # ── Impresión ──────────────────────────────────────────────────────────────
    def _imprimir(self):
        lineas = []
        for arm in ('left', 'right'):
            est = self._estado_arm(arm)
            q_real, q_gello = self._snap[arm]
            marca = "LISTO" if (est and est['aligned']) else ""
            lineas.append(f"{arm.upper():<6} {marca:>6}   real(°)   gello(°)   d(°)")
            for i, etq in enumerate(LABELS):
                r = q_real[i] if q_real else None
                g = q_gello[i] if q_gello else None
                rr = f"{math.degrees(r):>8.1f}" if r is not None else "     --"
                gg = f"{math.degrees(g):>8.1f}" if g is not None else "     --"
                if r is not None and g is not None:
                    d = math.degrees(wrap_a_pi(g - r))
                    flag = " " if abs(d) <= TOL_DEG else "*"
                    dd = f"{d:>7.1f}{flag}"
                else:
                    dd = "     --"
                lineas.append(f"  {etq:<5} {rr}  {gg}  {dd}")
        print("\n".join(lineas))
        print("=" * 46)

    # ── Bucle principal ────────────────────────────────────────────────────────
    def run(self, loop_hz=30, print_hz=5):
        rate = rospy.Rate(loop_hz)
        cada = max(1, int(loop_hz / print_hz))
        i = 0
        try:
            while not rospy.is_shutdown() and p.isConnected():
                self._actualizar()
                if i % cada == 0:
                    self._imprimir()
                i += 1
                rate.sleep()
        except KeyboardInterrupt:
            print("\nDeteniendo...")
        finally:
            apagar_gravedad()


if __name__ == '__main__':
    try:
        VisualizadorFantasma().run(loop_hz=30, print_hz=5)
    except rospy.ROSInterruptException:
        apagar_gravedad()