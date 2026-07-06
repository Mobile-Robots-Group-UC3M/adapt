import math
import time
import threading
import pybullet as p
import pybullet_data
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite
import os
import sys
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import RevoluteDH

# Importamos ADAMSim
sys.path.append(os.path.join(os.path.dirname(__file__), "AdamSim"))
from scripts.adam import ADAM

# URDF robot path and create ADAM instance
base_path = os.path.dirname(__file__)
robot_urdf_path = "/home/alumnos/tfg-laura/AdamSim/models/robot/rb1_base_description/robots/robotDummy.urdf"
adam = ADAM(robot_urdf_path, useRealTimeSimulation=True, used_fixed_base=True, use_ros=False)

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACION DE LOS DOS BRAZOS
# ════════════════════════════════════════════════════════════════════════════
PORT_2      = "/dev/ttyUSB0"
MOTOR_IDS_2 = [1, 2, 3, 4, 5, 6]
SIGNS_2     = [1, 1, -1, 1, 1, 1]
OFFSETS_2 = [-4.4477, -4.4879, 3.0840, 3.2282, -3.0217, 0]

PORT_1      = "/dev/ttyUSB1"
MOTOR_IDS_1 = [8, 9, 10, 11, 12, 13]
SIGNS_1     = [1, 1, -1, 1, 1, 1]
OFFSETS_1 = [-6.2262, -3.3671, 3.1002, -1.6222, -3.2817, -3.123]

BAUDRATE  = 57600
PROTOCOL  = 2.0

ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_CURRENT     = 102
ADDR_GOAL_POSITION    = 116  
ADDR_PRESENT_POSITION = 132

TICKS_POR_VUELTA      = 4096
MODE_CURRENT_BASED_POSITION = 5

# ════════════════════════════════════════════════════════════════════════════
#  MODELO FÍSICO Y PARAMETROS DH
# ════════════════════════════════════════════════════════════════════════════
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

# ── PARÁMETROS DEL EMBRAGUE ──────────────────────────────────────────────────
GAIN        = 1.2
K_CURRENT   = 1500.0

# Tope ABSOLUTO de seguridad del hardware (mA). En el XL330 Goal Current va en
# unidades de 1 mA. Stall = 1470 mA. NUNCA mantener corriente alta mucho rato:
# para hold continuo lo seguro es no pasar de ~700 mA por junta.
MAX_CURRENT = 700           # clip interno; ninguna corriente calculada lo supera

# ── Techos de comportamiento (separados: mover vs aguantar) ──────────────────
# Al MOVER queremos los brazos sueltos -> techo bajo y común
MAX_CURRENT_MOV  = 180

# Al SOLTAR queremos que clave la posición -> techo por junta.
# J2 (índice 1) es el hombro y carga todo el brazo: necesita el techo más alto.
#                  J1   J2   J3   J4   J5   J6
MAX_CURRENT_HOLD = [350, 500, 350, 220, 160, 160]

# "Freno" extra que se suma a la corriente dinámica en estado estático (por junta).
# J5/J6 (muñeca) apenas caen por gravedad -> freno mínimo para que vayan sueltas.
# J4 algo intermedio. Si J5/J6 NO derivan al soltarlas, puedes bajarlos a ~0.
#                  J1   J2   J3   J4   J5   J6
MARGEN_AGARRE    = [120, 220, 120,  40,  20,  20]

# ── Histéresis de detección de movimiento (ticks por ciclo de 20 ms), POR JUNTA ──
# Entrar en "movimiento" cuesta más (umbral alto) que salir (umbral bajo):
# así se mueve solo cuando empujas de verdad, y bloquea en cuanto sueltas.
# La muñeca (J4-J6) usa umbrales bajos -> se libera con un empujón muy suave.
#                  J1   J2   J3   J4   J5   J6
UMBRAL_ENTRAR = [ 35,  35,  35,  20,  12,  12]
#UMBRAL_ENTRAR = [ 25,  25,  25,  15,  10,  10]
UMBRAL_SALIR  = [ 12,  12,  12,   8,   5,   5]

FILTRO_RUIDO = 0.9

ACTIVO_1 = [1, 1, 1, 1, 0, 0]
ACTIVO_2 = [1, 1, 1, 1, 0, 0]

CURRENT_SIGN_1 = [-1, -1, 1, 1, 1, 1]
CURRENT_SIGN_2 = [-1, -1, 1, 1, 1, 1]

GRAVEDAD_1 = [6.936, 0.0, 6.936] 
GRAVEDAD_2 = [6.936, 0.0, 6.936] 

ESCALA_1 = [4.5,  6.5,  0.0,  0.1,  0.1,  0.1]
ESCALA_2 = [4.5,  6.5,  0.0,  0.1,  0.1,  0.1]

def ticks_a_rad(ticks):
    return ((ticks % TICKS_POR_VUELTA) / TICKS_POR_VUELTA) * 2 * math.pi

def normalizar_grados(deg):
    return ((deg + 180) % 360) - 180

def aplicar_transformacion(ticks_list, signs, offsets):
    return [signs[i] * ticks_a_rad(t) + offsets[i] for i, t in enumerate(ticks_list)]

def derivar_y_filtrar(actual, previo, derivada_previa, dt, alpha=FILTRO_RUIDO):
    derivada_bruta = [(a - p) / dt for a, p in zip(actual, previo)]
    return [alpha * bruta + (1 - alpha) * prev for bruta, prev in zip(derivada_bruta, derivada_previa)]

def corriente_dinamica(q, qd, qdd, signs, current_sign, activo, gravedad, escala):
    tau = robot.rne(q, qd, qdd, gravity=gravedad)
    corrientes = []
    for i in range(6):
        c = activo[i] * escala[i] * current_sign[i] * signs[i] * GAIN * K_CURRENT * tau[i]
        corrientes.append(int(np.clip(c, -MAX_CURRENT, MAX_CURRENT)))
    return corrientes

def leer_gripper_deg(port, packet, gid, lock_port):
    with lock_port:                                   # mismo lock que los brazos
        pos, res, _ = packet.read4ByteTxRx(port, gid, ADDR_PRESENT_POSITION)
    if res != 0:
        return None
    if pos > 0x7FFFFFFF:
        pos -= 0x100000000
    return (pos % TICKS_POR_VUELTA) / TICKS_POR_VUELTA * 360.0

def gripper_a_dofs(angle_deg):
    if angle_deg is None:
        return None
    norm = (angle_deg - GRIP_CLOSED_DEG) / (GRIP_OPEN_DEG - GRIP_CLOSED_DEG)
    dof  = float(np.clip(norm * 1000.0, 0.0, 1000.0))
    return [dof, dof, dof, dof, dof, THUMB_ABD]       # 5 flexiones iguales + abducción fija

# ── Estado compartido ───────────────────────────────────────────────────────
q_brazo1 = [0.0] * 6
q_brazo2 = [0.0] * 6

ticks_actuales_1 = [0] * 6
ticks_actuales_2 = [0] * 6

lock_q1  = threading.Lock()
lock_q2  = threading.Lock()
running  = True

# ── Dynamixel: abrir puertos ────────────────────────────────────────────────
def init_puerto(port):
    ph = PortHandler(port)
    pk = PacketHandler(PROTOCOL)
    if not ph.openPort(): raise IOError(f"No se pudo abrir {port}")
    if not ph.setBaudRate(BAUDRATE): raise IOError(f"No se pudo configurar baudrate en {port}")
    return ph, pk

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

lock_port1 = threading.Lock()
lock_port2 = threading.Lock()

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

# ── Hilo de Lectura Sincronizada ────────────────────────────────────────────
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
                        if pos > 0x7FFFFFFF: pos -= 0x100000000
                        t_list.append(pos)
                
                if len(t_list) == 6:
                    q_nuevo = aplicar_transformacion(t_list, signs, offsets)
                    with lock_q:
                        q_ref[:] = q_nuevo
                        ticks_ref[:] = t_list
        except Exception:
            pass
        time.sleep(0.01)

configurar_modo5(port_handler_1, packet_handler_1, MOTOR_IDS_1, "brazo 1 (USB0)")
configurar_modo5(port_handler_2, packet_handler_2, MOTOR_IDS_2, "brazo 2 (USB1)")

# ════════════════════════════════════════════════════════════════════════════
#  GRIPPER (PINZA)
# ════════════════════════════════════════════════════════════════════════════
GRIPPER_ID_1 = 14          # mismo bus que brazo 1 (PORT_1) -> mano 'right'
GRIPPER_ID_2 = 7           # mismo bus que brazo 2 (PORT_2) -> mano 'left'

GRIP_CLOSED_DEG = 236.0    # cerrado
GRIP_OPEN_DEG   = 289.0    # abierto
THUMB_ABD       = 200      # abducción fija del pulgar (oposición); ajústalo

# El operador mueve los grippers a mano -> torque OFF, solo leemos posición
packet_handler_1.write1ByteTxRx(port_handler_1, GRIPPER_ID_1, ADDR_TORQUE_ENABLE, 0)
packet_handler_2.write1ByteTxRx(port_handler_2, GRIPPER_ID_2, ADDR_TORQUE_ENABLE, 0)

threading.Thread(target=hilo_lectura_sync, args=(MOTOR_IDS_1, SIGNS_1, OFFSETS_1, sync_read_1, lock_port1, lock_q1, q_brazo1, ticks_actuales_1), daemon=True).start()
threading.Thread(target=hilo_lectura_sync, args=(MOTOR_IDS_2, SIGNS_2, OFFSETS_2, sync_read_2, lock_port2, lock_q2, q_brazo2, ticks_actuales_2), daemon=True).start()

# ── Lógica del Embrague Reforzado ───────────────────────────────────────────
# mov_state[i]: 0 = estático (freno duro), 1 = en movimiento (suelto).
def procesar_embrague(motor_ids, ticks_actuales, ticks_anteriores, locked_ticks, mov_state,
                      corrientes_dinamicas, sync_cur, sync_pos, lock_port):
    sync_cur.clearParam()
    sync_pos.clearParam()
    
    for i, mid in enumerate(motor_ids):
        vel = abs(ticks_actuales[i] - ticks_anteriores[i])
        
        # ── Histéresis (por junta): entrar en movimiento cuesta más que salir ──
        if mov_state[i]:
            if vel < UMBRAL_SALIR[i]:
                mov_state[i] = 0          # se ha quedado quieto -> bloquear ya
        else:
            if vel > UMBRAL_ENTRAR[i]:
                mov_state[i] = 1          # empuje deliberado -> liberar
        
        if mov_state[i]:
            # ESTADO: MOVIMIENTO -> sigue la posición con corriente baja (suelto)
            locked_ticks[i] = ticks_actuales[i]
            goal_current = min(abs(int(corrientes_dinamicas[i])), MAX_CURRENT_MOV)
        else:
            # ESTADO: ESTÁTICO -> "freno de mano" con techo por junta
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

GRAV_PERIODO = 0.02

# ── Hilo de Control Dinámico ────────────────────────────────────────────────
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
                
            c1 = corriente_dinamica(np.array(q1), np.array(qd1), np.array(qdd1), SIGNS_1, CURRENT_SIGN_1, ACTIVO_1, GRAVEDAD_1, ESCALA_1)
            c2 = corriente_dinamica(np.array(q2), np.array(qd2), np.array(qdd2), SIGNS_2, CURRENT_SIGN_2, ACTIVO_2, GRAVEDAD_2, ESCALA_2)
            
            procesar_embrague(MOTOR_IDS_1, t1, last_t1, locked_t1, mov_state_1, c1, sync_write_cur_1, sync_write_pos_1, lock_port1)
            procesar_embrague(MOTOR_IDS_2, t2, last_t2, locked_t2, mov_state_2, c2, sync_write_cur_2, sync_write_pos_2, lock_port2)
            
            q1_prev[:] = q1; qd1_prev[:] = qd1; qdd1_prev[:] = qdd1
            q2_prev[:] = q2; qd2_prev[:] = qd2; qdd2_prev[:] = qdd2

        except Exception:
            pass
        time.sleep(GRAV_PERIODO)

threading.Thread(target=hilo_dinamica_embrague, daemon=True).start()
print("\n[+] Sistema Activo: Embrague Reforzado de Alta Retención. Ctrl+C para salir.\n")

# ════════════════════════════════════════════════════════════════════════════
#  OBJETOS DE LA ESCENA (mesa + cubo para pick & place)
# ════════════════════════════════════════════════════════════════════════════
'''p.setAdditionalSearchPath(pybullet_data.getDataPath())   # acceso a las URDF que trae PyBullet
p.setGravity(0, 0, -9.81)                                # SOLO si ADAM no la pone ya (ver nota)

# Mesa: su origen está en el suelo y la superficie queda a ~0.63 m
table_id = p.loadURDF("table/table.urdf",
                      basePosition=[1.0, 0.0, -0.1],
                      useFixedBase=True)

# Cubo pequeño (5 cm). Lo suelto un poco por encima de la mesa para que caiga y asiente
CUBE_START = [0.3, 0.0, 0.66]
cube_id = p.loadURDF("cube_small.urdf", basePosition=CUBE_START, globalScaling=0.6)
CUBE_START = [0.3, 0.3, 0.66]
cube_id = p.loadURDF("cube_small.urdf", basePosition=CUBE_START, globalScaling=0.6)
# Más agarre: el cubo no debe resbalar entre los dedos
p.changeDynamics(cube_id, -1, lateralFriction=2.5)'''

# ════════════════════════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL (ADAMSim Sync)
# ════════════════════════════════════════════════════════════════════════════
try:
    while p.isConnected():
        with lock_q1: q1 = list(q_brazo1)
        with lock_q2: q2 = list(q_brazo2)

        adam.arm_kinematics.set_arm_pose('left',  'joint', q2)
        adam.arm_kinematics.set_arm_pose('right', 'joint', q1)

        ang1 = leer_gripper_deg(port_handler_1, packet_handler_1, GRIPPER_ID_1, lock_port1)
        ang2 = leer_gripper_deg(port_handler_2, packet_handler_2, GRIPPER_ID_2, lock_port2)
        dofs1, dofs2 = gripper_a_dofs(ang1), gripper_a_dofs(ang2)

        if dofs1 and dofs2:
            adam.hand_kinematics.move_hand_to_dofs('both', dofs1, dofs2)  # dofs1->right, dofs2->left
        elif dofs1:
            adam.hand_kinematics.move_hand_to_dofs('right', dofs1)
        elif dofs2:
            adam.hand_kinematics.move_hand_to_dofs('left', dofs2)

        b1 = "  ".join(f"J{i+1}:{normalizar_grados(math.degrees(q1[i])):>6.1f}°" for i in range(6))
        b2 = "  ".join(f"J{i+1}:{normalizar_grados(math.degrees(q2[i])):>6.1f}°" for i in range(6))
        print(f"B1: {b1}")
        print(f"B2: {b2}")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nDeteniendo el sistema manualmente...")
except Exception as e:
    print(f"\nError: {e}")
finally:
    running = False
    time.sleep(0.2)
    apagar(MOTOR_IDS_1, port_handler_1, packet_handler_1, lock_port1)
    apagar(MOTOR_IDS_2, port_handler_2, packet_handler_2, lock_port2)
    port_handler_1.closePort()
    port_handler_2.closePort()
    print("\nMotores liberados correctamente.")