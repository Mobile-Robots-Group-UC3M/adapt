import math
import time
import threading
import pybullet as p
import pybullet_data
from dynamixel_sdk import PortHandler, PacketHandler
import rospy
import os
import sys
from sensor_msgs.msg import JointState

#Importamos ADAMSim para cargar el robot en PyBullet y tener su ID y joints
sys.path.append(os.path.join(os.path.dirname(__file__), "AdamSim"))
from scripts.adam import ADAM

# URDF robot path and create ADAM instance
base_path = os.path.dirname(__file__)
#robot_urdf_path = os.path.join(base_path,"..","models","robot", "rb1_base_description", "robots", "robotDummy.urdf")
robot_urdf_path = "/home/alumnos/tfg-laura/AdamSim/models/robot/rb1_base_description/robots/robotDummy.urdf"
adam = ADAM(robot_urdf_path, useRealTimeSimulation=True, used_fixed_base=True, use_ros=False)
print([a for a in dir(adam) if not a.startswith('_')])


# ── Configuración brazo 2 (DERECHA) ─────────────────────────────────────────
PORT_2      = "/dev/ttyUSB0"
MOTOR_IDS_2 = [1, 2, 3, 4, 5, 6]
SIGNS_2     = [1, 1, -1, 1, 1, 1]
OFFSETS_2 = [-4.4477, -4.4879, 3.0840, 3.2282, -3.0217, 1.5567]

# ── Configuración brazo 1 (IZQUIERDA) ───────────────────────────────────────────
PORT_1      = "/dev/ttyUSB1"
MOTOR_IDS_1 = [8, 9, 10, 11, 12, 13]
SIGNS_1     = [1, 1, -1, 1, 1, 1]
OFFSETS_1 = [-6.2262, -2.8435, 3.1002, -1.6222, -3.2817, -4.6493]


# ── Configuración común ───────────────────────────────────────────────────────
BAUDRATE  = 57600
PROTOCOL  = 2.0
URDF_PATH = "/home/alumnos/tfg-laura/robotics-toolbox/xacro_generated/ur_description/urdf/ur3.urdf"

ADDR_PRESENT_POSITION = 132
TICKS_POR_VUELTA      = 4096

# ── Orden de joints que espera ADAMSim (orden robot real UR3) ─────────────────
# ADAMSim hace swap [0]↔[2] al recibir, así que publicamos:
# [elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3]
# Tu q_brazo viene en orden GELLO: [pan, lift, elbow, w1, w2, w3]
# → hay que reordenar: [2, 1, 0, 3, 4, 5]

JOINT_NAMES_LEFT = [
    'robot_left_arm_elbow_joint',
    'robot_left_arm_shoulder_lift_joint',
    'robot_left_arm_shoulder_pan_joint',
    'robot_left_arm_wrist_1_joint',
    'robot_left_arm_wrist_2_joint',
    'robot_left_arm_wrist_3_joint',
]
JOINT_NAMES_RIGHT = [
    'robot_right_arm_elbow_joint',
    'robot_right_arm_shoulder_lift_joint',
    'robot_right_arm_shoulder_pan_joint',
    'robot_right_arm_wrist_1_joint',
    'robot_right_arm_wrist_2_joint',
    'robot_right_arm_wrist_3_joint',
]

def reorder_para_adamsim(q):
    """
    Tu q viene en orden GELLO [pan, lift, elbow, w1, w2, w3].
    ADAMSim espera [elbow, lift, pan, w1, w2, w3] y luego hace swap internamente.
    """
    return [q[2], q[1], q[0], q[3], q[4], q[5]]

# ── Conversión ────────────────────────────────────────────────────────────────
def ticks_a_rad(ticks):
    ticks_norm = ticks % TICKS_POR_VUELTA
    return (ticks_norm / TICKS_POR_VUELTA) * 2 * math.pi

def normalizar_grados(deg):
    return ((deg + 180) % 360) - 180

def aplicar_transformacion(ticks_list, signs, offsets):
    return [signs[i] * ticks_a_rad(t) + offsets[i]
            for i, t in enumerate(ticks_list)]

def wrap_centrado(q, centro):
    """Coloca q en el intervalo [centro-π, centro+π), moviendo el corte a centro+π."""
    return centro + (q - centro + math.pi) % (2 * math.pi) - math.pi

# ── Estado compartido ─────────────────────────────────────────────────────────
q_brazo1 = [0.0] * 6
q_brazo2 = [0.0] * 6
lock_q1  = threading.Lock()
lock_q2  = threading.Lock()
running  = True

# ── Inicializar ROS ───────────────────────────────────────────────────────────
rospy.init_node('gello_publisher', anonymous=True)
pub_right  = rospy.Publisher('/robot/left_arm/joint_states',  JointState, queue_size=1)
pub_left = rospy.Publisher('/robot/right_arm/joint_states', JointState, queue_size=1)

def make_joint_state_msg(q_reordenado, names):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name     = names
    msg.position = q_reordenado
    msg.velocity = [0.0] * 6
    msg.effort   = [0.0] * 6
    return msg

# ── Inicializar comunicación Dynamixel ────────────────────────────────────────
def init_puerto(port):
    ph = PortHandler(port)
    pk = PacketHandler(PROTOCOL)
    if not ph.openPort():
        raise IOError(f"No se pudo abrir {port}")
    if not ph.setBaudRate(BAUDRATE):
        raise IOError(f"No se pudo configurar baudrate en {port}")
    print(f"✓ Conectado a {port}")
    return ph, pk

port_handler_1, packet_handler_1 = init_puerto(PORT_1)
port_handler_2, packet_handler_2 = init_puerto(PORT_2)

lock_port1 = threading.Lock()
lock_port2 = threading.Lock()

def leer_posicion(motor_id, port_handler, packet_handler, lock_port):
    try:
        with lock_port:
            pos, result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION)
        if result != 0 or error != 0:
            return None
        return pos
    except Exception:
        return None

# ── Hilos de lectura ──────────────────────────────────────────────────────────
def hilo_lectura(motor_ids, signs, offsets, port_handler, packet_handler,
                 lock_port, lock_q, q_ref, label):
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
        except Exception:
            pass

threading.Thread(
    target=hilo_lectura,
    args=(MOTOR_IDS_1, SIGNS_1, OFFSETS_1,
          port_handler_1, packet_handler_1,
          lock_port1, lock_q1, q_brazo1, "B1"),
    daemon=True
).start()

threading.Thread(
    target=hilo_lectura,
    args=(MOTOR_IDS_2, SIGNS_2, OFFSETS_2,
          port_handler_2, packet_handler_2,
          lock_port2, lock_q2, q_brazo2, "B2"),
    daemon=True
).start()


try:
    while p.isConnected() and not rospy.is_shutdown():
        with lock_q1:
            q1 = list(q_brazo1)
        with lock_q2:
            q2 = list(q_brazo2)

        # B1 (motores 8-13) y B2 (motores 1-6) → lados de ADAM.
        # OJO: confirma qué brazo es 'left' y cuál 'right' moviendo un joint.
        # OJO: empieza SIN reorder; si un joint mueve el equivocado, mete reorder_para_adamsim().
        adam.arm_kinematics.set_arm_pose('left',  'joint', q2)
        adam.arm_kinematics.set_arm_pose('right', 'joint', q1)

        b1 = "  ".join(f"J{i+1}:{normalizar_grados(math.degrees(q1[i])):>6.1f}°" for i in range(6))
        b2 = "  ".join(f"J{i+1}:{normalizar_grados(math.degrees(q2[i])):>6.1f}°" for i in range(6))
        print(f"B1: {b1}")
        print(f"B2: {b2}")

        time.sleep(0.05)

except Exception as e:
    print(f"\nError: {e}")
finally:
    running = False
    time.sleep(0.2)
    port_handler_1.closePort()
    port_handler_2.closePort()
    print("\nPuertos cerrados.")