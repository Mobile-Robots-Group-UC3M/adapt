#!/usr/bin/env python3
"""
Lee las posiciones de los dos brazos de ADAM desde ROS y las muestra en ADAMSim.

Se suscribe a:
    /robot/left_arm/joint_states
    /robot/right_arm/joint_states

- Guarda la última posición de cada brazo indexada por NOMBRE de junta
  (robusto frente al orden del array).
- Pinta esa posición en ADAMSim con set_arm_pose(..., 'joint', ...).
- Imprime cada junta en radianes y en grados.

Conversión de orden (clave):
    El topic publica en orden robot real:  [elbow, lift, pan, w1, w2, w3]
    set_arm_pose('joint', q) espera orden: [pan, lift, elbow, w1, w2, w3]
    -> es el swap [0]<->[2] que hace ros_connection.py. Aquí se resuelve
       simplemente pidiendo cada junta por su nombre en SIM_ORDER.

Este es el paso previo al home-in: aquí solo LEEMOS y MOSTRAMOS.
"""

import os
import sys
import math
import threading

import rospy
import pybullet as p
from sensor_msgs.msg import JointState

# ── Cargar ADAMSim ────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "AdamSim"))
from scripts.adam import ADAM

ROBOT_URDF_PATH = "/home/alumnos/tfg-laura/AdamSim/models/robot/rb1_base_description/robots/robotDummy.urdf"

# ── Órdenes de junta (sin el prefijo robot_<arm>_arm_) ────────────────────────
# Orden en que set_arm_pose('joint', q) espera los valores (orden GELLO/sim):
SIM_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
# Etiquetas cortas para imprimir, en el mismo orden que SIM_ORDER:
LABELS = ['pan', 'lift', 'elbow', 'w1', 'w2', 'w3']


class MostrarBrazosSim:
    def __init__(self, urdf_path=ROBOT_URDF_PATH):
        rospy.init_node('mostrar_brazos_sim', anonymous=True)

        # ADAMSim en PyBullet (sin su propia conexión ROS: la gestionamos aquí)
        self.adam = ADAM(urdf_path, useRealTimeSimulation=True,
                         used_fixed_base=True, use_ros=False)

        # Estado: {arm: {nombre_junta_completo: rad}}
        self._estado = {'left': None, 'right': None}
        self._lock = threading.Lock()

        rospy.Subscriber('/robot/left_arm/joint_states', JointState,
                         self._callback, callback_args='left', queue_size=1)
        rospy.Subscriber('/robot/right_arm/joint_states', JointState,
                         self._callback, callback_args='right', queue_size=1)

        rospy.loginfo("Suscrito a left_arm y right_arm. Esperando datos...")

    # ── ROS ───────────────────────────────────────────────────────────────────
    def _callback(self, msg, arm):
        pos_por_nombre = dict(zip(msg.name, msg.position))
        with self._lock:
            self._estado[arm] = pos_por_nombre

    def _q_sim(self, arm):
        """Devuelve la posición en orden SIM_ORDER (lista de rad), o None."""
        with self._lock:
            estado = self._estado[arm]
            if estado is None:
                return None
            estado = dict(estado)
        prefijo = f'robot_{arm}_arm_'
        try:
            return [estado[prefijo + nombre] for nombre in SIM_ORDER]
        except KeyError as e:
            rospy.logwarn_throttle(5.0, f"Junta no encontrada en {arm}: {e}")
            return None

    # ── Visualización ──────────────────────────────────────────────────────────
    def _actualizar_sim(self):
        for arm in ('left', 'right'):
            q = self._q_sim(arm)
            if q is not None:
                self.adam.arm_kinematics.set_arm_pose(arm, 'joint', q)

    def _imprimir(self):
        lineas = []
        for arm in ('left', 'right'):
            q = self._q_sim(arm)
            lineas.append(f"{arm.upper()}")
            if q is None:
                lineas.append("  (sin datos)")
                continue
            for etq, val in zip(LABELS, q):
                lineas.append(f"  {etq:<5}: {val:>8.4f} rad   {math.degrees(val):>8.2f}°")
        print("\n".join(lineas))
        print("-" * 42)

    # ── Bucle principal ────────────────────────────────────────────────────────
    def run(self, loop_hz=30, print_hz=5):
        rate = rospy.Rate(loop_hz)
        cada = max(1, int(loop_hz / print_hz))  # imprimir 1 de cada N iteraciones
        i = 0
        while not rospy.is_shutdown() and p.isConnected():
            self._actualizar_sim()
            if i % cada == 0:
                self._imprimir()
            i += 1
            rate.sleep()


if __name__ == '__main__':
    try:
        MostrarBrazosSim().run(loop_hz=30, print_hz=5)
    except rospy.ROSInterruptException:
        pass