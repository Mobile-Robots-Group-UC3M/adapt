import time
from dynamixel_sdk import PortHandler, PacketHandler

# ── Configuración ─────────────────────────────────────────────────────────────
#PORT        = "COM5"
BAUDRATE    = 57600        # baudrate por defecto de los XL330
PROTOCOL    = 2.0          # los XL330 usan Protocol 2.0
#MOTOR_IDS   = [1, 2, 3, 4, 5, 6, 7]
#MOTOR_IDS   = [1]

PORT     = "/dev/ttyUSB0"
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
# Dirección del registro de posición actual en los XL330 (Protocol 2.0)
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION  = 4   # 4 bytes

# Rango del encoder XL330: 0 – 4095 → 0° – 360°
TICKS_POR_VUELTA = 4096

def ticks_a_grados(ticks):
    return round((ticks / TICKS_POR_VUELTA) * 360.0, 2)

# ── Inicializar comunicación ──────────────────────────────────────────────────
port_handler   = PortHandler(PORT)
packet_handler = PacketHandler(PROTOCOL)

if not port_handler.openPort():
    raise IOError(f"No se pudo abrir el puerto {PORT}. "
                  "¿Está el U2D2 conectado y el puerto correcto?")

if not port_handler.setBaudRate(BAUDRATE):
    raise IOError(f"No se pudo configurar el baudrate {BAUDRATE}.")

print(f"Conectado a {PORT} a {BAUDRATE} bps\n")
print(f"{'Motor':<8} {'Ticks':<10} {'Grados':>8}")
print("-" * 30)

# ── Bucle de lectura ──────────────────────────────────────────────────────────
try:
    while True:
        for motor_id in MOTOR_IDS:
            pos, result, error = packet_handler.read4ByteTxRx(
                port_handler, motor_id, ADDR_PRESENT_POSITION
            )

            if result != 0:  # COMM_SUCCESS = 0
                estado = f"  ERROR comm (código {result})"
            elif error != 0:
                estado = f"  ERROR motor (código {error})"
            else:
                grados = ticks_a_grados(pos)
                estado = f"{pos:<10} {grados:>7}°"

            nombre = f"Motor {motor_id}" + (" (gripper)" if motor_id == 14 else "")
            print(f"{nombre:<18} {estado}")

        print()  # línea en blanco entre lecturas
        time.sleep(1)  # lee cada 100ms, ajusta si quieres más frecuencia

except KeyboardInterrupt:
    print("\nDetenido por el usuario.")

finally:
    port_handler.closePort()
    print("Puerto cerrado.")