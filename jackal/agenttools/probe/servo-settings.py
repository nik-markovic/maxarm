#!/usr/bin/env python3
"""What each bus servo holds about itself: read-only bus commands, nothing moved by this script.

Opening the port resets the ESP32 and its main.py homes the arm (transport.py): that is the only motion.
Then, for servos 1-3, the LX-protocol reads the board's BusServo class has no method for, sent through its own
send_data() / servo_receive_handle(): angle offset (19), angle limits (21), voltage limits (23), temperature
limit (25), temperature (26), voltage (27), position (28), servo-or-motor mode (30), torque on or off (32).
Each reply printed as its parameter bytes.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/servo-settings.py
"""
from maxarm.protocol import ReplProtocol
from maxarm.transport import SerialTransport

READS = {19: "angle offset", 21: "angle limits", 23: "voltage limits", 25: "temperature limit", 26: "temperature",
         27: "voltage", 28: "position", 30: "mode", 32: "torque"}

HELPER = """
def _probe(i, cmd):
    b = bus_servo
    b.uart.read()
    t = [0x55, 0x55, i, 3, cmd]
    t.append(b.check_sum(t))
    b.send_data(t)
    n = time.ticks_ms()
    while b.uart.any() == 0:
        if time.ticks_ms() - n > 50:
            return None
    return list(b.rx_buf) if b.servo_receive_handle() else None
"""


def signed(low, high):
    value = low | high << 8
    return value - 65536 if value >= 32768 else value


def describe(cmd, params):
    if cmd == 19:
        offset = params[0] - 256 if params[0] >= 128 else params[0]
        return f"{offset:+d} (x 0.24 deg = {offset * 0.24:+.2f} deg)"
    if cmd in (21, 23):
        return f"{signed(params[0], params[1])} .. {signed(params[2], params[3])}"
    if cmd in (27, 28):
        return str(signed(params[0], params[1]))
    if cmd == 30:
        return f"mode {params[0]} ({'motor' if params[0] else 'servo'}), speed {signed(params[2], params[3])}"
    return str(params[0])


protocol = ReplProtocol(SerialTransport())
protocol.connect()
protocol.run("import time")
protocol.run(HELPER)
for servo in (1, 2, 3):
    print(f"servo {servo}")
    for cmd, name in READS.items():
        reply = protocol.evaluate(f"_probe({servo}, {cmd})")
        if not reply or reply[4] != cmd:
            print(f"  {name:18} no answer ({reply})")
            continue
        params = reply[5:-1]
        print(f"  {name:18} {describe(cmd, params):30} raw {params}")
protocol.transport.close()
