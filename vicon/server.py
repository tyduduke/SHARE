#!/usr/bin/env python3


import argparse
import copy
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST             = "0.0.0.0"
DEFAULT_SYNC_PORT        = 6666    # devices register and receive timestamps here
DEFAULT_DEVICE_PORT      = 8888    # port on each device that receives sync packets
DEFAULT_VICON_CTRL_PORT  = 10000   # ViconLogger listens here for collection commands
DEFAULT_SENSOR_PORT      = 11111   # SensorCollector listens here

SYNC_DURATION            = 8.0     # seconds to stream timestamps per device
SYNC_SEND_INTERVAL       = 0.1     # seconds between each timestamp packet

LISTENER_SCRIPT          = "listener_time.py"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Most-recent Vicon timestamp, updated by timestamp_updater thread.
current_timestamp: float = 0.0
timestamp_lock = threading.Lock()

# Registered devices.
# devices[name] = {'status': 'syncing'|'synced', 'ip': str, 'addr': (ip, port)}
devices: Dict[str, Any] = {}
devices_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Vicon timestamp relay  (subprocess reader)
# ---------------------------------------------------------------------------

def timestamp_updater() -> None:
    """
    Launch listener_time.py as a subprocess and continuously read the Vicon
    timestamps it prints to stdout, updating ``current_timestamp``.

    Runs as a daemon thread; terminates when the main process exits.
    """
    # Earlier approach: use local system time instead of ROS timestamps.
    # Abandoned because time.time() is not synchronised to the Vicon clock,
    # so device offsets computed against it cannot be applied to Vicon-stamped
    # trajectory CSV files during post-processing.
    #
    # def _simple_time_loop():
    #     global current_timestamp
    #     while True:
    #         try:
    #             current_timestamp = time.time()
    #             time.sleep(0.01)
    #         except Exception as e:
    #             print(f"Error updating timestamp: {e}")

    global current_timestamp

    process = subprocess.Popen(
        ["python3", "-u", LISTENER_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,   # line-buffered so each timestamp arrives without delay
    )

    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                new_ts = float(line)
                with timestamp_lock:
                    current_timestamp = new_ts
            except ValueError:
                print(f"[timestamp_updater] Unrecognised output from listener: {line!r}",
                      flush=True)
    except Exception as exc:
        print(f"[timestamp_updater] Unexpected error: {exc}", flush=True)
    finally:
        process.terminate()
        process.wait()


# ---------------------------------------------------------------------------
# Per-device synchronisation thread
# ---------------------------------------------------------------------------

def synchronise_device(device_name: str,
                        device_addr: tuple,
                        sock: socket.socket) -> None:
    """
    Stream Vicon timestamps to one device for SYNC_DURATION seconds, then
    send a stop marker and mark the device as synced.

    Parameters
    ----------
    device_name : str
        Human-readable identifier registered by the device.
    device_addr : tuple
        ``(ip, port)`` to send timestamps to.
    sock : socket.socket
        The shared UDP socket.
    """
    deadline = time.time() + SYNC_DURATION

    while time.time() < deadline:
        with timestamp_lock:
            ts = current_timestamp
        sock.sendto(struct.pack("d", ts), device_addr)
        time.sleep(SYNC_SEND_INTERVAL)

    # Signal the device that synchronisation is complete.
    sock.sendto(b"Stop Sync", device_addr)
    print(f"[sync] Sent Stop Sync to {device_name} at {device_addr}", flush=True)

    with devices_lock:
        devices[device_name]["status"] = "synced"
        all_synced = all(d["status"] == "synced" for d in devices.values())

    if all_synced:
        with timestamp_lock:
            final_ts = current_timestamp
        _write_timestamp_file("sync_completion.txt", final_ts)
        print(f"[sync] All devices synced at Vicon time {final_ts:.6f}", flush=True)


# ---------------------------------------------------------------------------
# Console command handler
# ---------------------------------------------------------------------------

def command_listener(sock: socket.socket,
                      vicon_ctrl_port: int,
                      sensor_port: int) -> None:
    """
    Read operator commands from stdin and broadcast collection-control
    messages to devices and the Vicon logger.

    Supported commands
    ------------------
    begin   Start data collection on all synced devices and ViconLogger.
    end     Stop data collection on all devices and ViconLogger.
    exit    Shut down the server.
    """
    print("\nAvailable commands:", flush=True)
    print("  begin — start collection on all synced devices", flush=True)
    print("  end   — stop  collection on all devices", flush=True)
    print("  exit  — shut down the server\n", flush=True)

    while True:
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            # stdin closed (e.g. non-interactive mode)
            break

        try:
            if cmd == "begin":
                _handle_begin(sock, vicon_ctrl_port, sensor_port)

            elif cmd == "end":
                _handle_end(sock, vicon_ctrl_port, sensor_port)

            elif cmd == "exit":
                print("Shutting down server …", flush=True)
                sys.exit(0)

            else:
                print("Unknown command.  Valid commands: begin, end, exit", flush=True)

        except Exception as exc:
            print(f"[commands] Error processing '{cmd}': {exc}", flush=True)


def _handle_begin(sock: socket.socket,
                   vicon_ctrl_port: int,
                   sensor_port: int) -> None:
    """
    Broadcast start-collection to all synced devices and to ViconLogger.
    Records the Vicon timestamp of collection start to collection_start.txt.
    """
    with devices_lock:
        unsynced = [n for n, d in devices.items() if d["status"] != "synced"]
        if unsynced:
            print(f"[begin] Cannot start: devices not yet synced: {unsynced}", flush=True)
            return

        all_names      = list(devices.keys())
        sensor_addr    = (devices["SensorCollector"]["addr"][0], sensor_port) \
                         if "SensorCollector" in devices else None

    # Build the device list that ViconLogger should track: exclude
    # SensorCollector because it is not a Vicon-tracked object.
    tracked_names = [n for n in all_names if n != "SensorCollector"]

    # Inform ViconLogger which Vicon objects to log.
    vicon_msg = f"Start Collection: {tracked_names}".encode("utf-8")
    sock.sendto(vicon_msg, ("255.255.255.255", vicon_ctrl_port))

    # Inform SensorCollector if present.
    if sensor_addr:
        sock.sendto(b"Start Collection", sensor_addr)

    with timestamp_lock:
        start_ts = current_timestamp
    _write_timestamp_file("collection_start.txt", start_ts)
    print(f"[begin] Collection started at Vicon time {start_ts:.6f}", flush=True)
    print(f"[begin] Tracking Vicon objects: {tracked_names}", flush=True)


def _handle_end(sock: socket.socket,
                 vicon_ctrl_port: int,
                 sensor_port: int) -> None:
    """
    Broadcast stop-collection to all devices and to ViconLogger.
    Records the Vicon timestamp of collection end to collection_end.txt.
    """
    with devices_lock:
        device_snapshot = {n: dict(d) for n, d in devices.items()}

    # Stop each registered device.
    for name, info in device_snapshot.items():
        sock.sendto(b"Stop Collection", info["addr"])
        print(f"[end] Sent Stop Collection to {name}", flush=True)

    # Stop ViconLogger.
    sock.sendto(b"End Collection", ("255.255.255.255", vicon_ctrl_port))

    # Stop SensorCollector if present.
    if "SensorCollector" in device_snapshot:
        sensor_addr = (device_snapshot["SensorCollector"]["addr"][0], sensor_port)
        sock.sendto(b"End Collection", sensor_addr)

    with timestamp_lock:
        end_ts = current_timestamp
    _write_timestamp_file("collection_end.txt", end_ts)
    print(f"[end] Collection stopped at Vicon time {end_ts:.6f}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_timestamp_file(filename: str, timestamp: float) -> None:
    """Write a single Vicon timestamp to a plain-text file."""
    try:
        with open(filename, "w") as f:
            f.write(f"{timestamp:.9f}\n")
    except OSError as exc:
        print(f"[warning] Could not write {filename}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHARE Vicon synchronisation and collection control server")
    parser.add_argument("--host",            default=DEFAULT_HOST)
    parser.add_argument("--sync-port",       type=int, default=DEFAULT_SYNC_PORT,
                        help="UDP port for device registration and timestamp streaming")
    parser.add_argument("--device-port",     type=int, default=DEFAULT_DEVICE_PORT,
                        help="UDP port on each device that receives sync packets")
    parser.add_argument("--vicon-ctrl-port", type=int, default=DEFAULT_VICON_CTRL_PORT,
                        help="UDP port that ViconLogger listens on")
    parser.add_argument("--sensor-port",     type=int, default=DEFAULT_SENSOR_PORT,
                        help="UDP port that SensorCollector listens on")
    args = parser.parse_args()

    # Start the Vicon timestamp relay subprocess in a daemon thread.
    threading.Thread(target=timestamp_updater, daemon=True, name="TimestampUpdater").start()

    # Create the shared UDP socket.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind((args.host, args.sync_port))
    print(f"Vicon sync server listening on {args.host}:{args.sync_port}", flush=True)

    # Console command thread.
    threading.Thread(
        target=command_listener,
        args=(sock, args.vicon_ctrl_port, args.sensor_port),
        daemon=True,
        name="CommandListener",
    ).start()

    # Main loop: handle device registrations.
    while True:
        try:
            raw, addr = sock.recvfrom(1024)
            message   = raw.decode("utf-8").strip()

            # Expected format: "DeviceName:DeviceIP"
            if ":" not in message:
                print(f"[main] Unexpected message from {addr}: {message!r}", flush=True)
                continue

            device_name, device_ip = message.split(":", 1)
            device_addr = (device_ip, args.device_port)

            with devices_lock:
                if device_name in devices:
                    print(f"[main] {device_name} already registered; re-syncing", flush=True)
                devices[device_name] = {
                    "status": "syncing",
                    "ip":     device_ip,
                    "addr":   device_addr,
                }

            threading.Thread(
                target=synchronise_device,
                args=(device_name, device_addr, sock),
                daemon=True,
                name=f"Sync-{device_name}",
            ).start()
            print(f"[main] Started sync for {device_name} at {device_ip}", flush=True)

        except KeyboardInterrupt:
            print("\nInterrupted — shutting down.", flush=True)
            sys.exit(0)
        except Exception as exc:
            print(f"[main] Error in registration loop: {exc}", flush=True)


if __name__ == "__main__":
    main()
