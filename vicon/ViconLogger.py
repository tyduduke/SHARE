#!/usr/bin/env python3
"""
ViconLogger.py — Multi-Object Vicon Trajectory Logger

Subscribes to Vicon ROS topics for a list of tracked objects and writes
their ground-truth poses to CSV files.  Collection is controlled remotely
via UDP commands broadcast by server.py.

For each tracked object ``<Obj>`` the node subscribes to:
    /vicon/<Obj>/<Obj>  (geometry_msgs/TransformStamped)

Each CSV row has the format:
    timestamp  tx  ty  tz  qx  qy  qz  qw
where ``timestamp`` is the ROS header stamp in seconds (float, nanosecond
precision) and the translation and rotation are in Vicon's world frame.

UDP control protocol  (port configurable via --udp-port, default 10000):
    ``Start Collection: ['Obj1', 'Obj2', ...]``
        Begin logging for the named objects.  If logging is already active,
        the current session is stopped cleanly before starting a new one.
    ``End Collection``
        Flush and close all open CSV files.

Output files are written to ``--output-dir`` (default ./vicon_trajectory_logs)
as  ``<ObjectName>_<YYYY-MM-DD_HH-MM-SS>.csv``.

Usage
-----
    python3 ViconLogger.py [--output-dir ./vicon_trajectory_logs]
                           [--udp-port 10000]
"""

import argparse
import os
import signal
import socket
import sys
import threading
from datetime import datetime
from typing import Dict, List, Optional

import rospy
from geometry_msgs.msg import TransformStamped


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "./vicon_trajectory_logs"
DEFAULT_UDP_PORT   = 10000
CSV_HEADER         = "timestamp tx ty tz qx qy qz qw\n"


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------

class MultiTrajectoryLogger:
    """
    Manages ROS subscriptions and CSV file handles for a set of Vicon objects.

    Typical lifecycle:
        1. Instantiate once.
        2. Call ``start(object_names)`` to begin a logging session.
        3. Call ``stop()`` to end the session and flush all files.
        4. Repeat from step 2 for subsequent sessions.
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        """
        Parameters
        ----------
        output_dir : str
            Directory where CSV files will be written.  Created if absent.
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._subscribers:   list                  = []
        self._file_handles:  Dict[str, object]     = {}
        self._session_stamp: str                   = ""
        self.is_collecting:  bool                  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, object_names: List[str]) -> None:
        """
        Open CSV files and subscribe to Vicon topics for each named object.

        If a session is already active it is stopped first so files are
        closed cleanly before new ones are opened.

        Parameters
        ----------
        object_names : list of str
            Vicon object names as configured in the Vicon system, e.g.
            ``['User1', 'User2', 'Robot']``.
        """
        # Guard blocks from when this class was designed to be fully
        # self-contained (owning its own ROS node and guarding empty input).
        # Removed after the architecture shifted: rospy.init_node() is now
        # called in __main__, and empty object_names is handled by the UDP
        # listener before start() is ever invoked.
        #
        # if not object_names:
        #     rospy.logwarn("No objects to track. Not starting.")
        #     return
        # if not rospy.is_initialized():
        #     rospy.init_node('multi_trajectory_logger', anonymous=True)

        if self.is_collecting:
            rospy.logwarn("ViconLogger: previous session still active — stopping it first")
            self.stop()

        if not object_names:
            rospy.logwarn("ViconLogger: no object names provided; nothing to log")
            return

        self._session_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._open_files(object_names)
        self._subscribe(object_names)
        self.is_collecting = True
        rospy.loginfo(f"ViconLogger: started session '{self._session_stamp}' "
                      f"for objects: {object_names}")

    def stop(self) -> None:
        """
        Unsubscribe from all Vicon topics and close all open CSV files.
        Safe to call even if no session is active.
        """
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers.clear()

        for obj, fh in self._file_handles.items():
            try:
                fh.flush()
                fh.close()
                rospy.loginfo(f"ViconLogger: closed log for {obj}")
            except OSError as exc:
                rospy.logerr(f"ViconLogger: error closing file for {obj}: {exc}")
        self._file_handles.clear()
        self.is_collecting = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_files(self, object_names: List[str]) -> None:
        """Create one CSV file per tracked object for this session."""
        for obj in object_names:
            filename  = f"{obj}_{self._session_stamp}.csv"
            file_path = os.path.join(self.output_dir, filename)
            try:
                fh = open(file_path, "w")
                fh.write(CSV_HEADER)
                fh.flush()
                self._file_handles[obj] = fh
                rospy.loginfo(f"ViconLogger: opened {file_path}")
            except OSError as exc:
                rospy.logerr(f"ViconLogger: cannot open file for {obj}: {exc}")

    def _subscribe(self, object_names: List[str]) -> None:
        """Subscribe to /vicon/<obj>/<obj> for each object in the list."""
        for obj in object_names:
            topic = f"/vicon/{obj}/{obj}"
            try:
                sub = rospy.Subscriber(
                    topic,
                    TransformStamped,
                    self._make_callback(obj),
                    queue_size=100,
                )
                self._subscribers.append(sub)
                rospy.loginfo(f"ViconLogger: subscribed to {topic}")
            except Exception as exc:
                rospy.logerr(f"ViconLogger: failed to subscribe to {topic}: {exc}")

    def _make_callback(self, object_name: str):
        """
        Return a ROS message callback bound to ``object_name``.

        Each callback appends one space-separated row to the object's CSV:
            timestamp  tx  ty  tz  qx  qy  qz  qw
        """
        def callback(msg: TransformStamped) -> None:
            if object_name not in self._file_handles:
                return

            timestamp: float = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
            t = msg.transform.translation
            r = msg.transform.rotation

            row = (f"{timestamp:.9f} "
                   f"{t.x:.9f} {t.y:.9f} {t.z:.9f} "
                   f"{r.x:.9f} {r.y:.9f} {r.z:.9f} {r.w:.9f}\n")
            try:
                self._file_handles[object_name].write(row)
                self._file_handles[object_name].flush()
            except OSError as exc:
                rospy.logerr(f"ViconLogger: write error for {object_name}: {exc}")

        return callback


# ---------------------------------------------------------------------------
# UDP control listener
# ---------------------------------------------------------------------------

def udp_listener(logger: MultiTrajectoryLogger, port: int) -> None:
    """
    Listen for collection-control commands from server.py.

    Expected messages (UTF-8 encoded, sent to broadcast or this machine):
        ``Start Collection: ['Obj1', 'Obj2', ...]``
        ``End Collection``

    Parameters
    ----------
    logger : MultiTrajectoryLogger
        The shared logger instance to control.
    port : int
        UDP port to bind on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    rospy.loginfo(f"ViconLogger: UDP control listener on port {port}")

    while True:
        try:
            raw, addr = sock.recvfrom(1024)
            message   = raw.decode("utf-8").strip()
            rospy.loginfo(f"ViconLogger: received '{message}' from {addr}")

            if message.startswith("Start Collection:"):
                # Parse "Start Collection: ['Obj1', 'Obj2']"
                _, obj_part = message.split(":", 1)
                object_names = [
                    name.strip().strip("'\"")
                    for name in obj_part.strip().strip("[]").split(",")
                    if name.strip().strip("'\"")
                ]
                logger.start(object_names)

            elif message == "End Collection":
                logger.stop()
                rospy.loginfo("ViconLogger: session ended via UDP command")

            else:
                rospy.logwarn(f"ViconLogger: unrecognised UDP message: {message!r}")

        except Exception as exc:
            rospy.logerr(f"ViconLogger: error in UDP listener: {exc}")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _make_signal_handler(logger: MultiTrajectoryLogger):
    """Return a SIGINT/SIGTERM handler that cleanly stops the logger."""
    def handler(sig, frame):
        rospy.loginfo("ViconLogger: shutdown signal received — stopping collection")
        logger.stop()
        rospy.signal_shutdown("User interruption")
        sys.exit(0)
    return handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-object Vicon trajectory logger with UDP control")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV output files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--udp-port", type=int, default=DEFAULT_UDP_PORT,
        help=f"UDP port for collection control commands (default: {DEFAULT_UDP_PORT})")
    args = parser.parse_args()

    rospy.init_node("vicon_trajectory_logger", anonymous=True)

    logger = MultiTrajectoryLogger(output_dir=args.output_dir)

    # Register clean-shutdown handlers.
    handler = _make_signal_handler(logger)
    signal.signal(signal.SIGINT,  handler)
    signal.signal(signal.SIGTERM, handler)

    # Start UDP listener in a daemon thread so it does not block rospy.spin().
    udp_thread = threading.Thread(
        target=udp_listener,
        args=(logger, args.udp_port),
        daemon=True,
        name="UDPControlListener",
    )
    udp_thread.start()

    rospy.loginfo("ViconLogger: ready and waiting for UDP commands")
    rospy.spin()


if __name__ == "__main__":
    main()
