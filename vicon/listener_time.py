#!/usr/bin/env python3


import argparse
import sys

import rospy
from geometry_msgs.msg import TransformStamped


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_TOPIC = "/vicon/wand/wand"


# ---------------------------------------------------------------------------
# ROS callback
# ---------------------------------------------------------------------------

def timestamp_callback(msg: TransformStamped) -> None:
    """
    Receive one Vicon wand message and print its timestamp to stdout.

    Combines the integer seconds and nanosecond fields of the ROS header
    stamp into a single floating-point value with nanosecond precision,
    then immediately flushes stdout so the parent subprocess pipe is
    unblocked.

    Parameters
    ----------
    msg : geometry_msgs/TransformStamped
        Incoming message from the Vicon bridge.
    """
    timestamp: float = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
    print(timestamp, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Vicon wand ROS timestamps to stdout")
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"ROS topic to subscribe to (default: {DEFAULT_TOPIC})")
    args = parser.parse_args()

    rospy.init_node("vicon_timestamp_relay", anonymous=True)
    rospy.Subscriber(args.topic, TransformStamped, timestamp_callback)
    rospy.loginfo(f"Vicon timestamp relay listening on {args.topic}")
    rospy.spin()


if __name__ == "__main__":
    main()
