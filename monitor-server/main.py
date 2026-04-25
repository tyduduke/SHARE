#!/usr/bin/env python3
"""
SHARE Monitoring Server — Entry Point

Examples
--------
# Full SHARE system (QoE + PID + merge approval)
python main.py --baseline-mode full --clock-offset-ms 97.38 --alpha-m 0.5

# Baseline 1: fixed 30 % frame drop, no PID
python main.py --baseline-mode monitor_only --fixed-drop-rate 0.3

# Baseline 2: linear latency-based drop (Linear Baseline in paper §6.3)
python main.py --baseline-mode delay_linear --delay-threshold 20 --max-delay 1000

# Baseline 3: equal QoE for all agents (Equal Baseline in paper §6.3)
python main.py --baseline-mode same_qoe
"""

import argparse
import signal
import sys
import time
import logging

from monitoring_server import MonitoringServer
from volume_overlap import VolumeOverlapCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SHARE-Server")


def signal_handler(sig, frame):
    logger.info("Shutdown signal received …")
    if 'server' in globals():
        server.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='SHARE Multi-Agent QoE Monitoring Server')

    # Network
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--robot-eval-port',  type=int, default=9999)
    parser.add_argument('--human1-eval-port', type=int, default=9997)
    parser.add_argument('--human2-eval-port', type=int, default=9998)

    # General
    parser.add_argument('--debug',    action='store_true')
    parser.add_argument('--csv-file', default=None)

    # Baseline mode
    parser.add_argument('--baseline-mode', default='full',
                        choices=['full', 'monitor_only', 'delay_linear', 'same_qoe'])
    parser.add_argument('--fixed-drop-rate',  type=float, default=0.5)
    parser.add_argument('--delay-threshold',  type=float, default=20.0,
                        help='Latency threshold (ms) for delay_linear baseline')
    parser.add_argument('--max-delay',        type=float, default=1000.0,
                        help='Max latency (ms) for delay_linear baseline scaling')

    # Clock correction
    parser.add_argument('--clock-offset-ms', type=float, default=0.0,
                        help='Constant clock offset (ms) to subtract from raw RTT. '
                             'Measure via NTP analysis between agent and server clocks.')

    # QoE targets
    parser.add_argument('--qoe-target-human1', type=float, default=0.9)
    parser.add_argument('--qoe-target-human2', type=float, default=0.9)
    parser.add_argument('--qoe-target-robot',  type=float, default=0.9)

    # Independent PID gains (paper §4.3, grid-searched: Kp=0.8, Ki=0.15, Kd=0.2)
    for agent in ('human1', 'human2', 'robot'):
        parser.add_argument(f'--pid-kp-{agent}', type=float, default=0.8)
        parser.add_argument(f'--pid-ki-{agent}', type=float, default=0.15)
        parser.add_argument(f'--pid-kd-{agent}', type=float, default=0.2)

    # Merge approval: alpha_m controls sensitivity (paper §4.4.2, Eq. 6)
    parser.add_argument('--alpha-m', type=float, default=0.5,
                        help='alpha_m in P_skip = max(0, 1 - alpha_m * V_new/V_bar). '
                             'Higher → more merges approved.')

    # Camera / overlap parameters
    parser.add_argument('--camera-hfov',       type=float, default=90.0)
    parser.add_argument('--camera-vfov',       type=float, default=60.0)
    parser.add_argument('--camera-near',       type=float, default=0.1)
    parser.add_argument('--camera-far',        type=float, default=10.0)
    parser.add_argument('--overlap-resolution', type=float, default=0.2,
                        help='Grid sampling step δ (metres) for frustum overlap')

    # Cache timeouts
    parser.add_argument('--pose-cache-timeout',    type=float, default=5.0)
    parser.add_argument('--overlap-cache-timeout', type=float, default=2.0)

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global server
    server = MonitoringServer(
        host=args.host,
        robot_eval_port=args.robot_eval_port,
        human1_eval_port=args.human1_eval_port,
        human2_eval_port=args.human2_eval_port,
        csv_filename=args.csv_file,
    )

    # Baseline mode
    server.baseline_mode   = args.baseline_mode
    server.fixed_drop_rate = args.fixed_drop_rate
    server.delay_threshold = args.delay_threshold
    server.max_delay       = args.max_delay

    # Clock offset correction
    server.clock_offset_ms = args.clock_offset_ms

    # Merge approval: alpha_m (paper Eq. 6)
    server.alpha_m = args.alpha_m

    # Cache timeouts
    server.pose_cache_timeout    = args.pose_cache_timeout
    server.overlap_cache_timeout = args.overlap_cache_timeout

    # QoE targets
    server.qoe_targets['human1'] = args.qoe_target_human1
    server.qoe_targets['human2'] = args.qoe_target_human2
    server.qoe_targets['robot']  = args.qoe_target_robot

    # PID gains
    for agent in ('human1', 'human2', 'robot'):
        server.pid_params[agent] = {
            'Kp': getattr(args, f'pid_kp_{agent}'),
            'Ki': getattr(args, f'pid_ki_{agent}'),
            'Kd': getattr(args, f'pid_kd_{agent}'),
        }

    # Volume overlap calculator
    server.overlap_calculator = VolumeOverlapCalculator({
        'horizontal_fov_deg': args.camera_hfov,
        'vertical_fov_deg':   args.camera_vfov,
        'near_distance':      args.camera_near,
        'far_distance':       args.camera_far,
        'sampling_resolution': args.overlap_resolution,
    })

    # Initialise QoE models after all parameters are set
    server.initialize_qoe_models()

    logger.info(f"Baseline mode  : {args.baseline_mode}")
    logger.info(f"Clock offset   : {args.clock_offset_ms} ms")
    logger.info(f"alpha_m        : {args.alpha_m}")
    logger.info(f"QoE targets    : human1={args.qoe_target_human1}, "
                f"human2={args.qoe_target_human2}, robot={args.qoe_target_robot}")
    logger.info(f"PID (human1)   : Kp={args.pid_kp_human1} Ki={args.pid_ki_human1} "
                f"Kd={args.pid_kd_human1}")
    logger.info(f"PID (human2)   : Kp={args.pid_kp_human2} Ki={args.pid_ki_human2} "
                f"Kd={args.pid_kd_human2}")
    logger.info(f"PID (robot)    : Kp={args.pid_kp_robot}  Ki={args.pid_ki_robot}  "
                f"Kd={args.pid_kd_robot}")

    server.start()

    try:
        logger.info("SHARE monitoring server running.  Press Ctrl+C to stop.")
        logger.info(f"  Robot  eval → {args.host}:{args.robot_eval_port}")
        logger.info(f"  Human1 eval → {args.host}:{args.human1_eval_port}")
        logger.info(f"  Human2 eval → {args.host}:{args.human2_eval_port}")
        logger.info("  Pose streams → ports 6161 (robot), 6162 (human2), 6163 (human1)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        logger.info("Server shutdown complete")


if __name__ == '__main__':
    main()
