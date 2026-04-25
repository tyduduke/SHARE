#!/usr/bin/env python3
"""
SHARE Monitoring Server

Edge-server component that implements:
  1. Agent-specific QoE modeling  (paper §4.2)
  2. PID scheduler for per-agent frame transmission rate  (paper §4.3)
  3. Visual overlap-based map merge approval  (paper §4.4, Eq. 6)

Key design decisions aligned with the paper:
  - AR users prioritised via different QoE model (d0 = 20 ms threshold).
  - PID output maps directly to frame-rate resource via R(t) = R_prev + u(t)
    clipped to [R_min, R_max] (Algorithm 1, NOT a sigmoid transform).
  - Map merge skip probability: P_skip = max(0, 1 - alpha_m * V_new / V_bar)
    where V_bar is the running mean of overlap volumes from COMPLETED merges
    (paper §4.4.2, Eq. 6).
"""

import socket
import struct
import json
import time
import threading
import logging
import os
import csv
from datetime import datetime
from collections import deque
import numpy as np

from qoe_model import QoEQoSModel
from volume_overlap import VolumeOverlapCalculator

logger = logging.getLogger(__name__)


class MonitoringServer:
    """
    Multi-agent QoE monitoring server with merge approval.
    """

    def __init__(self, host='0.0.0.0',
                 robot_eval_port=9999,
                 human1_eval_port=9997,
                 human2_eval_port=9998,
                 merge_signal_port=9955,
                 merge_approval_port=9956,
                 csv_filename=None):

        self.host = host
        self.robot_eval_port  = robot_eval_port
        self.human1_eval_port = human1_eval_port
        self.human2_eval_port = human2_eval_port
        self.merge_signal_port   = merge_signal_port
        self.merge_approval_port = merge_approval_port

        self.eval_ports = {
            'robot':  robot_eval_port,
            'human1': human1_eval_port,
            'human2': human2_eval_port,
        }
        self.pose_ports = [6161, 6162, 6163]   # robot, human2, human1

        # ----------------------------------------------------------------
        # Baseline mode  (full | monitor_only | delay_linear | same_qoe)
        # ----------------------------------------------------------------
        self.baseline_mode   = 'full'
        self.fixed_drop_rate = 0.5
        self.delay_threshold = 20.0     # ms  (same as AR threshold for linear baseline)
        self.max_delay       = 1000.0   # ms

        # ----------------------------------------------------------------
        # Sockets
        # ----------------------------------------------------------------
        self.eval_socks  = {'robot': None, 'human1': None, 'human2': None}
        self.pose_socks  = [None, None, None]
        self.merge_signal_sock   = None
        self.merge_approval_sock = None
        self.running = False

        # ----------------------------------------------------------------
        # Locks
        # ----------------------------------------------------------------
        self.agent_locks = {
            'human1': threading.RLock(),
            'human2': threading.RLock(),
            'robot':  threading.RLock(),
        }
        self.csv_data_lock   = threading.RLock()
        self.pose_cache_lock = threading.RLock()
        self.overlap_cache_lock = threading.RLock()

        # ----------------------------------------------------------------
        # Pose cache
        # ----------------------------------------------------------------
        self.cached_poses           = [None, None, None]
        self.pose_cache_timestamps  = [0.0, 0.0, 0.0]
        self.pose_cache_timeout     = 5.0   # seconds

        # ----------------------------------------------------------------
        # Overlap cache  (updated continuously, used for P_skip)
        # V_overlap is in m³ — the paper uses absolute volume, not IoU ratio.
        # ----------------------------------------------------------------
        self.cached_overlap_volume    = 0.0   # V_new  (current, m³)
        self.overlap_cache_timestamp  = 0.0
        self.overlap_cache_timeout    = 2.0   # seconds

        # Buffer of overlap volumes from COMPLETED merge operations → V_bar
        # (paper §4.4.2: "a buffer maintains overlap volumes from completed
        # merge operations … historical mean V_bar = (1/N) * sum(Vi)")
        self.merge_volume_buffer = deque(maxlen=100)

        # ----------------------------------------------------------------
        # Merge approval parameters  (paper §4.4.2, Eq. 6)
        # ----------------------------------------------------------------
        # alpha_m: controls sensitivity of redundancy removal.
        # Higher alpha_m → lower P_skip → more merges approved.
        self.alpha_m = 0.5

        # ----------------------------------------------------------------
        # Clock offset: constant ms offset subtracted from raw RTT to correct
        # for systematic clock differences between devices.  Calibrate via NTP
        # and pass with --clock-offset-ms.
        # ----------------------------------------------------------------
        self.clock_offset_ms = 0.0

        # ----------------------------------------------------------------
        # CSV output
        # ----------------------------------------------------------------
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.eval_csv_filename          = csv_filename or f"slam_monitoring_{ts}.csv"
        self.pose_csv_filenames         = [f"pose_data_agent{i+1}_{ts}.csv" for i in range(3)]
        self.overlap_csv_filename       = f"volume_overlap_{ts}.csv"
        self.merge_approval_csv_filename = f"merge_approvals_{ts}.csv"

        # ----------------------------------------------------------------
        # Per-agent data
        # ----------------------------------------------------------------
        self.eval_data = {'robot': [], 'human1': [], 'human2': []}
        self.pose_data = [[], [], []]
        self.overlap_data = []
        self.merge_approval_data = []

        self.agent_data = {
            name: {
                'eval_results': {'delay': 50.0, 'accuracy': 0.5, 'timestamp': time.time()},
                'total_frames': 0,
                'rtt_history':  deque(maxlen=1000),
                'eval_history': deque(maxlen=1000),
                'clients':      {},
            }
            for name in ('robot', 'human1', 'human2')
        }

        self.map_merge_count = 0
        self.merge_history   = deque(maxlen=100)

        # ----------------------------------------------------------------
        # QoE models (initialised later via initialize_qoe_models())
        # ----------------------------------------------------------------
        self.qoe_models = {}

        # ----------------------------------------------------------------
        # Independent PID controllers  (paper §4.3, Algorithm 1)
        # ----------------------------------------------------------------
        self.pid_params = {
            'human1': {'Kp': 0.8, 'Ki': 0.15, 'Kd': 0.2},
            'human2': {'Kp': 0.8, 'Ki': 0.15, 'Kd': 0.2},
            'robot':  {'Kp': 0.8, 'Ki': 0.15, 'Kd': 0.2},
        }
        self.pid_state = {
            name: {
                'prev_error': 0.0, 'integral': 0.0, 'prev_derivative': 0.0,
                'prev_resource': 1.0, 'prev_time': time.time(),
            }
            for name in ('human1', 'human2', 'robot')
        }

        self.qoe_targets = {'human1': 0.9, 'human2': 0.9, 'robot': 0.9}

        # Derivative filter coefficient (alpha_filter in Algorithm 1)
        self.alpha_filter = 0.3

        # Resource allocation limits  (R_min, R_max in paper)
        self.R_min, self.R_max = 0.1, 1.0
        # Integral anti-windup cap
        self.I_max = 0.5

        # ----------------------------------------------------------------
        # Volume overlap calculator
        # ----------------------------------------------------------------
        self.overlap_calculator = VolumeOverlapCalculator()

        # ----------------------------------------------------------------
        # Background threads (started before running flag is set)
        # ----------------------------------------------------------------
        self.pose_cache_thread = threading.Thread(
            target=self._pose_cache_updater, daemon=True)
        self.pose_cache_thread.start()

        self.continuous_overlap_thread = threading.Thread(
            target=self._continuous_overlap_calculation, daemon=True)
        self.continuous_overlap_thread.start()

    # ==================================================================
    # QoE model initialisation
    # ==================================================================

    def initialize_qoe_models(self):
        """
        Initialise agent-specific QoE models based on baseline_mode.

        Human (AR) model parameters (paper §4.2):
          d_max = 20 ms  — the AR motion-to-photon latency threshold
          beta  controls steepness of QoE collapse above 20 ms

        Robot model parameters:
          a_min = 1 cm   — minimum acceptable tracking accuracy (paper §4.2)
          k (theta)      — latency normalisation constant
        """
        # AR / human model  (latency-dominant, paper §4.2)
        h_params = {
            'alpha': 1.0,     # will be overwritten by normalize_alpha()
            'beta':  0.2,     # steepness of collapse above d0 = 20 ms
            'gamma': 0.2,     # diminishing returns on accuracy
            'd_max': 20.0,    # d0 = 20 ms  (paper §3.1.2)
        }

        # Robot model  (accuracy-dominant, paper §4.2)
        r_params = {
            'alpha': 1.0,     # will be overwritten by normalize_alpha()
            'beta':  2.0,     # steepness of collapse below a_min
            'gamma': 0.2,     # diminishing returns on latency improvement
            'a_min': 1.0,     # minimum acceptable accuracy (1 cm, paper §4.2)
            'k':     100.0,   # theta: latency normalisation constant
        }

        if self.baseline_mode == 'same_qoe':
            # Equal-QoE baseline: all agents share the same model.
            # Use the human model for all (including robot) to remove
            # agent-type differentiation (paper ablation §6.3).
            logger.info("same_qoe baseline: using identical human QoE model for all agents")
            unified = dict(h_params)
            self.qoe_models = {
                'human1': QoEQoSModel('human', dict(unified)),
                'human2': QoEQoSModel('human', dict(unified)),
                'robot':  QoEQoSModel('human', dict(unified)),
            }
        else:
            self.qoe_models = {
                'human1': QoEQoSModel('human', dict(h_params)),
                'human2': QoEQoSModel('human', dict(h_params)),
                'robot':  QoEQoSModel('robot', dict(r_params)),
            }

    # ==================================================================
    # Baseline drop-rate selection
    # ==================================================================

    def _calculate_baseline_drop_rate(self, agent, delay_ms, accuracy, qoe):
        """Return (drop_rate, base_drop_rate, modifier) for the active mode."""

        if self.baseline_mode == 'monitor_only':
            # Fixed drop rate — no QoE or PID logic
            dr = self.fixed_drop_rate
            return np.clip(dr, 0.0, 1.0), dr, 1.0

        elif self.baseline_mode == 'delay_linear':
            # Linear frame drop proportional to latency above threshold
            if delay_ms <= self.delay_threshold:
                dr = 0.0
            else:
                excess  = delay_ms - self.delay_threshold
                max_exc = max(self.max_delay - self.delay_threshold, 1e-6)
                dr = min(1.0, excess / max_exc)
            return np.clip(dr, 0.0, 1.0), dr, 1.0

        else:
            # 'full' or 'same_qoe': QoE-driven PID control
            base_drop = max(0.0, min(1.0, 1.0 - qoe))
            modifier  = self._calculate_pid_modifier(agent, qoe)
            dr        = np.clip(base_drop * modifier, 0.0, 1.0)
            return dr, base_drop, modifier

    # ==================================================================
    # PID scheduler  (paper §4.3, Algorithm 1, Equations 2-4)
    # ==================================================================

    def _calculate_pid_modifier(self, agent: str, current_qoe: float) -> float:
        """
        PID controller that maps QoE error to a frame-rate resource modifier.

        Algorithm 1 (paper §4.3):
          err(t)  = QoE_target - QoE_current
          u(t)    = Kp*err + Ki*∫err dt + Kd*(derr/dt)   filtered derivative
          R(t)    = R_prev + u(t),  clipped to [R_min, R_max]
          modifier = 1 - R(t)   (higher resource → lower drop rate)
        """
        try:
            with self.agent_locks[agent]:
                params = self.pid_params[agent]
                state  = self.pid_state[agent]
                target = self.qoe_targets[agent]

                current_time = time.time()
                dt = max(current_time - state['prev_time'], 0.01)

                # Proportional term
                err    = target - current_qoe
                P_term = params['Kp'] * err

                # Integral term with anti-windup
                integral_candidate = state['integral'] + params['Ki'] * err * dt
                state['integral']  = np.clip(integral_candidate, -self.I_max, self.I_max)
                I_term = state['integral']

                # Derivative term with exponential filter  (Algorithm 1 line 10)
                raw_deriv      = (err - state['prev_error']) / dt
                filt_deriv     = (self.alpha_filter * raw_deriv +
                                  (1.0 - self.alpha_filter) * state['prev_derivative'])
                D_term = params['Kd'] * filt_deriv

                # Resource allocation  R(t) = R_prev + u(t)  — Algorithm 1 line 13
                u_t   = P_term + I_term + D_term
                R_new = state['prev_resource'] + u_t

                # Clip to [R_min, R_max]  (paper Algorithm 1 — NOT a sigmoid)
                # Earlier version mapped u(t) through a sigmoid to keep R in [0,1]
                # without hard clipping. Replaced because sigmoid introduced a
                # non-linear dead-band near saturation that slowed QoE recovery
                # after a latency spike, and Algorithm 1 specifies clipping explicitly.
                # R_new = 1.0 / (1.0 + np.exp(-u_t))
                R_new = np.clip(R_new, self.R_min, self.R_max)

                # Anti-windup: undo integral if at saturation and error same sign
                if (R_new == self.R_max or R_new == self.R_min) and \
                   (err * state['prev_error'] > 0):
                    state['integral'] -= params['Ki'] * err * dt

                # Update state
                state['prev_error']      = err
                state['prev_derivative'] = filt_deriv
                state['prev_resource']   = R_new
                state['prev_time']       = current_time

            modifier = np.clip(1.0 - R_new, 0.0, 1.0)
            logger.debug(f"[PID] {agent}: err={err:.4f} P={P_term:.4f} I={I_term:.4f} "
                         f"D={D_term:.4f} R={R_new:.4f} modifier={modifier:.4f}")
            return modifier

        except Exception as e:
            logger.error(f"PID error for {agent}: {e}")
            return 1.0

    # ==================================================================
    # Map merge approval  (paper §4.4.2, Eq. 6)
    # ==================================================================

    def _calculate_merge_approval(self, agent_list: list) -> dict:
        """
        Decide whether to approve a map merge request using Eq. 6:

            P_skip = max(0, 1 - alpha_m * V_new / V_bar)

        where:
            V_new  = current overlap volume (m³) from the continuous cache
            V_bar  = mean overlap volume from completed merge operations (m³)

        A merge is approved if the random draw r >= P_skip, i.e., the
        merge is NOT skipped.  Higher overlap relative to the historical
        mean → lower P_skip → more likely to be approved (kept).
        """
        try:
            # V_new: current cached overlap volume (m³)
            with self.overlap_cache_lock:
                age    = time.time() - self.overlap_cache_timestamp
                V_new  = self.cached_overlap_volume if age < self.overlap_cache_timeout \
                         else 0.0

            # V_bar: mean volume from completed merge operations (paper §4.4.2)
            if len(self.merge_volume_buffer) > 0:
                V_bar = float(np.mean(self.merge_volume_buffer))
            else:
                # No history yet: approve all merges to build initial baseline
                V_bar = None

            if V_bar is None or V_bar < 1e-9:
                # No prior merge history → approve unconditionally
                p_skip = 0.0
                reason = "No prior merge history; unconditional approval"
            else:
                # Eq. 6
                # Earlier version used a linear threshold: approve if V_new > alpha_m * V_bar,
                # skip otherwise. Replaced with the probabilistic formulation so that
                # near-threshold merges are not deterministically blocked, which was
                # causing oscillatory map fragmentation in sequences with steady overlap.
                # approved = V_new > self.alpha_m * V_bar
                p_skip = max(0.0, 1.0 - self.alpha_m * (V_new / V_bar))
                reason = (f"Eq.6: P_skip={p_skip:.3f} "
                          f"(V_new={V_new:.4f} m³, V_bar={V_bar:.4f} m³, "
                          f"alpha_m={self.alpha_m})")

            # Stochastic decision
            r        = float(np.random.random())
            approved = r >= p_skip          # approved ↔ NOT skipped

            if not approved:
                reason += f" → SKIP (r={r:.3f} < P_skip={p_skip:.3f})"
            else:
                reason += f" → APPROVE (r={r:.3f} >= P_skip={p_skip:.3f})"

            return {
                'approved':    approved,
                'p_skip':      p_skip,
                'V_new':       V_new,
                'V_bar':       V_bar,
                'random_value': r,
                'reason':      reason,
            }

        except Exception as e:
            logger.error(f"Error in merge approval calculation: {e}")
            return {'approved': True, 'p_skip': 0.0, 'V_new': 0.0,
                    'V_bar': 0.0, 'random_value': 0.0,
                    'reason': f"Error, defaulting to approval: {e}"}

    # ==================================================================
    # Continuous overlap calculation (background thread)
    # ==================================================================

    def _continuous_overlap_calculation(self):
        """
        Periodically recompute the current pairwise overlap volume from
        cached agent poses and update self.cached_overlap_volume (V_new).
        """
        logger.info("Continuous volume overlap calculation started")
        interval = 1.0   # seconds

        while True:
            try:
                with self.pose_cache_lock:
                    poses      = [p.copy() if p else None for p in self.cached_poses]
                    timestamps = list(self.pose_cache_timestamps)

                now        = time.time()
                valid_poses = [p for p, ts in zip(poses, timestamps)
                               if p and (now - ts) < self.pose_cache_timeout]

                if len(valid_poses) >= 2:
                    overlaps = self.overlap_calculator.calculate_all_overlaps(poses)

                    if overlaps:
                        # Use the first available pairwise volume as V_new.
                        # In a 2-agent scenario this is the only pair; in a
                        # 3-agent scenario the minimum (most conservative) is used.
                        # Earlier version used IoU ratio (V_overlap / V_union) as V_new
                        # so that V_bar would be unit-free and scale-invariant across
                        # different room sizes. Switched to absolute volume (m³) after
                        # finding that IoU saturated near 1.0 in close-proximity
                        # sequences, making P_skip insensitive to actual overlap changes.
                        # V_new = min(v['overlap_ratio_iou'] for v in overlaps.values())
                        V_candidates = [v['volume_overlap']
                                        for v in overlaps.values()
                                        if v['volume_overlap'] >= 0]
                        V_new = float(np.mean(V_candidates)) if V_candidates else 0.0

                        with self.overlap_cache_lock:
                            self.cached_overlap_volume   = V_new
                            self.overlap_cache_timestamp = now

                        # Build CSV record (IoU ratio kept for human-readable logs)
                        record = {'timestamp': datetime.fromtimestamp(now)
                                               .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                  'agents_with_poses': len(valid_poses),
                                  'V_new_m3': V_new}
                        for key, stats in overlaps.items():
                            for sk, sv in stats.items():
                                record[f"{key}_{sk}"] = sv

                        with self.csv_data_lock:
                            self.overlap_data.append(record)

                        logger.debug(f"Overlap: V_new={V_new:.4f} m³ "
                                     f"across {len(overlaps)} pair(s)")

            except Exception as e:
                logger.error(f"Error in overlap calculation: {e}")

            time.sleep(interval)

    # ==================================================================
    # Pose cache updater (background thread)
    # ==================================================================

    def _pose_cache_updater(self):
        """Periodically check for stale pose cache entries."""
        while True:
            try:
                with self.pose_cache_lock:
                    now = time.time()
                    for i in range(3):
                        if self.cached_poses[i] is not None and \
                           (now - self.pose_cache_timestamps[i]) > self.pose_cache_timeout:
                            logger.debug(f"Pose cache expired for agent {i+1}")
            except Exception as e:
                logger.error(f"Pose cache updater error: {e}")
            time.sleep(0.5)

    # ==================================================================
    # UDP merge signal listener (COVINS-G → server)
    # ==================================================================

    def _monitor_merge_signals_udp(self):
        """
        Listen for merge_start / merge_complete notifications from COVINS-G.

        On merge_complete: record the current overlap volume into the
        merge_volume_buffer so V_bar is based on completed merges only
        (paper §4.4.2).
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.merge_signal_port))
            sock.settimeout(1.0)
            self.merge_signal_sock = sock

            logger.info(f"UDP merge signal listener on port {self.merge_signal_port}")
            current_merge = {}

            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    msg = json.loads(data.decode('utf-8'))
                    event      = msg.get('event', '')
                    agents     = msg.get('agents', [])
                    ts_ms      = msg.get('timestamp', time.time() * 1000)
                    ts         = ts_ms / 1000.0

                    if event == 'merge_start':
                        logger.info(f"Merge START — agents {agents}")
                        current_merge = {'start_time': ts, 'agents': agents}

                    elif event == 'merge_complete':
                        kf_count = msg.get('keyframes', 0)
                        lm_count = msg.get('landmarks', 0)
                        self.map_merge_count += 1

                        # Record the current overlap volume into the buffer
                        # (this is V_i in the paper's history buffer)
                        with self.overlap_cache_lock:
                            V_completed = self.cached_overlap_volume
                        self.merge_volume_buffer.append(V_completed)

                        duration = ts - current_merge.get('start_time', ts)
                        logger.info(f"Merge COMPLETE — agents {agents}, "
                                    f"KFs={kf_count}, LMs={lm_count}, "
                                    f"duration={duration:.2f}s, "
                                    f"V_recorded={V_completed:.4f} m³, "
                                    f"buffer_size={len(self.merge_volume_buffer)}")
                        current_merge = {}

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Merge signal error: {e}")

        except Exception as e:
            logger.error(f"Merge signal listener fatal error: {e}")
        finally:
            if self.merge_signal_sock:
                self.merge_signal_sock.close()

    # ==================================================================
    # UDP merge approval listener (COVINS-G → server, bidirectional)
    # ==================================================================

    def _monitor_merge_approval_requests(self):
        """
        Receive merge_approval_request from COVINS-G, apply the P_skip
        criterion (Eq. 6), and send the binary approved/denied response.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.merge_approval_port))
            sock.settimeout(1.0)
            self.merge_approval_sock = sock

            logger.info(f"Merge approval listener on port {self.merge_approval_port}")

            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    req = json.loads(data.decode('utf-8'))

                    if req.get('event') != 'merge_approval_request':
                        continue

                    agents        = req.get('agents', [])
                    response_port = req.get('response_port')
                    req_ts        = req.get('timestamp', time.time() * 1000) / 1000.0

                    if response_port is None:
                        logger.error("No response_port in merge approval request")
                        continue

                    decision = self._calculate_merge_approval(agents)

                    # Send binary response back to COVINS-G
                    resp_msg = {
                        'event':               'merge_approval_response',
                        'timestamp':           time.time() * 1000,
                        'request_timestamp':   req_ts * 1000,
                        'agents':              agents,
                        'approved':            decision['approved'],
                        'p_skip':              decision['p_skip'],
                        'V_new':               decision['V_new'],
                        'reason':              decision['reason'],
                    }
                    resp_bytes = json.dumps(resp_msg).encode('utf-8')
                    resp_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    resp_sock.sendto(resp_bytes, (addr[0], response_port))
                    resp_sock.close()

                    # Log decision
                    record = {
                        'timestamp':        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'agents':           str(agents),
                        'approved':         decision['approved'],
                        'p_skip':           decision['p_skip'],
                        'V_new_m3':         decision['V_new'],
                        'V_bar_m3':         decision['V_bar'],
                        'decision_reason':  decision['reason'],
                    }
                    with self.csv_data_lock:
                        self.merge_approval_data.append(record)

                    logger.info(f"Merge decision: {decision['approved']} — {decision['reason']}")

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Merge approval error: {e}")

        except Exception as e:
            logger.error(f"Merge approval listener fatal: {e}")
        finally:
            if self.merge_approval_sock:
                self.merge_approval_sock.close()

    # ==================================================================
    # Evaluation message processing (agent → server)
    # ==================================================================

    def _process_eval_message(self, message_data: bytes, client_id: str,
                              agent: str) -> float:
        """
        Decode one evaluation message, compute QoE and drop rate, return drop rate.

        Message JSON fields:
          frame_id           -- integer frame counter
          evaluation_result  -- DNN SLAM error estimate (cm)
          processing_time_ms -- time to process this keyframe on the agent (ms)
          timestamp          -- Unix timestamp when message was sent (seconds)
        """
        try:
            msg           = json.loads(message_data.decode('utf-8'))
            frame_id      = msg['frame_id']
            eval_result   = msg['evaluation_result']
            proc_time_ms  = msg['processing_time_ms']
            sent_ts       = msg['timestamp']

            recv_ts  = time.time()
            # RTT corrected for clock offset between agent and server clocks
            rtt_ms   = (recv_ts - sent_ts) * 1000.0 - self.clock_offset_ms
            total_delay = rtt_ms + proc_time_ms
            accuracy    = abs(eval_result)     # DNN output (SLAM error in cm)

            # Update agent state
            with self.agent_locks[agent]:
                self.agent_data[agent]['eval_results'] = {
                    'delay':     total_delay,
                    'accuracy':  accuracy,
                    'timestamp': recv_ts,
                }
                self.agent_data[agent]['total_frames'] += 1
                self.agent_data[agent]['rtt_history'].append(rtt_ms)
                self.agent_data[agent]['eval_history'].append(eval_result)

            # QoE and drop rate
            qoe = self.qoe_models[agent].calculate_qoe(total_delay, accuracy)
            drop_rate, base_drop, modifier = self._calculate_baseline_drop_rate(
                agent, total_delay, accuracy, qoe)

            # PID state snapshot for logging
            with self.agent_locks[agent]:
                pid_resource = self.pid_state[agent]['prev_resource']
                qoe_error    = self.qoe_targets[agent] - qoe

            frame_record = {
                'timestamp':             recv_ts,
                'frame_id':              frame_id,
                'client_id':             client_id,
                'agent':                 agent,
                'rtt_ms':                rtt_ms,
                'evaluation_result':     eval_result,
                'processing_time_ms':    proc_time_ms,
                'accuracy':              accuracy,
                'qoe':                   qoe,
                'base_drop_rate':        base_drop,
                'drop_rate_modifier':    modifier,
                'final_drop_rate':       drop_rate,
                'total_frames_at_time':  self.agent_data[agent]['total_frames'],
                'pid_resource_allocation': pid_resource,
                'qoe_error':             qoe_error,
                'agent_delay':           total_delay,
                'baseline_mode':         self.baseline_mode,
            }
            with self.csv_data_lock:
                self.eval_data[agent].append(frame_record)

            return drop_rate

        except Exception as e:
            logger.error(f"Error processing eval message for {agent}: {e}")
            return 0.5

    # ==================================================================
    # TCP evaluation server loops
    # ==================================================================

    def _eval_server_loop(self, agent: str):
        """Accept TCP connections from one agent's evaluation client."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.bind((self.host, self.eval_ports[agent]))
            s.listen(5)
            s.settimeout(1.0)
            self.eval_socks[agent] = s
            logger.info(f"Eval server [{agent}] listening on {self.host}:{self.eval_ports[agent]}")

            while self.running:
                try:
                    conn, addr = s.accept()
                    logger.info(f"New eval client [{agent}] from {addr}")
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    t = threading.Thread(target=self._handle_eval_client,
                                         args=(conn, addr, agent), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Eval accept error [{agent}]: {e}")
        finally:
            if self.eval_socks[agent]:
                try:
                    self.eval_socks[agent].close()
                except Exception:
                    pass

    def _handle_eval_client(self, conn, addr, agent: str):
        """Handle one evaluation TCP connection (length-prefixed JSON protocol)."""
        client_id = f"{addr[0]}:{addr[1]}"
        with self.agent_locks[agent]:
            self.agent_data[agent]['clients'][client_id] = {
                'connected_at': time.time(), 'frames_received': 0, 'last_seen': time.time()}

        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.settimeout(60.0)

            while self.running:
                # Read 4-byte length prefix
                hdr = self._recv_exactly(conn, 4)
                if not hdr:
                    break
                msg_len = struct.unpack('!I', hdr)[0]
                if msg_len <= 0 or msg_len > 1024 * 1024:
                    logger.error(f"Invalid message length {msg_len} from [{agent}] {client_id}")
                    break

                body = self._recv_exactly(conn, msg_len)
                if not body or len(body) != msg_len:
                    break

                drop_rate = self._process_eval_message(body, client_id, agent)
                if not 0.0 <= drop_rate <= 1.0:
                    drop_rate = 0.5

                resp       = json.dumps({'drop_rate': float(drop_rate)}).encode('utf-8')
                resp_frame = struct.pack('!I', len(resp)) + resp
                self._send_completely(conn, resp_frame)

                with self.agent_locks[agent]:
                    if client_id in self.agent_data[agent]['clients']:
                        self.agent_data[agent]['clients'][client_id]['frames_received'] += 1
                        self.agent_data[agent]['clients'][client_id]['last_seen'] = time.time()

        except socket.timeout:
            logger.warning(f"Eval client timeout [{agent}] {client_id}")
        except ConnectionResetError:
            logger.info(f"Eval client reset [{agent}] {client_id}")
        except Exception as e:
            logger.error(f"Eval client error [{agent}] {client_id}: {e}")
        finally:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            with self.agent_locks[agent]:
                self.agent_data[agent]['clients'].pop(client_id, None)
            logger.info(f"Eval client disconnected [{agent}] {client_id}")

    # ==================================================================
    # TCP pose server loops
    # ==================================================================

    def _pose_server_loop(self, agent_id: int):
        """Accept TCP connections for pose stream from one agent."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.pose_ports[agent_id]))
            s.listen(5)
            s.settimeout(1.0)
            self.pose_socks[agent_id] = s
            logger.info(f"Pose server [agent{agent_id+1}] on {self.host}:{self.pose_ports[agent_id]}")

            while self.running:
                try:
                    conn, addr = s.accept()
                    t = threading.Thread(target=self._handle_pose_client,
                                         args=(conn, addr, agent_id), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Pose accept error [agent{agent_id+1}]: {e}")
        finally:
            if self.pose_socks[agent_id]:
                try:
                    self.pose_socks[agent_id].close()
                except Exception:
                    pass

    def _handle_pose_client(self, conn, addr, agent_id: int):
        """Handle one pose-stream TCP connection (newline-delimited text)."""
        buf = ""
        try:
            while self.running:
                chunk = conn.recv(1024).decode('utf-8')
                if not chunk:
                    break
                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    self._process_pose_data(line, addr, agent_id)
        except Exception as e:
            logger.error(f"Pose client error [agent{agent_id+1}] {addr}: {e}")
        finally:
            conn.close()

    def _process_pose_data(self, line: str, addr, agent_id: int):
        """Parse 'x y z qx qy qz qw' line and update the pose cache."""
        try:
            parts = line.strip().split()
            if len(parts) < 7:
                return
            x, y, z   = float(parts[0]), float(parts[1]), float(parts[2])
            qx, qy, qz, qw = (float(parts[3]), float(parts[4]),
                               float(parts[5]), float(parts[6]))

            now = time.time()
            with self.pose_cache_lock:
                self.cached_poses[agent_id]          = {'x': x, 'y': y, 'z': z,
                                                         'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw}
                self.pose_cache_timestamps[agent_id] = now

            # Log pose for CSV
            roll, pitch, yaw = self._quat_to_euler(qx, qy, qz, qw)
            record = {
                'timestamp':   datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'client_addr': f"{addr[0]}:{addr[1]}",
                'x': x, 'y': y, 'z': z,
                'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
                'roll_deg': roll, 'pitch_deg': pitch, 'yaw_deg': yaw,
            }
            with self.csv_data_lock:
                self.pose_data[agent_id].append(record)

        except Exception as e:
            logger.error(f"Pose parse error [agent{agent_id+1}]: {e}")

    # ==================================================================
    # Server lifecycle
    # ==================================================================

    def start(self):
        """Start all server threads."""
        self.running = True

        # UDP listeners
        threading.Thread(target=self._monitor_merge_signals_udp,   daemon=True).start()
        threading.Thread(target=self._monitor_merge_approval_requests, daemon=True).start()

        # Per-agent TCP evaluation servers
        for agent in ('human1', 'human2', 'robot'):
            threading.Thread(target=self._eval_server_loop, args=(agent,), daemon=True).start()

        # Per-agent pose servers
        for i in range(3):
            threading.Thread(target=self._pose_server_loop, args=(i,), daemon=True).start()

        logger.info(f"SHARE monitoring server started (mode={self.baseline_mode})")

    def stop(self):
        """Stop all threads and flush data to CSV."""
        logger.info("Stopping monitoring server …")
        self.running = False

        for sock in list(self.eval_socks.values()) + self.pose_socks + \
                    [self.merge_signal_sock, self.merge_approval_sock]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        self.save_to_csv()
        logger.info("Monitoring server stopped")

    # ==================================================================
    # CSV persistence
    # ==================================================================

    def save_to_csv(self):
        """Write all collected data to CSV files."""
        try:
            with self.csv_data_lock:
                eval_snap   = {a: list(d) for a, d in self.eval_data.items()}
                pose_snap   = [list(d) for d in self.pose_data]
                overlap_snap = list(self.overlap_data)
                approval_snap = list(self.merge_approval_data)

            eval_headers = [
                'timestamp', 'frame_id', 'client_id', 'agent',
                'rtt_ms', 'evaluation_result', 'processing_time_ms',
                'accuracy', 'qoe', 'base_drop_rate', 'drop_rate_modifier',
                'final_drop_rate', 'total_frames_at_time',
                'pid_resource_allocation', 'qoe_error',
                'agent_delay', 'baseline_mode',
            ]

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            for agent in ('robot', 'human1', 'human2'):
                if eval_snap[agent]:
                    fname = f"{agent}_evaluation_{self.baseline_mode}_{ts}.csv"
                    with open(fname, 'w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=eval_headers)
                        w.writeheader()
                        w.writerows(eval_snap[agent])
                    logger.info(f"Saved {len(eval_snap[agent])} records → {fname}")

            all_eval = sorted(
                eval_snap['robot'] + eval_snap['human1'] + eval_snap['human2'],
                key=lambda r: r['timestamp'])
            if all_eval:
                fname = self.eval_csv_filename.replace('.csv', f'_{self.baseline_mode}.csv')
                with open(fname, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=eval_headers)
                    w.writeheader()
                    w.writerows(all_eval)
                logger.info(f"Saved {len(all_eval)} combined records → {fname}")

            pose_headers = ['timestamp', 'client_addr',
                            'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                            'roll_deg', 'pitch_deg', 'yaw_deg']
            for i, records in enumerate(pose_snap):
                if records:
                    with open(self.pose_csv_filenames[i], 'w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=pose_headers)
                        w.writeheader()
                        w.writerows(records)

            if overlap_snap:
                headers = list(overlap_snap[0].keys())
                with open(self.overlap_csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=headers)
                    w.writeheader()
                    w.writerows(overlap_snap)

            if approval_snap:
                approval_headers = ['timestamp', 'agents', 'approved',
                                    'p_skip', 'V_new_m3', 'V_bar_m3', 'decision_reason']
                with open(self.merge_approval_csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=approval_headers)
                    w.writeheader()
                    w.writerows(approval_snap)
                logger.info(f"Saved {len(approval_snap)} merge decisions → "
                             f"{self.merge_approval_csv_filename}")

        except Exception as e:
            logger.error(f"CSV save error: {e}")

    # ==================================================================
    # Helpers
    # ==================================================================

    def _recv_exactly(self, sock, length: int):
        data = b''
        while len(data) < length:
            try:
                chunk = sock.recv(min(length - len(data), 8192))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                return None
            except Exception as e:
                logger.error(f"recv error: {e}")
                return None
        return data

    def _send_completely(self, sock, data: bytes):
        sent = 0
        while sent < len(data):
            n = sock.send(data[sent:])
            if n == 0:
                raise RuntimeError("Socket broken during send")
            sent += n

    def _quat_to_euler(self, qx, qy, qz, qw):
        roll  = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)))
        yaw   = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        return roll, pitch, yaw
