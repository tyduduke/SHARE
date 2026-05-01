using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;


namespace CameraPoseController
{
    public class CameraPoseController : MonoBehaviour
    {
        [Header("Camera Settings")]
        public Camera targetCamera;
        public float fov = 60f;

        [Header("Pose Data")]
        public TextAsset poseDataFile;
        public Vector3 startingCoordinateOffset = Vector3.zero;
        public bool autoStartOnPlay = true;

        // Paper §5: for dataset generation the camera must follow the exact
        // recorded trajectory.  Only set this true in the live AR deployment
        // when cameras start from different Unity scene positions but must
        // replay the same movement pattern.
        public bool useRelativePositioning = false;

        [Header("Screenshot Settings")]
        // Leave empty to auto-name from gameObject.name (recommended).
        // This ensures each agent saves to its own directory with no manual
        // configuration per camera.
        public string saveDirectory = "";
        public bool saveCameraFrames = true;
        public int targetFrameRate = 30;    // 30 fps — matches field study recording rate

        [Header("High Resolution Capture")]
        public bool useHighResolution = false;
        public int resolutionMultiplier = 2;
        public bool useCustomResolution = false;
        public int customWidth  = 1920;
        public int customHeight = 1080;

        // Internal state
        private bool isPlaying = false;
        private int currentIndex = 0;
        private List<PoseData> poseDataList = new List<PoseData>();
        private Coroutine poseUpdateCoroutine;

        // Relative-positioning offset variables (used only when useRelativePositioning=true)
        private Vector3    positionOffset  = Vector3.zero;
        private Quaternion rotationOffset  = Quaternion.identity;
        private bool       offsetCalculated = false;

        // Public monitoring properties
        public bool  IsPlaying         => isPlaying;
        public int   CurrentPoseIndex  => currentIndex;
        public int   TotalPoses        => poseDataList.Count;
        public float PlaybackProgress  => poseDataList.Count > 0
                                          ? (float)currentIndex / poseDataList.Count : 0f;
        public Vector3    PositionOffset     => positionOffset;
        public Quaternion RotationOffset     => rotationOffset;
        public bool       IsOffsetCalculated => offsetCalculated;

        [System.Serializable]
        private struct PoseData
        {
            public double    timestamp;
            public Quaternion rotation;
            public Vector3    position;
        }

        // ==================================================================
        // Unity lifecycle
        // ==================================================================

        private void Awake()
        {
            if (targetCamera == null)
                targetCamera = GetComponent<Camera>();
            if (targetCamera != null)
                targetCamera.fieldOfView = fov;
        }

        private void Start()
        {
            Application.targetFrameRate = targetFrameRate;

            // Auto-name save directory from the GameObject name so that each
            // agent camera (User1, User2, Robot) saves to its own folder.
            if (string.IsNullOrEmpty(saveDirectory))
            {
                saveDirectory = gameObject.name;
                Debug.Log($"[{gameObject.name}] Save directory set to: {saveDirectory}");
            }

            Debug.Log($"[{gameObject.name}] Camera Pose Controller started. "
                    + $"Target FPS: {targetFrameRate}, "
                    + $"Relative positioning: {useRelativePositioning}");

            if (poseDataFile != null)
            {
                Debug.Log($"[{gameObject.name}] Parsing pose data file: {poseDataFile.name}");
                ParsePoseData();

                if (autoStartOnPlay && poseDataList.Count > 0)
                {
                    Debug.Log($"[{gameObject.name}] Auto-starting with {poseDataList.Count} poses");
                    StartPosePlayback();
                }
                else if (poseDataList.Count == 0)
                {
                    Debug.LogError($"[{gameObject.name}] No valid pose data found in file!");
                }
            }
            else
            {
                Debug.LogWarning($"[{gameObject.name}] No pose data file assigned!");
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.P))
            {
                if (isPlaying) StopPosePlayback();
                else           StartPosePlayback();
            }
            if (Input.GetKeyDown(KeyCode.R)) RestartPlayback();
            if (Input.GetKeyDown(KeyCode.O)) RecalculatePoseOffset();
            if (Input.GetKeyDown(KeyCode.T)) ToggleRelativePositioning();
        }

        // ==================================================================
        // Pose data parsing
        // ==================================================================

        /// <summary>
        /// Parse the pose data .txt file.
        ///
        /// Expected format: space-separated, one pose per line:
        ///     timestamp  qx  qy  qz  qw  px  py  pz
        ///
        /// Coordinate conversion (applied per line):
        ///   Input:  ROS/ENU right-handed (X-forward, Y-left, Z-up)
        ///           — this is the convention of the Vicon ground-truth files
        ///             used in the SHARE field study.
        ///   Output: Unity left-handed    (X-right, Y-up, Z-forward)
        ///   Transform: rotation = new Quaternion(qx, -qy, qz, qw)
        ///              position = new Vector3(-px, pz, py)
        ///
        /// If your source data uses a different frame, adjust the two lines
        /// marked "COORDINATE CONVERSION" below.
        ///
        /// Note: timestamps are parsed as double (not float) because large Unix
        /// timestamps (e.g. 1749911387.520850) exceed float precision and would
        /// collapse all poses to the same timestamp, producing zero-duration
        /// intervals and stopping playback after the first frame.
        /// </summary>
        private void ParsePoseData()
        {
            if (poseDataFile == null)
            {
                Debug.LogError($"[{gameObject.name}] No pose data file assigned!");
                return;
            }

            poseDataList.Clear();
            string[] lines = poseDataFile.text.Split('\n');
            Debug.Log($"[{gameObject.name}] Parsing {lines.Length} lines");

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] data = line.Trim().Split(' ');

                if (data.Length == 8)
                {
                    try
                    {
                        double ts = double.Parse(data[0]);

                        float qx = float.Parse(data[1]);
                        float qy = float.Parse(data[2]);
                        float qz = float.Parse(data[3]);
                        float qw = float.Parse(data[4]);
                        float px = float.Parse(data[5]);
                        float py = float.Parse(data[6]);
                        float pz = float.Parse(data[7]);

                        // COORDINATE CONVERSION: ROS/ENU → Unity left-handed
                        Quaternion rotation = new Quaternion(qx, -qy, qz, qw);
                        Vector3    position = new Vector3(-px, pz, py) + startingCoordinateOffset;

                        poseDataList.Add(new PoseData {
                            timestamp = ts,
                            rotation  = rotation,
                            position  = position,
                        });

                        if (poseDataList.Count <= 3)
                        {
                            Debug.Log($"[{gameObject.name}] Entry {poseDataList.Count}: "
                                    + $"ts={ts:F6}, "
                                    + $"q=({qx:F4},{qy:F4},{qz:F4},{qw:F4}), "
                                    + $"p=({px:F4},{py:F4},{pz:F4})");
                        }
                    }
                    catch (FormatException e)
                    {
                        Debug.LogWarning($"[{gameObject.name}] Parse error: {line} — {e.Message}");
                    }
                }
                else
                {
                    Debug.LogWarning($"[{gameObject.name}] Expected 8 values, got {data.Length}: {line}");
                }
            }

            Debug.Log($"[{gameObject.name}] Parsed {poseDataList.Count} poses");

            if (poseDataList.Count > 1)
            {
                double t0   = poseDataList[0].timestamp;
                double tN   = poseDataList[poseDataList.Count - 1].timestamp;
                double dur  = tN - t0;
                double avg  = dur / (poseDataList.Count - 1);
                Debug.Log($"[{gameObject.name}] Duration: {dur:F3}s  avg Δt: {avg:F6}s");

                for (int i = 0; i < Math.Min(5, poseDataList.Count - 1); i++)
                {
                    double diff = poseDataList[i + 1].timestamp - poseDataList[i].timestamp;
                    Debug.Log($"[{gameObject.name}] Δt[{i}] = {diff:F6}s");
                }
            }
        }

        // ==================================================================
        // Playback control
        // ==================================================================

        public void StartPosePlayback()
        {
            if (poseDataList.Count == 0)
            {
                Debug.LogWarning($"[{gameObject.name}] No pose data to play!");
                return;
            }
            if (isPlaying)
            {
                Debug.LogWarning($"[{gameObject.name}] Playback already running!");
                return;
            }

            if (useRelativePositioning)
                CalculatePoseOffset();
            else
            {
                positionOffset   = Vector3.zero;
                rotationOffset   = Quaternion.identity;
                offsetCalculated = true;
            }

            isPlaying     = true;
            currentIndex  = 0;

            if (poseUpdateCoroutine != null)
                StopCoroutine(poseUpdateCoroutine);

            poseUpdateCoroutine = StartCoroutine(UpdateCameraPoseRoutine());
            Debug.Log($"[{gameObject.name}] Playback started — {poseDataList.Count} poses, "
                    + $"saveCameraFrames={saveCameraFrames}, "
                    + $"useRelativePositioning={useRelativePositioning}");
        }

        public void StopPosePlayback()
        {
            isPlaying = false;
            if (poseUpdateCoroutine != null)
            {
                StopCoroutine(poseUpdateCoroutine);
                poseUpdateCoroutine = null;
            }
            Debug.Log($"[{gameObject.name}] Playback stopped");
        }

        public void RestartPlayback()
        {
            StopPosePlayback();
            if (useRelativePositioning) ResetPoseOffset();
            StartPosePlayback();
        }

        // ==================================================================
        // Main coroutine
        // ==================================================================

        /// <summary>
        /// Drives the camera along the recorded trajectory.
        ///
        /// For each consecutive pose pair (i, i+1):
        ///   1. Compute interpolationSteps = round(Δt / frameTime).
        ///   2. Loop step = 0 .. interpolationSteps-1  (EXCLUSIVE upper bound
        ///      to avoid duplicating the boundary pose at the start of the
        ///      next interval).
        ///   3. Set camera transform from interpolated pose.
        ///   4. Capture exactly ONE frame per step — no wall-clock gating.
        ///      The coroutine waits frameTime seconds between steps, so the
        ///      output frame rate exactly matches targetFrameRate.
        /// </summary>
        private IEnumerator UpdateCameraPoseRoutine()
        {
            float frameTime = 1f / targetFrameRate;

            Debug.Log($"[{gameObject.name}] Coroutine started — {poseDataList.Count} poses, "
                    + $"frameTime={frameTime:F4}s");

            while (currentIndex < poseDataList.Count - 1 && isPlaying)
            {
                PoseData cur  = poseDataList[currentIndex];
                PoseData next = poseDataList[currentIndex + 1];

                double dt = next.timestamp - cur.timestamp;

                // Skip degenerate intervals (zero or negative Δt)
                if (dt <= 0.0 || double.IsNaN(dt) || double.IsInfinity(dt))
                {
                    Debug.LogWarning($"[{gameObject.name}] Invalid Δt={dt:F6} at index "
                                   + $"{currentIndex}; skipping to next pose");
                    SetCameraPose(next);
                    currentIndex++;
                    continue;
                }

                // Number of frames to emit between these two keyframes.
                int steps = Mathf.Max(1, Mathf.RoundToInt((float)dt / frameTime));

                // Throttled log (every 100 poses to avoid overhead)
                if (currentIndex % 100 == 0)
                    Debug.Log($"[{gameObject.name}] Pose {currentIndex}/{poseDataList.Count - 1}  "
                            + $"Δt={dt:F4}s  steps={steps}");

                // EXCLUSIVE upper bound: step 0 .. steps-1
                // At step=0:     t=0 → current pose
                // At step=steps: t=1 → next pose  (emitted as step=0 of the NEXT interval)
                // This prevents one duplicated frame at every keyframe boundary.
                for (int step = 0; step < steps && isPlaying; step++)
                {
                    float t = (float)step / steps;
                    t = Mathf.Clamp01(t);

                    GetAdjustedInterpolatedPose(cur, next, t,
                                               out Vector3    pos,
                                               out Quaternion rot);

                    if (IsValidPose(rot, pos))
                    {
                        targetCamera.transform.position = pos;
                        targetCamera.transform.rotation = rot;
                    }

                    // Capture one frame per interpolation step.
                    // No wall-clock gating — the coroutine already advances at
                    // exactly frameTime seconds per step, so this produces
                    // exactly targetFrameRate frames per second.
                    if (saveCameraFrames)
                    {
                        double interpTs = cur.timestamp + dt * t;
                        StartCoroutine(CaptureFrame(interpTs));
                    }

                    yield return new WaitForSeconds(frameTime);
                }

                currentIndex++;
            }

            // Emit the final pose of the trajectory
            if (currentIndex < poseDataList.Count)
            {
                SetCameraPose(poseDataList[currentIndex]);
                if (saveCameraFrames)
                    StartCoroutine(CaptureFrame(poseDataList[currentIndex].timestamp));
            }

            isPlaying = false;
            Debug.Log($"[{gameObject.name}] Playback complete");
        }

        // ==================================================================
        // Frame capture
        // ==================================================================

        /// <summary>
        /// Capture and save one PNG frame after the current render completes.
        /// </summary>
        private IEnumerator CaptureFrame(double timestamp)
        {
            yield return new WaitForEndOfFrame();

            try
            {
                Texture2D screenshot = useHighResolution
                    ? CaptureHighResolutionScreenshot()
                    : CaptureStandardScreenshot();

                string directoryPath = Path.Combine(
                    Application.persistentDataPath, saveDirectory);

                if (!Directory.Exists(directoryPath))
                {
                    Directory.CreateDirectory(directoryPath);
                    Debug.Log($"[{gameObject.name}] Created directory: {directoryPath}");
                }

                // Filename = timestamp to 6 decimal places, matching the naming
                // convention expected by pyslam_integration.py's process_sequence().
                string filename = $"{timestamp:F6}.png";
                string filePath = Path.Combine(directoryPath, filename);

                byte[] imageBytes = screenshot.EncodeToPNG();
                File.WriteAllBytes(filePath, imageBytes);
                DestroyImmediate(screenshot);
            }
            catch (Exception e)
            {
                Debug.LogError($"[{gameObject.name}] Frame capture failed: {e.Message}");
            }
        }

        /// <summary>
        /// Capture at screen resolution by rendering explicitly from targetCamera.
        ///
        /// This method mirrors the high-resolution path: both paths render
        /// from targetCamera into a RenderTexture so the captured image is
        /// always the correct camera view, regardless of whether the Unity
        /// editor window is focused or other cameras are active.
        ///
        /// The original ReadPixels-from-display approach was replaced because
        /// it reads from the active display framebuffer, not from targetCamera,
        /// which produces wrong images when the window is minimised or when
        /// multiple cameras are rendering to the same display.
        /// </summary>
        private Texture2D CaptureStandardScreenshot()
        {
            int width  = Screen.width;
            int height = Screen.height;

            RenderTexture  rt             = new RenderTexture(width, height, 24);
            RenderTexture  originalTarget = targetCamera.targetTexture;

            targetCamera.targetTexture = rt;
            targetCamera.Render();

            RenderTexture.active = rt;
            Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();

            targetCamera.targetTexture = originalTarget;
            RenderTexture.active       = null;
            DestroyImmediate(rt);

            return screenshot;
        }

        /// <summary>
        /// Capture at high or custom resolution.
        /// </summary>
        private Texture2D CaptureHighResolutionScreenshot()
        {
            int width, height;
            if (useCustomResolution)
            {
                width  = customWidth;
                height = customHeight;
            }
            else
            {
                width  = Screen.width  * resolutionMultiplier;
                height = Screen.height * resolutionMultiplier;
            }

            RenderTexture rt = new RenderTexture(width, height, 24);
            rt.antiAliasing  = 8;

            RenderTexture originalTarget = targetCamera.targetTexture;

            targetCamera.targetTexture = rt;
            targetCamera.Render();

            RenderTexture.active = rt;
            Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();

            targetCamera.targetTexture = originalTarget;
            RenderTexture.active       = null;
            DestroyImmediate(rt);

            return screenshot;
        }

        // ==================================================================
        // Pose helpers
        // ==================================================================

        private void SetCameraPose(PoseData pose)
        {
            PoseData adj = ApplyPoseOffset(pose);
            if (IsValidPose(adj.rotation, adj.position))
            {
                targetCamera.transform.rotation = adj.rotation;
                targetCamera.transform.position = adj.position;
            }
        }

        private void GetAdjustedInterpolatedPose(PoseData a, PoseData b, float t,
                                                  out Vector3    position,
                                                  out Quaternion rotation)
        {
            Vector3    interPos = Vector3.Lerp(a.position, b.position, t);
            Quaternion interRot = Quaternion.Slerp(a.rotation, b.rotation, t);

            if (offsetCalculated)
            {
                position = interPos + positionOffset;
                rotation = rotationOffset * interRot;
            }
            else
            {
                position = interPos;
                rotation = interRot;
            }
        }

        private PoseData ApplyPoseOffset(PoseData pose)
        {
            if (!offsetCalculated)
            {
                Debug.LogWarning($"[{gameObject.name}] Offset not calculated; using raw pose");
                return pose;
            }
            return new PoseData {
                timestamp = pose.timestamp,
                position  = pose.position + positionOffset,
                rotation  = rotationOffset * pose.rotation,
            };
        }

        private bool IsValidPose(Quaternion rotation, Vector3 position)
        {
            bool rotOk = !float.IsNaN(rotation.x) && !float.IsNaN(rotation.y)
                      && !float.IsNaN(rotation.z) && !float.IsNaN(rotation.w);
            bool posOk = !float.IsNaN(position.x) && !float.IsNaN(position.y)
                      && !float.IsNaN(position.z);

            if (!rotOk) Debug.LogError($"[{gameObject.name}] NaN rotation: {rotation}");
            if (!posOk) Debug.LogError($"[{gameObject.name}] NaN position: {position}");
            return rotOk && posOk;
        }

        // ==================================================================
        // Relative positioning (live deployment only, not for dataset gen)
        // ==================================================================

        private void CalculatePoseOffset()
        {
            if (poseDataList.Count == 0 || targetCamera == null)
            {
                Debug.LogError($"[{gameObject.name}] Cannot calculate offset — no data or camera");
                return;
            }

            Vector3    curPos = targetCamera.transform.position;
            Quaternion curRot = targetCamera.transform.rotation;
            PoseData   first  = poseDataList[0];

            positionOffset   = curPos - first.position;
            rotationOffset   = curRot * Quaternion.Inverse(first.rotation);
            offsetCalculated = true;

            Debug.Log($"[{gameObject.name}] Pose offset calculated:");
            Debug.Log($"  position offset: {positionOffset}");
            Debug.Log($"  rotation offset: {rotationOffset.eulerAngles}");
        }

        public void ResetPoseOffset()
        {
            positionOffset   = Vector3.zero;
            rotationOffset   = Quaternion.identity;
            offsetCalculated = false;
            Debug.Log($"[{gameObject.name}] Pose offset reset");
        }

        [ContextMenu("Recalculate Pose Offset")]
        public void RecalculatePoseOffset()
        {
            if (useRelativePositioning) CalculatePoseOffset();
            else Debug.LogWarning($"[{gameObject.name}] Relative positioning is disabled");
        }

        [ContextMenu("Toggle Relative Positioning")]
        public void ToggleRelativePositioning()
        {
            useRelativePositioning = !useRelativePositioning;
            Debug.Log($"[{gameObject.name}] Relative positioning: {useRelativePositioning}");
            if (useRelativePositioning) ResetPoseOffset();
        }

        // ==================================================================
        // Editor context menu shortcuts
        // ==================================================================

        [ContextMenu("Parse Pose Data")]
        public void ParsePoseDataFromEditor() => ParsePoseData();

        [ContextMenu("Start Playback")]
        public void StartPlaybackFromEditor() => StartPosePlayback();

        [ContextMenu("Stop Playback")]
        public void StopPlaybackFromEditor() => StopPosePlayback();

        [ContextMenu("Restart Playback")]
        public void RestartPlaybackFromEditor() => RestartPlayback();

        [ContextMenu("Reset Offset")]
        public void ResetOffsetFromEditor() => ResetPoseOffset();
    }
}
