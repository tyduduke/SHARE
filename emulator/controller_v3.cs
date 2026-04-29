using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

/*
 * Camera Pose Controller - Fixed Version with Relative Positioning
 * 
 * Key fixes for the "stopping after one frame" issue:
 * 1. Fixed timestamp precision issue: Changed from float to double for timestamps
 *    - Large Unix timestamps (like 1749911387.520850) lose precision with float
 *    - This caused all timestamps to become identical, making duration = 0 seconds
 * 2. Changed timing logic from real-time matching to fixed timestep interpolation
 * 3. Moved frame capture outside the blocking coroutine call 
 * 4. Added frame rate limiting to prevent excessive capture attempts
 * 5. Improved error handling and debugging output
 * 6. Added progress monitoring and manual controls (P to play/pause, R to restart)
 * 7. Enhanced timestamp difference logging for debugging
 * 
 * NEW FEATURE - Relative Positioning:
 * 8. Added relative positioning system that allows each camera to follow the same
 *    trajectory pattern but starting from their current position/orientation
 * 9. When useRelativePositioning is enabled, the script calculates an offset between
 *    the current camera pose and the first pose in the data file
 * 10. This offset is then applied to all subsequent poses, making each camera follow
 *     the same movement pattern relative to their starting position
 */

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
        public bool useRelativePositioning = true; // New option to enable relative positioning

        [Header("Screenshot Settings")]
        public string saveDirectory = "CameraFrames";
        public bool saveCameraFrames = true;
        public int targetFrameRate = 30;
        
        [Header("High Resolution Capture")]
        public bool useHighResolution = false;
        public int resolutionMultiplier = 2;
        public bool useCustomResolution = false;
        public int customWidth = 1920;
        public int customHeight = 1080;

        // Internal variables
        private bool isPlaying = false;
        private int currentIndex = 0;
        private List<PoseData> poseDataList = new List<PoseData>();
        private Coroutine poseUpdateCoroutine;

        // Offset calculation variables
        private Vector3 positionOffset = Vector3.zero;
        private Quaternion rotationOffset = Quaternion.identity;
        private bool offsetCalculated = false;

        // Public properties for monitoring
        public bool IsPlaying => isPlaying;
        public int CurrentPoseIndex => currentIndex;
        public int TotalPoses => poseDataList.Count;
        public float PlaybackProgress => poseDataList.Count > 0 ? (float)currentIndex / poseDataList.Count : 0f;
        public Vector3 PositionOffset => positionOffset;
        public Quaternion RotationOffset => rotationOffset;
        public bool IsOffsetCalculated => offsetCalculated;

        [System.Serializable]
        private struct PoseData
        {
            public double timestamp;
            public Quaternion rotation;
            public Vector3 position;
        }

        private void Awake()
        {
            // Initialize camera settings
            if (targetCamera == null)
                targetCamera = GetComponent<Camera>();
            
            if (targetCamera != null)
                targetCamera.fieldOfView = fov;
        }

        private void Start()
        {
            Application.targetFrameRate = targetFrameRate;
            Debug.Log($"[{gameObject.name}] Camera Pose Controller started. Target FPS: {targetFrameRate}");
            
            if (poseDataFile != null)
            {
                Debug.Log($"[{gameObject.name}] Parsing pose data file: {poseDataFile.name}");
                ParsePoseData();
                
                if (autoStartOnPlay && poseDataList.Count > 0)
                {
                    Debug.Log($"[{gameObject.name}] Auto-starting pose playback with {poseDataList.Count} poses");
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
            // Debug key controls (optional)
            if (Input.GetKeyDown(KeyCode.P))
            {
                if (isPlaying)
                    StopPosePlayback();
                else
                    StartPosePlayback();
            }
            
            if (Input.GetKeyDown(KeyCode.R))
            {
                RestartPlayback();
            }

            if (Input.GetKeyDown(KeyCode.O))
            {
                RecalculatePoseOffset();
            }

            if (Input.GetKeyDown(KeyCode.T))
            {
                ToggleRelativePositioning();
            }
        }

        /// <summary>
        /// Parse the pose data from the txt file
        /// Expected format: timestamp qx qy qz qw px py pz
        /// Note: If your data format is different (e.g., timestamp px py pz qx qy qz qw), 
        /// adjust the parsing order below accordingly.
        /// </summary>
        private void ParsePoseData()
        {
            if (poseDataFile == null)
            {
                Debug.LogError("No pose data file assigned!");
                return;
            }

            poseDataList.Clear();
            string[] lines = poseDataFile.text.Split('\n');
            
            Debug.Log($"[{gameObject.name}] Parsing {lines.Length} lines from pose data file");
            
            foreach (string line in lines)
            {
                // Skip empty lines
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                    
                string[] data = line.Trim().Split(' ');
                
                // Ensure we have exactly 8 values: timestamp + 4 quaternion + 3 position
                if (data.Length == 8)
                {
                    try
                    {
                        // Parse according to format: timestamp qx qy qz qw px py pz
                        // IMPORTANT: Adjust this parsing order if your data format is different!
                        double timestamp = double.Parse(data[0]);
                        
                        // Quaternion components (x, y, z, w)
                        float qx = float.Parse(data[1]);
                        float qy = float.Parse(data[2]);
                        float qz = float.Parse(data[3]);
                        float qw = float.Parse(data[4]);
                        
                        // Position components (x, y, z)
                        float px = float.Parse(data[5]);
                        float py = float.Parse(data[6]);
                        float pz = float.Parse(data[7]);
                        
                        // Apply coordinate system transformation if needed
                        // Adjust these transformations based on your coordinate system
                        Quaternion rotation = new Quaternion(qx, -qy, qz, qw);
                        Vector3 position = new Vector3(-px, pz, py) + startingCoordinateOffset;
                        
                        PoseData poseData = new PoseData
                        {
                            timestamp = timestamp,
                            rotation = rotation,
                            position = position
                        };
                        
                        poseDataList.Add(poseData);
                        
                        // Debug first few entries to verify parsing
                        if (poseDataList.Count <= 3)
                        {
                            Debug.Log($"[{gameObject.name}] Parsed entry {poseDataList.Count}: " +
                                     $"timestamp={timestamp:F6}, " +
                                     $"quat=({qx:F6},{qy:F6},{qz:F6},{qw:F6}), " +
                                     $"pos=({px:F6},{py:F6},{pz:F6})");
                        }
                    }
                    catch (System.FormatException e)
                    {
                        Debug.LogWarning($"Failed to parse line: {line}. Error: {e.Message}");
                    }
                }
                else
                {
                    Debug.LogWarning($"Invalid data format in line: {line}. Expected 8 values, got {data.Length}");
                }
            }
            
            Debug.Log($"[{gameObject.name}] Parsed {poseDataList.Count} pose data entries");
            
            if (poseDataList.Count > 1)
            {
                double firstTimestamp = poseDataList[0].timestamp;
                double lastTimestamp = poseDataList[poseDataList.Count - 1].timestamp;
                double totalDuration = lastTimestamp - firstTimestamp;
                Debug.Log($"[{gameObject.name}] Pose data spans {totalDuration:F6} seconds (from {firstTimestamp:F6} to {lastTimestamp:F6})");
                
                // Check average time between poses
                double avgTimeDiff = totalDuration / (poseDataList.Count - 1);
                Debug.Log($"[{gameObject.name}] Average time between poses: {avgTimeDiff:F6} seconds");
                
                // Show first few timestamp differences for debugging
                for (int i = 0; i < Math.Min(5, poseDataList.Count - 1); i++)
                {
                    double diff = poseDataList[i + 1].timestamp - poseDataList[i].timestamp;
                    Debug.Log($"[{gameObject.name}] Time diff {i}: {diff:F6} seconds");
                }
            }
        }

        /// <summary>
        /// Start the pose playback
        /// </summary>
        public void StartPosePlayback()
        {
            if (poseDataList.Count == 0)
            {
                Debug.LogWarning($"[{gameObject.name}] No pose data to play!");
                return;
            }

            if (isPlaying)
            {
                Debug.LogWarning($"[{gameObject.name}] Pose playback already in progress!");
                return;
            }

            // Calculate offset between current camera pose and first pose in data
            if (useRelativePositioning)
            {
                CalculatePoseOffset();
            }
            else
            {
                // Reset offsets if not using relative positioning
                positionOffset = Vector3.zero;
                rotationOffset = Quaternion.identity;
                offsetCalculated = true;
            }

            isPlaying = true;
            currentIndex = 0;
            
            if (poseUpdateCoroutine != null)
                StopCoroutine(poseUpdateCoroutine);
            
            poseUpdateCoroutine = StartCoroutine(UpdateCameraPoseRoutine());
            Debug.Log($"[{gameObject.name}] Started camera pose playback with {poseDataList.Count} poses. Save frames: {saveCameraFrames}. Relative positioning: {useRelativePositioning}");
        }

        /// <summary>
        /// Calculate the offset between current camera pose and first pose in the data
        /// </summary>
        private void CalculatePoseOffset()
        {
            if (poseDataList.Count == 0 || targetCamera == null)
            {
                Debug.LogError($"[{gameObject.name}] Cannot calculate pose offset - no data or camera!");
                return;
            }

            // Get current camera pose
            Vector3 currentPosition = targetCamera.transform.position;
            Quaternion currentRotation = targetCamera.transform.rotation;

            // Get first pose from data
            PoseData firstPose = poseDataList[0];

            // Calculate position offset
            positionOffset = currentPosition - firstPose.position;

            // Calculate rotation offset 
            // This is the rotation needed to go from first pose rotation to current rotation
            rotationOffset = currentRotation * Quaternion.Inverse(firstPose.rotation);

            offsetCalculated = true;

            Debug.Log($"[{gameObject.name}] Calculated pose offset:");
            Debug.Log($"  Current position: {currentPosition}");
            Debug.Log($"  First data position: {firstPose.position}");
            Debug.Log($"  Position offset: {positionOffset}");
            Debug.Log($"  Current rotation: {currentRotation.eulerAngles}");
            Debug.Log($"  First data rotation: {firstPose.rotation.eulerAngles}");
            Debug.Log($"  Rotation offset: {rotationOffset.eulerAngles}");
        }

        /// <summary>
        /// Stop the pose playback
        /// </summary>
        public void StopPosePlayback()
        {
            isPlaying = false;
            
            if (poseUpdateCoroutine != null)
            {
                StopCoroutine(poseUpdateCoroutine);
                poseUpdateCoroutine = null;
            }
            
            Debug.Log($"[{gameObject.name}] Stopped camera pose playback");
        }

        /// <summary>
        /// Reset the pose offset calculation
        /// </summary>
        public void ResetPoseOffset()
        {
            positionOffset = Vector3.zero;
            rotationOffset = Quaternion.identity;
            offsetCalculated = false;
            Debug.Log($"[{gameObject.name}] Reset pose offset");
        }

        /// <summary>
        /// Manually recalculate the pose offset based on current camera position
        /// </summary>
        [ContextMenu("Recalculate Pose Offset")]
        public void RecalculatePoseOffset()
        {
            if (useRelativePositioning)
            {
                CalculatePoseOffset();
            }
            else
            {
                Debug.LogWarning($"[{gameObject.name}] Relative positioning is disabled");
            }
        }

        /// <summary>
        /// Coroutine that updates camera pose over time
        /// </summary>
        private IEnumerator UpdateCameraPoseRoutine()
        {
            float lastCaptureTime = 0f;
            float captureInterval = 1f / targetFrameRate; // Time between captures for target FPS
            
            Debug.Log($"Starting pose playback with {poseDataList.Count} poses");
            
            while (currentIndex < poseDataList.Count - 1 && isPlaying)
            {
                PoseData currentPose = poseDataList[currentIndex];
                PoseData nextPose = poseDataList[currentIndex + 1];

                // Calculate time difference between poses (using double precision)
                double timeDifference = nextPose.timestamp - currentPose.timestamp;
                
                // Handle edge cases
                if (timeDifference <= 0.0 || double.IsNaN(timeDifference) || double.IsInfinity(timeDifference))
                {
                    Debug.LogWarning($"Invalid time difference at index {currentIndex}: {timeDifference}. Moving to next pose.");
                    
                    // Set pose directly and move to next
                    SetCameraPose(nextPose);
                    currentIndex++;
                    continue;
                }

                // Use a fixed time step for smooth playback instead of real-time matching
                float frameTime = 1f / targetFrameRate;
                int interpolationSteps = Mathf.Max(1, Mathf.RoundToInt((float)timeDifference / frameTime));
                
                Debug.Log($"[{gameObject.name}] Processing pose {currentIndex}: time diff = {timeDifference:F6}s, steps = {interpolationSteps}");
                
                for (int step = 0; step <= interpolationSteps && isPlaying; step++)
                {
                    float t = (float)step / interpolationSteps;
                    t = Mathf.Clamp01(t);

                    // Get interpolated pose with offset applied
                    GetAdjustedInterpolatedPose(currentPose, nextPose, t, out Vector3 interpolatedPosition, out Quaternion interpolatedRotation);
                    
                    // Validate and apply pose
                    if (IsValidPose(interpolatedRotation, interpolatedPosition))
                    {
                        targetCamera.transform.rotation = interpolatedRotation;
                        targetCamera.transform.position = interpolatedPosition;
                    }

                    // Capture frame at target framerate
                    if (saveCameraFrames && Time.time - lastCaptureTime >= captureInterval)
                    {
                        // Use interpolated timestamp
                        double interpolatedTimestamp = currentPose.timestamp + (nextPose.timestamp - currentPose.timestamp) * t;
                        StartCoroutine(CaptureFrame(interpolatedTimestamp));
                        lastCaptureTime = Time.time;
                    }

                    // Wait for next frame
                    yield return new WaitForSeconds(frameTime);
                }

                currentIndex++;
                Debug.Log($"Processed pose {currentIndex}/{poseDataList.Count}");
            }
            
            // Set final pose
            if (currentIndex < poseDataList.Count)
            {
                SetCameraPose(poseDataList[currentIndex]);
                if (saveCameraFrames)
                {
                    StartCoroutine(CaptureFrame(poseDataList[currentIndex].timestamp));
                }
            }

            isPlaying = false;
            Debug.Log("Camera pose playback completed successfully");
        }

        /// <summary>
        /// Set camera pose directly
        /// </summary>
        private void SetCameraPose(PoseData pose)
        {
            PoseData adjustedPose = ApplyPoseOffset(pose);
            
            if (IsValidPose(adjustedPose.rotation, adjustedPose.position))
            {
                targetCamera.transform.rotation = adjustedPose.rotation;
                targetCamera.transform.position = adjustedPose.position;
            }
        }

        /// <summary>
        /// Apply the calculated offset to a pose
        /// </summary>
        private PoseData ApplyPoseOffset(PoseData originalPose)
        {
            if (!offsetCalculated)
            {
                Debug.LogWarning($"[{gameObject.name}] Offset not calculated yet, using original pose");
                return originalPose;
            }

            PoseData adjustedPose = new PoseData();
            adjustedPose.timestamp = originalPose.timestamp;
            
            // Apply position offset
            adjustedPose.position = originalPose.position + positionOffset;
            
            // Apply rotation offset
            adjustedPose.rotation = rotationOffset * originalPose.rotation;
            
            return adjustedPose;
        }

        /// <summary>
        /// Get an adjusted pose with offset applied (for interpolation)
        /// </summary>
        private void GetAdjustedInterpolatedPose(PoseData currentPose, PoseData nextPose, float t, out Vector3 position, out Quaternion rotation)
        {
            // First interpolate between original poses
            Vector3 interpolatedPosition = Vector3.Lerp(currentPose.position, nextPose.position, t);
            Quaternion interpolatedRotation = Quaternion.Slerp(currentPose.rotation, nextPose.rotation, t);
            
            // Then apply offset
            if (offsetCalculated)
            {
                position = interpolatedPosition + positionOffset;
                rotation = rotationOffset * interpolatedRotation;
            }
            else
            {
                position = interpolatedPosition;
                rotation = interpolatedRotation;
            }
        }

        /// <summary>
        /// Validate pose data for NaN values
        /// </summary>
        private bool IsValidPose(Quaternion rotation, Vector3 position)
        {
            bool rotationValid = !float.IsNaN(rotation.x) && !float.IsNaN(rotation.y) && 
                                !float.IsNaN(rotation.z) && !float.IsNaN(rotation.w);
            
            bool positionValid = !float.IsNaN(position.x) && !float.IsNaN(position.y) && !float.IsNaN(position.z);
            
            if (!rotationValid)
                Debug.LogError($"Invalid rotation detected: {rotation}");
            
            if (!positionValid)
                Debug.LogError($"Invalid position detected: {position}");
            
            return rotationValid && positionValid;
        }

        /// <summary>
        /// Capture a screenshot frame
        /// </summary>
        private IEnumerator CaptureFrame(double timestamp)
        {
            yield return new WaitForEndOfFrame();

            try
            {
                Texture2D screenshot = useHighResolution ? CaptureHighResolutionScreenshot() : CaptureStandardScreenshot();
                
                // Create save directory
                string directoryPath = Path.Combine(Application.persistentDataPath, saveDirectory);
                if (!Directory.Exists(directoryPath))
                {
                    Directory.CreateDirectory(directoryPath);
                    Debug.Log($"Created directory: {directoryPath}");
                }
                
                // Save with timestamp as filename (using double precision)
                string filename = $"{timestamp:F6}.png";
                string filePath = Path.Combine(directoryPath, filename);
                
                byte[] imageBytes = screenshot.EncodeToPNG();
                File.WriteAllBytes(filePath, imageBytes);
                
                // Cleanup
                DestroyImmediate(screenshot);
                
                Debug.Log($"[{gameObject.name}] Frame saved: {filename} to {directoryPath}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[{gameObject.name}] Failed to capture frame: {e.Message}");
            }
        }

        /// <summary>
        /// Capture high resolution screenshot
        /// </summary>
        private Texture2D CaptureHighResolutionScreenshot()
        {
            int width, height;
            
            if (useCustomResolution)
            {
                width = customWidth;
                height = customHeight;
            }
            else
            {
                width = Screen.width * resolutionMultiplier;
                height = Screen.height * resolutionMultiplier;
            }

            RenderTexture renderTexture = new RenderTexture(width, height, 24);
            renderTexture.antiAliasing = 8;
            
            // Store original settings
            RenderTexture originalTarget = targetCamera.targetTexture;
            
            // Render to high resolution texture
            targetCamera.targetTexture = renderTexture;
            targetCamera.Render();
            
            // Read pixels
            RenderTexture.active = renderTexture;
            Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();
            
            // Restore settings
            targetCamera.targetTexture = originalTarget;
            RenderTexture.active = null;
            DestroyImmediate(renderTexture);
            
            return screenshot;
        }

        /// <summary>
        /// Capture standard resolution screenshot
        /// </summary>
        private Texture2D CaptureStandardScreenshot()
        {
            int width = Screen.width;
            int height = Screen.height;
            
            Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshot.Apply();
            
            return screenshot;
        }

        /// <summary>
        /// Public method to restart playback
        /// </summary>
        public void RestartPlayback()
        {
            StopPosePlayback();
            
            // Reset offset calculation so it recalculates from current position
            if (useRelativePositioning)
            {
                ResetPoseOffset();
            }
            
            StartPosePlayback();
        }

        // Editor/Debug methods
        [ContextMenu("Parse Pose Data")]
        public void ParsePoseDataFromEditor()
        {
            ParsePoseData();
        }

        [ContextMenu("Start Playback")]
        public void StartPlaybackFromEditor()
        {
            StartPosePlayback();
        }

        [ContextMenu("Stop Playback")]
        public void StopPlaybackFromEditor()
        {
            StopPosePlayback();
        }

        [ContextMenu("Recalculate Offset")]
        public void RecalculateOffsetFromEditor()
        {
            RecalculatePoseOffset();
        }

        [ContextMenu("Reset Offset")]
        public void ResetOffsetFromEditor()
        {
            ResetPoseOffset();
        }

        [ContextMenu("Toggle Relative Positioning")]
        public void ToggleRelativePositioning()
        {
            useRelativePositioning = !useRelativePositioning;
            Debug.Log($"[{gameObject.name}] Relative positioning: {useRelativePositioning}");
            
            if (useRelativePositioning)
            {
                ResetPoseOffset();
            }
        }
    }
}