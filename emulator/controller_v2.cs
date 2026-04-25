using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;

namespace MultiController
{
    public class HumanOneController : MonoBehaviour
    {
        private Rigidbody rb;

        #region Camera Variables
        public Camera playerCamera;
        public float fov = 60f;
        public bool invertCamera = false;
        public bool cameraCanMove = true;
        public float mouseSensitivity = 2f;
        public float maxLookAngle = 50f;
        public bool lockCursor = true;

        // Internal Variables
        private float yaw = 0.0f;
        private float pitch = 0.0f;
        
        // Add smoothing for camera movement
        public float cameraSmoothing = 5f;
        private Vector2 currentMouseDelta;
        private Vector2 currentMouseVelocity;
        #endregion

        #region Camera Translation 
        public bool useControlledTranslations = false;
        public TextAsset txtFile;
        public string pathToSaveData;
        public bool saveCameraFrames = true;
        public Vector3 startingCoordinateOffset = Vector3.zero;

        private float t_startTime;
        private float t_elapsedTime;
        private bool isPlaying = false;
        private int index;

        private struct CameraData
        {
            public float timestamp;
            public Quaternion quat;
            public Vector3 vec;
        }

        private List<CameraData> cameraDataList = new List<CameraData>();
        #endregion

        #region Movement Variables
        public bool playerCanMove = true;
        public float walkSpeed = 5f;
        public float maxVelocityChange = 10f;

        // Human Specific - WASD Keys
        public KeyCode forwardKey = KeyCode.W;
        public KeyCode backwardKey = KeyCode.S;
        public KeyCode leftKey = KeyCode.A;
        public KeyCode rightKey = KeyCode.D;

        // Internal Variables
        private bool isWalking = false;
        
        // Add input buffering for better key detection
        private Dictionary<KeyCode, bool> keyStates = new Dictionary<KeyCode, bool>();

        #region Sprint
        public bool enableSprint = true;
        public bool unlimitedSprint = false;
        public KeyCode sprintKey = KeyCode.LeftShift;
        public float sprintSpeed = 7f;
        public float sprintDuration = 5f;
        public float sprintCooldown = .5f;

        // Internal Variables
        private bool isSprinting = false;
        private float sprintRemaining;
        private bool isSprintCooldown = false;
        private float sprintCooldownReset;
        #endregion

        #region Jump
        public bool enableJump = true;
        public KeyCode jumpKey = KeyCode.Space;
        public float jumpPower = 5f;

        // Internal Variables
        private bool isGrounded = false;
        #endregion
        #endregion

        #region Head Bob
        public bool enableHeadBob = true;
        public Transform joint;
        public float bobSpeed = 10f;
        public Vector3 bobAmount = new Vector3(.15f, .05f, 0f);

        // Internal Variables
        private Vector3 jointOriginalPos;
        private float timer = 0;
        #endregion

        #region Screenshot Capture
        public int targetFrameRate = 30;
        private double startTime = 1680000000.00001;
        #endregion

        private void Awake()
        {
            rb = GetComponent<Rigidbody>();

            // Set internal variables
            playerCamera.fieldOfView = fov;
            jointOriginalPos = joint.localPosition;

            if (!unlimitedSprint)
            {
                sprintRemaining = sprintDuration;
                sprintCooldownReset = sprintCooldown;
            }
            
            // Initialize key states
            InitializeKeyStates();
        }

        void Start()
        {
            if (lockCursor)
            {
                Cursor.lockState = CursorLockMode.Locked;
            }

            Application.targetFrameRate = targetFrameRate;

            cameraCanMove = !useControlledTranslations; 
            playerCanMove = !useControlledTranslations;

            if (useControlledTranslations)
            {
                ParseTXT();
            }
            
            // Make this controller's camera layer unique 
            playerCamera.cullingMask &= ~(1 << LayerMask.NameToLayer("Human2") | 1 << LayerMask.NameToLayer("Robot"));
        }

        private void InitializeKeyStates()
        {
            keyStates[forwardKey] = false;
            keyStates[backwardKey] = false;
            keyStates[leftKey] = false;
            keyStates[rightKey] = false;
            keyStates[sprintKey] = false;
            keyStates[jumpKey] = false;
        }

        private void UpdateKeyStates()
        {
            keyStates[forwardKey] = Input.GetKey(forwardKey);
            keyStates[backwardKey] = Input.GetKey(backwardKey);
            keyStates[leftKey] = Input.GetKey(leftKey);
            keyStates[rightKey] = Input.GetKey(rightKey);
            keyStates[sprintKey] = Input.GetKey(sprintKey);
            keyStates[jumpKey] = Input.GetKeyDown(jumpKey); // Use GetKeyDown for jump
        }

        private void Update()
        {
            // Update input states first
            UpdateKeyStates();
            StartTranslations();
            //if (Input.GetKeyDown(KeyCode.Alpha1))
            //{
            //    Debug.Log("Starting Camera Translations for Human 1");
            //    StartTranslations();
            //}

            // Control camera movement with smoothing
            if (cameraCanMove)
            {
                Vector2 targetMouseDelta = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));

                // Smooth the mouse input to prevent jittery movement
                currentMouseDelta = Vector2.SmoothDamp(currentMouseDelta, targetMouseDelta, ref currentMouseVelocity, 1f / cameraSmoothing);

                yaw += currentMouseDelta.x * mouseSensitivity;

                if (!invertCamera)
                {
                    pitch -= mouseSensitivity * currentMouseDelta.y;
                }
                else
                {
                    pitch += mouseSensitivity * currentMouseDelta.y;
                }

                // Clamp pitch between lookAngle
                pitch = Mathf.Clamp(pitch, -maxLookAngle, maxLookAngle);

                transform.localEulerAngles = new Vector3(0, yaw, 0);
                playerCamera.transform.localEulerAngles = new Vector3(pitch, 0, 0);
            }

            #region Sprint
            if (enableSprint)
            {
                if (isSprinting)
                {
                    if (!unlimitedSprint)
                    {
                        sprintRemaining -= Time.deltaTime;
                        if (sprintRemaining <= 0)
                        {
                            isSprinting = false;
                            isSprintCooldown = true;
                        }
                    }
                }
                else
                {
                    sprintRemaining = Mathf.Clamp(sprintRemaining + Time.deltaTime, 0, sprintDuration);
                }

                if (isSprintCooldown)
                {
                    sprintCooldown -= Time.deltaTime;
                    if (sprintCooldown <= 0)
                    {
                        isSprintCooldown = false;
                        sprintCooldown = sprintCooldownReset; // Reset cooldown here
                    }
                }
            }
            #endregion

            #region Jump
            if (enableJump && keyStates[jumpKey] && isGrounded)
            {
                Jump();
            }
            #endregion

            CheckGround();

            if (enableHeadBob)
            {
                HeadBob();
            }

            if (saveCameraFrames)
            {
                StartCoroutine(CaptureCameraFrame());
            }
        }

        void FixedUpdate()
        {
            #region Movement
            if (playerCanMove)
            {
                // Calculate how fast we should be moving
                Vector3 targetVelocity = Vector3.zero;
                
                // Using custom key mappings with improved input detection
                if (keyStates[forwardKey])
                    targetVelocity += Vector3.forward;
                if (keyStates[backwardKey])
                    targetVelocity += Vector3.back;
                if (keyStates[leftKey])
                    targetVelocity += Vector3.left;
                if (keyStates[rightKey])
                    targetVelocity += Vector3.right;

                // Normalize diagonal movement
                if (targetVelocity.magnitude > 1f)
                {
                    targetVelocity.Normalize();
                }

                // Checks if player is walking and isGrounded
                if ((targetVelocity.x != 0 || targetVelocity.z != 0) && isGrounded)
                {
                    isWalking = true;
                }
                else
                {
                    isWalking = false;
                }

                // All movement calculations while sprint is active
                if (enableSprint && keyStates[sprintKey] && sprintRemaining > 0f && !isSprintCooldown)
                {
                    targetVelocity = transform.TransformDirection(targetVelocity) * sprintSpeed;

                    // Apply a force that attempts to reach our target velocity
                    Vector3 velocity = rb.velocity;
                    Vector3 velocityChange = (targetVelocity - velocity);
                    velocityChange.x = Mathf.Clamp(velocityChange.x, -maxVelocityChange, maxVelocityChange);
                    velocityChange.z = Mathf.Clamp(velocityChange.z, -maxVelocityChange, maxVelocityChange);
                    velocityChange.y = 0;

                    // Player is only moving when velocity change != 0
                    if (velocityChange.x != 0 || velocityChange.z != 0)
                    {
                        isSprinting = true;
                    }

                    rb.AddForce(velocityChange, ForceMode.VelocityChange);
                }
                // All movement calculations while walking
                else
                {
                    isSprinting = false;

                    targetVelocity = transform.TransformDirection(targetVelocity) * walkSpeed;

                    // Apply a force that attempts to reach our target velocity
                    Vector3 velocity = rb.velocity;
                    Vector3 velocityChange = (targetVelocity - velocity);
                    velocityChange.x = Mathf.Clamp(velocityChange.x, -maxVelocityChange, maxVelocityChange);
                    velocityChange.z = Mathf.Clamp(velocityChange.z, -maxVelocityChange, maxVelocityChange);
                    velocityChange.y = 0;

                    rb.AddForce(velocityChange, ForceMode.VelocityChange);
                }
            }
            #endregion
        }

        
        private void ParseTXT()
        {
            string[] lines = txtFile.text.Split('\n');
            
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
                        // Parse the data according to TXT format:
                        // timestamp, qx, qy, qz, qw, px, py, pz
                        float timestamp = float.Parse(data[0]);
                        
                        // Quaternion components (x, y, z, w)
                        float qx = float.Parse(data[1]);
                        float qy = float.Parse(data[2]);
                        float qz = float.Parse(data[3]);
                        float qw = float.Parse(data[4]);
                        
                        // Position components (x, y, z)
                        float px = float.Parse(data[5]);
                        float py = float.Parse(data[6]);
                        float pz = float.Parse(data[7]);
                        
                        // Apply coordinate system transformation and add offset
                        // Original CSV transformation: quat = (x, -y, z, w), vec = (-x, z, y)
                        // For TXT: assuming same coordinate system transformation is needed
                        Quaternion transformedQuat = new Quaternion(qx, -qy, qz, qw);
                        Vector3 transformedVec = new Vector3(-px, pz, py) + startingCoordinateOffset;
                        
                        CameraData cameraData = new CameraData
                        {
                            timestamp = timestamp,
                            quat = transformedQuat,
                            vec = transformedVec
                        };
                        
                        cameraDataList.Add(cameraData);
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
            
            Debug.Log($"Parsed {cameraDataList.Count} camera data entries from TXT file");
        }
        private void StartTranslations()
        {
            isPlaying = true;
            t_startTime = Time.time;
            t_elapsedTime = 0f;
            index = 0;

            StartCoroutine(UpdateCameraPositionRoutine());
        }

        IEnumerator UpdateCameraPositionRoutine()
        {
            while (index < cameraDataList.Count - 1)
            {
                CameraData startData = cameraDataList[index];
                CameraData endData = cameraDataList[index + 1];

                float t_startTime = Time.time;
                float t_elapsedTime = 0f;
                
                // Calculate time difference and handle edge cases
                float timeDifference = endData.timestamp - startData.timestamp;
                
                // If timestamps are identical or invalid, skip to next frame immediately
                if (timeDifference <= 0f || float.IsNaN(timeDifference) || float.IsInfinity(timeDifference))
                {
                    Debug.LogWarning($"Invalid time difference at index {index}: {timeDifference}. Skipping frame.");
                    
                    // Set final position/rotation directly
                    playerCamera.transform.rotation = endData.quat;
                    playerCamera.transform.position = endData.vec;
                    
                    index++;
                    continue;
                }

                while (t_elapsedTime < 1f)
                {
                    t_elapsedTime = (Time.time - t_startTime) / timeDifference;
                    
                    // Clamp t_elapsedTime to prevent overshooting
                    t_elapsedTime = Mathf.Clamp01(t_elapsedTime);

                    // Validate interpolation values before applying
                    Quaternion targetRotation = Quaternion.Slerp(startData.quat, endData.quat, t_elapsedTime);
                    Vector3 targetPosition = Vector3.Lerp(startData.vec, endData.vec, t_elapsedTime);
                    
                    // Check for NaN values before assignment
                    if (!float.IsNaN(targetRotation.x) && !float.IsNaN(targetRotation.y) && 
                        !float.IsNaN(targetRotation.z) && !float.IsNaN(targetRotation.w))
                    {
                        playerCamera.transform.rotation = targetRotation;
                    }
                    else
                    {
                        Debug.LogError($"NaN detected in rotation at index {index}");
                    }
                    
                    if (!float.IsNaN(targetPosition.x) && !float.IsNaN(targetPosition.y) && !float.IsNaN(targetPosition.z))
                    {
                        playerCamera.transform.position = targetPosition;
                    }
                    else
                    {
                        Debug.LogError($"NaN detected in position at index {index}: {targetPosition}");
                    }

                    Debug.Log("Interpolation Done: " + pathToSaveData);

                    yield return null; // Wait for the next frame
                }

                index++;
            }
            Debug.LogFormat("Done with camera translations for Human 1!");
            //isPlaying = false;
        }

        [Header("High Resolution Capture Settings")]
        public bool useHighRes = false;
        public int captureResMultiplier = 2;
        public int customCaptureWidth = 1920;
        public int cuestomCaptureHight = 1080;
        public bool useCustomRes = false;

        IEnumerator CaptureCameraFrame()
        {
            yield return new WaitForEndOfFrame();
            Debug.Log($"Judgement Results: {useControlledTranslations},{isPlaying},{saveCameraFrames}");
            if (useControlledTranslations && isPlaying && saveCameraFrames)
            {
                // Capture screenshot
                Texture2D imgTexture = useHighRes ? CaptureHighResScreenshot() : CaptureScreenshot();

                // Save data
                string dirPath = pathToSaveData + "/frames/";
                Directory.CreateDirectory(dirPath);

                byte[] imgBytes = imgTexture.EncodeToPNG();
                File.WriteAllBytes(dirPath + index.ToString() + ".png", imgBytes);
                Debug.Log(pathToSaveData);
                Debug.Log($"Frame Saved!");

                Destroy(imgTexture);
                //index++;
            }
        }

        private Texture2D CaptureHighResScreenshot()
        {
            int width = Screen.width * captureResMultiplier, height = Screen.height * captureResMultiplier;
            if (useCustomRes)
            {
                width = customCaptureWidth;
                height = cuestomCaptureHight;
            }
            RenderTexture renderTexture = new RenderTexture(width, height, 24);
            renderTexture.antiAliasing = 8; // Maximum anti-aliasing
            
            // Store original camera settings
            RenderTexture originalTargetTexture = playerCamera.targetTexture;
            int originalPixelWidth = playerCamera.pixelWidth;
            int originalPixelHeight = playerCamera.pixelHeight;
            
            // Set camera to render to our high-res RenderTexture
            playerCamera.targetTexture = renderTexture;
            
            // Render the camera
            playerCamera.Render();
            
            // Create Texture2D and read pixels from RenderTexture
            RenderTexture.active = renderTexture;
            Texture2D screenshotTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            screenshotTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            screenshotTexture.Apply();
            
            // Cleanup: restore original camera settings
            playerCamera.targetTexture = originalTargetTexture;
            RenderTexture.active = null;
            
            // Destroy the temporary RenderTexture
            DestroyImmediate(renderTexture);
            
            return screenshotTexture;
        }

        private Texture2D CaptureScreenshot()
        {
            int width = Screen.width;
            int height = Screen.height;
            Texture2D screenshotTexture = new Texture2D(width, height, TextureFormat.ARGB32, false);
            Rect rect = new Rect(0, 0, width, height);
            screenshotTexture.ReadPixels(rect, 0, 0);
            screenshotTexture.Apply();
            return screenshotTexture;
        }

        private void CheckGround()
        {
            Vector3 origin = new Vector3(transform.position.x, transform.position.y - (transform.localScale.y * .5f), transform.position.z);
            Vector3 direction = transform.TransformDirection(Vector3.down);
            float distance = .75f;

            if (Physics.Raycast(origin, direction, out RaycastHit hit, distance))
            {
                isGrounded = true;
            }
            else
            {
                isGrounded = false;
            }
        }

        private void Jump()
        {
            if (isGrounded)
            {
                rb.AddForce(0f, jumpPower, 0f, ForceMode.Impulse);
                isGrounded = false;
            }
        }

        private void HeadBob()
        {
            if (isWalking)
            {
                if (isSprinting)
                {
                    timer += Time.deltaTime * (bobSpeed + sprintSpeed);
                }
                else
                {
                    timer += Time.deltaTime * bobSpeed;
                }
                
                joint.localPosition = new Vector3(
                    jointOriginalPos.x + Mathf.Sin(timer) * bobAmount.x,
                    jointOriginalPos.y + Mathf.Sin(timer) * bobAmount.y,
                    jointOriginalPos.z + Mathf.Sin(timer) * bobAmount.z
                );
            }
            else
            {
                timer = 0;
                joint.localPosition = new Vector3(
                    Mathf.Lerp(joint.localPosition.x, jointOriginalPos.x, Time.deltaTime * bobSpeed),
                    Mathf.Lerp(joint.localPosition.y, jointOriginalPos.y, Time.deltaTime * bobSpeed),
                    Mathf.Lerp(joint.localPosition.z, jointOriginalPos.z, Time.deltaTime * bobSpeed)
                );
            }
        }
    }
}