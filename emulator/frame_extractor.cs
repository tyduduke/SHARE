using System.Collections;
using UnityEngine;
using System.IO;

public class SimpleCameraController : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera playerCamera;
    public float mouseSensitivity = 2f;
    public float maxLookAngle = 90f;
    public bool lockCursor = true;

    [Header("Frame Capture Settings")]
    public string saveDirectory = "CapturedFrames";
    public bool useHighResolution = false;
    public int resolutionMultiplier = 2;

    // Internal camera variables
    private float yaw = 0f;
    private float pitch = 0f;
    
    // Movement detection variables
    private Vector3 lastCameraPosition;
    private Quaternion lastCameraRotation;
    private bool isMoving = false;
    private float movementStopDelay = 0.5f; // Time to wait after movement stops
    private float timeSinceLastMovement = 0f;
    
    // Frame capture variables
    private int framesCaptured = 0;
    private int maxFramesToCapture = 3;
    private bool isCapturingFrames = false;
    private int totalFramesSaved = 0; // Global counter for unique filenames

    void Start()
    {
        // Initialize camera
        if (playerCamera == null)
            playerCamera = Camera.main;

        // Lock cursor if enabled
        if (lockCursor)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        // Initialize position tracking
        lastCameraPosition = playerCamera.transform.position;
        lastCameraRotation = playerCamera.transform.rotation;

        // Create save directory if it doesn't exist
        string fullPath = Path.Combine(Application.dataPath, saveDirectory);
        Directory.CreateDirectory(fullPath);
        
        Debug.Log($"Frames will be saved to: {fullPath}");
    }

    void Update()
    {
        HandleCameraMovement();
        DetectMovement();
        HandleFrameCapture();
        
        // Toggle cursor lock with Escape key
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleCursorLock();
        }
    }

    void HandleCameraMovement()
    {
        // Get mouse input
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

        // Update yaw and pitch
        yaw += mouseX;
        pitch -= mouseY;
        
        // Clamp pitch to prevent over-rotation
        pitch = Mathf.Clamp(pitch, -maxLookAngle, maxLookAngle);

        // Apply rotation
        transform.rotation = Quaternion.Euler(0f, yaw, 0f);
        playerCamera.transform.localRotation = Quaternion.Euler(pitch, 0f, 0f);
    }

    void DetectMovement()
    {
        // Check if camera position or rotation has changed
        bool positionChanged = Vector3.Distance(playerCamera.transform.position, lastCameraPosition) > 0.001f;
        bool rotationChanged = Quaternion.Angle(playerCamera.transform.rotation, lastCameraRotation) > 0.1f;

        if (positionChanged || rotationChanged)
        {
            isMoving = true;
            timeSinceLastMovement = 0f;
            
            // Update last known position and rotation
            lastCameraPosition = playerCamera.transform.position;
            lastCameraRotation = playerCamera.transform.rotation;
        }
        else
        {
            // If we were moving and now we've stopped
            if (isMoving)
            {
                timeSinceLastMovement += Time.deltaTime;
                
                // If enough time has passed since last movement, consider movement stopped
                if (timeSinceLastMovement >= movementStopDelay)
                {
                    isMoving = false;
                    
                    // Start frame capture sequence if not already capturing
                    if (!isCapturingFrames)
                    {
                        StartFrameCapture();
                    }
                }
            }
        }
    }

    void HandleFrameCapture()
    {
        // This is handled by the coroutine, but we can add any additional logic here if needed
    }

    void StartFrameCapture()
    {
        if (!isCapturingFrames)
        {
            framesCaptured = 0;
            isCapturingFrames = true;
            StartCoroutine(CaptureFramesSequence());
            Debug.Log("Starting frame capture sequence...");
        }
    }

    IEnumerator CaptureFramesSequence()
    {
        while (framesCaptured < maxFramesToCapture && isCapturingFrames)
        {
            yield return new WaitForEndOfFrame();
            
            // Capture and save frame
            CaptureFrame();
            framesCaptured++;
            
            Debug.Log($"Captured frame {framesCaptured}/{maxFramesToCapture}");
            
            // Wait a frame between captures to ensure stability
            yield return null;
        }
        
        isCapturingFrames = false;
        Debug.Log($"Frame capture sequence completed. Total frames saved: {totalFramesSaved}");
    }

    void CaptureFrame()
    {
        Texture2D screenshot;
        
        if (useHighResolution)
        {
            screenshot = CaptureHighResScreenshot();
        }
        else
        {
            screenshot = CaptureStandardScreenshot();
        }

        // Save the frame
        SaveFrame(screenshot);
        
        // Clean up
        Destroy(screenshot);
    }

    Texture2D CaptureHighResScreenshot()
    {
        int width = Screen.width * resolutionMultiplier;
        int height = Screen.height * resolutionMultiplier;
        
        RenderTexture renderTexture = new RenderTexture(width, height, 24);
        renderTexture.antiAliasing = 8;
        
        // Store original settings
        RenderTexture originalTargetTexture = playerCamera.targetTexture;
        
        // Set camera to render to high-res texture
        playerCamera.targetTexture = renderTexture;
        playerCamera.Render();
        
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();
        
        // Restore original settings
        playerCamera.targetTexture = originalTargetTexture;
        RenderTexture.active = null;
        
        // Clean up
        DestroyImmediate(renderTexture);
        
        return screenshot;
    }

    Texture2D CaptureStandardScreenshot()
    {
        int width = Screen.width;
        int height = Screen.height;
        
        Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();
        
        return screenshot;
    }

    void SaveFrame(Texture2D screenshot)
    {
        byte[] imageBytes = screenshot.EncodeToPNG();
        string fullPath = Path.Combine(Application.dataPath, saveDirectory);
        string fileName = $"frame_{totalFramesSaved:D4}.png"; // Use 4-digit padding for better sorting
        string filePath = Path.Combine(fullPath, fileName);
        
        File.WriteAllBytes(filePath, imageBytes);
        totalFramesSaved++;
        
        Debug.Log($"Frame saved: {fileName}");
    }

    void ToggleCursorLock()
    {
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
        else
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }

    void OnApplicationFocus(bool hasFocus)
    {
        if (hasFocus && lockCursor)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }
}

