using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class WebcamSource : MonoBehaviour
{
    [Tooltip("If set, this picture texture will be used instead of the web camera.")]
    [HideInInspector]
    public Texture2D pictureTex;

    [Tooltip("Whether the web-camera output needs to be flipped horizontally or not.")]
    public bool flipHorizontally = false;

    [Tooltip("Selected web-camera name, if any.")]
    public string webcamName;

    // the web-camera texture
    private WebCamTexture webcamTex;

    // whether the texture is updated
    private bool isTexUpdated;

    // whether the output aspect ratio is set
    private bool bTexResolutionSet = false;


    ///// <summary>
    ///// Checks for web camera
    ///// </summary>
    //public bool HasCamera
    //{
    //    get
    //    {
    //        return webcamTex && !string.IsNullOrEmpty(webcamTex.deviceName);
    //    }
    //}

    /// <summary>
    /// Webcam texture, or null
    /// </summary>
    public WebCamTexture WebcamTex
    {
        get
        {
            return webcamTex;
        }
    }

    /// <summary>
    /// Check if the texture was updated
    /// </summary>
    public bool IsTexUpdated
    {
        get { return isTexUpdated; }
        set { isTexUpdated = value; }
    }


    void Awake()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        
        if (devices != null && devices.Length > 0)
        {
            // print available webcams
            StringBuilder sbWebcams = new StringBuilder();
            sbWebcams.Append("Available webcams:").AppendLine();

            for(int i = 0; i < devices.Length; i++)
            {
                WebCamDevice device = devices[i];
                sbWebcams.Append(device.name).AppendLine();

                // get 1st front-facing webcam name, if not set
                if (string.IsNullOrEmpty(webcamName) && device.isFrontFacing)
                {
                    webcamName = device.name;
                }
            }

            Debug.Log(sbWebcams.ToString());
            Debug.Log("Selected webcam: " + webcamName);

            if(pictureTex != null)
            {
                RawImage rawimage = GetComponent<RawImage>();
                if (rawimage)
                {
                    rawimage.texture = pictureTex;
                }

                isTexUpdated = true;
            }
            else if(!string.IsNullOrEmpty(webcamName))
            {
                // create webcam tex
                webcamTex = new WebCamTexture(webcamName);

                RawImage rawimage = GetComponent<RawImage>();
                if (rawimage)
                {
                    rawimage.texture = webcamTex;
                }
            }

            bTexResolutionSet = false;
        }

        if (flipHorizontally)
        {
            Vector3 scale = transform.localScale;
            transform.localScale = new Vector3(-scale.x, scale.y, scale.z);
        }

        if (webcamTex != null)
        {
            webcamTex.Play();
        }
    }


    // set new test image
    public void SetTestImage(Texture2D picture)
    {
        pictureTex = picture;

        if (pictureTex != null)
        {
            RawImage rawimage = GetComponent<RawImage>();
            if (rawimage)
            {
                rawimage.texture = pictureTex;
            }

            isTexUpdated = true;
            bTexResolutionSet = false;
        }
    }


    void Update()
    {
        // set aspect ratio if needed
        bool bTexAvailable = pictureTex != null || (webcamTex != null && webcamTex.isPlaying);
        if (!bTexResolutionSet && bTexAvailable)
        {
            AspectRatioFitter ratioFitter = GetComponent<AspectRatioFitter>();
            if (ratioFitter)
            {
                ratioFitter.aspectRatio = pictureTex != null ? (float)pictureTex.width / (float)pictureTex.height :
                    (float)webcamTex.width / (float)webcamTex.height;
            }

            bTexResolutionSet = true;
        }

        // check for texture update
        if(webcamTex != null && webcamTex.isPlaying)
        {
            isTexUpdated = webcamTex.didUpdateThisFrame;
        }
    }

}
