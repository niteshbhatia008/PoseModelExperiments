using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;
using System.Json;

public class Yolo2Estimator : MonoBehaviour
{
    [Tooltip("CNN model name (must exist in resources).")]
    public string modelName = "yolov2-tiny";

    [Tooltip("Minimum confidence of the detected keypoints, in order to be considered valid.")]
    [Range(0.1f, 0.9f)]
    public float minConfidence = 0.25f;

    [Tooltip("Overlay threshold used by the NMS-filter.")]
    [Range(0.1f, 0.9f)]
    public float overlayThreshold = 0.3f;

    [Tooltip("Whether to show the detected object boxes.")]
    public bool showBoxes = true;

    [Tooltip("Whether to show the detected object labels.")]
    public bool showLabels = true;

    [Tooltip("Static image to be used for testing, instead of the webcam.")]
    public Texture2D testImage = null;

    [Tooltip("Whether the image needs to be flipped horizontally or not.")]
    public bool flipHorizontally = false;

    [Tooltip("UI debug image.")]
    public UnityEngine.UI.RawImage debugImage;

    private WebcamSource webcamSource = null;
    private WebCamTexture webcamTexture = null;
    private Texture2D webcamTex2D = null;
    private RenderHelper renderHelper;

    private bool modelInited = false;
    private JsonValue modelMetaData = null;

    // yolo parameters
    private int inputWidth = 416, inputHeight = 416;
    private int outputWidth = 13, outputHeight = 13;
    private int numBoxes = 5, numClasses = 80;

    private float[] anchors = new float[] { 0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f };
    private string[] classLabels = null;
    private Color[] classColors = null;

    private float[,,,] output = null;
    private float[,,,] boxPred = null;
    private float[,,,] classProb = null;
    private List<Yolo2Helper.YoloObject> detectedObjects = null;

    private TFSession session;
    private TFGraph graph;

    private MyAsyncTask<Color[], TFTensor> taskYolo = null;


    void Start()
    {
        // load the model
        string modelResName = modelName + ".pb";
        TextAsset graphModel = Resources.Load(modelResName) as TextAsset;

        if (graphModel != null)
        {
            graph = new TFGraph();
            graph.Import(graphModel.bytes);
            session = new TFSession(graph);

            modelInited = true;
        }
        else
        {
            throw new System.Exception("Model not found: " + modelResName);
        }

        // load the meta-data
        string metaFileResName = modelName + ".meta";
        TextAsset metaFileRes = Resources.Load(metaFileResName) as TextAsset;

        if (metaFileRes != null)
        {
            modelMetaData = JsonValue.Parse(metaFileRes.text);
            LoadYoloParams();
        }
        else
        {
            throw new System.Exception("Meta-file not found: " + metaFileResName);
        }

        // get reference to the webcam-texture
        webcamSource = FindObjectOfType<WebcamSource>();
        webcamTexture = webcamSource != null && testImage == null ? webcamSource.WebcamTex : null;

        if(webcamSource)
        {
            // set static image & horizontal flip, if needed
            if (testImage != null)
            {
                webcamSource.SetTestImage(testImage);
            }

            webcamSource.SetHorizontalFlip(flipHorizontally);
        }

        // get reference to gl-renderer
        renderHelper = FindObjectOfType<RenderHelper>();

        if(renderHelper)
        {
            // set render-helper parameters
            renderHelper.sizePoseImage = new Vector2(inputWidth, inputHeight);
            renderHelper.SetCamImageRect(testImage ? testImage.width : webcamTexture.width, 
                testImage ? testImage.height : webcamTexture.height);

            // set renderer horizontal flip
            renderHelper.flipHorizontally = !flipHorizontally;
        }

        // create async task for pose prediction
        CreateAsyncTask();

        Debug.Log("Yolo estimator inited.");
    }


    void Update()
    {
        if (!modelInited || !webcamSource || !webcamSource.IsTexUpdated)
            return;

        if(testImage != null)
        {
            webcamTex2D = testImage;
        }
        else if(webcamTexture != null)
        {
            if (webcamTex2D == null)
            {
                webcamTex2D = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.ARGB32, false);
            }

            var color32 = webcamTexture.GetPixels32();
            webcamTex2D.SetPixels32(color32);
            webcamTex2D.Apply();
        }

        if (taskYolo != null)
        {
            if (taskYolo.State == AsyncTaskState.NotRunning)
            {
                // scale the texture
                Texture2D scaledTex = MyUtils.ScaleTexture(webcamTex2D, inputWidth, inputHeight);
                if (debugImage != null && debugImage.gameObject.activeSelf && debugImage.texture == null)
                {
                    debugImage.texture = scaledTex;
                }

                // start model prediction
                taskYolo.Start(scaledTex.GetPixels());
            }
            else if (taskYolo.IsCompleted)
            {
                if (taskYolo.State == AsyncTaskState.Failed)
                {
                    Debug.LogError("Task failed: " + taskYolo.ErrorMessage);
                }

                if(taskYolo.State == AsyncTaskState.Succeeded)
                {
                    if(testImage != null)
                    {
                        LogDetectedObjects();
                    }
                }

                // mark as processed
                webcamSource.IsTexUpdated = false;

                taskYolo.State = AsyncTaskState.NotRunning;
            }
        }

    }

    // logs the currently found objects
    private void LogDetectedObjects()
    {
        if (detectedObjects != null)
        {
            Debug.Log("Found " + detectedObjects.Count + " objects");

            for(int i = 0; i < detectedObjects.Count; i++)
            {
                Yolo2Helper.YoloObject obj = detectedObjects[i];

                Debug.Log(string.Format("{0} {1:F0}%, box({2:F0}, {3:F0}, {4:F0}, {5:F0})", obj.classLabel, obj.classConf * 100f,
                    obj.boxLeft, obj.boxTop, obj.boxRight, obj.boxBottom));
            }
        }
    }


    void OnRenderObject()
    {
        if(showBoxes && detectedObjects != null && renderHelper != null)
        {
            // draw boxes around the detected objects
            for(int i = 0; i < detectedObjects.Count; i++)
            {
                Yolo2Helper.YoloObject obj = detectedObjects[i];

                Vector3 topLeft = renderHelper.GetOverlayPos2D(obj.boxLeft, obj.boxTop);
                Vector3 bottomRight = renderHelper.GetOverlayPos2D(obj.boxRight, obj.boxBottom);

                renderHelper.DrawRect2D(topLeft, bottomRight, 2f, obj.classColor);
            }
        }
    }


    void OnGUI()
    {
        if (showLabels && detectedObjects != null)
        {
            Color guiColor = GUI.color;

            //GUIStyle guiStyle = new GUIStyle();
            //guiStyle.fontStyle = FontStyle.Bold;

            float screenW = Screen.width;
            float screenH = Screen.height;
            
            for (int i = 0; i < detectedObjects.Count; i++)
            {
                Yolo2Helper.YoloObject obj = detectedObjects[i];

                Vector3 topLeft = renderHelper.GetOverlayPos2D(obj.boxLeft, obj.boxTop);
                Vector3 bottomRight = renderHelper.GetOverlayPos2D(obj.boxRight, obj.boxBottom);

                float rX = 10f + Mathf.Min(topLeft.x, bottomRight.x);
                float rY = screenH - Mathf.Max(topLeft.y, bottomRight.y);

                float rW = screenW - rX; // Mathf.Abs(bottomRight.x - topLeft.x);
                float rH = Mathf.Abs(bottomRight.y - topLeft.y);

                Rect guiRect = new Rect(rX, rY, rW, rH);
                GUI.color = obj.classColor;

                string sObjLabel = string.Format("{0} {1:F0}%", obj.classLabel, obj.classConf * 100f);
                GUI.Label(guiRect, sObjLabel);
            }

            GUI.color = guiColor;
        }
    }


    // loads yolo parameters from the meta-file
    private void LoadYoloParams()
    {
        JsonArray inpSize = (JsonArray)modelMetaData["inp_size"];
        inputWidth = (int)inpSize[0];
        inputHeight = (int)inpSize[1];

        JsonArray outSize = (JsonArray)modelMetaData["out_size"];
        outputWidth = (int)outSize[0];
        outputHeight = (int)outSize[1];

        numBoxes = (int)modelMetaData["num"];
        numClasses = (int)modelMetaData["classes"];

        JsonArray arrAnchors = (JsonArray)modelMetaData["anchors"];
        anchors = new float[arrAnchors.Count];
        for (int i = 0; i < arrAnchors.Count; i++)
            anchors[i] = (float)arrAnchors[i];

        JsonArray arrLabels = (JsonArray)modelMetaData["labels"];
        classLabels = new string[arrLabels.Count];
        for (int i = 0; i < arrLabels.Count; i++)
            classLabels[i] = arrLabels[i];

        JsonArray arrColors = (JsonArray)modelMetaData["colors"];
        classColors = new Color[arrColors.Count];
        for (int i = 0; i < arrColors.Count; i++)
        {
            JsonArray arrColor = (JsonArray)arrColors[i];
            classColors[i] = new Color(arrColor[0] / 255f, arrColor[1] / 255f, arrColor[2] / 255f, 1f);
        }
    }


    // create the task
    void CreateAsyncTask()
    {
        taskYolo = new MyAsyncTask<Color[], TFTensor>((param) =>
        {
            Color[] imagePixels = (Color[])param;
            var inputTensor = TransformInput(imagePixels, inputWidth, inputHeight);
            //var inputTensorVal = (float[,,,])inputTensor.GetValue(jagged: false);

            var runner = session.GetRunner();
            runner.AddInput(graph["input"][0], inputTensor);
            runner.Fetch(
                graph["output"][0]
            );

            TFTensor[] result = runner.Run();
            output = (float[,,,])result[0].GetValue(jagged: false);

            Yolo2Helper.EstimateOutBoxes(output, anchors, outputHeight, outputWidth, numBoxes, numClasses, 
                ref boxPred, ref classProb);
            List<Yolo2Helper.YoloObject> foundObjs = Yolo2Helper.GetDetectedObjects(boxPred, classProb, classLabels, classColors, 
                minConfidence, (float)inputWidth, (float)inputHeight);
            detectedObjects = Yolo2Helper.NonMaxSuppression(foundObjs, overlayThreshold);

            return output;
        });
    }


    // transforms the image to input tensor
    private static TFTensor TransformInput(Color[] pic, int width, int height)
    {
        float[] fValues = new float[width * height * 3];

        //float colorFactor = 2.0f / 255.0f;
        for (int i = 0, j = (fValues.Length - 3); i < pic.Length; i++, j -= 3)
        {
            var color = pic[i];
            int ji = j;

            fValues[ji++] = color.b;
            fValues[ji++] = color.g;
            fValues[ji++] = color.r;
        }

        TFShape shape = new TFShape(1, width, height, 3);

        return TFTensor.FromBuffer(shape, fValues, 0, fValues.Length);
    }

}
