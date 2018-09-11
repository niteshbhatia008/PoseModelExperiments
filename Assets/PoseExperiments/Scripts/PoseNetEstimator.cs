using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class PoseNetEstimator : MonoBehaviour
{
    public enum ModelTypeEnum : int
    {
        x101 = 101,
        x100 = 100,
        x075 = 75,
        x050 = 50
    }

    [Tooltip("CNN model type (The multiplier in terms of PoseNet/MobileNet).")]
    public ModelTypeEnum modelType = ModelTypeEnum.x101;

    public enum OutputStrideEnum : int
    {
        x32 = 32,
        x16 = 16,
        x8 = 8
    }

    [Tooltip("Output stride used by the CNN model.")]
    public OutputStrideEnum outputStride = OutputStrideEnum.x16;

    [Tooltip("Input image size required by the CNN model (for both width & height).")]
    public int modelImageSize = 513;

    [Tooltip("Minimum confidence of the detected keypoints, in order to be considered valid.")]
    [Range(0.1f, 0.9f)]
    public float minConfidence = 0.1f;

    [Tooltip("Whether to show the keypoint heatmaps.")]
    public bool showHeatmap = false;

    [Tooltip("Whether to show the detected poses (keypoints & bones).")]
    public bool showPoses = true;

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
    private PoseNet posenet = new PoseNet();
    private PoseNet.Pose[] poses = null;
    private float[,,,] heatmap = null;

    private TFSession session;
    //TFSession.Runner runner;
    private TFGraph graph;

    //private bool isPosing;
    private MyAsyncTask<Color[], PoseNet.Pose[]> taskPoseNet = null;


    void Start()
    {
        // load the model
        string modelName = "posenet_model_" + modelType.ToString() + "_" + outputStride.ToString() + ".pb";
        TextAsset graphModel = Resources.Load(modelName) as TextAsset;

        if (graphModel != null)
        {
            graph = new TFGraph();
            graph.Import(graphModel.bytes);
            session = new TFSession(graph);

            modelInited = true;
        }
        else
        {
            throw new System.Exception("Model not found: " + modelName);
        }

        // get reference to the webcam-texture
        webcamSource = FindObjectOfType<WebcamSource>();
        webcamTexture = webcamSource != null && testImage == null ? webcamSource.WebcamTex : null;

        if (webcamSource)
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

        if (renderHelper)
        {
            // set render-helper parameters
            renderHelper.sizePoseImage = new Vector2(modelImageSize, modelImageSize);
            renderHelper.SetCamImageRect(testImage ? testImage.width : webcamTexture.width,
                testImage ? testImage.height : webcamTexture.height);

            // set renderer horizontal flip
            renderHelper.flipHorizontally = !flipHorizontally;
        }

        // create async task for pose prediction
        CreateAsyncTask();

        Debug.Log("Pose-Net estimator inited.");
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

        if(taskPoseNet != null)
        {
            if(taskPoseNet.State == AsyncTaskState.NotRunning)
            {
                // scale the texture
                Texture2D scaledTex = MyUtils.ScaleTexture(webcamTex2D, modelImageSize, modelImageSize);
                if (debugImage != null && debugImage.gameObject.activeSelf && debugImage.texture == null)
                {
                    debugImage.texture = scaledTex;
                }

                // start model prediction
                taskPoseNet.Start(scaledTex.GetPixels());
            }
            else if(taskPoseNet.IsCompleted)
            {
                if(taskPoseNet.State == AsyncTaskState.Failed)
                {
                    Debug.LogError("Task failed: " + taskPoseNet.ErrorMessage);
                }

                // mark as processed
                webcamSource.IsTexUpdated = false;

                taskPoseNet.State = AsyncTaskState.NotRunning;
            }
        }

    }


    void OnRenderObject()
    {
        if(renderHelper != null)
        {
            if (showHeatmap && heatmap != null)
                DrawHeatmap(heatmap, (int)outputStride, minConfidence);

            if (showPoses && poses != null)
                DrawPoses(poses, minConfidence);
        }
    }


    // create the task
    void CreateAsyncTask()
    {
        taskPoseNet = new MyAsyncTask<Color[], PoseNet.Pose[]>((param) =>
        {
            Color[] imagePixels = (Color[])param;
            var inputTensor = TransformInput(imagePixels, modelImageSize, modelImageSize);
            //var inputTensorVal = (float[,,,])inputTensor.GetValue(jagged: false);

            var runner = session.GetRunner();
            runner.AddInput(graph["image"][0], inputTensor);
            runner.Fetch(
                graph["heatmap"][0],
                graph["offset_2"][0],
                graph["displacement_fwd_2"][0],
                graph["displacement_bwd_2"][0]
            );

            TFTensor[] result = runner.Run();
            heatmap = (float[,,,])result[0].GetValue(jagged: false);
            var offsets = (float[,,,])result[1].GetValue(jagged: false);
            var displacementsFwd = (float[,,,])result[2].GetValue(jagged: false);
            var displacementsBwd = (float[,,,])result[3].GetValue(jagged: false);

            poses = posenet.DecodeMultiplePoses(
                heatmap, offsets,
                displacementsFwd,
                displacementsBwd,
                outputStride: (int)outputStride, maxPoseDetections: 15,
                scoreThreshold: minConfidence, nmsRadius: 20);

            //inputTensor.Dispose();
            //inputTensor = null;

            //for (int i = 0; i < result.Length; i++)
            //{
            //    result[i].Dispose();
            //    result[i] = null;
            //}

            return poses;
        });
    }


    private static TFTensor TransformInput(Color[] pic, int width, int height)
    {
        //System.Array.Reverse(pic);
        float[] floatValues = new float[width * height * 3];
        //double sum = 0f;

        //float colorFactor = 2.0f / 255.0f;
        for (int i = 0, j = (floatValues.Length - 3); i < pic.Length; i++, j -= 3)
        {
            var color = pic[i];
            int ji = j;

            floatValues[ji++] = color.r * 2f - 1f;
            floatValues[ji++] = color.g * 2f - 1f;
            floatValues[ji++] = color.b * 2f - 1f;
        }

        TFShape shape = new TFShape(1, width, height, 3);

        return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
    }


    private void DrawHeatmap(float[,,,] heatmap, int outputStride, float minConfidence)
    {
        if (heatmap == null || renderHelper == null)
            return;

        List<Vector3> alPoints = new List<Vector3>();

        int lenI = heatmap.GetLength(1);
        int lenJ = heatmap.GetLength(2);
        int lenK = heatmap.GetLength(3);

        for (int i = 0; i < lenI; i++)
        {
            for (int j = 0; j < lenJ; j++)
            {
                float mean = 0f;

                for (int k = 0; k < lenK; k++)
                {
                    mean += heatmap[0, i, j, k];
                }

                //mean /= (float)lenK;

                if (mean >= minConfidence)
                {
                    float x = (float)(j * outputStride);
                    float y = (float)(i * outputStride);

                    Vector3 pos = renderHelper.GetOverlayPos2D(x, y);
                    alPoints.Add(pos);
                }
            }
        }

        if (alPoints.Count > 0)
        {
            renderHelper.DrawPoints2D(alPoints, 3f, Color.yellow);  // 0.05f
        }

        alPoints.Clear();
        alPoints = null;
    }


    private void DrawPoses(PoseNet.Pose[] poses, float minConfidence)
    {
        if (poses == null || renderHelper == null)
            return;

        // draw skeletons
        foreach (var pose in poses)
        {
            if (pose.score >= minConfidence)
            {
                DrawSkeleton(pose.keypoints, minConfidence);
            }
        }

        // draw keypoints
        foreach (var pose in poses)
        {
            if (pose.score >= minConfidence)
            {
                DrawKeypoints(pose.keypoints, minConfidence);
            }
        }
    }

    private void DrawSkeleton(PoseNet.Keypoint[] keypoints, float minConfidence)
    {
        if (keypoints == null || renderHelper == null)
            return;

        List<Vector3> alLinePoints = new List<Vector3>();
        var adjacentKeyPoints = posenet.GetAdjacentKeyPoints(keypoints, minConfidence);

        foreach (var keypoint in adjacentKeyPoints)
        {
            Vector3 pos1 = renderHelper.GetOverlayPos2D(keypoint.Item1.position.x, keypoint.Item1.position.y);
            Vector3 pos2 = renderHelper.GetOverlayPos2D(keypoint.Item2.position.x, keypoint.Item2.position.y);

            alLinePoints.Add(pos1);
            alLinePoints.Add(pos2);
        }

        if (alLinePoints.Count > 0)
        {
            renderHelper.DrawLines2D(alLinePoints, 2f, Color.cyan);
        }

        alLinePoints.Clear();
        alLinePoints = null;
    }

    private void DrawKeypoints(PoseNet.Keypoint[] keypoints, float minConfidence)
    {
        if (keypoints == null || renderHelper == null)
            return;

        List<Vector3> alPoints = new List<Vector3>();

        foreach (var keypoint in keypoints)
        {
            if (keypoint.score >= minConfidence)
            {
                Vector3 pos = renderHelper.GetOverlayPos2D(keypoint.position.x, keypoint.position.y);
                alPoints.Add(pos);
            }
        }

        if (alPoints.Count > 0)
        {
            renderHelper.DrawCircles2D(alPoints, 3f, Color.cyan);  // 0.08f;
        }

        alPoints.Clear();
        alPoints = null;
    }

}
