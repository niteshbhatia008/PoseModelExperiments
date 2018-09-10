using System.Collections;
using System.Collections.Generic;
using System.Json;
using UnityEngine;

public class Yolo2Helper
{

    public struct YoloObject
    {
        public int classIndex;
        public float classConf;

        public string classLabel;
        public Color classColor;

        public float boxLeft, boxRight, boxTop, boxBottom;
    }


    // calculates sigmoid function for the given x-value
    private static float Sigmoid(float x)
    {
        float y = 1f / (1f + Mathf.Exp(-x));
        return y;
    }


    // estimates output boxes from the cnn output
    public static void EstimateOutBoxes(float[,,,] output, float[] anchors, int height, int width, int boxNum, int classNum,
        ref float[,,,] boxPred, ref float[,,,] classProb)
    {
        int boxLen = output.GetLength(3) / boxNum;

        boxPred = new float[height, width, boxNum, 5];
        classProb = new float[height, width, boxNum, boxLen - 5 + 1];

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                for (int box = 0; box < boxNum; box++)
                {
                    int boxOfs = box * boxLen;

                    boxPred[row, col, box, 0] = (col + Sigmoid(output[0, row, col, boxOfs + 0])) / (float)width;
                    boxPred[row, col, box, 1] = (row + Sigmoid(output[0, row, col, boxOfs + 1])) / (float)height;
                    boxPred[row, col, box, 2] = Mathf.Exp(output[0, row, col, boxOfs + 2]) * anchors[2 * box + 0] / (float)width;
                    boxPred[row, col, box, 3] = Mathf.Exp(output[0, row, col, boxOfs + 3]) * anchors[2 * box + 1] / (float)height;
                    boxPred[row, col, box, 4] = Sigmoid(output[0, row, col, boxOfs + 4]);

                    // SOFTMAX
                    float classMax = 0f;
                    for(int c = 0; c < classNum; c++)
                    {
                        float classVal = output[0, row, col, boxOfs + 5 + c];
                        if (classMax < classVal)
                            classMax = classVal;
                    }

                    float classSum = 0f;
                    for (int c = 0; c < classNum; c++)
                    {
                        classProb[row, col, box, c] = Mathf.Exp(output[0, row, col, boxOfs + 5 + c] - classMax);
                        classSum += classProb[row, col, box, c];
                    }

                    classMax = 0f;
                    int classIdx = -1;

                    for (int c = 0; c < classNum; c++)
                    {
                        classProb[row, col, box, c] *= boxPred[row, col, box, 4] / classSum;

                        if (classMax < classProb[row, col, box, c])
                        {
                            classMax = classProb[row, col, box, c];
                            classIdx = c;
                        }
                    }

                    // max-class index
                    classProb[row, col, box, classNum] = (float)classIdx;
                }
            }
        }

    }


    // finds out the currently detected objects
    public static List<YoloObject> GetDetectedObjects(float[,,,] boxPred, float[,,,] classProb,
        string[] classLabels, Color[] classColors, float threshold, float imageWidth, float imageHeight)
    {
        List<YoloObject> alObjects = new List<YoloObject>();

        int height = classProb.GetLength(0);
        int width = classProb.GetLength(1);
        int boxNum = classProb.GetLength(2);
        int classNum = classProb.GetLength(3) - 1;

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                for (int box = 0; box < boxNum; box++)
                {
                    int maxIndex = (int)classProb[row, col, box, classNum];

                    if(maxIndex >= 0 && maxIndex < classNum && classProb[row, col, box, maxIndex] >= threshold)
                    {
                        YoloObject obj = new YoloObject();

                        obj.classIndex = maxIndex;
                        obj.classConf = classProb[row, col, box, maxIndex];

                        obj.classLabel = classLabels != null ? classLabels[maxIndex] : "Class " + maxIndex;
                        obj.classColor = classColors != null ? classColors[maxIndex] : Color.cyan;

                        obj.boxLeft = Mathf.Max(0f, boxPred[row, col, box, 0] - boxPred[row, col, box, 2] * 0.5f) * imageWidth;
                        obj.boxRight = Mathf.Max(0f, boxPred[row, col, box, 0] + boxPred[row, col, box, 2] * 0.5f) * imageHeight;
                        obj.boxTop = Mathf.Max(0f, boxPred[row, col, box, 1] - boxPred[row, col, box, 3] * 0.5f) * imageWidth;
                        obj.boxBottom = Mathf.Max(0f, boxPred[row, col, box, 1] + boxPred[row, col, box, 3] * 0.5f) * imageHeight;

                        alObjects.Add(obj);
                    }
                }
            }
        }

        return alObjects;
    }


    // applies non-max suppression filter to the given objects' bounding boxes
    public static List<YoloObject> NonMaxSuppression(List<YoloObject> objs, float overlayThreshold)
    {
        List<YoloObject> pickedObjs = new List<YoloObject>();
        if (objs == null || objs.Count == 0)
            return pickedObjs;

        // sort objs by box bottom-Y
        objs.Sort((x, y) => x.boxBottom.CompareTo(y.boxBottom));

        //// calculate all box areas
        //float[] boxAreas = new float[objs.Count];

        //for(int i = objs.Count - 1; i >= 0; i--)
        //{
        //    YoloObject thisObj = objs[i];
        //    boxAreas[i] = (thisObj.boxRight - thisObj.boxLeft + 1f) * (thisObj.boxBottom - thisObj.boxTop + 1f);
        //}

        while(objs.Count > 0)
        {
            int lastI = objs.Count - 1;
            YoloObject lastObj = objs[lastI];

            pickedObjs.Add(lastObj);
            objs.RemoveAt(lastI);

            float lastArea = (lastObj.boxRight - lastObj.boxLeft + 1f) * (lastObj.boxBottom - lastObj.boxTop + 1f);

            for(int i = objs.Count - 1; i >= 0; i--)
            {
                YoloObject thisObj = objs[i];

                float x1 = thisObj.boxLeft > lastObj.boxLeft ? thisObj.boxLeft : lastObj.boxLeft;  // max
                float y1 = thisObj.boxTop > lastObj.boxTop ? thisObj.boxTop : lastObj.boxTop;  // max
                float x2 = thisObj.boxRight < lastObj.boxRight ? thisObj.boxRight : lastObj.boxRight;  // min
                float y2 = thisObj.boxBottom < lastObj.boxBottom ? thisObj.boxBottom : lastObj.boxBottom;  // min

                float xW = x2 - x1 + 1f; if (xW < 0f) xW = 0f;
                float yH = y2 - y1 + 1f; if (yH < 0f) yH = 0f;

                float overlay = (xW * yH) / lastArea;
                if(overlay > overlayThreshold)
                {
                    // supress this obj
                    objs.RemoveAt(i);
                }
            }
        }

        return pickedObjs;
    }

}
