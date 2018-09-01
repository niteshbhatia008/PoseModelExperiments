using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderHelper : MonoBehaviour
{
    private Material matRender = null;

    [HideInInspector]
    public Vector2 sizePoseImage = new Vector2(513f, 513f);

    public bool flipHorizontally = false;

    public bool flipVertically = false;

    //public float defaultDepth = 10f;

    private Vector2 sizeCamImage = new Vector2(640f, 480f);
    private Rect rectScreenImage = new Rect();
    //private Vector2 sizeScreen;

    //private Camera mainCamera = null;
    private int screenW, screenH;


    public void SetCamImageRect(float imageW, float imageH)
    {
        sizeCamImage = new Vector2(imageW, imageH);
        EstimateScreenRect();
    }


    void Awake()
    {
        if (!matRender)
        {
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            matRender = new Material(shader);

            matRender.hideFlags = HideFlags.HideAndDontSave;
            matRender.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            matRender.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            matRender.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            matRender.SetInt("_ZWrite", 0);
        }

        screenW = Screen.width;
        screenH = Screen.height;
        //mainCamera = Camera.main;

        EstimateScreenRect();
    }


    void Update()
    {
        if(Screen.width != screenW || Screen.height != screenH)
        {
            screenW = Screen.width;
            screenH = Screen.height;

            EstimateScreenRect();
        }
    }


    // estimates screen rectangle
    private void EstimateScreenRect()
    {
        float fScreenW = (float)Screen.width;
        float fScreenH = (float)Screen.height;
        //sizeScreen = new Vector2(fScreenW, fScreenH);

        float fImageW = fScreenH * sizeCamImage.x / sizeCamImage.y;
        float fImageX = (fScreenW - fImageW) * 0.5f;
        rectScreenImage = new Rect(fImageX, 0f, fImageW, fScreenH);
    }


    // estimates screen overlay position
    public Vector3 GetOverlayPos2D(float poseX, float poseY)
    {
        float xScaled = poseX * rectScreenImage.width / sizePoseImage.x;
        float yScaled = poseY * rectScreenImage.height / sizePoseImage.y;

        float xScreen = rectScreenImage.x + (!flipHorizontally ? xScaled : rectScreenImage.width - xScaled);
        float yScreen = rectScreenImage.y + (!flipVertically ? yScaled : rectScreenImage.height - yScaled);

        //Vector3 posOverlay = mainCamera.ScreenToWorldPoint(new Vector3(xScreen, yScreen, poseZ));
        //Vector3 posOverlay = new Vector3(xScreen / sizeScreen.x, yScreen / sizeScreen.y, 0f);
        Vector3 posOverlay = new Vector3(xScreen, yScreen, 0f);

        return posOverlay;
    }


    public void DrawPoints2D(List<Vector3> alPoints, float quadSize, Color color)
    {
        if (alPoints == null)
            return;

        GL.PushMatrix();
        matRender.SetPass(0);

        GL.LoadPixelMatrix();
        GL.Begin(GL.QUADS);
        GL.Color(color);

        foreach (Vector3 v in alPoints)
        {
            DrawPoint2D(v, quadSize);
        }

        GL.End();
        GL.PopMatrix();
    }

    public void DrawPoint2D(Vector3 v, float quadSize)
    {
        float q2 = quadSize / 2f;
        GL.Vertex3(v.x - q2, v.y - q2, 0f);
        GL.Vertex3(v.x - q2, v.y + q2, 0f);
        GL.Vertex3(v.x + q2, v.y + q2, 0f);
        GL.Vertex3(v.x + q2, v.y - q2, 0f);
    }


    public void DrawLines2D(List<Vector3> alLinePoints, float lineWidth, Color color)
    {
        if (alLinePoints == null)
            return;

        GL.PushMatrix();
        matRender.SetPass(0);

        GL.LoadPixelMatrix();
        GL.Begin(GL.QUADS);
        GL.Color(color);

        for(int i = 0; i < alLinePoints.Count; i += 2)
        {
            Vector3 v0 = alLinePoints[i];
            Vector3 v1 = alLinePoints[i + 1];

            DrawLine2D(v0, v1, lineWidth);
        }

        GL.End();
        GL.PopMatrix();
    }

    public void DrawLine2D(Vector3 v0, Vector3 v1, float lineWidth)
    {
        Vector3 n = ((new Vector3(v1.y, v0.x, 0f)) - (new Vector3(v0.y, v1.x, 0f))).normalized * lineWidth;
        GL.Vertex3(v0.x - n.x, v0.y - n.y, 0f);
        GL.Vertex3(v0.x + n.x, v0.y + n.y, 0f);
        GL.Vertex3(v1.x + n.x, v1.y + n.y, 0f);
        GL.Vertex3(v1.x - n.x, v1.y - n.y, 0f);
    }


    public void DrawCircles2D(List<Vector3> alCenters, float radius, Color color)
    {
        foreach (Vector3 vCenter in alCenters)
        {
            DrawCircle2D(vCenter, radius, color);
        }
    }


    public void DrawCircle2D(Vector3 vCenter, float radius, Color color)
    {
        GL.PushMatrix();
        matRender.SetPass(0);

        GL.LoadPixelMatrix();
        GL.Begin(GL.LINES);
        GL.Color(color);

        for (float theta = 0.0f; theta < (2 * Mathf.PI); theta += 0.01f)
        {
            GL.Vertex3(
                //Mathf.Cos(theta) * radius + keypoint.position.x * scale,
                //Mathf.Sin(theta) * radius + keypoint.position.y * scale, 0f);
                Mathf.Cos(theta) * radius + vCenter.x,
                Mathf.Sin(theta) * radius + vCenter.y, 0f);
        }

        GL.End();
        GL.PopMatrix();
    }


}
