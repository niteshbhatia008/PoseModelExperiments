using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class MyUtils
{

    private static string savePath = string.Empty;
    
    /// <summary>
    /// Returns the persistent folder path
    /// </summary>
    /// <returns>Persistent folder</returns>
    public static string GetSaveFolder()
    {
        if(string.IsNullOrEmpty(savePath))
        {
            savePath = Application.persistentDataPath;

            if (savePath.Length > 0 && savePath[savePath.Length - 1] == Path.DirectorySeparatorChar)
                savePath = savePath.Substring(0, savePath.Length - 1);
        }

        return savePath;
    }


    /// <summary>
    /// Downloads file from the given URL
    /// </summary>
    /// <param name="url">Url</param>
    /// <param name="saveFilePath">Save file</param>
    /// <returns></returns>
    public static IEnumerator DownloadFileFromUrl(string fileName, string url, string saveFilePath)
    {
        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.LogError("Error downloading '" + fileName + "': " + www.error);
            }
            else
            {
                byte[] btData = www.downloadHandler.data;
                File.WriteAllBytes(saveFilePath, btData);

                Debug.Log("Downloaded '" + fileName + "' to: " + saveFilePath);
            }
        }
    }

    /// <summary>
    /// Scale 2D-texture
    /// </summary>
    /// <param name="src"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="mode"></param>
    /// <returns></returns>
    public static Texture2D ScaleTexture(Texture2D src, int width, int height, FilterMode mode = FilterMode.Bilinear)
    {
        if (src.width == width && src.height == height)
            return src;

        Rect texR = new Rect(0, 0, width, height);
        _gpu_scale(src, width, height, mode);

        //Get rendered data back to a new texture
        if(resultTex == null)
        {
            resultTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        }

        //result.Resize(width, height);
        resultTex.ReadPixels(texR, 0, 0, false);
        resultTex.Apply();

        return resultTex;
    }

    private static RenderTexture renderTex = null;
    private static Texture2D resultTex = null;

    // used to scale the given texture
    private static void _gpu_scale(Texture2D src, int width, int height, FilterMode fmode)
    {
        //We need the source texture in VRAM because we render with it
        if(src.filterMode != fmode)
        {
            src.filterMode = fmode;
            src.Apply(false);
        }

        //Using RTT for best quality and performance. Thanks, Unity 5
        if (renderTex == null)
        {
            renderTex = new RenderTexture(width, height, 0);  // 32
        }

        //Set the render texture, to render to it
        Graphics.SetRenderTarget(renderTex);

        //Setup 2D matrix in range 0..1, so nobody needs to care about sizes
        GL.LoadPixelMatrix(0, 1, 1, 0);

        //Then clear & draw the texture to fill the entire RTT.
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), src);
    }

}
