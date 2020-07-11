
import nu.pattern.OpenCV;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.Core.normalize;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

public class GeneralHoughHelper {

    public OpenCVHelper helper = null;
    private List<Vector<Float>> houghSample;
    private float sampleHeight;
    private float sampleWidth;
    private Mat targetImage = null;
    private Mat targetMag = null;

    public List<float[]> GHT(){
        int imageHeight = targetImage.rows();
        int imageWidth = targetImage.cols();
        // (X, Y, Width, Height, Rotate): first scale then rotate
        // 24-124 scale, 30 degree rotation
        float[][] houghSpace = new float[imageWidth][imageHeight];
        GeneralHoughPoint[][] maxResultSpace = new GeneralHoughPoint[256][256];

        for (int height = 100; height <= 300; height ++){
            float scaleH = height / 128.0F;
        for (int width = 30; width <= 300; width ++){
        float scaleW = width / 128.0F;

        // clear value
        for (int j = 0; j <= imageHeight - 1; j++) {
            for (int i = 0; i <= imageWidth - 1; i++) {
                houghSpace[j][i] = 0.0F;
            }
        }

        for (int j = 0; j <= imageHeight - 1; j++) {
            for (int i = 0; i <= imageWidth - 1; i++) {
                float pixel = helper.matGetValue4(targetMag, j, i);
                if (pixel > 0.9F) {

                    for (Vector<Float> vec : houghSample) {
                        //Dim x1 As Single = Math.Cos(theta) * vec(0) * scaleW - Math.Sin(theta) * vec(1) * scaleH
                        //Dim y1 As Single = Math.Sin(theta) * vec(0) * scaleW + Math.Cos(theta) * vec(1) * scaleH
                        float x1 = vec.elementAt(0) * scaleW;
                        float y1 = vec.elementAt(1) * scaleH;
                        List<Float> arr = new ArrayList<Float>();
                        arr.add(-x1);
                        arr.add(-y1);
                        Vector<Float> invVec = new Vector<Float>(arr);

                        int x2 = (int) (i + invVec.elementAt(0));
                        int y2 = (int) (j + invVec.elementAt(1));
                        if (x2 >= 0 && x2 < imageWidth && y2 >= 0 && y2 < imageHeight) {
                            houghSpace[x2][y2] += 1.0F;
                        }

                    }
                }
            }
        }

        // getmax
        float maxValue = 0.0F;
        int maxArgs[] = new int[2];
        for (int i = 0; i <= imageWidth - 1; i++) {
            for (int j = 0; j <= imageHeight - 1; j++) {
                if (houghSpace[i][j] > maxValue) {
                    maxValue = houghSpace[i][j];
                    maxArgs[0] = i;
                    maxArgs[1] = j;
                }
            }
        }

            GeneralHoughPoint tmpPoint = new GeneralHoughPoint();
            tmpPoint.X = maxArgs[0];
            tmpPoint.Y = maxArgs[1];
            tmpPoint.Value = maxValue;

            maxResultSpace[width][height] = tmpPoint;
        }
        }

        // get maximum
        GeneralHoughPoint maxValue2 = new GeneralHoughPoint();
        int tmpW = 0;
        int tmpH = 0;
        for (int j = 0; j <= 255; j++) {
            for (int i = 0; i <= 255; i++){
                if (maxResultSpace[i][j].Value > maxValue2.Value){
                    maxValue2 = maxResultSpace[i][j];
					tmpW = i;
					tmpH = j;
                }
            }

        }

//        Dim source As Mat = Imread("C:\Users\asdfg\Desktop\ocvjtest\materials\darttest.png", ImreadModes.Color)
//        Resize(source, source, New Size(256, 256))
//        CvInvoke.Rectangle(source, New Rectangle(maxValue2.X - 0.5 * tmpW, maxValue2.Y - 0.5 * tmpH, tmpW, tmpH), New MCvScalar(255, 0, 0))
//        Debug.WriteLine(tmpW & "," & tmpH)
//
//        Imshow("result", source)
//        WaitKey(0)

        float[] resultArr = new float[4];
		resultArr[0] = maxValue2.X;
		resultArr[1] = maxValue2.Y;
		resultArr[2] = tmpW;
		resultArr[3] = tmpH;
		
		List<float[]> resultList = new ArrayList<float[]>();
		resultList.add(resultArr);
		return resultList;
    }

    public void LoadTarget(String path){
        targetImage = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        resize(targetImage, targetImage, new Size(256, 256));
        Mat mag = GetEdgeMagnitude(targetImage);
        threshold(mag, mag, 0.5, 1.0, THRESH_BINARY);
        targetMag = mag;
//        Imshow("tarmag", mag)
    }

    public void LoadSample(String path) {
        Mat sampleImage = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        resize(sampleImage, sampleImage, new Size(128, 128));
        // sobel
        Mat mag = GetEdgeMagnitude(sampleImage);
        // Dim mag As Mat = GetEdgeMagnitudeCanny(sampleImage)

        Mat dir = GetEdgeDirection(sampleImage);
        threshold(mag, mag, 0.75, 1.0, THRESH_BINARY);

        float originX = sampleImage.cols() / 2;
        float originY = sampleImage.rows() / 2;

        for (int j = 0; j <= mag.rows() - 1; j += 2) {
            for (int i = 0; i <= mag.cols() - 1; i += 2){
                float pixel = helper.matGetValue4(mag,j,i);
                // Dim pixel As Byte = mag.GetRawData(j, i)(0)
                if (pixel > 0.9F) {
                    float pixDir = helper.matGetValue4(dir,j,i);
                    List<Float> array = new ArrayList<Float>();
                    array.add(i - originX);
                    array.add(j - originY);
                    array.add(pixDir);
                    Vector<Float> vec = new Vector<Float>(array);
                    houghSample.add(vec);
                }
            }
        }

        sampleHeight = sampleImage.rows();
        sampleWidth = sampleImage.cols();

//        Dim sampleDisplay As Mat = Mat.Zeros(128, 128, DepthType.Cv8U, 1)
//        For Each tmpVec As VectorOfFloat In HoughSample
//        Mat_SetPixel_1(sampleDisplay, 64 + tmpVec(1), 64 + tmpVec(0), 255)
//        Next
//        Imshow("sample", sampleDisplay)

    }

    public Mat GetEdgeMagnitude(Mat image) {
        Mat resultH = new Mat(image.size(), CvType.CV_32FC1);
        Mat resultV = new Mat(image.size(), CvType.CV_32FC1);
        Mat resultM = new Mat(image.size(), CvType.CV_32FC1);

        for (int j = 1; j <= image.rows() - 2; j++) {
            for (int i = 1; i <= image.cols() - 2; i++) {
                float vh = 0.0f;
                vh = - helper.matGetValue1(image,j-1,i-1);
                vh -= helper.matGetValue1(image, j, i - 1) * 2;
                vh -= helper.matGetValue1(image,j + 1, i - 1);
                vh += helper.matGetValue1(image,j-1,i+1);
                vh += helper.matGetValue1(image,j,i+1) * 2;
                vh += helper.matGetValue1(image,j+1,i+1);
                vh /= 8.0f;

                helper.matSetValue4(resultH, j, i, vh / 255.0F);

                float vv = 0.0f;
                vv = -helper.matGetValue1(image,j-1,i-1);
                vv -= helper.matGetValue1(image, j-1, i) * 2;
                vv -= helper.matGetValue1(image,j-1,i+1);
                vv += helper.matGetValue1(image,j+1,i-1);
                vv += helper.matGetValue1(image,j+1,i) * 2;
                vv += helper.matGetValue1(image,j+1,i+1);
                vv /= 8.0f;

                helper.matSetValue4(resultV, j, i, vv / 255.0F);
            }
        }

        for (int j = 1; j <= image.rows() - 2; j++){
            for (int i = 1; i <= image.cols() - 2; i++){
                float v1 = helper.matGetValue4(resultH,j,i);
                float v2 = helper.matGetValue4(resultV,j,i);
                float v = (float)Math.sqrt(v1*v1+v2*v2);
                helper.matSetValue4(resultM,j,i,v);
            }
        }

        normalize(resultM, resultM, 0.0F, 1.0F, NORM_MINMAX);

        resultH.release();
        resultV.release();

        return resultM;

    }

    public Mat GetEdgeDirection(Mat image) {
        Mat resultH = new Mat(image.size(), CvType.CV_32FC1);
        Mat resultV = new Mat(image.size(), CvType.CV_32FC1);
        Mat resultT = new Mat(image.size(), CvType.CV_32FC1);

        for (int j = 1; j <= image.rows() - 2; j++){
            for (int i = 1; i <= image.cols() - 2; i++){
                float vh = 0.0f;
                vh = - helper.matGetValue1(image,j-1,i-1);
                vh -= helper.matGetValue1(image, j, i - 1) * 2;
                vh -= helper.matGetValue1(image,j + 1, i - 1);
                vh += helper.matGetValue1(image,j-1,i+1);
                vh += helper.matGetValue1(image,j,i+1) * 2;
                vh += helper.matGetValue1(image,j+1,i+1);
                vh /= 8.0f;

                helper.matSetValue4(resultH, j, i, vh / 255.0F);

                float vv = 0.0f;
                vv = -helper.matGetValue1(image,j-1,i-1);
                vv -= helper.matGetValue1(image, j-1, i) * 2;
                vv -= helper.matGetValue1(image,j-1,i+1);
                vv += helper.matGetValue1(image,j+1,i-1);
                vv += helper.matGetValue1(image,j+1,i) * 2;
                vv += helper.matGetValue1(image,j+1,i+1);
                vv /= 8.0f;

                helper.matSetValue4(resultV, j, i, vv / 255.0F);
            }
        }

        for (int j = 1; j <= image.rows() - 2; j++){
            for (int i = 1; i <= image.cols() - 2; i++){
                float v1 = helper.matGetValue4(resultH,j,i);
                float v2 = helper.matGetValue4(resultV,j,i);

                double phi = Math.atan2(v2, v1);
                double theta = phi - Math.PI / 2;
                if (theta < -Math.PI){ theta += 2 * Math.PI;}

                helper.matSetValue4(resultT, j, i, (float)theta);
            }
        }

        resultH.release();
        resultV.release();

        return resultT;

    }


}
