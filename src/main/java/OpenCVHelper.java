import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class OpenCVHelper {

    OpenCVHelper(){
        nu.pattern.OpenCV.loadShared();  // 引用 OpenCV 库
    }

    /**
     * 给 8-bit 1-chan 的 mat 赋值，通常用于灰度 GrayScale
     *
     * */
    public void matSetValue1(Mat mat, int row, int col, int value) {
        byte[] tmp = new byte[1];
        if (value > 127){
            tmp[0] = (byte)(value - 256);
        }
        else{
            tmp[0] = (byte)value;
        }
        mat.put(row, col, tmp);
    }

    /**
     * 给 8-bit 3-chan 的 mat 赋值，通常用于 RGB 彩色（BGR）或 HSV
     * 顺序为 B, G, R 或 H, S, V
     *
     * */
    public void matSetValue3(Mat mat, int row, int col, int[] value) {
        byte[] tmp = new byte[3];
        for (int i = 0; i < 3; i++){
            if (value[i] > 127){
                tmp[i] = (byte)(value[i] - 256);
            }
            else{
                tmp[i] = (byte)value[i];
            }
        }
        mat.put(row, col, tmp);
    }

    /**
     * 给 32-bit 1-chan 的 mat 赋值，通常用于 float GrayScale 灰度
     *
     * */
    public void matSetValue4(Mat mat, int row, int col, float value) {
        float[] tmp = new float[1];
        tmp[0] = value;
        mat.put(row, col, tmp);
    }

    /**
     * 将 32Float 图像转换成可以显示的灰度图
     *
     * */
    public Mat convertImage32F(Mat source){
        Mat result = new Mat();
        source.convertTo(result, CvType.CV_8UC1,255,0);
        return result;
    }

    /**
     * 获取 8-bit 灰度像素值
     *
     * */
    public int matGetValue1(Mat source, int row, int col){
        byte[] value = new byte[1];
        source.get(row, col, value);
        int result = value[0];
        if (result < 0) {
            return (result + 256);
        }
        return result;
    }

    /**
     * 获取 RGB 图像的颜色
     * 顺序为 B, G, R
     *
     * */
    public int[] matGetValue3(Mat source, int row, int col){
        byte[] value = new byte[3];
        source.get(row, col, value);
        int[] result = new int[3];
        for (int i = 0; i < 3; i++){
            result[i] = value[i];
            if (result[i] < 0){result[i] += 256;}
        }
        return result;
    }

    /**
     * 获取 32F 图像一个点的值
     * */
    public float matGetValue4(Mat source, int row, int col){
        float[] value = new float[1];
        source.get(row, col, value);
        return value[0];
    }


    public Mat convertHoughImage(int[][][] houghSpace, int max, int x, int y, int r){
        int width = x;
        int height = y;
        int depth = r;
        Mat result = new Mat(height, width, CvType.CV_8UC3);

        for (int j = 0; j < height; j++){
            for (int i = 0; i< width; i++){
                int maxK = 0;
                int maxK_index = 0;
                for (int k = 0; k < depth; k++){
                    if (houghSpace[i][j][k] > maxK){
                        maxK = houghSpace[i][j][k];
                        maxK_index = k;
                    }
                }
                int value1 = (int)(255.0*maxK/max);
                int value2 = (int)((1.0-maxK/r) * value1);
                int[] value = new int[3];
                value[0] = value1;
                value[1] = value2;
                value[2] = value2;
                matSetValue3(result, j, i, value);
            }
        }

        return result;
    }

}