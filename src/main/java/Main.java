import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import sun.tools.jconsole.AboutDialog;

import java.lang.reflect.Array;
import java.util.*;

import static org.opencv.core.Core.*;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE;

public class Main {

    /** Global variables */
    static String cascade_name = "materials/dartcascade/cascade.xml";
    public static CascadeClassifier cascade;
    private static OpenCVHelper helper = new OpenCVHelper();

    public static void main(String[] args) {
        //Manually store the correct location of dartboards of different images, for validation purpose

        List<BoundingBox> dart13BoundingBoxes = new ArrayList<BoundingBox>();
        dart13BoundingBoxes.add(new BoundingBox(425, 140, 95, 110));

        List<BoundingBox> dart14BoundingBoxes = new ArrayList<BoundingBox>();
        dart14BoundingBoxes.add(new BoundingBox(472, 212, 80, 100));
        dart14BoundingBoxes.add(new BoundingBox(734, 186, 103, 98));

        List<BoundingBox> dartboard14 = new ArrayList<BoundingBox>();
        dartboard14.add(new BoundingBox(100, 89, 160, 140));
        dartboard14.add(new BoundingBox(980, 86, 140, 140));

        List<BoundingBox> dartboard4 = new ArrayList<BoundingBox>();
        dartboard4.add(new BoundingBox(155, 55, 250, 260));

        List<BoundingBox> dartboard5 = new ArrayList<BoundingBox>();
        dartboard5.add(new BoundingBox(420, 127, 142, 142));

        List<BoundingBox> dartboard0 = new ArrayList<BoundingBox>();
        dartboard0.add(new BoundingBox(413 , 4, 213, 213));

        List<BoundingBox> dartboard1 = new ArrayList<BoundingBox>();
        dartboard1.add(new BoundingBox(175, 117, 235, 192));

        List<BoundingBox> dartboard2 = new ArrayList<BoundingBox>();
        dartboard2.add(new BoundingBox(84, 77, 121, 125));

        List<BoundingBox> dartboard3 = new ArrayList<BoundingBox>();
        dartboard3.add(new BoundingBox(321, 144, 78, 85));

        List<BoundingBox> dartboard6 = new ArrayList<BoundingBox>();
        dartboard6.add(new BoundingBox(200, 105, 90, 90));

        List<BoundingBox> dartboard7 = new ArrayList<BoundingBox>();
        dartboard7.add(new BoundingBox(232, 146, 196, 186));

        List<BoundingBox> dartboard8 = new ArrayList<BoundingBox>();
        dartboard8.add(new BoundingBox(63, 230, 78, 119));
        dartboard8.add(new BoundingBox(826, 208, 156, 150));

        List<BoundingBox> dartboard9 = new ArrayList<BoundingBox>();
        dartboard9.add(new BoundingBox(180, 39, 260, 250));

        List<BoundingBox> dartboard10 = new ArrayList<BoundingBox>();
        dartboard10.add(new BoundingBox(79, 102, 125, 125));
        dartboard10.add(new BoundingBox(590, 117, 74, 112));
        dartboard10.add(new BoundingBox(913, 145, 45, 91));

        List<BoundingBox> dartboard11 = new ArrayList<BoundingBox>();
        dartboard11.add(new BoundingBox(175, 97, 63, 64));

        List<BoundingBox> dartboard12 = new ArrayList<BoundingBox>();
        dartboard12.add(new BoundingBox(150, 70, 80, 140));

        List<BoundingBox> dartboard13 = new ArrayList<BoundingBox>();
        dartboard13.add(new BoundingBox(260, 110, 153, 141));

        List<BoundingBox> dartboard15 = new ArrayList<BoundingBox>();
        dartboard15.add(new BoundingBox(132, 37, 170, 166));

        // 1. Read Input Image
        Mat frame = imread(args[0], CV_LOAD_IMAGE_COLOR);

        // 2. Load the Strong Classifier in a structure called `Cascade'
        cascade = new CascadeClassifier();

        if (!cascade.load(cascade_name)) {
            System.out.println("--(!)Error loading");

            return;
        }

        //decide the dartboard to detect
        List<BoundingBox> dartBoard = dartboard1;

        //image transform
        Mat detected = cannyEdgedetector(frame);

        //using viola jones detector to detect potential dartboard (this detector has high false positive rate)
        List<BoundingBox> detectedBoxes = detectAndDisplay(frame);

        //get line detection result
        Mat transformed = HoughTransform(detected);

        //transform line result into coordinate system
        List<MyTuple> lines = DrawLines(frame, transformed, 80);

        //find intersecting lines
        List<Coordinate> intersections = intersectingPoints(lines,frame);

        //get circle detection result
        int[][][] imageSpace = HoughTransformCircle(detected, frame.cols(), frame.rows(), 200);

        //transform circle result into coordinate system
        List<Circle> circles = drawCircle(frame, imageSpace, frame.cols(), frame.rows(), 200, lines);

        //transform circles into boxes
        List<BoundingBox> boxesOfCircles = DrawBoxes(circles);

        //combine vj detector, circleBoxes and intersections to locate dartboards
        List<BoundingBox> combined = combineVjAndCircles(frame, detectedBoxes, boxesOfCircles, intersections);

        //3. calculate f1-score
        int successNumber = successfulDetection(combined, dartBoard);
        float precision = successNumber / (float) combined.size();
        float recall = successNumber / (float) dartBoard.size();
        float f1_score = 2 * (precision * recall) / (precision + recall);

        System.out.println("Truth positive number: " + successNumber);
        System.out.println("Ground truth number: " + dartBoard.size());
        System.out.println("Truth positive rate: " + recall);
        System.out.println("F1 score: " + f1_score);

        imwrite("edgeMagnitude.png", detected);
        imwrite("DetectedCircle.png", frame);
        imwrite("transformedSpace.png", transformed);

        System.exit(0);
    }

    static List<BoundingBox> combineVjAndCircles(Mat frame, List<BoundingBox> detectedBoxes,
                                                 List<BoundingBox> circleBoxes, List<Coordinate> intersetcions) {
        List<BoundingBox> combinedboxes = new ArrayList<BoundingBox>();
        List<BoundingBox> finalbox = new ArrayList<BoundingBox>();
        Map<BoundingBox, Coordinate> detectedCentrePoints = new HashMap<BoundingBox, Coordinate>();
        Map<BoundingBox, Coordinate> circleCentrePoints = new HashMap<BoundingBox, Coordinate>();

        for (BoundingBox dbox : detectedBoxes) {
            detectedCentrePoints.put(dbox, new Coordinate(dbox.x + dbox.width/2, dbox.y + dbox.height / 2));
        }

        for (BoundingBox cbox : circleBoxes) {
            circleCentrePoints.put(cbox, new Coordinate(cbox.x + cbox.width/2, cbox.y + cbox.height / 2));
        }

        for (BoundingBox cbox : circleBoxes) {
            int minDist = 100000;

            for (BoundingBox dbox : detectedBoxes) {
                int currentDist = (int) Math.hypot(circleCentrePoints.get(cbox).x - detectedCentrePoints.get(dbox).x,
                        circleCentrePoints.get(cbox).y - detectedCentrePoints.get(dbox).y);

                if (currentDist < minDist) {
                    minDist = currentDist;
                }
            }

            if (minDist < 0.2 * cbox.height) {
                combinedboxes.add(cbox);
            }
        }

        for (BoundingBox box : combinedboxes) {
            for (Coordinate cor : intersetcions) {
                int Dist = (int) Math.hypot(box.x + box.width / 2 - cor.x, box.y + box.height / 2 - cor.y);

                if (Dist < 0.2 * box.width) {
                    finalbox.add(box);
                }
            }
        }

        if (combinedboxes.size() == 0) {
            for (BoundingBox cbox : circleBoxes){
                float max = 0;
                BoundingBox currentbox = null;

                for (BoundingBox dbox : detectedBoxes) {
                    float currentIOU = IntersectionOverUnion(dbox, cbox);

                    if (currentIOU > max && currentIOU > 0.4) {
                        max = currentIOU;
                        currentbox = cbox;
                    }
                }

                if (currentbox != null) {
                    finalbox.add(currentbox);
                }
            }
        }

        finalbox = combineBoxes(finalbox);

        for (BoundingBox b : finalbox) {
            rectangle(
                    frame, new Point(b.x, b.y), new Point(b.x + b.width, b.y + b.height),
                    new Scalar(0, 255 , 0), 2);
        }

        return finalbox;
    }

    static int successfulDetection(List<BoundingBox> detected, List<BoundingBox> groundTruth) {
        int successNumber = 0;

        for (BoundingBox box : groundTruth) {
            for (int x = 0; x < detected.size(); x++) {
                float ratio = IntersectionOverUnion(detected.get(x), box);

                if (ratio >= 0.5f) {
                    successNumber++;
                }
            }
        }

        return successNumber;
    }

    static List<BoundingBox> detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat frame_gray = new Mat();
        List<BoundingBox> detectedBoundingBoxes = new ArrayList<BoundingBox>();

        // 1. Prepare Image by turning it into Grayscale and normalising lighting
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        // 2. Perform Viola-Jones Object Detection
        cascade.detectMultiScale(frame_gray, faces,
                1.1, 1, 0| CASCADE_SCALE_IMAGE,
                new Size(50, 50), new Size(500,500));

        Rect[] faceArray = faces.toArray();

        // 3. Print number of Faces found
        System.out.println(faceArray.length);

        // 4. Draw box around faces found
        for (int i = 0; i < faceArray.length; i++) {
            detectedBoundingBoxes.add(
                    new BoundingBox(faceArray[i].x,faceArray[i].y,faceArray[i].width,faceArray[i].height));
        }

        return  detectedBoundingBoxes;
    }

    static boolean IfOverlap(BoundingBox detected, BoundingBox groundTruth) {
        if (detected == null || groundTruth == null) {
            return false;
        }

        boolean IfOverlap = false;

        if (groundTruth.x > detected.x - groundTruth.width &&
                groundTruth.x < detected.x + detected.width &&
                groundTruth.y > detected.y - detected.height &&
                groundTruth.y < detected.y + groundTruth.height) {
            IfOverlap = true;
        }

        return IfOverlap;
    }

    static int IntersectingArea(BoundingBox detected, BoundingBox groundTruth) { //calculate the intersecting area of two rectangles
        int overlappingWidth = 0;
        int overlappingHeight = 0;

        if (detected.x > groundTruth.x &&
                detected.x + detected.width < groundTruth.x + groundTruth.width) {
            overlappingWidth = detected.width;
        } else if (detected.x < groundTruth.x &&
                detected.x +detected.width > groundTruth.x + groundTruth.width) {
            overlappingWidth = groundTruth.width;
        } else {
            overlappingWidth = detected.width + groundTruth.width -
                    (Math.max(groundTruth.x + groundTruth.width, detected.x + detected.width) -
                            Math.min(groundTruth.x, detected.x));
        }

        if (detected.y > groundTruth.y &&
                detected.y + detected.height < groundTruth.y + groundTruth.height) {
            overlappingHeight = detected.height;
        } else if (detected.y < groundTruth.y &&
                detected.y + detected.height > groundTruth.y + groundTruth.height) {
            overlappingHeight = groundTruth.height;
        } else {
            overlappingHeight = detected.height + groundTruth.height -
                    (Math.max(groundTruth.y + groundTruth.height, detected.y + detected.height) -
                            Math.min(groundTruth.y,detected.y));
        }

        return overlappingHeight * overlappingWidth;
    }

    static float IntersectionOverUnion(BoundingBox detected, BoundingBox groundtruth) {
        int intersectingArea = 0;
        int detectedSize = detected.width * detected.height;
        int groundTruthSize = groundtruth.width * groundtruth.height;

        if (IfOverlap(detected, groundtruth)) {
            intersectingArea = IntersectingArea(detected, groundtruth);
        } else {
            return 0.0f;
        }

        float ratio = (intersectingArea / (float)(detectedSize + groundTruthSize - intersectingArea));

        return ratio;
    }

    static Mat cannyEdgedetector(Mat frame) {
        Mat gray = new Mat();
        Mat detected = new Mat();

        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
        GaussianBlur(gray, detected, new Size(7, 7), 0, 0);
        Canny(detected, detected, 50, 200);
        
        return detected;
    }

    static Mat sobel(Mat image){
        Mat gray_image = new Mat();
        Mat drad_x = new Mat();
        Mat drad_y = new Mat();
        Mat dest = new Mat();

        cvtColor(image, gray_image, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(gray_image, gray_image, new Size(3, 3));

        Sobel(gray_image, drad_x, CvType.CV_16S, 1, 0, 3, 1, 0,  BORDER_DEFAULT);
        convertScaleAbs(drad_x, drad_x);

        Sobel(gray_image, drad_y, CvType.CV_16S, 0, 1, 3, 1, 0,  BORDER_DEFAULT);
        convertScaleAbs(drad_y, drad_y);

        addWeighted(drad_x, 0.4, drad_y, 0.4, 0, dest);

        return dest;
    }

    //Hough Transform for Straight lines.
    static Mat HoughTransform(Mat frame) {
        int maxDiagonalDist = (int) Math.ceil(Math.hypot(frame.cols(), frame.rows()));
        Mat outputSpace = new Mat(360, maxDiagonalDist, CvType.CV_8UC1);

        for (int i = 0; i < outputSpace.cols(); i++) {
            for (int j = 0; j < outputSpace.rows(); j++) {
                helper.matSetValue1(outputSpace, j, i, 0);
            }
        }

        for (int y = 0; y < frame.rows(); y++) {
            for (int x = 0; x < frame.cols(); x++) {
                if (helper.matGetValue1(frame, y, x) == 255) {
                    for (int theta = 1; theta <= 360; theta++) {
                        if (theta % 45 != 0) {
                            double degree = (((double) theta * Math.PI) / 180);
                            double rou = 0.0;

                            rou = x * Math.sin(degree) + y * Math.cos(degree);

                            if (theta > 10 && theta < 80 || theta > 100 && theta < 170 || theta > 190 && theta < 260 ||
                                    theta > 280 && theta < 350) {
                                int adjustedRou = (int) Math.round(rou);

                                accumulate(outputSpace, theta, adjustedRou, 1);
                            }
                        }
                    }
                }
            }
        }

        return outputSpace;
    }

    //accumulator
    static void accumulate(Mat frame, int x, int y, int increment) {
        int currentValue = helper.matGetValue1(frame, x, y);

        if (currentValue != 255) {
            helper.matSetValue1(frame, x, y, currentValue + increment);
        }
    }

    //Circle HoughTransform
    static int[][][] HoughTransformCircle(Mat frame, int imageWidth, int imageHeight, int maxR){
        int[][][] outputSpace = new int[imageWidth][imageHeight][maxR];

        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                for (int r = 0; r < maxR; r++) {
                    outputSpace[x][y][r] = 0;
                }
            }
        }

        for (int y = 0; y < frame.rows() - 1; y++) {
            for (int x = 0; x < frame.cols() - 1; x++) {
                if (helper.matGetValue1(frame, y, x) == 255) {
                    for (int r = 35; r < maxR; r++) {
                        for (int theta = 1; theta <= 360; theta += 2) {
                            int a = (int)(x + r * Math.cos(( theta * Math.PI) / 180));
                            int b = (int)(y - r * Math.sin(( theta * Math.PI) / 180));

                            if (a >= 0 && a < imageWidth && b >= 0 && b < imageHeight) {
                                outputSpace[a][b][r] += 1;
                            }
                        }
                    }
                }
            }
        }

        return outputSpace;
    }

    //Find maximum value in a 3d array
    static int FindMaximum(int[][][] frame, int imageWidth, int imageHeight, int maxR) {
        int maximum = 0;

        for (int y = 0; y < imageHeight ; y++) {
            for (int x = 0; x < imageWidth ; x++) {
                for (int r = 0; r < maxR; r++) {
                    if (frame[x][y][r] > maximum) {
                        maximum = frame[x][y][r];
                    }
                }
            }
        }

        return maximum;
    }

    //select circles from hough space, and reduce circles inside big circles
    static List<Circle> drawCircle(
            Mat original, int[][][] frame,  int imageWidth, int imageHeight, int maxR, List<MyTuple> intersects) {
        int maxValue = FindMaximum(frame, imageWidth, imageHeight, maxR);
        List<Circle> circles = new ArrayList<Circle>();

        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                for (int r = 0; r < maxR; r++) {
                    if (frame[x][y][r] >= maxValue * 0.8) {
                        circles.add(new Circle(x, y, r));
                    }
                }
            }
        }

        List<Circle> smallCircles = SmallCircles(circles);
        List<Circle> result = removeSmallCircle(smallCircles, circles);

        return result;
    }

    // reduce overlapping boxes to one
    static List<BoundingBox> combineBoxes(List<BoundingBox> box) {
        List<BoundingBox> bigbox = new ArrayList<BoundingBox>();

        for (BoundingBox b : box){
            BoundingBox current = null;
            for (BoundingBox comp : box) {
                if (IfOverlap(b, comp)) {
                    if (current == null) {
                        current = comp;
                    } else {
                        if (comp.height * comp.width > current.height * current.width &&
                                Math.abs(current.x + current.width/2 - comp.x + comp.width/2) < comp.width * 0.2 &&
                                Math.abs(current.y + current.height/2 - comp.y + comp.height/2) < comp.height * 0.2) {
                            current = comp;
                        }
                    }
                }
            }

            if (bigbox.size() == 0) {
                bigbox.add(current);
            } else {
                int doNotIntersect = 0;

                for (BoundingBox boundingBox : bigbox) {
                    if (!IfOverlap(current, boundingBox)) {
                        doNotIntersect++;
                    }
                }

                if (doNotIntersect == bigbox.size()) {
                    bigbox.add(current);
                }
            }
        }

        return bigbox;
    }

    static List<BoundingBox> DrawBoxes(List<Circle> circles) {
        List<BoundingBox> box = new ArrayList<BoundingBox>();

        for (Circle circle: circles) {
            box.add(new BoundingBox(circle.x - circle.radius, circle.y - circle.radius,
                    circle.radius * 2 + 5, circle.radius * 2 + 5));
        }

        return box;
    }

    public static int findThreshold(Mat frame) {
        int[] histarray = new int[256];

        for (int i = 0; i < 256; i++) {
            histarray[i] = 0;
        }

        for (int i = 0; i < frame.rows(); i++) {
            for (int j = 0; j < frame.cols(); j++) {
                histarray[helper.matGetValue1(frame, i, j)] += 1;
            }
        }

        int threshold = (int) (histarray.length / 2.0);
        int deltaT = histarray.length;
        int oldT = threshold;

        while (deltaT > 2) {
            int avgLeft = 0;
            int totalLeft = 0;
            int avgRight = 0;
            int totalRight = 0;
            int numberLeft = 0;
            int numberRight = 0;

            for (int i = 0; i < threshold; i++) {
                totalLeft += histarray[i] * i;
                numberLeft += histarray[i];
            }

            avgLeft = (int) (totalLeft / (double) numberLeft);

            for (int i = threshold; i < histarray.length; i++) {
                totalRight += histarray[i] * i;
                numberRight += histarray[i];
            }

            avgRight = (int) (totalRight / (double) numberRight);
            oldT = threshold;
            threshold = (avgLeft + avgRight) / 2;
            deltaT = Math.abs(threshold - oldT);
        }

        return threshold;
    }

    static int getMax(int array[]) {
        int max = 0;

        for (int i = 0; i < array.length; i++){
            if(array[i] > max){
                max = array[i];
            }
        }

        return max;
    }

    static List<Circle> SmallCircles(List<Circle> circles){
        List<Circle> smallCircle = new ArrayList<Circle>();

        for (Circle c1 : circles){
            for (Circle c2 : circles){
                double centreDist = Math.sqrt(Math.pow(c1.x - c2.x, 2) + Math.pow(c1.y - c2.y, 2));
                if (!(centreDist == 0.0 && c1.radius == c2.radius)){
                    if (centreDist + c1.radius < c2.radius){
                        smallCircle.add(c1);
                    }
                }
            }
        }

        return smallCircle;
    }

    static List<Circle> removeSmallCircle(List<Circle> smallCircles, List<Circle> circles) {
        List<Circle> resCircles = new ArrayList<Circle>();

        for (Circle c : circles) {
            boolean flag = true;

            for (Circle sc : smallCircles) {
                if (c.x == sc.x && c.y == sc.y && c.radius == sc.radius) {
                    flag = false;

                    break;
                }
            }

            if (flag) {
                resCircles.add(c);
            }
        }

        return resCircles;
    }

    static List<Coordinate> intersectingPoints(List<MyTuple> lines, Mat frame){
        List<Coordinate> threeLinesintersections = new ArrayList<Coordinate>();
        List<Coordinate> twoLinesIntersection = new ArrayList<Coordinate>();

        for (MyTuple tuple : lines) {
            //y = kx + b
            double x1 = 0.0;
            double x2 = frame.cols();
            double y1 = tuple.rho / Math.cos(tuple.theta * Math.PI / 180.0);
            double y2 = tuple.rho / Math.cos(tuple.theta * Math.PI / 180.0) - frame.cols() * Math.tan(tuple.theta * Math.PI / 180.0);
            double theta = (double) tuple.theta * Math.PI / 180.0;
            double k1 = ((y2 - y1) / (x2 - x1));
            double b1 = (double) tuple.rho / Math.cos(theta);

            for (MyTuple comp : lines) {
                double x3 = 0.0;
                double x4 = frame.cols();
                double y3 = comp.rho / Math.cos(comp.theta * Math.PI / 180.0);
                double y4 = comp.rho / Math.cos(comp.theta * Math.PI / 180.0) - frame.cols() * Math.tan(comp.theta * Math.PI / 180.0);
                double theta_comp = (double) comp.theta * Math.PI / 180.0;
                double k2 = (y4 - y3) / (x4 - x3);
                double b2 = (double) comp.rho / Math.cos(theta_comp);

                if (k1 != k2 && k1 != 0 && k2 != 0) {
                    int x = (int) ((b2 - b1) / (k1 - k2));
                    int y = (int) (k1 * x + b1);
                    if (x < frame.cols() - 2 && y < frame.rows() - 2 && x > 0 && y > 0){
                        twoLinesIntersection.add(new Coordinate(x, y));
                    }
                }
            }
        }

        if(twoLinesIntersection.size() < 10) {
            for (Coordinate cor : twoLinesIntersection){
                threeLinesintersections.add(cor);
            }
        } else {
            for (int y = 0; y < frame.rows() - 4; y++) {
                for (int x = 0; x < frame.cols() - 4; x++) {
                    List<Coordinate> block = new ArrayList<Coordinate>();

                    for (int a = y; a < y + 4; a++) {
                        for (int b = x; b < x + 4; b++) {
                            block.add(new Coordinate(b, a));
                        }
                    }

                    block.retainAll(twoLinesIntersection);

                    if (block.size() > 3) {
                        threeLinesintersections.add(new Coordinate(x + 2, y + 2));
                    }
                }
            }
        }

        return threeLinesintersections;
    }

    static List<MyTuple> DrawLines(Mat origin, Mat houghSpace, double threshold){
        List<MyTuple> lines = new ArrayList<MyTuple>();
        int[][] image = new int[houghSpace.cols() + 10][houghSpace.rows() + 10];

        for (int y = 0; y < houghSpace.rows(); y ++) {
            for (int x = 0; x < houghSpace.cols(); x ++) {
                image[x][y] = helper.matGetValue1(houghSpace, y, x);
            }
        }

        Mat updateimage = Mat.zeros(houghSpace.rows(), houghSpace.cols(), CvType.CV_8UC1);

        for (int y = 14; y < houghSpace.rows() - 14; y ++) {
            for (int x = 14; x < houghSpace.cols() - 14; x ++) {
                int thisvalue = image[x][y];
                int max = 0;

                for (int sy = y - 14; sy < y + 14; sy++) {
                    for (int sx = x - 14; sx < x + 14; sx++) {
                        int current = image[sx][sy];

                        if (current >= max) {
                            max = current;
                        }
                    }
                }

                if(thisvalue == max) {
                    helper.matSetValue1(updateimage, y, x, max);
                }
            }
        }

        //row = theta   col = rho
        int lineNum = 0;
        int thres = 70;

        while (lineNum < 60) {    //adaptive threshold changing
            for (int y = 1; y < updateimage.rows() - 1 ; y++) {
                for (int x = 1; x < updateimage.cols() - 1 ; x++) {
                    if (helper.matGetValue1(updateimage, y, x) > thres && helper.matGetValue1(updateimage, y, x) < 180) {
                        lines.add(new MyTuple(y, x));
                    }
                }
            }

            lineNum = lines.size();

            if (lineNum < 60) {
                lines.clear();
            }

            if (thres > 2) {
                thres -= 2;
            }
        }

        return lines;
    }
}
