package org.opencv.samples.colorblobdetect;

import android.os.Environment;
import android.provider.ContactsContract;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static android.content.ContentValues.TAG;
import static org.opencv.imgproc.Imgproc.HoughLinesP;
import static org.opencv.imgproc.Imgproc.minAreaRect;

public class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);

    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;

    // Color radius for range checking in HSV color space with touch color
    private Scalar mColorRadius = new Scalar(25, 50, 50, 0);

    private Mat mSpectrum = new Mat();
    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();
    private List<MatOfPoint> mPolys = new ArrayList<MatOfPoint>();


    // Cache
    Mat mPyrDownMat = new Mat();
    Mat mHsvMat = new Mat();
    Mat mMask = new Mat();
    Mat mDilatedMask = new Mat();
    Mat mHierarchy = new Mat();
    Mat mLines = new Mat();
    //Mat mRectangle = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(7,7),new Point(7,7));
    ArrayList<Integer> Xs = new ArrayList<>(), Ys = new ArrayList<>();
    int totalX = 0, totalY = 0;
    Point centerP = new Point();

    LineClusters clusters = new LineClusters();


    public void setColorRadius(Scalar radius) {
        mColorRadius = radius;
    }

    public void setColorRange(Scalar minHSV, Scalar maxHSV) {
        mLowerBound = minHSV;
        mUpperBound = maxHSV;
    }

    public void checkContours(List<MatOfPoint> contours) {

        mPolys.clear();
        for (MatOfPoint c : contours) {
            MatOfPoint2f thisContour2f = new MatOfPoint2f();
            MatOfPoint approxContour = new MatOfPoint();
            MatOfPoint2f approxContour2f = new MatOfPoint2f();

            c.convertTo(thisContour2f, CvType.CV_32FC2);

            Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

            approxContour2f.convertTo(approxContour, CvType.CV_32S);
            if (Imgproc.contourArea(approxContour) > 50) mPolys.add(approxContour);
        }
    }

    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.val[0] >= mColorRadius.val[0]) ? hsvColor.val[0] - mColorRadius.val[0] : 0;
        double maxH = (hsvColor.val[0] + mColorRadius.val[0] <= 255) ? hsvColor.val[0] + mColorRadius.val[0] : 255;

        mLowerBound.val[0] = minH;
        mUpperBound.val[0] = maxH;

        //mLowerBound.val[1] = hsvColor.val[1] - mColorRadius.val[1];
        //mUpperBound.val[1] = hsvColor.val[1] + mColorRadius.val[1];
        mLowerBound.val[1] = 0;
        mUpperBound.val[1] = 255;

        //mLowerBound.val[2] = hsvColor.val[2] - mColorRadius.val[2];
        //mUpperBound.val[2] = hsvColor.val[2] + mColorRadius.val[2];
        mLowerBound.val[2] = 0;
        mUpperBound.val[2] = 255;

        mLowerBound.val[3] = 0;
        mUpperBound.val[3] = 255;

        Log.i(TAG, "Set HSV Color Min: (" +
                mLowerBound.val[0] + "," +
                mLowerBound.val[1] + "," +
                mLowerBound.val[2] + ")");

        Log.i(TAG, "Set HSV Color Max: (" +
                mUpperBound.val[0] + "," +
                mUpperBound.val[1] + "," +
                mUpperBound.val[2] + ")");

        Mat spectrumHsv = new Mat(1, (int) (maxH - minH), CvType.CV_8UC3);

        for (int j = 0; j < maxH - minH; j++) {
            byte[] tmp = {(byte) (minH + j), (byte) 255, (byte) 255};
            spectrumHsv.put(0, j, tmp);
        }

        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public void setMinContourArea(double area) {
        mMinContourArea = area;
    }

    public void processPolys(Mat mRgba, CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        try {
            mRgba = inputFrame.rgba();
            Imgproc.cvtColor(mRgba, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);
            //Imgproc.pyrDown(mHsvMat,mHsvMat);

            if (mUpperBound.val[0] < mLowerBound.val[0]) {

                Mat mMask1 = new Mat();
                Mat mMask2 = new Mat();


                Scalar tLowerBound = mLowerBound.clone();
                Scalar tUpperBound = mUpperBound.clone();

                tLowerBound.val[0] = 0.0;
                tUpperBound.val[0] = mUpperBound.val[0];

                Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask1);

                tLowerBound = mLowerBound.clone();
                tUpperBound.val[0] = 255.0;

                Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask2);

                Core.add(mMask1, mMask2, mMask);

            } else {
                Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
            }
            List<MatOfPoint> polys = new ArrayList<MatOfPoint>();

            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

            Imgproc.findContours(mRgba, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            checkContours(contours);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void showPolys(Mat mRgba) {
        try {

            for (int i = 0; i < mPolys.size(); i++) {
                Imgproc.drawContours(mRgba, mPolys, i, new Scalar(255, 255, 255), 5);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    final int PYR = 1;

    public void processLines(Mat mRgba, CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);
//        Imgproc.pyrDown(mHsvMat,mHsvMat);
//        Imgproc.pyrDown(mHsvMat,mHsvMat);

        if (mUpperBound.val[0] < mLowerBound.val[0]) {

            Mat mMask1 = new Mat();
            Mat mMask2 = new Mat();


            Scalar tLowerBound = mLowerBound.clone();
            Scalar tUpperBound = mUpperBound.clone();

            tLowerBound.val[0] = 0.0;
            tUpperBound.val[0] = mUpperBound.val[0];

            Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask1);

            tLowerBound = mLowerBound.clone();
            tUpperBound.val[0] = 255.0;

            Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask2);

            Core.add(mMask1, mMask2, mMask);

        } else {
            Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
        }

        Imgproc.erode(mMask, mMask, Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(5,5)));
        Imgproc.dilate(mMask,mMask,Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT,new Size(13,13)));


        Imgproc.Canny(mMask, mMask, 50, 100);

        HoughLinesP(mMask, mLines, 5, Math.PI / 180, 7, 60 / PYR, 20 / PYR);

        //Imgproc.morphologyEx(mMask,mMask,Imgproc.MORPH_CLOSE,mRectangle);

    }

    public void showLines(Mat mRgba) {

        try {
            double val0, val1, val2, val3;
            clusters = new LineClusters();
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i, 0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);

                val0 = val[0] * PYR;
                val1 = val[1] * PYR;
                val2 = val[2] * PYR;
                val3 = val[3] * PYR;


                double angle = ((Math.atan2(val1 - val3, val0 - val2) * 180 / Math.PI) + 180) % 180;

                //Log.println(Log.ASSERT, "TAG", angle+"degrees loop:" + i);

                if ((isWithin(angle, 30, 150) && !isWithin(angle, 80, 100)) &&
                        val1 > mRgba.rows() / 4 && val3 > mRgba.rows() / 4) {
                    Imgproc.line(mRgba,
                            new Point(val0, val1),
                            new Point(val2, val3),
                            new Scalar(255, 255, 255),
                            10);
                    clusters.add(new Line(new Point(val0, val1), new Point(val2, val3), angle));
                    Log.println(Log.ASSERT, "TAG", angle + " is the angle of line " + i);
                }
//                else Imgproc.line(mRgba,
//                            new Point(val0, val1),
//                            new Point(val2, val3),
//                            new Scalar(255, 0, 0),
//                            10);
            }
            Log.println(Log.ASSERT, "TAG", clusters.toString() + "\n");
            final int SHOW_THRESH = 2;
            for (int i = 0; i < clusters.clusterGroups.size() && i < 2; i++) {
                if (clusters.clusterGroups.get(i).lines.size() >= SHOW_THRESH) {
                    Point[] rectPoints = new Point[4];
                    MatOfPoint2f mp2f = new MatOfPoint2f();
                    mp2f.fromList(clusters.clusterGroups.get(i).points);
                    RotatedRect rRect = minAreaRect(mp2f);
                    rRect.points(rectPoints);
                    MatOfPoint mPoints = new MatOfPoint(rectPoints);
                    List<MatOfPoint> lPoints = new ArrayList<>();
                    lPoints.add(mPoints);
                    Log.println(Log.ASSERT, "TAG", Arrays.toString(rectPoints) + "Points");


                    Imgproc.polylines(mRgba, lPoints, true, new Scalar(0, 255, 0), 10);
                }
            }

            int countedCount = 0;
            totalX = 0;
            totalY = 0;

            for (int i = 0; i < 2 && i < clusters.clusterGroups.size(); i++) {
                LineCluster l = clusters.clusterGroups.get(i);
                if (l.lines.size() >= SHOW_THRESH) {
                    totalX += l.center().x;
                    totalY += l.center().y;
                    countedCount++;
                }
            }

            totalX = totalX / (countedCount);
            totalY = totalY / (countedCount);

            final int QUEUE_SIZE = 15;

            if (!(Xs.size() < QUEUE_SIZE) || !(Ys.size() < QUEUE_SIZE)) {
                Xs.remove(0);
                Ys.remove(0);
            }

            Xs.add(totalX);
            Ys.add(totalY);
            centerP = new Point(median(Xs), median(Ys));
        }catch (Exception e) {
            e.printStackTrace();
        }
        try{
            Imgproc.circle(mRgba, centerP, 2, new Scalar(255, 0, 0), 5);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static double median(ArrayList<Integer> median) {
        ArrayList<Integer> m = new ArrayList<>();
        for (int i : median) m.add(i);
        Collections.sort(m);
        if (median.size() % 2 == 0)
            return (m.get(m.size() / 2) + m.get(m.size() / 2 - 1)) / 2;
        else return m.get(m.size() / 2);
    }

    public void showHorizontalLines(Mat mRgba) {
        try {
            Imgproc.pyrUp(mLines, mLines);
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i, 0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);


                double angle = Math.atan2(val[1] - val[3], val[0] - val[2]) * 180 / Math.PI;

                Log.println(Log.ASSERT, "TAG", angle + "");

                if (isWithin(angle, 160, 180) || isWithin(angle, -180, -160) || isWithin(angle, -20, 20))
                    Imgproc.line(mRgba,
                            new Point(val[0], val[1]),
                            new Point(val[2], val[3]),
                            new Scalar(200, 128, 0),
                            10);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void showVerticalLines(Mat mRgba) {
        try {
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i, 0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);


                double angle = Math.atan2(val[1] - val[3], val[0] - val[2]) * 180 / Math.PI;

                Log.println(Log.ASSERT, "TAG", angle + "");

                if (isWithin(angle, 80, 100) || isWithin(angle, -100, -80)) Imgproc.line(mRgba,
                        new Point(val[0], val[1]),
                        new Point(val[2], val[3]),
                        new Scalar(0, 255, 0),
                        10);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public boolean isWithin(double val, double low, double high) {
        return val >= low && val <= high;
    }

    public void process(Mat rgbaImage) {
        Imgproc.pyrDown(rgbaImage, mPyrDownMat);
        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);

        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);

        if (mUpperBound.val[0] < mLowerBound.val[0]) {

            Mat mMask1 = new Mat();
            Mat mMask2 = new Mat();


            Scalar tLowerBound = mLowerBound.clone();
            Scalar tUpperBound = mUpperBound.clone();

            tLowerBound.val[0] = 0.0;
            tUpperBound.val[0] = mUpperBound.val[0];

            Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask1);

            tLowerBound = mLowerBound.clone();
            tUpperBound.val[0] = 255.0;

            Core.inRange(mHsvMat, tLowerBound, tUpperBound, mMask2);

            Core.add(mMask1, mMask2, mMask);

        } else {
            Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
        }

        Imgproc.dilate(mMask, mDilatedMask, new Mat());

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(mDilatedMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Find max contour area
        double maxArea = 0;
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint wrapper = each.next();
            double area = Imgproc.contourArea(wrapper);
            if (area > maxArea)
                maxArea = area;
        }

        // Filter contours by area and resize to fit the original image size
        mContours.clear();
        each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint contour = each.next();
            if (Imgproc.contourArea(contour) > mMinContourArea * maxArea) {
                Core.multiply(contour, new Scalar(4, 4), contour);
                mContours.add(contour);
            }
        }
        
        
    }


    class Line {
        Point p1, p2;
        double angle;

        public Line(Point p1, Point p2, double angle) {
            this.p1 = p1;
            this.p2 = p2;
            this.angle = angle;
        }
    }

    class LineCluster implements Comparable{

        RotatedRect rRect = new RotatedRect();
        List<Line> lines = new ArrayList<>();
        List<Point> points = new ArrayList<>();
        double angle = 0;
        double area = 0;

        public LineCluster(Line line) {
            addLine(line);
        }

        public void addLine(Line line) {
            lines.add(line);
            points.add(line.p1);
            points.add(line.p2);
            avgAngle();
            updateRect();
        }

        private void updateRect() {
            MatOfPoint2f mp2f = new MatOfPoint2f();
            mp2f.fromList(points);
            rRect = minAreaRect(mp2f);
            area = rRect.size.area();
        }

        public boolean isClose(Point p, double tolerance) {
            for (int i = 0; i < points.size(); i++) {
                if (Math.hypot(p.x - points.get(i).x, p.y - points.get(i).y) <= tolerance) {
                    return true;
                }
            }
            return false;
        }

        public Point center() {
            return rRect.center;
        }

        public void avgAngle() {
            int n = 0;
            for (int i = 0; i < lines.size(); i++) {
                n += lines.get(i).angle;
            }
            angle = n / lines.size();
        }

        public String toString() {
            avgAngle();
            return "Angle is " + angle + "Lines are " + lines.size();
        }

        @Override
        public int compareTo(Object o) {
            if(o instanceof LineCluster) {
                if (area > ((LineCluster)o).area) return 1;
                else if(area < ((LineCluster)o).area) return -1;
                else return 0;
            }else return 0;
        }
    }

    class LineClusters {
        List<LineCluster> clusterGroups = new ArrayList<>();

        public void add(Line line) {
            boolean foundCluster = false;
            outerloop:
            for (int i = 0; i < clusterGroups.size(); i++) {
                for (int j = 0; j < clusterGroups.get(i).lines.size(); j++) {
                    if (isWithin(line.angle, clusterGroups.get(i).angle - 7, clusterGroups.get(i).angle + 7)) {
                        if (clusterGroups.get(i).isClose(line.p1, 80) || clusterGroups.get(i).isClose(line.p2, 80)) {
                            clusterGroups.get(i).addLine(line);
                            foundCluster = true;
                            break outerloop;
                        }
                    }
                }
            }
            
            if (!foundCluster) {
                clusterGroups.add(new LineCluster(line));
            }
            Collections.sort(clusterGroups);

        }


        public void writeSolutionPoint() {
            if(clusterGroups.size() >= 2){
                LineCluster lc = clusterGroups.get(0);
                LineCluster lc0 = clusterGroups.get(1);
                logToFile(calculateIntersect(lc.angle, lc.center(), lc0.angle, lc0.center()));
            }
        }


        public String toString() {
            String returnVal = "";
            for (int i = 0; i < clusterGroups.size(); i++) {
                returnVal += "cluster " + i + " " + clusterGroups.get(i).toString() + "\n";
            }
            return returnVal;
        }

    }


    public List<MatOfPoint> getContours() {
        return mContours;
    }


    //VERY IMPORTANT! THE 'Y' VALUE THAT IS RETURNED NEEDS TO BE NEGATED TO BE USEFULL
    //TODO
    //TODO
    //TODO
    public static String calculateIntersect(double angle1, Point p1, double angle2, Point p2) {
        Point intersect = new Point();
        angle1 *= Math.PI/180;
        angle2 *= Math.PI/180;
        int x1 = (int) p1.x;
        int y1 = -(int) p1.y;
        int x2 = (int) p2.x;
        int y2 = -(int) p2.y;
        double m1 = Math.tan(angle1);
        double m2 = Math.tan(angle2);
        int xf = (int) (((-m2*x2) + y2 + (m1 * x1) - y1)/(m1-m2));
        int yf = (int) ((m1*xf) - (m1*x1) + y1);
        intersect = new Point(xf,yf);
        return intersect.x+"\t"+intersect.y;
    }

    public static void logToFile(Object o)
    {
        String text = o.toString();
        File logFile = new File(Environment.getExternalStorageDirectory()+"/Angles/Data/data.txt");
        if (!logFile.exists())
        {
            try
            {
                logFile.createNewFile();
            }
            catch (Exception e)
            {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try
        {
            //BufferedWriter for performance, true to set append to file flag
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
            buf.append(text);
            buf.append(System.lineSeparator());
            buf.close();
        }
        //this is a commit
        catch (Exception e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

}
