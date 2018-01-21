package org.opencv.samples.colorblobdetect;

import android.provider.ContactsContract;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static android.content.ContentValues.TAG;
import static org.opencv.imgproc.Imgproc.HoughLinesP;
import static org.opencv.imgproc.Imgproc.initUndistortRectifyMap;
import static org.opencv.imgproc.Imgproc.minAreaRect;

public class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);

    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;

    // Color radius for range checking in HSV color space with touch color
    private Scalar mColorRadius = new Scalar(25,50,50,0);

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


    public void setColorRadius(Scalar radius) {
        mColorRadius = radius;
    }

    public void setColorRange(Scalar minHSV, Scalar maxHSV) {
        mLowerBound = minHSV;
        mUpperBound = maxHSV;
    }

    public void checkContours(List<MatOfPoint> contours) {

        mPolys.clear();
        for (MatOfPoint c : contours){
            MatOfPoint2f thisContour2f = new MatOfPoint2f();
            MatOfPoint approxContour = new MatOfPoint();
            MatOfPoint2f approxContour2f = new MatOfPoint2f();

            c.convertTo(thisContour2f, CvType.CV_32FC2);

            Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

            approxContour2f.convertTo(approxContour, CvType.CV_32S);
            if(Imgproc.contourArea(approxContour) > 50) mPolys.add(approxContour);
        }
    }

    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.val[0] >= mColorRadius.val[0]) ? hsvColor.val[0]-mColorRadius.val[0] : 0;
        double maxH = (hsvColor.val[0]+mColorRadius.val[0] <= 255) ? hsvColor.val[0]+mColorRadius.val[0] : 255;

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

        Mat spectrumHsv = new Mat(1, (int)(maxH-minH), CvType.CV_8UC3);

        for (int j = 0; j < maxH-minH; j++) {
            byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
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

    public void processPolys(Mat mRgba,CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
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

            Imgproc.findContours(mMask, contours, mHierarchy , Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            checkContours(contours);

        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void showPolys(Mat mRgba){
        try {

           for(int i = 0; i < mPolys.size();i++) {
               Imgproc.drawContours(mRgba,mPolys,i,new Scalar(255,255,255),5);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    final int PYR = 1;

    public void processLines(Mat mRgba, CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba,mHsvMat,Imgproc.COLOR_RGB2HSV_FULL);
        //Imgproc.pyrDown(mHsvMat,mHsvMat);
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

            Core.add(mMask1,mMask2,mMask);

        } else {
            Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
        }

        Imgproc.Canny(mMask, mMask, 50, 100);

        HoughLinesP(mMask, mLines, 5, Math.PI/180, 7,60/PYR, 20/PYR);

        //Imgproc.morphologyEx(mMask,mMask,Imgproc.MORPH_CLOSE,mRectangle);

    }

    public void showLines(Mat mRgba){
        
        try {
            double val0,val1,val2,val3;
            LineClusters clusters = new LineClusters();
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i,0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);

                val0 = val[0] * PYR;
                val1 = val[1] * PYR;
                val2 = val[2] * PYR;
                val3 = val[3] * PYR;
                
                
                double angle = ((Math.atan2(val1-val3,val0-val2)*180/Math.PI) + 180)%180;

                //Log.println(Log.ASSERT, "TAG", angle+"degrees loop:" + i);

                if(isWithin(angle,30,150) && !isWithin(angle,80,100)) {
                    Imgproc.line(mRgba,
                            new Point(val0, val1),
                            new Point(val2, val3),
                            new Scalar(255, 255, 255),
                            10);
                    clusters.add(new Line(new Point(val0,val1),new Point(val2,val3),angle));
                    Log.println(Log.ASSERT,"TAG",angle + " is the angle of line " + i);
                }
//                else Imgproc.line(mRgba,
//                            new Point(val0, val1),
//                            new Point(val2, val3),
//                            new Scalar(255, 0, 0),
//                            10);
            }
            Log.println(Log.ASSERT, "TAG", clusters.toString()+"\n");
            for(int i = 0; i < clusters.clusters.size(); i ++) {
               if(clusters.clusters.get(i).lines.size() > 2) {
                   Point[] rectPoints = new Point[4];
                   MatOfPoint2f mp2f = new MatOfPoint2f();
                   mp2f.fromList(clusters.clusters.get(i).points);
                   RotatedRect rRect = minAreaRect(mp2f);
                   rRect.points(rectPoints);
                   MatOfPoint mPoints = new MatOfPoint(rectPoints);
                   List<MatOfPoint> lPoints = new ArrayList<>();
                   lPoints.add(mPoints);
                   Log.println(Log.ASSERT, "TAG", Arrays.toString(rectPoints) + "Points");

                   Imgproc.polylines(mRgba, lPoints, true, new Scalar(0, 255, 0), 10);
               }

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void showHorizontalLines(Mat mRgba){
        try {
            Imgproc.pyrUp(mLines,mLines);
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i,0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);


                double angle = Math.atan2(val[1]-val[3],val[0]-val[2])*180/Math.PI;

                Log.println(Log.ASSERT, "TAG", angle+"");

                if(isWithin(angle, 160, 180) || isWithin(angle, -180, -160) || isWithin(angle, -20,20))Imgproc.line(mRgba,
                        new Point(val[0], val[1]),
                        new Point(val[2], val[3]),
                        new Scalar(200, 128, 0),
                        10);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void showVerticalLines(Mat mRgba){
        try {
            for (int i = 0; i < mLines.rows(); i++) {
                double[] val = mLines.get(i,0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);


                double angle = Math.atan2(val[1]-val[3],val[0]-val[2])*180/Math.PI;

                Log.println(Log.ASSERT, "TAG", angle+"");

                if(isWithin(angle, 80, 100) || isWithin(angle, -100, -80))Imgproc.line(mRgba,
                        new Point(val[0], val[1]),
                        new Point(val[2], val[3]),
                        new Scalar(0, 255, 0),
                        10);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public boolean isWithin(double val, double low, double high){
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

            Core.add(mMask1,mMask2,mMask);

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
            if (Imgproc.contourArea(contour) > mMinContourArea*maxArea) {
                Core.multiply(contour, new Scalar(4,4), contour);
                mContours.add(contour);
            }
        }
    }

    class Line {
        Point p1, p2;
        double angle;
        public Line(Point p1,Point p2, double angle) {
            this.p1 = p1;
            this.p2 = p2;
            this.angle = angle;
        }
    }
    class LineCluster {
        public LineCluster(Line line) {
            addLine(line);
        }
        public void addLine(Line line) {
            lines.add(line);
            points.add(line.p1);
            points.add(line.p2);
            avgAngle();
        }

        public boolean isClose(Point p, double tolerance) {
            for(int i = 0; i < points.size(); i++) {
                if(Math.hypot(p.x-points.get(i).x,p.y-points.get(i).y) <= tolerance) {
                    return true;
                }
            }
            return false;
        }

        List<Line> lines = new ArrayList<>();
        List<Point> points = new ArrayList<Point>();
        double angle = 0;
        public void avgAngle() {
            int n = 0;
            for(int i = 0; i < lines.size(); i++) {
                n += lines.get(i).angle;
            }
            angle = n / lines.size();
        }
        public String toString() {
            avgAngle();
            return "Angle is " + angle + "Lines are " + lines.size();
        }
    }
    class LineClusters {
        List<LineCluster> clusters = new ArrayList<>();

        public void add(Line line){
            boolean foundCluster = false;
            outerloop:
            for(int i = 0; i < clusters.size(); i ++) {
                for(int j = 0; j < clusters.get(i).lines.size(); j++) {
                    if(isWithin(line.angle ,clusters.get(i).angle - 7, clusters.get(i).angle + 7)) {
                        if(clusters.get(i).isClose(line.p1,80) ||clusters.get(i).isClose(line.p2,80)) {
                            clusters.get(i).addLine(line);
                            foundCluster = true;
                            break outerloop;
                        }
                    }
                }
            }
            if(!foundCluster) {
                clusters.add(new LineCluster(line));

            }
        }

        public String toString() {
            String returnVal = "";
            for(int i = 0; i < clusters.size();i++) {
                returnVal += "cluster " + i + " " +clusters.get(i).toString() + "\n";
            }
            return returnVal;
        }
    }

    public List<MatOfPoint> getContours() {
        return mContours;
    }
}