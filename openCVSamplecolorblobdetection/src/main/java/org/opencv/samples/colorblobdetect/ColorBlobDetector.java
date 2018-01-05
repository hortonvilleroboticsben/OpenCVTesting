package org.opencv.samples.colorblobdetect;

import android.util.Log;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static android.content.ContentValues.TAG;
import static org.opencv.imgproc.Imgproc.HoughLinesP;

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


    // Cache
    Mat mPyrDownMat = new Mat();
    Mat mHsvMat = new Mat();
    Mat mMask = new Mat();
    Mat mDilatedMask = new Mat();
    Mat mHierarchy = new Mat();
    Mat lines = new Mat();

    public void setColorRadius(Scalar radius) {
        mColorRadius = radius;
    }

    public void setColorRange(Scalar minHSV, Scalar maxHSV) {
        mLowerBound = minHSV;
        mUpperBound = maxHSV;
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

    public void processLines(Mat mRgba, CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Imgproc.cvtColor(mRgba,mHsvMat,Imgproc.COLOR_RGB2HSV_FULL);

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

        HoughLinesP(mMask, lines, 5, Math.PI/180, 7,60, 15);

    }

    public void showLines(Mat mRgba){
        try {
            for (int i = 0; i < lines.rows(); i++) {
                double[] val = lines.get(i,0);
//                double rho = val[0], theta = val[1];
//                double cosTheta = Math.cos(theta);
//                double sinTheta = Math.sin(theta);
//                double x = cosTheta * rho;
//                double y = sinTheta * rho;
//                Point p1 = new Point(x + 10000 * -sinTheta, y + 10000 * cosTheta);
//                Point p2 = new Point(x - 10000 * -sinTheta, y - 10000 * cosTheta);


                double angle = Math.atan2(val[1]-val[3],val[0]-val[2])*180/Math.PI;

                Log.println(Log.ASSERT, "TAG", angle+"");

                if(isWithin(angle, 170, 180) || isWithin(angle, -180, -170))Imgproc.line(mRgba,
                        new Point(val[0], val[1]),
                        new Point(val[2], val[3]),
                        new Scalar(255, 0, 0),
                        10);
                else if(isWithin(angle, 80, 90) || isWithin(angle, -90, -80))Imgproc.line(mRgba,
                        new Point(val[0], val[1]),
                        new Point(val[2], val[3]),
                        new Scalar(0, 0, 255),
                        10);
                else Imgproc.line(mRgba,
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

    public List<MatOfPoint> getContours() {
        return mContours;
    }
}