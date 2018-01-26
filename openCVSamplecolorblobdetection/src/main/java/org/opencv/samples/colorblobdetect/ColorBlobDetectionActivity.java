package org.opencv.samples.colorblobdetect;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;
import android.widget.Toast;

import static android.R.attr.radius;
import static android.R.attr.x;
import static android.R.attr.y;
import static org.opencv.imgproc.Imgproc.HoughLines;
import static org.opencv.imgproc.Imgproc.HoughLinesP;
import static org.opencv.imgproc.Imgproc.minEnclosingCircle;
import static org.opencv.imgproc.Imgproc.moments;

public class ColorBlobDetectionActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    private int colorSelectCount = 0;

    private Mat mRgba;
    private Mat mROI;
    private Rect rROI;
    private Point centerROI;
    int count = 1;

    private Scalar mBlobColorRgba;
    private Scalar mBlobColorHsv;

    private ColorBlobDetector mDetectorRed;
    private ColorBlobDetector mDetectorBlue;
    private ColorBlobDetector mDetectorWhite;

    private Mat mSpectrumRed;
    private Mat mSpectrumBlue;
    private Size SPECTRUM_SIZE;

    private Scalar CONTOUR_COLOR_RED;
    private Scalar CONTOUR_COLOR_BLUE;

    private Scalar LINE_COLOR_RED;
    private Scalar LINE_COLOR_BLUE;

    private Scalar ROI_COLOR;

    String RB = "R";

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public ColorBlobDetectionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.color_blob_detection_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);


        rROI = new Rect((int) width / 3, (int) height / 3, (int) width / 3, (int) height / 3);
        centerROI = new Point(rROI.width / 2, rROI.height / 2);
        mROI = new Mat(mRgba, rROI);

        Log.d(TAG, "onCamerViewStarted Width: " + width + " Height: " + height);

        mDetectorRed = new ColorBlobDetector();
        mDetectorBlue = new ColorBlobDetector();

        mSpectrumRed = new Mat();
        mSpectrumBlue = new Mat();

        mBlobColorRgba = new Scalar(255);

        SPECTRUM_SIZE = new Size(200, 64);

        CONTOUR_COLOR_RED = new Scalar(255, 0, 0, 255);
        CONTOUR_COLOR_BLUE = new Scalar(0, 0, 255, 255);

        LINE_COLOR_BLUE = new Scalar(0,0,200,200);


        ROI_COLOR = new Scalar(255, 255, 255, 255);

        mDetectorRed.setColorRange(new Scalar(217, 150, 150, 0), new Scalar(45, 255, 255, 255));
        mDetectorBlue.setColorRange(new Scalar(125, 120, 130, 0), new Scalar(187, 255, 255, 255));
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {

        try{

            if(RB.equals("B")){
                mDetectorBlue.clusters.writeInteriorAngle();
            }else{
                if(count <= 50) {
                    mDetectorRed.clusters.writeInteriorAngle();
                    if (mDetectorRed.clusters.clusterGroups.size() >= 2) {
                        Toast.makeText(this, "LOGGED:"+count, Toast.LENGTH_SHORT).show();
                    }
                    count++;
                    return true;
                }else {
                    ColorBlobDetector.logToFile("");
                    count = 1;
                    return false;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return true;
        // don't need subsequent touch events
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        try{

            if(RB.equals("B")){
                mDetectorBlue.processLines(mRgba,inputFrame);
                mDetectorBlue.showLines(mRgba);
            }else{
                mDetectorRed.processLines(mRgba, inputFrame);
                mDetectorRed.showLines(mRgba);
            }

            Imgproc.rectangle(mRgba, new Point(mRgba.cols()/2-10, 0), new Point(mRgba.cols()/2+10, mRgba.rows()), new Scalar(0,255,0), 4);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return mRgba;
    }

    //here ya go ben

    public boolean isWithin(double val, double low, double high){
        return val >= low && val <= high;
    }



    private double distance(Point center, Point check) {
        return Math.hypot((center.x - check.x), (center.y - check.y));
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}