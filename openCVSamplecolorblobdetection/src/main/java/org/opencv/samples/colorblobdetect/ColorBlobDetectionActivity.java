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

import static android.R.attr.radius;
import static android.R.attr.x;
import static android.R.attr.y;
import static org.opencv.imgproc.Imgproc.minEnclosingCircle;
import static org.opencv.imgproc.Imgproc.moments;

public class ColorBlobDetectionActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
    private static final String  TAG              = "OCVSample::Activity";

    private int colorSelectCount = 0;

    private Mat                  mRgba;
    private Mat                  mROI;

    private Scalar               mBlobColorRgba;
    private Scalar               mBlobColorHsv;

    private ColorBlobDetector    mDetectorRed;
    private ColorBlobDetector    mDetectorBlue;

    private Mat                  mSpectrumRed;
    private Mat                  mSpectrumBlue;
    private Size                 SPECTRUM_SIZE;

    private Scalar               CONTOUR_COLOR_RED;
    private Scalar               CONTOUR_COLOR_BLUE;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public ColorBlobDetectionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
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
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
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

        Log.d(TAG, "DJN: Width: " + width + " Height: " + height );

        mDetectorRed = new ColorBlobDetector();
        mDetectorBlue = new ColorBlobDetector();

        mSpectrumRed = new Mat();
        mSpectrumBlue = new Mat();

        mBlobColorRgba = new Scalar(255);

        SPECTRUM_SIZE = new Size(200, 64);

        CONTOUR_COLOR_RED = new Scalar(255,0,0,255);
        CONTOUR_COLOR_BLUE = new Scalar(0,0,255,255);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;

        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        Rect touchedRect = new Rect();

        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] +
                ", " + mBlobColorRgba.val[2] + ", " + mBlobColorRgba.val[3] + ")");

        if (colorSelectCount++ % 2 == 0) {
            mDetectorRed.setHsvColor(mBlobColorHsv);
            Imgproc.resize(mDetectorRed.getSpectrum(), mSpectrumRed, SPECTRUM_SIZE);
        } else {
            mDetectorBlue.setHsvColor(mBlobColorHsv);
            Imgproc.resize(mDetectorBlue.getSpectrum(), mSpectrumBlue, SPECTRUM_SIZE);
        }

        touchedRegionRgba.release();
        touchedRegionHsv.release();

        return false; // don't need subsequent touch events
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (colorSelectCount > 0) {
            Imgproc.blur(mRgba,mRgba,new Size(20,20));

            mDetectorRed.process(mRgba);
            mDetectorBlue.process(mRgba);

            List<MatOfPoint> contoursRed = mDetectorRed.getContours();
            Log.e(TAG, "Contours count: " + contoursRed.size());
            //Imgproc.drawContours(mRgba, contoursRed, -1, CONTOUR_COLOR_RED);

            List<MatOfPoint> contoursBlue = mDetectorBlue.getContours();
            Log.e(TAG, "Contours count: " + contoursBlue.size());
            //Imgproc.drawContours(mRgba, contoursBlue, -1, CONTOUR_COLOR_BLUE);

            List<Moments> muRed = new ArrayList<Moments>(contoursRed.size());

            Point centerRed = new Point();

            List<MatOfPoint2f> contours2fRed   = new ArrayList<MatOfPoint2f>();
            List<MatOfPoint2f> polyMOP2fRed    = new ArrayList<MatOfPoint2f>();
            List<MatOfPoint> polyMOPRed        = new ArrayList<MatOfPoint>();

            float[] radiusRed     = new float[contoursRed.size()];

            for (int i = 0; i < contoursRed.size(); i++) {
                muRed.add(i, Imgproc.moments(contoursRed.get(i), false));
                Moments p = muRed.get(i);
                int x = (int) (p.get_m10() / p.get_m00());
                int y = (int) (p.get_m01() / p.get_m00());
                Imgproc.circle(mRgba, new Point(x, y), 16, CONTOUR_COLOR_RED, 16);

                contours2fRed.add(new MatOfPoint2f());
                polyMOP2fRed.add(new MatOfPoint2f());
                polyMOPRed.add(new MatOfPoint());

                contoursRed.get(i).convertTo(contours2fRed.get(i), CvType.CV_32FC2);
                Imgproc.approxPolyDP(contours2fRed.get(i), polyMOP2fRed.get(i), 3, true);
                polyMOP2fRed.get(i).convertTo(polyMOPRed.get(i), CvType.CV_32S);

                minEnclosingCircle(polyMOP2fRed.get(i),centerRed,radiusRed);
                Imgproc.circle(mRgba, new Point(x, y), 16, CONTOUR_COLOR_RED, 16);
                Imgproc.circle(mRgba, centerRed, (int)radiusRed[i], CONTOUR_COLOR_RED, 16);
                Log.e(TAG, "Red Center: (" + (int)centerRed.x + "," + (int)centerRed.y + ") with radius: " + (int)radiusRed[i]);
            }

            List<Moments> muBlue = new ArrayList<Moments>(contoursBlue.size());
            Point centerBlue = new Point();

            List<MatOfPoint2f> contours2fBlue   = new ArrayList<MatOfPoint2f>();
            List<MatOfPoint2f> polyMOP2fBlue    = new ArrayList<MatOfPoint2f>();
            List<MatOfPoint> polyMOPBlue        = new ArrayList<MatOfPoint>();

            float[] radiusBlue     = new float[contoursBlue.size()];

            for (int i = 0; i < contoursBlue.size(); i++) {
                muBlue.add(i, Imgproc.moments(contoursBlue.get(i), false));
                Moments p = muBlue.get(i);
                int x = (int) (p.get_m10() / p.get_m00());
                int y = (int) (p.get_m01() / p.get_m00());
                Imgproc.circle(mRgba, new Point(x, y), 16, CONTOUR_COLOR_BLUE, 16);

                contours2fBlue.add(new MatOfPoint2f());
                polyMOP2fBlue.add(new MatOfPoint2f());
                polyMOPBlue.add(new MatOfPoint());

                contoursBlue.get(i).convertTo(contours2fBlue.get(i), CvType.CV_32FC2);
                Imgproc.approxPolyDP(contours2fBlue.get(i), polyMOP2fBlue.get(i), 3, true);
                polyMOP2fBlue.get(i).convertTo(polyMOPBlue.get(i), CvType.CV_32S);

                minEnclosingCircle(polyMOP2fBlue.get(i),centerBlue,radiusBlue);
                Imgproc.circle(mRgba, new Point(x, y), 16, CONTOUR_COLOR_BLUE, 16);
                Imgproc.circle(mRgba, centerBlue, (int)radiusBlue[i], CONTOUR_COLOR_BLUE, 16);
                Log.e(TAG, "Blue Center: (" + (int)centerBlue.x + "," + (int)centerBlue.y + ") with radius: " + (int)radiusBlue[i]);
            }


            Mat colorLabel = mRgba.submat(4, 68, 4, 68);
            colorLabel.setTo(mBlobColorRgba);

            //Mat spectrumLabelRed = mRgba.submat(4, 4 + mSpectrumRed.rows(), 70, 70 + mSpectrumRed.cols());
            //mSpectrumRed.copyTo(spectrumLabelRed);

            //Mat spectrumLabelBlue = mRgba.submat(4, 4 + mSpectrumBlue.rows(), 70, 70 + mSpectrumBlue.cols());
            //mSpectrumBlue.copyTo(spectrumLabelBlue);

            if (centerBlue.x < centerRed.x) {
                Log.e(TAG, "Blue Left: (" + (int)centerBlue.x + "," + (int)centerBlue.y + ") Red Right: (" + (int)centerRed.x + "," + (int)centerRed.y + ")");
            } else {
                Log.e(TAG, "Red Left: (" + (int)centerRed.x + "," + (int)centerRed.y + ") Blue Right: (" + (int)centerBlue.x + "," + (int)centerBlue.y + ")");
            }

        }

        return mRgba;
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
