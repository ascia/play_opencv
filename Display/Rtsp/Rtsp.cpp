#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>


using namespace cv;
using namespace std;
int main(int argc, char* argv[])
{
    //cvlc -vvv rtsp://admin:123456@172.16.15.194:7070/track1 --sout="#std{access=file,mux=ps,dst=video}"

    CvCapture* capture = 0; 
    capture = cvCaptureFromAVI(argv[1]);
    if (!capture)  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

   double dWidth = cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
   double dHeight = cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    cout << "Frame size : " << dWidth << " x " << dHeight << endl;

    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
    namedWindow("MyNegativeVideo",CV_WINDOW_AUTOSIZE);

    while (1)
    {
        Mat frame;
        Mat contours;
        IplImage* iplImg = cvQueryFrame( capture );  
        //bool bSuccess = cap.read(frame); // read a new frame from video
        frame = cv::cvarrToMat(iplImg);
         

        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        
        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
       {
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;
}
