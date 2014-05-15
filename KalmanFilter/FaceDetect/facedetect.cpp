#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video/tracking.hpp"


#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <vector>
using namespace std;
using namespace cv;
//--------------factor need to change-----
float frame_size_multiple = 30;
bool face_found_once = false;

//----------------------------------------


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

IplImage* img_resize(IplImage* src_img, int new_width,int new_height);
//for kalman filter
struct face_location_struct { 
    float x,y,width,height; 
};
struct face_location_struct face_location = {0,0,0,0},last_location = {0,0,0,0};

    //for kalman filter
    vector<Point> facev,kalmanv;

    
    KalmanFilter KFX(3, 1, 0);
    KalmanFilter KFY(3, 1, 0);
    Mat stateX(3, 1, CV_32F);
    Mat stateY(3, 1, CV_32F);
    Mat processNoiseX(3, 1, CV_32F);
    Mat processNoiseY(3, 1, CV_32F);
    Mat_<float> measurementX(1,1); 
    Mat_<float> measurementY(1,1); 

    //---------------------- 
//----------------------
string cascadeName = "./haarcascade_frontalface_alt2.xml";

int main( int argc, const char** argv )
{
    

    CvCapture* capture = 0;       
    Mat frame, frameCopy, image;
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string nestedCascadeOpt = "--nested-cascade";
    size_t nestedCascadeOptLen = nestedCascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
    bool tryflip = false;


    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

  
    capture = cvCaptureFromAVI( argv[1] );
    if(!capture) cout << "Capture from AVI didn't work" << endl;
    
    cvNamedWindow( "result", 1 );
        
    if( capture )
    {
        
        int pos_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES) + 1;
        int total_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);
        cout << "In capture ..." << endl;
        //-----kalman xy-----
        KFX.statePre.at<float>(0) = face_location.x;
        KFY.statePre.at<float>(0) = face_location.y;
        KFX.statePre.at<float>(1) = 0;
        KFY.statePre.at<float>(1) = 0;

        KFX.statePre.at<float>(2) = 0;
        KFY.statePre.at<float>(2) = 0;
        

        
        KFX.transitionMatrix = (Mat_<float>(3, 3) << 1, 1, 0.5,    0, 1, 1   , 0, 0, 1);
        

        
        KFY.transitionMatrix = (Mat_<float>(3, 3) << 1, 1, 0.5,    0, 1, 1   , 0, 0, 1);
        
        setIdentity(KFX.measurementMatrix);
        setIdentity(KFX.processNoiseCov, Scalar::all(1e-3));
        setIdentity(KFX.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KFX.errorCovPost, Scalar::all(0.1));
        
        setIdentity(KFY.measurementMatrix);
        setIdentity(KFY.processNoiseCov, Scalar::all(1e-4));
        setIdentity(KFY.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KFY.errorCovPost, Scalar::all(0.1));

        facev.clear();
        kalmanv.clear();
        //---------kalman end-----------

        for(; pos_frame < total_frame ;)
        {
            
            pos_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES) + 1;
            printf("Total %d ,NO. %d ",total_frame, pos_frame );
            IplImage* iplImg = cvQueryFrame( capture );
            frame = cv::cvarrToMat(img_resize(iplImg, 16 * frame_size_multiple, 9 * frame_size_multiple));
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}
IplImage* img_resize(IplImage* src_img, int new_width,int new_height)
{
    IplImage* des_img;
    des_img=cvCreateImage(cvSize(new_width,new_height),src_img->depth,src_img->nChannels);
    cvResize(src_img,des_img,CV_INTER_LINEAR);
    return des_img;
} 
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    
    int i = 0;
    double t = 0;
    int face_found = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(120,120,200),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    t = (double)cvGetTickCount();
    
    cascade.detectMultiScale( smallImg, faces,
        1.3, 2, 0
        |CASCADE_FIND_BIGGEST_OBJECT
        |CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    
    t = (double)cvGetTickCount() - t;
    //printf( "detection time = %g ms", t/((double)cvGetTickFrequency()*1000.) );
    
        
    //}
    if(faces.size() > 0) face_found = true ; 
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
     
        Scalar color = colors[i%8];
   
        //-----For Kalman-----
        face_location.x      = r->x;    
        face_location.y      = r->y; 
        face_location.width  = r->width;
        face_location.height = r->height;  
        //-----End Kalman-----
        // modified by CJ
        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    color, 1, 1, 0);
       
        
    }
    
    if(face_found > 0 ){ 

  
        Point measPt( face_location.x, face_location.y+face_location.height/2);

        facev.push_back(measPt);

        face_found_once = true;
        

    } 
    
    if(face_found_once){
                        
        //----kalman xy-----
        Point statePt(stateX.at<float>(0),stateY.at<float>(0)+face_location.height/2);
        Mat predictionX = KFX.predict();
        Mat predictionY = KFY.predict();
        
        Point predictPt(predictionX.at<float>(0),predictionY.at<float>(0)+face_location.height/2);
        kalmanv.push_back(predictPt);
        
        randn( measurementX, Scalar::all(0), Scalar::all(KFX.measurementNoiseCov.at<float>(0)));
        randn( measurementY, Scalar::all(0), Scalar::all(KFY.measurementNoiseCov.at<float>(0)));

        
        if(face_found > 0){    
            measurementX(0) = face_location.x;// + measurement(2);
            measurementY(0) = face_location.y;// + measurement(3);
        } 
        
        if(theRNG().uniform(0,4) != 0 && face_found > 0)
                KFX.correct(measurementX);
        if(theRNG().uniform(0,4) != 0 && face_found > 0)
                KFY.correct(measurementY);    
            
            randn( processNoiseX, Scalar(0), Scalar::all(sqrt(KFX.processNoiseCov.at<float>(0, 0))));
            stateX = KFX.transitionMatrix*stateX + processNoiseX;
            randn( processNoiseY, Scalar(0), Scalar::all(sqrt(KFY.processNoiseCov.at<float>(0, 0))));
            stateY = KFY.transitionMatrix*stateY + processNoiseY;

        //----end xy


        #define drawCross( center, color, d )                                        \
                line( img, Point( center.x - d, center.y - d ),                          \
                             Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                          \
                             Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )
            //drawCross( statePt, Scalar(255,255,255), 3 );
            //drawCross( measPt, Scalar(0,0,255), 3 );
        drawCross( predictPt, Scalar(0,255,0), 3 );

        if(!face_found){
            if(face_location.width >0) face_location.width*=1.01;
            if(face_location.height >0) face_location.height*=1.01;

            rectangle( img, cvPoint(cvRound(predictPt.x*scale), cvRound(predictPt.y*scale - face_location.height/2)),
                cvPoint(cvRound((predictPt.x + face_location.width)*scale), cvRound((predictPt.y + face_location.height/2)*scale)),
                CV_RGB(255,0,0), 2, 1, 0);
        }
            
            
   
    }
    for (int i = 1; i < facev.size(); i++) {
        line(img, facev[i-1], facev[i], Scalar(255,255,255), 1);
    }
    for (int i = 1; i < kalmanv.size(); i++) {
        line(img, kalmanv[i-1], kalmanv[i], Scalar(0,0,255), 1);
    }
    //-------End Kalman-------------    
    printf("\n");
    cv::imshow( "result", img );
}
