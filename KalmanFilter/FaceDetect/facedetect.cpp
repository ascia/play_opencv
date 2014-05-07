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
int frame_size_multiple = 20;

//----------------------------------------
static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );
IplImage* img_resize(IplImage* src_img, int new_width,int new_height);
//for kalman filter
struct face_location_struct { int x,y; };
struct face_location_struct face_location = {-1,-1}, last_location;

    //for kalman filter
    vector<Point> facev,kalmanv;
    KalmanFilter KF(4, 2, 0);
    Mat state(4, 1, CV_32F); /* (x, y, Vx, Vy) */
    Mat processNoise(4, 1, CV_32F);
    Mat measurement = Mat::zeros( 2, 1,CV_32F); 
    //---------------------- 
//----------------------
string cascadeName = "./data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "./data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

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

    help();

    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
        {
            if( argv[i][nestedCascadeOpt.length()] == '=' )
                nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
            if( !nestedCascade.load( nestedCascadeName ) )
                cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
        {
            tryflip = true;
            cout << " will try to flip image horizontally to detect assymetric objects\n";
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        image = imread( "lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read lena.jpg" << endl;
    }

    cvNamedWindow( "result", 1 );
        
    if( capture )
    {
        
        int pos_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES) + 1;
        int total_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);
        cout << "In capture ..." << endl;
        
        for(; pos_frame < total_frame ;)
        {
            //--------For kalman------------ 
            KF.statePre.at<float>(0) = face_location.x;
            KF.statePre.at<float>(1) = face_location.y;
            KF.statePre.at<float>(2) = 0;
            KF.statePre.at<float>(3) = 0;
            KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);

            setIdentity(KF.measurementMatrix);
            setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
            setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
            setIdentity(KF.errorCovPost, Scalar::all(1));

            randn(KF.statePost, Scalar::all(0), Scalar::all(1));

            facev.clear();
            kalmanv.clear();

            //---------kalman end-----------
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
        1.1, 2, 0
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
    printf( "detection time = %g ms", t/((double)cvGetTickFrequency()*1000.) );
    face_found = faces.size();
    if(face_found > 0 ) printf(" face found %d",face_found);
    //------For kalman------------
    

    //-------End Kalman------------- 
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        // modified by CJ
        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    color, 1, 1, 0);
       
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    printf("\n");
    cv::imshow( "result", img );
}
