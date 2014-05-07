#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/objdetect_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/softcascade.hpp>

#include <iostream>
#include <vector>
// #include <string>
// #include <fstream>
using namespace::std;
void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects);
IplImage* img_resize(IplImage* src_img, int new_width,int new_height);
int frame_size_multiple = 20;
int main(int argc, char** argv)
{
    cv::HOGDescriptor hog;
    CvCapture* capture = 0;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::cout << "going to use HOG detector." << std::endl; 
    
    capture = cvCaptureFromAVI( argv[1] ); 
    if(!capture){ 
        cout << "Capture from AVI didn't work" << endl;
        return -1;
    }
    int pos_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES) + 1;
    int total_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);


    for( ; pos_frame < total_frame ; ) 
    {

        cv::Mat frame;
        pos_frame = cvGetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES) + 1;
        printf("Total %d ,NO. %d ",total_frame, pos_frame );
        IplImage* iplImg = cvQueryFrame( capture );
        frame = cv::cvarrToMat(img_resize(iplImg, 16 * frame_size_multiple, 9 * frame_size_multiple));

        if(frame.empty())
        {

            continue;
        }
        std::vector<cv::Rect> found, found_filtered;
        // run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
       // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        hog.detectMultiScale(frame, found, 0, cv::Size(0,0), cv::Size(100,100), 1.05, 3);

        filter_rects(found, found_filtered);
        std::cout << "collected: " << (int)found_filtered.size() << " detections." << std::endl;

        for (size_t ff = 0; ff < found_filtered.size(); ++ff)
        {
            cv::Rect r = found_filtered[ff];
            cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 1);

        }
        

            cv::imshow("pedestrian detector", frame);
            cv::waitKey(10);
    }

    return 0;
}
IplImage* img_resize(IplImage* src_img, int new_width,int new_height)
{
    IplImage* des_img;
    des_img=cvCreateImage(cvSize(new_width,new_height),src_img->depth,src_img->nChannels);
    cvResize(src_img,des_img,CV_INTER_LINEAR);
    return des_img;
} 
void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects)
{
    size_t i, j;
    for (i = 0; i < candidates.size(); ++i)
    {
        cv::Rect r = candidates[i];

        for (j = 0; j < candidates.size(); ++j)
            if (j != i && (r & candidates[j]) == r)
                break;

        if (j == candidates.size())
            objects.push_back(r);
    }
}
