#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <ctime>
#include <sys/stat.h>
#include <vector>

using namespace cv;

int main()
{
    // Load camera calibration parameters
    FileStorage fs("calibration_params.xml", FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Error: Unable to open calibration parameters file" << std::endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    std::cout << cameraMatrix << std::endl;
    std::cout << distCoeffs << std::endl;

    // Open the default camera
    VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Create a window to display the camera feed
    // namedWindow("Camera Feed", WINDOW_NORMAL);
    // resizeWindow("Camera Feed", 640, 480);

    aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    // Load the ArUco dictionary
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_1000);
    aruco::ArucoDetector detector(dictionary, detectorParams);

    while (cap.grab())
    {
        Mat frame;

        // Capture frame-by-frame
        // cap >> frame;

        cap.retrieve(frame);

        // Check if the frame is empty
        if (frame.empty())
        {
            std::cerr << "Error: Unable to capture frame" << std::endl;
            break;
        }

        // Undistort the frame using camera calibration parameters
        Mat undistortedFrame;
        undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);

        // Detect markers
        std::vector<int> markerIds;
        std::vector<std::vector<Point2f>> markerCorners;
        // aruco::detectMarkers(undistortedFrame, dictionary, markerCorners, markerIds);
        detector.detectMarkers(undistortedFrame, markerCorners, markerIds);

        // Estimate pose of markers
        std::vector<Vec3d> rvecs, tvecs;
        aruco::estimatePoseSingleMarkers(markerCorners, 0.09, cameraMatrix, distCoeffs, rvecs, tvecs);

        // Draw markers if any are detected
        if (!markerIds.empty())
        {
            aruco::drawDetectedMarkers(undistortedFrame, markerCorners, markerIds);

            // Draw pose frames
            for (size_t i = 0; i < markerIds.size(); ++i)
            {
                drawFrameAxes(undistortedFrame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                std::cout << "Translational Vec: " << tvecs[i] << std::endl;
                std::cout << "Rotational Vec: " << rvecs[i] << std::endl;
            }
        }

        // Display the resulting frame
        imshow("Camera Feed", undistortedFrame);

        // Check for the escape key press to exit
        int key = waitKey(100);
        if (key == 27)
        { // ASCII code for escape key
            break;
        }
    }

    // Release the camera
    cap.release();

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
