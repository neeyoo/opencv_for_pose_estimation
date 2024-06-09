#include <opencv2/opencv.hpp>
#include <ctime>
#include <sys/stat.h>
#include <vector>

using namespace cv;

int main()
{
    // Create the "photos" subfolder if it doesn't exist
    const char *photosFolder = "photos";
    mkdir(photosFolder, 0777);

    // Open the default camera
    VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Create a window to display the camera feed
    namedWindow("Camera Feed", WINDOW_NORMAL);
    resizeWindow("Camera Feed", 640, 480);

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f>> objpoints;

    std::vector<std::vector<cv::Point2f>> imagePoints;
    Size boardSize(7, 10); // Change the board size according to your calibration pattern

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i{0}; i < boardSize.height; i++)
    {
        for (int j{0}; j < boardSize.width; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

    while (true)
    {
        Mat frame;

        // Capture frame-by-frame
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty())
        {
            std::cerr << "Error: Unable to capture frame" << std::endl;
            break;
        }

        // Display the resulting frame
        imshow("Camera Feed", frame);

        // Check for the spacebar press to capture calibration images
        int key = waitKey(10);
        if (key == 32)
        { // ASCII code for spacebar
            // Convert frame to grayscale
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            // Find chessboard corners
            std::vector<cv::Point2f> corners;
            bool found = findChessboardCorners(gray, boardSize, corners,
                                               CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

            if (found)
            {
                // Refine corner locations
                cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                             TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

                // Draw and display the corners
                drawChessboardCorners(frame, boardSize, Mat(corners), found);
                imshow("Camera Feed", frame);

                // Pause for 1 second
                waitKey(1000);

                // Save the image points for calibration
                objpoints.push_back(objp);
                imagePoints.push_back(corners);
                std::cout << "Calibration image captured (" << imagePoints.size() << " / 20)" << std::endl;

                // Stop capturing images after 20 images
                if (imagePoints.size() >= 20)
                {
                    break;
                }
            }
            else
            {
                std::cerr << "Chessboard corners not found in the image" << std::endl;
            }
        }
        else if (key == 27)
        { // ASCII code for escape key
            break;
        }
    }

    // Release the camera
    cap.release();

    // Close all OpenCV windows
    destroyAllWindows();

    // Perform camera calibration if enough images were captured
    if (imagePoints.size() >= 20)
    {
        std::cout << "Performing camera calibration..." << std::endl;

        Size imageSize = Size(640, 480); // Adjust the image size according to your camera
        Mat cameraMatrix, distCoeffs;
        std::vector<Mat> rvecs, tvecs;

        // Calibrate camera
        double rms = calibrateCamera(objpoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

        // Save calibration parameters
        FileStorage fs("calibration_params.xml", FileStorage::WRITE);
        fs << "cameraMatrix" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs.release();

        std::cout << "Calibration successful. RMS reprojection error: " << rms << std::endl;
    }
    else
    {
        std::cerr << "Insufficient calibration images captured. At least 20 images are required." << std::endl;
    }

    return 0;
}
