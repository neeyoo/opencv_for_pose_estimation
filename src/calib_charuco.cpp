#include <iostream>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "aruco_samples_utility.hpp"

using namespace cv;

namespace
{
    const char *about =
        "Calibration using a ChArUco board\n"
        "  To capture a frame for calibration, press 'c',\n"
        "  If input comes from video, press any key for next frame\n"
        "  To finish capturing, press 'ESC' key and calibration starts.\n";
    const char *keys =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{@outfile |<none> | Output file with calibrated camera parameters }"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       | false | Apply refind strategy }"
        "{zt       | false | Assume zero tangential distortion }"
        "{a        |       | Fix aspect ratio (fx/fy) to this value }"
        "{pc       | false | Fix the principal point at the center }"
        "{sc       | false | Show detected chessboard corners after calibration }";
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    // if (argc < 7)
    // {
    //     parser.printMessage();
    //     return 0;
    // }

    // int squaresX = parser.get<int>("w");
    // int squaresY = parser.get<int>("h");
    // float squareLength = parser.get<float>("sl");
    // float markerLength = parser.get<float>("ml");
    // std::string outputFile = parser.get<std::string>(0);
    int squaresX = 8;
    int squaresY = 8;
    float squareLength = 0.015;
    float markerLength = 0.011;
    std::string outputFile = "calibration_params.xml";

    // bool showChessboardCorners = parser.get<bool>("sc");
    bool showChessboardCorners = true;

    int calibrationFlags = 0;
    float aspectRatio = 1;
    if (parser.has("a"))
    {
        calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if (parser.get<bool>("zt"))
        calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.get<bool>("pc"))
        calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    if (parser.has("dp"))
    {
        FileStorage fs(parser.get<std::string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if (!readOk)
        {
            std::cerr << "Invalid detector parameters file" << std::endl;
            return 0;
        }
    }

    bool refindStrategy = parser.get<bool>("rs");
    int camId = parser.get<int>("ci");
    String video;

    if (parser.has("v"))
    {
        video = parser.get<String>("v");
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty())
    {
        inputVideo.open(video);
        waitTime = 0;
    }
    else
    {
        inputVideo.open(camId);
        waitTime = 10;
    }

    // aruco::Dictionary dictionary = aruco::getPredefinedDictionary(0);
    // if (parser.has("d"))
    // {
    //     int dictionaryId = parser.get<int>("d");
    //     dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));
    // }
    // else if (parser.has("cd"))
    // {
    //     FileStorage fs(parser.get<std::string>("cd"), FileStorage::READ);
    //     bool readOk = dictionary.aruco::Dictionary::readDictionary(fs.root());
    //     if (!readOk)
    //     {
    //         std::cerr << "Invalid dictionary file" << std::endl;
    //         return 0;
    //     }
    // }
    // else
    // {
    //     std::cerr << "Dictionary not specified" << std::endl;
    //     return 0;
    // }
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_1000);

    // Create charuco board object
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
    aruco::CharucoParameters charucoParams;

    if (refindStrategy)
    {
        charucoParams.tryRefineMarkers = true;
    }

    aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    // Collect data from each frame
    std::vector<Mat> allCharucoCorners;
    std::vector<Mat> allCharucoIds;

    std::vector<std::vector<Point2f>> allImagePoints;
    std::vector<std::vector<Point3f>> allObjectPoints;

    std::vector<Mat> allImages;
    Size imageSize;

    while (inputVideo.grab())
    {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        std::vector<int> markerIds;
        std::vector<std::vector<Point2f>> markerCorners, rejectedMarkers;
        Mat currentCharucoCorners;
        Mat currentCharucoIds;
        std::vector<Point3f> currentObjectPoints;
        std::vector<Point2f> currentImagePoints;

        // Detect ChArUco board
        detector.detectBoard(image, currentCharucoCorners, currentCharucoIds);

        // Draw results
        image.copyTo(imageCopy);
        if (!markerIds.empty())
        {
            aruco::drawDetectedMarkers(imageCopy, markerCorners);
        }

        if (currentCharucoCorners.total() > 3)
        {
            aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);
        }

        putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
                Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

        imshow("out", imageCopy);

        // Wait for key pressed
        char key = (char)waitKey(waitTime);

        if (key == 27)
        {
            break;
        }

        if (key == 'c' && currentCharucoCorners.total() > 3)
        {
            // Match image points
            board.matchImagePoints(currentCharucoCorners, currentCharucoIds, currentObjectPoints, currentImagePoints);

            if (currentImagePoints.empty() || currentObjectPoints.empty())
            {
                std::cout << "Point matching failed, try again." << std::endl;
                continue;
            }

            std::cout << "Frame captured" << std::endl;

            allCharucoCorners.push_back(currentCharucoCorners);
            allCharucoIds.push_back(currentCharucoIds);
            allImagePoints.push_back(currentImagePoints);
            allObjectPoints.push_back(currentObjectPoints);
            allImages.push_back(image);

            imageSize = image.size();
        }
    }
    if (allCharucoCorners.size() < 4)
    {
        std::cerr << "Not enough corners for calibration" << std::endl;
        return 0;
    }

    Mat cameraMatrix, distCoeffs;

    if (calibrationFlags & CALIB_FIX_ASPECT_RATIO)
    {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = aspectRatio;
    }

    // Calibrate camera using ChArUco
    double repError = calibrateCamera(
        allObjectPoints, allImagePoints, imageSize,
        cameraMatrix, distCoeffs, noArray(), noArray(), noArray(),
        noArray(), noArray(), calibrationFlags);

    bool saveOk = saveCameraParams(
        outputFile, imageSize, aspectRatio, calibrationFlags,
        cameraMatrix, distCoeffs, repError);

    if (!saveOk)
    {
        std::cerr << "Cannot save output file" << std::endl;
        return 0;
    }

    std::cout << "Rep Error: " << repError << std::endl;
    std::cout << "Calibration saved to " << outputFile << std::endl;

    // Show interpolated charuco corners for debugging
    if (showChessboardCorners)
    {
        for (size_t frame = 0; frame < allImages.size(); frame++)
        {
            Mat imageCopy = allImages[frame].clone();

            if (allCharucoCorners[frame].total() > 0)
            {
                aruco::drawDetectedCornersCharuco(
                    imageCopy, allCharucoCorners[frame], allCharucoIds[frame]);
            }

            imshow("out", imageCopy);
            char key = (char)waitKey(0);
            if (key == 27)
            {
                break;
            }
        }
    }

    return 0;
}
