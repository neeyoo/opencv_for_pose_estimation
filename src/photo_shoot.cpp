#include <opencv2/opencv.hpp>
#include <ctime>
#include <sys/stat.h>

using namespace cv;

int main() {
    // Create the "photos" subfolder if it doesn't exist
    const char* photosFolder = "photos";
    mkdir(photosFolder, 0777);

    // Open the default camera
    VideoCapture cap(0);
    
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Create a window to display the camera feed
    namedWindow("Camera Feed", WINDOW_NORMAL);
    resizeWindow("Camera Feed", 640, 480);

    while (true) {
        Mat frame;
        
        // Capture frame-by-frame
        cap >> frame;
        
        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture frame" << std::endl;
            break;
        }
        
        // Display the resulting frame
        imshow("Camera Feed", frame);
        
        // Check for the spacebar press
        int key = waitKey(10);
        if (key == 32) { // ASCII code for spacebar
            // Get current timestamp
            time_t now = time(0);
            struct tm* timeinfo = localtime(&now);
            char timestamp[80];
            strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", timeinfo);
            
            // Save the image with timestamp in the filename
            std::string filename = std::string(photosFolder) + "/captured_photo_" + std::string(timestamp) + ".jpg";
            imwrite(filename, frame);
            std::cout << "Photo captured with timestamp: " << filename << std::endl;
        } else if (key == 27) { // ASCII code for escape key
            break;
        }
    }
    
    // Release the camera
    cap.release();
    
    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}