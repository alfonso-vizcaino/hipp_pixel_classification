#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

const String filePath = "../images/hippocampal/";
const String labelFilePath = "../labels/";
const String outputFilePath = "../extracted_features/";
String fileName = "20191128_190652_HDR_2.jpg"; //First image in folder

int pixelRange = 8; // 9 x 9 window size (zero based)
int roiWindowSize = 20;

//Hippocampus Bounding Box 
std::map<String, std::list<int>> hipMapLocation = {
    {"20191128_190652_HDR_2.jpg", {501, 282, 644, 373}},
    {"20191128_190910_HDR_2.jpg", {549, 360, 708, 441}},
    {"20191128_191028_HDR_2.jpg", {512, 421, 628, 485}},
    {"20191128_191118_HDR_2.jpg", {541, 338, 696, 439}},
    {"20191128_191134_HDR_2.jpg", {540, 367, 695, 461}},
    {"20191128_191203_HDR_2.jpg", {520, 356, 694, 451}},
    {"20191128_191320_HDR_2.jpg", {549, 359, 746, 484}},
    {"20191128_191349_HDR_2.jpg", {537, 334, 736, 459}},
    {"20191128_191356_HDR_2.jpg", {542, 329, 727, 442}},
    {"20191128_191420_HDR_2.jpg", {548, 405, 769, 521}},
    {"20191128_191500_HDR_2.jpg", {540, 377, 752, 510}},
    {"20191128_191558_HDR_2.jpg", {522, 370, 739, 492}},
    {"20191128_191647_HDR_2.jpg", {513, 371, 722, 486}},
    {"20191128_192218_HDR_2.jpg", {483, 427, 619, 498}},
    {"20191128_192315_HDR_2.jpg", {533, 208, 693, 310}},
    {"20191128_192402_HDR_2.jpg", {503, 369, 676, 457}},
    {"20191128_192452_HDR_2.jpg", {509, 233, 707, 340}},
    {"20191128_192514_HDR_2.jpg", {459, 464, 669, 567}},
    {"20191128_192922_HDR_2.jpg", {493, 255, 703, 341}},
    {"Control_2_Bregma_-3.12_PS_(1x).jpg", {488, 213, 756, 337}},
    {"Control_3_Bregma_-2.jpg", {508, 147, 758, 290}},
    {"Control_A_3-4.jpg", {484, 256, 779, 389}},
    {"Control_A_-2.28.jpg", {465, 307, 699, 429}},
    {"Control_A_-2.76.jpg", {536, 157, 777, 272}},
    {"Control_A_-4.08.jpg", {517, 209, 820, 369}}};

std::vector<Point> matchList;
std::vector<Point> featurePointList;
std::vector<String> featureStringList;
Mat img, originalImg, grayImg;
Mat roiImg, negRoiImg, enhancedImg;
Vec3d toHSV(Vec3b rgbColor);
String windowTitle;
Vec3b greenMark, other;
String hipArea = "CA1"; 
Vec4i roiCoordinates;

void autoMaticFeatureExtraction(Point p, String hipArea);
std::vector<double> getFeatures(Point center, Mat img);
int savePixelFeatures();
void readPoints();
void reloadPoints();
int init();
void cleanup();
Vec4i getHippBoundingBoxCoord(String fileName);
void doImageEnhancement();
void fillROI();
void enhanceImage();
String concatFeatureString(String fromImage, std::vector<double> features);

void cleanup()
{
    matchList.clear();
    img.release();
    originalImg.release();
    roiImg.release();
    negRoiImg.release();
    enhancedImg.release();
    featurePointList.clear();
    featureStringList.clear();
}

int init()
{
    greenMark[0] = 34;
    greenMark[1] = 245;
    greenMark[2] = 50;

    other[0] = 255;
    other[1] = 255;
    other[2] = 0;

    // Read image from file
    img = imread(filePath + fileName);
    originalImg = imread(filePath + fileName);

    doImageEnhancement();

    windowTitle = "Hippocampus Pixel Feature Extractor";

    //if fail to read the image
    if (img.empty())
    {
        cout << "Error loading the image" << endl;
        return 1;
    }

    //Create a window
    namedWindow(windowTitle, CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);

    //show the image
    imshow(windowTitle, img);

    //preload labels if any
    readPoints();

    return 0;
}

void readPoints()
{
    string line;
    ifstream myfile(labelFilePath + fileName + ".txt");
    int count = 0, x = 0, y = 0;
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            stringstream ss(line);

            while (ss.good())
            {
                string substr;
                getline(ss, substr, ',');
                //cout << "count=" << to_string(count) << " read " <<substr << endl;
                if (count == 0)
                {
                    x = stoi(substr);
                    count++;
                }
                else if (count == 1)
                {
                    y = stoi(substr);

                    img.at<Vec3b>(Point(x, y)) = greenMark;
                    matchList.push_back(Point(x, y));
                    count = 0;
                    //cout << "x=" << to_string(x) << "y=" << to_string(y) << endl;
                }
            }
        }
        myfile.close();

        imwrite(filePath + fileName + "_painted.png", img);
        imshow(windowTitle, img);
    }

    myfile = ifstream(labelFilePath + fileName + "_labels.csv");
    count = 0, x = 0, y = 0;
    if (myfile.is_open())
    {

        int x1 = roiCoordinates[0] - roiWindowSize,
            y1 = roiCoordinates[1] - roiWindowSize;
        string substr;
        while (getline(myfile, line))
        {
            stringstream ss(line);

            while (ss.good())
            {

                getline(ss, substr, ','); //File Name
                getline(ss, substr, ','); //X
                x = stoi(substr);

                getline(ss, substr, ','); //Y
                y = stoi(substr);

                featurePointList.push_back(Point(x, y));
                //x, y are extracted from ROI window, relocate them to main image
                x += x1;
                y += y1;

                getline(ss, substr); //Hip Area Label 

                //cout << " read " << substr << endl;

                // CA1=CA2=CA3=DG= Hippocampus Pixel Label
                // Outer= Non Hippocampus pixel

                if (substr == "CA1")
                {
                    img.at<Vec3b>(Point(x, y)) = greenMark;
                }
                else if (substr == "CA2")
                {
                    img.at<Vec3b>(Point(x, y)) = greenMark;
                }
                else if (substr == "CA3")
                {
                    img.at<Vec3b>(Point(x, y)) = greenMark;
                }
                else if (substr == "DG")
                {
                    img.at<Vec3b>(Point(x, y)) = greenMark;
                }
                else if (substr == "Outer")
                {
                    Vec3b rgbPixel = img.at<Vec3b>(Point(x, y));
                    rgbPixel[0] = 255;
                    img.at<Vec3b>(Point(x, y)) = rgbPixel;
                }

                autoMaticFeatureExtraction(Point(x - x1, y - y1), substr);
            }
        }
        myfile.close();

        imwrite(filePath + fileName + "_FE_painted.png", img);

        imshow(windowTitle, img);
    }
}

void reloadPoints()
{
    cleanup();
    img = imread(filePath + fileName);
    originalImg = imread(filePath + fileName);

    doImageEnhancement();

    readPoints();
}

int savePixelFeatures()
{

    if (!featureStringList.empty())
    {
        ofstream featureStringListFile(outputFilePath + fileName + "_Features.csv", ios::out);
        if (featureStringListFile.is_open())
        {
            for (String f : featureStringList)
            {
                featureStringListFile << f << "\n";
            }
            featureStringListFile.flush();
            featureStringListFile.close();
        }
        else
        {
            cout << "Unable to open Feature file";
            return -1;
        }
    }

    return 0;
}

Vec4i getHippBoundingBoxCoord(String fileName)
{
    std::list<int> posLocation = hipMapLocation.at(fileName);
    std::list<int>::iterator it;
    int x1, y1, x2, y2;
    it = posLocation.begin();
    x1 = *it;
    ++it;
    y1 = *it;
    ++it;
    x2 = *it;
    ++it;
    y2 = *it;
    Vec4i coordinates = Vec4i(x1, y1, x2, y2);

    cout << "coordinates for file " << fileName << ", coordinates=" << coordinates << endl;

    return coordinates;
}

void enhanceImage()
{
    enhancedImg = Mat(negRoiImg.rows, negRoiImg.cols, CV_64F);
    enhancedImg = negRoiImg.clone();

    Mat op = Mat(negRoiImg.rows, negRoiImg.cols, CV_64F);

    // **** Remove salt & pepper noice
    blur(enhancedImg, op, Size(3, 3));

    // **** Smooth noice
    GaussianBlur(op, op, Size(9, 9), 1.8);

    // **** Background image
    blur(op, op, Size(69, 69));

    // **** Diff image
    op = enhancedImg - op;

    // **** Equalize hist
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(op, &minVal, &maxVal, &minLoc, &maxLoc);

    //cout << "maxVal=" << maxVal << endl;

    op = op + 128 - maxVal;
    op.setTo(255, op > 255);
    op.setTo(0, op < 0);

    // **** Top Hat
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));

    morphologyEx(op, enhancedImg, MORPH_TOPHAT, kernel);
    /*
    minMaxLoc(enhancedImg, &minVal, &maxVal, &minLoc, &maxLoc);


    cout << "maxVal" << maxVal << endl;
    cout << "minVal" << minVal << endl;

    enhancedImg = enhancedImg * 255  / maxVal;
    */    
}

void fillROI()
{
    Mat hsvImg;
    vector<Mat> hsvChannels(3);
    
    cvtColor(img, hsvImg, COLOR_BGR2HSV );

    split(hsvImg, hsvChannels);

    //get Hippocampus Bounding Box coordinates for the given image
    roiCoordinates = getHippBoundingBoxCoord(fileName);
    int x1 = roiCoordinates[0] - roiWindowSize,
        y1 = roiCoordinates[1] - roiWindowSize,
        x2 = roiCoordinates[2] + roiWindowSize,
        y2 = roiCoordinates[3] + roiWindowSize;

    hsvChannels[2](Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(roiImg);

    roiImg.copyTo(negRoiImg);
    negRoiImg = 255 - negRoiImg;
}

void doImageEnhancement()
{
    fillROI();
    namedWindow("ROI", 1);
    imshow("ROI", negRoiImg);

    enhanceImage();
    namedWindow("Enhanced", 1);
    imshow("Enhanced", enhancedImg);
}

std::vector<double> getFeatures(Point center, Mat img)
{
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Scalar mean, stdDev;
    
    //9 x 9 window size  
    //(4 pixel left + 1  center pixel + 4 pixel right, 4 pixel top + 1  center pixel + 4 pixel down)
    int windowSize = 4; 
    Rect rect = Rect(center.x - windowSize, center.y - windowSize, windowSize * 2 + 1, windowSize * 2 + 1);
    // cout << "r" << r << endl;

    Mat subImg = Mat(windowSize * 2 + 1, windowSize * 2 + 1, CV_64F);

    img(rect).copyTo(subImg);

    //force cast to double!(?)
    subImg.convertTo(subImg, CV_64F);

    minMaxLoc(subImg, &minVal, &maxVal, &minLoc, &maxLoc);
    meanStdDev(subImg, mean, stdDev);

    double f5 = subImg.at<double>(Point(windowSize, windowSize));
    double f1 = f5 - minVal;
    double f2 = maxVal - f5;
    double f3 = f5 - mean[0];
    double f4 = stdDev[0];

    //17 x 17 window size  
    //(8 pixel left + 1  center pixel + 8 pixel right, 8 pixel top + 1  center pixel + 8 pixel down)
    windowSize = 8; 
    rect = Rect(center.x - windowSize, center.y - windowSize, windowSize * 2 + 1, windowSize * 2 + 1);

    subImg = Mat(windowSize * 2 + 1, windowSize * 2 + 1, CV_32F);

    img(rect).copyTo(subImg);

    //force cast to double!(?)
    subImg.convertTo(subImg, CV_32F);

    Mat kernelX = getGaussianKernel(17, 1.7, CV_32F);
    Mat kernelY = getGaussianKernel(17, 1.7, CV_32F);
    Mat kernelXY = kernelX * kernelY.t();

    subImg = subImg.mul(kernelXY);

    //cout << "Kernel " << endl;
    //cout << kernelXY << endl;

    Moments mmnts = moments(subImg);

    // Calculate Hu Moments
    double huMoments[7];
    HuMoments(mmnts, huMoments);

    double f6 = huMoments[0] != 0 ? huMoments[0] >  0 ? log10(huMoments[0]) : -1* log10(abs(huMoments[0])) : 0;
    double f7 = huMoments[1] != 0 ? huMoments[1] > 0 ? log10(huMoments[1]) : -1* log10(abs(huMoments[1])) : 0;

    std::vector<double> features = {f1, f2, f3, f4, f5, f6, f7};

    //cout << "Features"  << endl;
    //cout << "f1=" << f1 << ",f2=" << f2 << ",f3=" << f3 << ",f4=" << f4 <<  ",f5=" << f5 << ",f6=" << f6 << ",f7=" << f7 << endl;

    return features;
}

String concatFeatureString(std::vector<double> features)
{

    String featureString = "";

    if (!features.empty())
    {
        for (double f : features)
        {
            featureString += to_string(f) + ",";
        }
    }

    return featureString;
}

void autoMaticFeatureExtraction(Point p, String hipArea)
{

    //cout << "Automatic Feature Extraction Starts" << endl;

    std::vector<double> features = getFeatures(p, enhancedImg);
    String featureString = fileName + "," + to_string(p.x) + "," + to_string(p.y) + "," + concatFeatureString(features);

    features = getFeatures(p, negRoiImg);

    featureString += concatFeatureString(features) + hipArea;

    featureStringList.push_back(featureString);

    //cout << "Automatic Feature Extraction DONE!" << endl;
}

Vec3d toHSV(Vec3b rgbColor)
{
    double r = rgbColor[0] / 255.0;
    double g = rgbColor[1] / 255.0;
    double b = rgbColor[2] / 255.0;

    double mx = max(r, max(g, b));
    double mn = min(r, min(g, b));
    double hue, sat, val;

    //cout << "mx min =" << to_string(mx) << "," << to_string(mn) << endl;

    if (mx == mn)
    {
        hue = 0.0;
    }
    else if (mx == r)
    {
        hue = 60.0 * (0.0 + (g - b) / (mx - mn));
    }
    else if (mx == g)
    {
        hue = 60.0 * (2.0 + (b - r) / (mx - mn));
    }
    else if (mx == b)
    {
        hue = 60.0 * (4.0 + (r - g) / (mx - mn));
    }

    if (hue < 0)
    {
        hue += 360.0;
    }

    if (mx == 0)
    {
        sat = 0.0;
    }
    else
    {
        sat = (mx - mn) / mx;
    }

    val = mx;

    //Vec3d hsv = Vec3d(hue, sat, val);
    //cout << "  rgb=" << to_string(rgbColor[0]) << "," << to_string(rgbColor[1]) <<"," << to_string(rgbColor[2]) << endl;
    //cout << "  hsv=" << to_string(hsv[0])      << "," << to_string(hsv[1])      <<"," << to_string(hsv[2]) << endl;
    return Vec3d(hue, sat, val);
}

int main(int argc, char **argv)
{

    String fileList[25] = {
        "20191128_190652_HDR_2.jpg",
        "20191128_190910_HDR_2.jpg",
        "20191128_191028_HDR_2.jpg",
        "20191128_191118_HDR_2.jpg",
        "20191128_191134_HDR_2.jpg",
        "20191128_191203_HDR_2.jpg",
        "20191128_191320_HDR_2.jpg",
        "20191128_191349_HDR_2.jpg",
        "20191128_191356_HDR_2.jpg",
        "20191128_191420_HDR_2.jpg",
        "20191128_191500_HDR_2.jpg",
        "20191128_191558_HDR_2.jpg",
        "20191128_191647_HDR_2.jpg",
        "20191128_192218_HDR_2.jpg",
        "20191128_192315_HDR_2.jpg",
        "20191128_192402_HDR_2.jpg",
        "20191128_192452_HDR_2.jpg",
        "20191128_192514_HDR_2.jpg",
        "20191128_192922_HDR_2.jpg",
        "Control_2_Bregma_-3.12_PS_(1x).jpg",
        "Control_3_Bregma_-2.jpg",
        "Control_A_-2.28.jpg",
        "Control_A_-2.76.jpg",
        "Control_A_3-4.jpg",
        "Control_A_-4.08.jpg"};

    char keyPressed = 0;
    int filelIndex = 0;
    int error = init();

    if (error)
    {
        cout << "Terminating Segmenter" << endl;
        return -1;
    }

    const int RELOAD_IMAGE = 114; // 'r' key board
    const int SAVE_EXTRACTED_FEATURES = 115; // 's' key board
    const int QUIT_PROGRAM = 113; // 'q' key board key
    const int GO_TO_PREV_IMAGE = 81; // 'previous arrow' keyboard key
    const int GO_TO_NEXT_IMAGE = 83; // 'next arrow' keyboard key

    // Wait until user presses some key
    while (keyPressed != QUIT_PROGRAM)
    {
        keyPressed = waitKey(0);
        switch (keyPressed)
        {
            case GO_TO_PREV_IMAGE :
            { 
                if (filelIndex == 0)
                {
                    break;
                };
                filelIndex--;
                fileName = fileList[filelIndex];
                reloadPoints();
                break;
            }
            case GO_TO_NEXT_IMAGE :
            { 
                if (filelIndex == 24)
                {
                    break;
                };
                filelIndex++;
                fileName = fileList[filelIndex];
                reloadPoints();
                break;
            }
            case RELOAD_IMAGE: 
            { 
                cout << "Reloading" << endl;
                reloadPoints();
                cout << "Done" << endl;
                break;
            }
            case SAVE_EXTRACTED_FEATURES:
            { 
                cout << "Saving" << endl;
                error = savePixelFeatures();
                cout << "Done, error=" << to_string(error) << endl;
                break;
            }
            case QUIT_PROGRAM:
            { 
                cout << "bye" << endl;
                break;
            }
            default:
            { // other
                cout << "key pressed=" << to_string(keyPressed) << endl;
            }
        }
    }
    cout << "Thanks!" << endl;

    cleanup();

    return 0;
}
