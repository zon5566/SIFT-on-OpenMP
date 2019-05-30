#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

/* Extrema point is for keypoint localization */
struct ExtremaPoint {
	int x;		// relatviely in that octave scale
	int y;
	int octave;
	int level;
	ExtremaPoint() {} 
	ExtremaPoint(int x_, int y_, int octave_, int level_): 
		x(x_), y(y_), octave(octave_), level(level_) {}
};

/* Key point is for the orientation assignment and descriptor after that */
struct Keypoint {
	int x;
	int y;
	int octave;
	int level;
	double magnitude;
	double orientation;
	Keypoint() {}
	Keypoint(int x_, int y_, int o_, int l_, double mag_, double orie_):
		x(x_), y(y_), octave(o_), level(l_), magnitude(mag_), orientation(orie_) {}
};

void pipeline(cv::Mat, std::vector<Keypoint>&, std::vector< std::vector<double> >&);

void plot_keypoint(cv::Mat, std::vector<Keypoint>, int, int, int, std::string);

void plot_pyramid(cv::Mat**, int, int);
