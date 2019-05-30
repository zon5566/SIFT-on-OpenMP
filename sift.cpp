#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <unistd.h>
#include <omp.h>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "util.hpp"
#include "sift.hpp"

#define PI 3.14159265

using namespace std;
using namespace cv;

static void initSigmaList(double base, double k, double* sigma, int n_octave, int n_level) {
	
	for (int i = 0; i < n_octave * n_level; i++) {
		int cur_octave = i / n_level;
		int cur_level = i % n_level;
		sigma[i] = pow(k, cur_octave) * base * pow(k, (double)cur_level / (n_level-3));
	}
}

static void initGaussian(double* sigma, Mat** gaussian_x, Mat** gaussian_y, int n_octave, int n_level) {

	int k_size;
	
	for (int i = 0; i < n_octave; i++) {
		for (int j = 0; j < n_level; j++) {
			/*
				n_level = s+3 images including original
				sigma, sigma * 2^(1/s), sigma * 2^(2/s), ..., sigma * 2^(s+1)/s
			*/
			k_size = int(sqrt(sigma[i * n_level + j]) * 6.0);
			if (k_size % 2 == 0) k_size += 1;
			gaussian_x[i][j] = getGaussianKernel(k_size, sigma[i * n_level + j], CV_64F);
			gaussian_y[i][j] = getGaussianKernel(k_size, sigma[i * n_level + j], CV_64F);	
		}
	}
}

static Mat convolution(Mat img, Mat kernel_x, Mat kernel_y) {

	int k_size = kernel_x.rows;
	int padding = (int)((k_size - 1) / 2);

	int n_row = img.rows, n_col = img.cols;
	Mat tmp_img = Mat::zeros(n_row, n_col, CV_64F);
	Mat res_img = Mat::zeros(n_row, n_col, CV_64F);

	copyMakeBorder(img, img, 0, 0, padding, padding, BORDER_REPLICATE);
	
	#pragma omp parallel
	#pragma omp for schedule(auto)	//collapse(2)
	for (int i = 0; i < n_row; i++) {
		for (int j = padding; j < n_col + padding; j++) {
			for (int w = 0; w < 2 * padding + 1; w++) {
				tmp_img.at<double>(i, j-padding) += kernel_x.at<double>(w) * img.at<double>(i, j - (w - padding));
			}
		}
	}
	
	copyMakeBorder(tmp_img, tmp_img, padding, padding, 0, 0, BORDER_REPLICATE);

	#pragma omp parallel
	#pragma omp for schedule(auto) //collapse(2)
	for (int i = padding; i < n_row + padding; i++) {
		for (int j = 0; j < n_col; j++) {
			for (int h = 0; h < 2 * padding + 1; h++) {
				res_img.at<double>(i-padding, j) += kernel_y.at<double>(h) * tmp_img.at<double>(i - (h - padding), j);
			}
		}
	}

	return res_img;
}


static void createPyramid(Mat img, Mat** gaussian_x, Mat** gaussian_y, Mat** pyramids, int n_octave, int n_level) {
	
	Mat* img_hierarchy = new Mat[n_octave];
	for (int i = 0; i < n_octave; i++) {
		img_hierarchy[i] = img;
		Size new_size(img.size().width / 2, img.size().height / 2);
		resize(img, img, new_size, 0, 0, INTER_LINEAR);
	}
	
	//#pragma omp parallel
	//#pragma omp for collapse(2)
	for (int i=0; i < n_octave; i++){
		for (int j=0; j < n_level; j++) {
			pyramids[i][j] = convolution(img_hierarchy[i], gaussian_x[i][j], gaussian_y[i][j]);
		}
		// pyrDown(img, img);
		/*
		Mat des;
		Size new_size(img.size().width / 2, img.size().height / 2);
		resize(img, des, new_size, 0, 0, INTER_LINEAR);
		img = des;
		*/
	}
	delete[] img_hierarchy;
}

static void createDoG(Mat** pyramids, Mat** DoG, int n_octave, int n_level) {
	#pragma omp parallel
	#pragma omp for collapse(2)
	for (int i = 0; i < n_octave; i++)
		for (int j = 0 ; j < n_level-1; j++) {
			subtract(pyramids[i][j], pyramids[i][j+1], DoG[i][j]);
		}
}

static void getExtrema(Mat** DoG, vector<ExtremaPoint>& points, int n_octave, int n_level) {

	int* local_size = new int[n_octave * (n_level - 3)];
	for (int i = 0; i < n_octave * (n_level - 3); i++)
		local_size[i] = 0;
	vector<ExtremaPoint>* local_vec = new vector<ExtremaPoint>[n_octave * (n_level - 3)];

	#pragma omp parallel for schedule(auto) collapse(2)
	for (int i = 0; i < n_octave; i++) {
		for (int j = 1; j < (n_level-1) - 1; j++) {
			int n_row = DoG[i][0].rows;
			int n_col = DoG[i][0].cols;
			for (int r = 1; r < n_row-1; r++){
				for (int c = 1; c < n_col-1; c++) {

					bool isMax = true, isMin = true;

					/* compare 26 neighbors */
					double current_point = DoG[i][j].at<double>(r, c);

					for (int d1 = -1; d1 <= 1; d1++) {
						for (int d2 = -1; d2 <= 1; d2++) {
							for (int d3 = -1; d3 <= 1; d3++) {
								if (current_point < DoG[i][j+d1].at<double>(r+d2, c+d3)){
									isMax = false;
									break;
								}
							}
							if (!isMax) break;
						}
						if (!isMax) break;
					}

					for (int d1 = -1; d1 <= 1; d1++) {
						for (int d2 = -1; d2 <= 1; d2++) {
							for (int d3 = -1; d3 <= 1; d3++) {
								if (current_point > DoG[i][j+d1].at<double>(r+d2, c+d3)){
									isMin = false;
									break;
								}
							}
							if (!isMin) break;
						}
						if (!isMin) break;
					}

					if (isMax || isMin){
						// points.push_back( ExtremaPoint(r, c, i, j) );
						local_vec[i * (n_level-3) + j - 1].push_back( ExtremaPoint(r, c, i, j) );
					}
				}
			}
			local_size[i * (n_level-3) + (j-1)] = local_vec[i * (n_level-3) + j - 1].size();
		}
	}

	int* prefix_sum = new int[n_octave * (n_level - 3)];
	int count = 0;
	for (int i = 0; i < n_octave * (n_level - 3); i++) {
		prefix_sum[i] = count;
		count += local_size[i]; 
	}

	ExtremaPoint* global_arr = new ExtremaPoint[count];	
	#pragma omp parallel
	#pragma omp for
	for (int i = 0; i < n_octave * (n_level-3); i++) {
		int start = prefix_sum[i];
		for (int j = start; j < start + local_size[i]; j++) {
			global_arr[j] = local_vec[i][j - start];
		}	
	}

	vector<ExtremaPoint> global_vec(global_arr, global_arr + count);
	points = global_vec;

	delete[] local_size;
	delete[] local_vec;
	delete[] prefix_sum;
	delete[] global_arr;
}

static void rejectOutlier(Mat** DoG, vector<ExtremaPoint>& points, vector<ExtremaPoint>& res_points) {
	
	double CONTRAST_THRE = 0.03, EDGE_THRE = 12.1;
	
	for (int i = 0; i < points.size(); i++) {
		/*
			For each potential key point, find its refined location by Taylor expansion series.
			Ref: http://dev.ipol.im/~reyotero/publi/phd/phd_thesis_AnatomyOfTheSIFTMethod_v2.pdf (3.13)
		*/
		int octave = points[i].octave;
		int level = points[i].level;
		int x = points[i].x;
		int y = points[i].y;

		double first_deriv[3];
		first_deriv[0] = (DoG[octave][level].at<double>(x + 1, y) - DoG[octave][level].at<double>(x - 1, y)) / 2.0;
		first_deriv[1] = (DoG[octave][level].at<double>(x, y + 1) - DoG[octave][level].at<double>(x, y - 1)) / 2.0;
		first_deriv[2] = (DoG[octave][level + 1].at<double>(x, y) - DoG[octave][level - 1].at<double>(x, y)) / 2.0;

		double second_deriv[3][3];
		second_deriv[0][0] = DoG[octave][level].at<double>(x + 1, y) + DoG[octave][level].at<double>(x - 1, y) - 2.0 * DoG[octave][level].at<double>(x, y);
		second_deriv[1][1] = DoG[octave][level].at<double>(x, y + 1) + DoG[octave][level].at<double>(x, y - 1) - 2.0 * DoG[octave][level].at<double>(x, y);
		second_deriv[2][2] = DoG[octave][level + 1].at<double>(x, y) + DoG[octave][level - 1].at<double>(x, y) - 2.0 * DoG[octave][level].at<double>(x, y);
		
		second_deriv[0][1] = (DoG[octave][level].at<double>(x + 1, y + 1) 
							- DoG[octave][level].at<double>(x + 1, y - 1)
							- DoG[octave][level].at<double>(x - 1, y + 1)
							+ DoG[octave][level].at<double>(x - 1, y - 1)) / 4.0;
		second_deriv[1][0] = second_deriv[0][1];

		second_deriv[0][2] = (DoG[octave][level + 1].at<double>(x + 1, y) 
							- DoG[octave][level + 1].at<double>(x - 1, y)
							- DoG[octave][level - 1].at<double>(x + 1, y)
							+ DoG[octave][level - 1].at<double>(x - 1, y)) / 4.0;
		second_deriv[2][0] = second_deriv[0][2];

		second_deriv[1][2] = (DoG[octave][level + 1].at<double>(x, y + 1) 
							- DoG[octave][level + 1].at<double>(x, y - 1)
							- DoG[octave][level - 1].at<double>(x, y + 1)
							+ DoG[octave][level - 1].at<double>(x, y - 1)) / 4.0;
		second_deriv[2][1] = second_deriv[2][1];

		double inverse_second_deriv[3][3];
		if (inverse_3x3(second_deriv, inverse_second_deriv) < 0) continue;

		double refined_points[3];
		for (int p = 0; p < 3; p++){
			refined_points[p] = 0.0;
			for (int q = 0; q < 3; q++) {
				refined_points[p] += -inverse_second_deriv[p][q] * first_deriv[q];
			}
		}

		/* discard the point if it is smaller than the contrast threshold */
		double tmp = 0.0;
		for (int p = 0; p < 3; p++)
			tmp += first_deriv[p] * refined_points[p];

		if (abs(DoG[octave][level].at<double>(x, y) + 0.5 * tmp) < CONTRAST_THRE)
			continue;

		/* discard the point that is on the edge, use Hessian matrix */
		double hessian_mat[2][2];
		hessian_mat[0][0] = second_deriv[0][0];
		hessian_mat[1][1] = second_deriv[1][1];
		hessian_mat[0][1] = second_deriv[0][1];
		hessian_mat[1][0] = second_deriv[1][0];

		double hessian_det = hessian_mat[0][0] * hessian_mat[1][1] - hessian_mat[1][0] * hessian_mat[0][1];
		double hessian_tr = hessian_mat[0][0] + hessian_mat[1][1];

		if (pow(hessian_tr, 2) / hessian_det > EDGE_THRE)
			continue; 
		
		res_points.push_back(ExtremaPoint(round(refined_points[0] + x), round(refined_points[1] + y), octave, level));
	}
}

static double interpolateAngle (double* histogram, int max_bin, int n_bin) {
	
	double max_left = histogram[(max_bin-1) % n_bin];
	double max_right = histogram[(max_bin+1) % n_bin];
	double maxval = histogram[max_bin];

	double interpolated_angle = (360 / (double)n_bin) * (max_bin);
	interpolated_angle += (360 / (double)n_bin) * (max_left - max_right) / (max_left - 2 * maxval + max_right);

	return interpolated_angle;
}

static void getOrientation(Mat** pyramids, vector<ExtremaPoint> points, double* sigmaList, 
					int N_OCTAVE, int N_LEVEL, vector<Keypoint>& kp_vec) {

	int N_ORIENTATION_BIN = 36;
	
	kp_vec.resize(points.size());

	vector<vector<Keypoint>> tmp(points.size(), vector<Keypoint>());

	#pragma omp parallel
	#pragma omp for schedule(auto)
	for (int i = 0; i < points.size(); i++) {
		int octave = points[i].octave;
		int level = points[i].level;
		int x = points[i].x;
		int y = points[i].y;
		double sigma = sigmaList[octave * N_OCTAVE + level];
		int radius = round((3*sigma - 1) / 2);

		Mat img = pyramids[octave][level];
		Size s = img.size();

		double* orienHist = new double[N_ORIENTATION_BIN];
		for (int t = 0; t < N_ORIENTATION_BIN; t++)
			orienHist[t] = 0.0;

		for (int r = -radius; r <= radius; r++) {
			/* also exclude the border pixel */
			if (x + r <= 0 || x + r >= s.height - 1) continue;

			for (int c = -radius; c <= radius; c++) {
				if (y + c <= 0 || y + c >= s.width - 1) continue;

				/* compute the magnitude and the orientation */
				double dx = img.at<double>(x+r+1, y+c) - img.at<double>(x+r-1, y+c);
				double dy = img.at<double>(x+r, y+c+1) - img.at<double>(x+r, y+c-1);
				double m = sqrt(pow(dx, 2) + pow(dy, 2));
				double theta = atan2(dx+1e-5, dy) * 180 / PI;

				if (theta < 0) theta += 360;
				else if (theta > 360) theta -= (int)(theta / 360) * 360;
				int bin = (int)round(theta / (360 / N_ORIENTATION_BIN)) % N_ORIENTATION_BIN;

				double weight = exp( -(r * r + c * c) / (2 * sigma * sigma) ) / (2 * PI * sigma * sigma);
				orienHist[bin] += weight * m;
			}
		}

		// smooth the histogram with 1/3 * [1 1 1] box filter
		double* smooth_orienHist = new double[N_ORIENTATION_BIN];
		for (int t = 0; t < N_ORIENTATION_BIN; t++) {
			smooth_orienHist[t] = (orienHist[(t-1) % N_ORIENTATION_BIN] 
									+ orienHist[t] 
									+ orienHist[(t+1) % N_ORIENTATION_BIN]) / 3.0;
		}

		double maxval = smooth_orienHist[0];
		int max_bin = 0;
		for (int bin = 0; bin < N_ORIENTATION_BIN; bin++){
			if (smooth_orienHist[bin] > maxval) {
				maxval = smooth_orienHist[bin];
				max_bin = bin;
			}
		}

		// interpolate the orientation, by doing so we can convert #bin into an approximate angle
		// ref: http://dev.ipol.im/~reyotero/publi/phd/phd_thesis_AnatomyOfTheSIFTMethod_v2.pdf (3.23)
		double maximum_orien = interpolateAngle (smooth_orienHist, max_bin, N_ORIENTATION_BIN); 
		//kp_vec.push_back(Keypoint(x, y, octave, level, maxval, maximum_orien));
		kp_vec[i] = Keypoint(x, y, octave, level, maxval, maximum_orien);

		// also add the local maxima orientation which is greater than 80% of max as a new key point
		for (int bin = 0; bin < N_ORIENTATION_BIN; bin++) {
			if (bin == max_bin) continue;

			double left = smooth_orienHist[(bin - 1) % N_ORIENTATION_BIN];
			double right = smooth_orienHist[(bin + 1) % N_ORIENTATION_BIN];

			if (smooth_orienHist[bin] > maxval * 0.8 && smooth_orienHist[bin] > left && smooth_orienHist[bin] > right) {
				double maximal_orien = interpolateAngle (smooth_orienHist, bin, N_ORIENTATION_BIN);
				//kp_vec.push_back(Keypoint(x, y, octave, level, smooth_orienHist[bin], maximal_orien));
				tmp[i].push_back(Keypoint(x, y, octave, level, smooth_orienHist[bin], maximal_orien));
			}
		}

		delete[] orienHist;
		delete[] smooth_orienHist;
	}

	for (int i = 0; i < tmp.size(); i++) {
                for (int j = 0; j < tmp[i].size(); j++) {
                        kp_vec.push_back(tmp[i][j]);
                }
        }
	


/*
	int count = 0;
	vector<int> prefix_sum(points.size(), 0);
	for (int i = 0; i < tmp.size(); i++) {
		prefix_sum[i] = count;
		for (int j = 0; j < tmp[i].size(); j++) {
			count++;
		}
	}

//	cout << "new points: " << count << "  out of  " << points.size() << endl;

	kp_vec.resize(points.size() + count);

	#pragma omp parallel
	#pragma omp for
	for (int i = 0; i < tmp.size(); i++) {
		for (int j = 0; j < tmp[i].size(); j++) {
			kp_vec[points.size() + prefix_sum[i] + j] = tmp[i][j];
		}
	}
*/
}

static void calcDescriptor(const Mat img, Keypoint point, vector<double>& descriptor, double* sigmaList, int N_OCTAVE) {

	// transform back to the original resolution
	int x = sub2index(point.x, point.octave), y = sub2index(point.y, point.octave);
	double sigma = sigmaList[point.octave * N_OCTAVE + point.level];

	double orien = point.orientation * PI / 180.0;
	double rotate_matrix[2][2] = {{cos(orien), -sin(orien)}, {sin(orien), cos(orien)}};

	int N_DESCRIPTOR_BIN = 8;
	Size s = img.size();

	// 16 x 16 window, each of one is a 4x4 block
	for (int block_r = -8; block_r <= 7; block_r += 4) {
		for (int block_c = -8; block_c <= 7; block_c += 4) {

			double* descHist = new double[N_DESCRIPTOR_BIN];
			for (int i = 0; i < N_DESCRIPTOR_BIN; i++)
				descHist[i] = 0.0;

			for (int r = 0; r < 4; r++) {
				if (x + block_r + r <= 0 || x + block_r + r >= s.height - 1) continue;

				for (int c = 0; c < 4; c++) {
					if (y + block_c + c <= 0 || y + block_c + c >= s.width - 1) continue;

					double dx = img.at<double>(x+block_r+r+1, y+block_c+c) - img.at<double>(x+block_r+r-1, y+block_c+c);
					double dy = img.at<double>(x+block_r+r, y+block_c+c+1) - img.at<double>(x+block_r+r, y+block_c+c-1);
					
					double norm_dy = rotate_matrix[0][0] * dy + rotate_matrix[0][1] * dx; 
					double norm_dx = rotate_matrix[1][0] * dy + rotate_matrix[1][1] * dx;
					double m = sqrt(pow(norm_dx, 2) + pow(norm_dy, 2));
					double theta = atan2(norm_dx + 1e-5, norm_dy) * 180 / PI;

					if (theta < 0) theta += 360;
					else if (theta > 360) theta -= (int)(theta / 360) * 360;
					int bin = (int)round(theta / (360 / N_DESCRIPTOR_BIN)) % N_DESCRIPTOR_BIN;

					double weight = exp( -(pow(block_r + r, 2) + pow(block_c + c, 2)) / (2 * sigma * sigma) ) / (2 * PI * sigma * sigma);
					descHist[bin] += weight * m;
				}
			}
			
			int offset = (4 * (block_r + 8) / 4 + (block_c + 8) / 4) * 8;
			for (int bin = 0; bin < N_DESCRIPTOR_BIN; bin++) {
				descriptor[offset + bin] = descHist[bin];
			}

			delete[] descHist;
		}
	}
}

static void normalize(vector<double>& desc) {

	double mean = accumulate(desc.begin(), desc.end(), 0.0) / desc.size();
	
	double stdev_nom = 0;
	for (int i = 0; i < desc.size(); i++) {
		stdev_nom += pow((desc[i] - mean), 2);
	}
	double stdev = sqrt(stdev_nom / (desc.size() - 1));
	for (int i = 0; i < desc.size(); i++) {
		desc[i] = (desc[i] - mean) / (stdev + 1e-10);
	}
}

static void getDescriptor(Mat img, const vector<Keypoint> kp_vec, double* sigmaList, int n_octave, vector< vector<double> >& desc_vec) {

	desc_vec.resize(kp_vec.size());

	#pragma omp parallel
        #pragma omp for schedule(auto)
	for (int i = 0; i < kp_vec.size(); i++) {
		vector<double> desc(128, 0);
		calcDescriptor(img, kp_vec[i], desc, sigmaList, n_octave);

		normalize(desc);
		desc_vec[i] = desc;
	}
}

static void opencv_sift(Mat img, string filename) {
	double start_time = omp_get_wtime();

	vector<KeyPoint> kps;
	Mat desc;
	SIFT sift(0, 3, 0.03, 12.1);
	sift(img, noArray(), kps, desc);

    	cout << "OpenCV detectAndCompute() takes: " << omp_get_wtime() - start_time << " seconds. " << endl;
	cout << "Detect " << kps.size() << " keypoints with opencv SIFT" << endl;
	// Add results to image and save
	Mat output;
	drawKeypoints(img, kps, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imwrite("res/opencv_" + filename + ".png", output);
}
/*
void plot_pyramid(Mat** pyramids, int n_octave, int n_level) {
	char name[100];
	Mat img;
	for (int i = 0; i < n_octave; i++) {
		for (int j = 0; j < n_level; j++) {
			img = pyramids[i][j];
			img.convertTo(img, CV_8U, 255.0);
			sprintf(name, "img/py_img_%d_%d.png", i+1, j+1);
			imwrite(name, img);
		}
	}
}
*/

void plot_keypoint(Mat img, vector<Keypoint> points, int N_ROW, int N_COL, int img_size, string filename) {
	
	int size = 0, amp = img_size / 4;
	Scalar color;

	sort(points.begin(), points.end(), [](Keypoint lhs, Keypoint rhs) {
		return lhs.magnitude > rhs.magnitude;
	});

	vector<Keypoint> visited;

	for (int i = 0; i < points.size(); i++) {
		if (points[i].x >= N_ROW || points[i].y >= N_COL
			|| points[i].x < 0 || points[i].y < 0)
			size ++;

		else {

			color = Scalar(rand()%256, rand()%256, rand()%256);
			int center_x = sub2index(points[i].x, points[i].octave), center_y = sub2index(points[i].y, points[i].octave);
				
			bool canShow = true;
			for (int p = 0; p < visited.size(); p++) {
				int visited_x = sub2index(visited[p].x, visited[p].octave);
				int visited_y = sub2index(visited[p].y, visited[p].octave);
				if (sqrt(pow(visited_x - center_x, 2) + pow(visited_y - center_y, 2)) < 10){
					canShow = false;
					break;
				}
			}
			if (!canShow) continue;
			
			double k = 2.0;
			amp = 120;
			//double vis_size = amp * 0.01 * pow(k, points[i].octave) * pow(k, (double)points[i].level / (6));
			double vis_size = amp * pow(k, points[i].octave) * pow(k, (double)points[i].level / (3)) * 0.02;//* points[i].magnitude;

			//double vis_size = amp * points[i].magnitude * 5;

			int lineEnd_x = center_x + vis_size * sin(points[i].orientation * PI / 180.0);
			int lineEnd_y = center_y + vis_size * cos(points[i].orientation * PI / 180.0);

			circle(img, Point(center_y, center_x), vis_size, color);
			arrowedLine(img, Point(center_y, center_x), Point(lineEnd_y, lineEnd_x), color, 1);

			visited.push_back(points[i]);
		}
	}
	cout << "There are " << points.size() << " keypoints, with " << size << " invalid points" << endl;
	imwrite("res/our_" + filename + ".png", img);
}

void pipeline(Mat img, vector<Keypoint>& kp_vec, vector< vector<double> >& desc_vec) {

	int N_ROW = img.rows, N_COL = img.cols;
	// int N_OCTAVE = 4;
	int N_OCTAVE = round(log( (double)std::min( img.rows, img.cols) ) / log(2.) - 2) + 1;
	int N_LEVEL = 6;

	printf("Octave: %d, Level: %d\n", N_OCTAVE, N_LEVEL);

	Mat** pyramids = new Mat*[N_OCTAVE];
	Mat** gaussian_x = new Mat*[N_OCTAVE];
	Mat** gaussian_y = new Mat*[N_OCTAVE];
	Mat** dog = new Mat*[N_OCTAVE];
	
	for (int i = 0; i < N_OCTAVE; i++){
		pyramids[i] = new Mat[N_LEVEL];
		gaussian_x[i] = new Mat[N_LEVEL];
		gaussian_y[i] = new Mat[N_LEVEL];
		dog[i] = new Mat[N_LEVEL-1];
	}

	double base = 1.6, k = 2.0;
	double* sigmaList = new double[N_OCTAVE * N_LEVEL];
	double start, end;
	
	// 1. N * M list of sigma
	start = omp_get_wtime();
	initSigmaList(base, k, sigmaList, N_OCTAVE, N_LEVEL); 
	end = omp_get_wtime();
	printf("init sigma list = %.3g secs\n", end-start);
	
	// 2. Init gaussain kernel by sigmaList
	start = omp_get_wtime();
	initGaussian(sigmaList, gaussian_x, gaussian_y, N_OCTAVE, N_LEVEL);
	end = omp_get_wtime();
	printf("init gaussian = %.3g secs\n", end-start);

	// 3. Build Pyramid
	start = omp_get_wtime();
	createPyramid(img, gaussian_x, gaussian_y, pyramids, N_OCTAVE, N_LEVEL);
	end = omp_get_wtime();
	printf("create pyramid = %.3g secs\n", end-start);
	
	// 4. Subtract DoG
	start = omp_get_wtime();
	createDoG(pyramids, dog, N_OCTAVE, N_LEVEL);
	end = omp_get_wtime();
	printf("create DoG = %.3g secs\n", end-start);

	// 5. S+3 --> S extrama in one octave
	vector<ExtremaPoint> points;
	start = omp_get_wtime();
	getExtrema(dog, points, N_OCTAVE, N_LEVEL);
	end = omp_get_wtime();
	printf("getExtrema = %.3g secs\n", end - start);

	// 6. Section 4
	vector<ExtremaPoint> refined_points;
	start = omp_get_wtime();
	rejectOutlier(dog, points, refined_points);
	end = omp_get_wtime();
	printf("reject outlier = %.3g secs\n", end - start);

	// 7.
	start = omp_get_wtime();
	getOrientation(pyramids, refined_points, sigmaList, N_OCTAVE, N_LEVEL, kp_vec);
	end = omp_get_wtime();
	printf("get orientation = %.3g secs\n", end - start);

	// 8.
	start = omp_get_wtime();
	getDescriptor(img, kp_vec, sigmaList, N_OCTAVE, desc_vec);
	end = omp_get_wtime();
	printf("get descriptor = %.3g secs\n", end - start);
}

int main(int argc, char* argv[]) {

	//cout << "OPENCV VERSION: " << CV_VERSION << endl;

	int nthreads = atoi(argv[2]);
	
	omp_set_dynamic(0);
	if (nthreads < 0) {
		omp_set_num_threads(omp_get_num_procs());
		printf("Using %d threads...\n", omp_get_num_procs());
	}
	else {
		omp_set_num_threads(nthreads);
		printf("Using %d threads...\n", nthreads);
	}

	string filename = argv[1];
	Mat img = imread(filename, IMREAD_GRAYSCALE);
	Mat tmp = img;

	img.convertTo(img, CV_64F, 1.0/255.0);
	vector<Keypoint> kp_vec;
	vector< vector<double> > desc_vec;

	// Algorithm Starts
	double start_time = omp_get_wtime();

	pipeline(img, kp_vec, desc_vec);

	// Algortihm Ends
	double time = omp_get_wtime() - start_time;
	cout << "Sift Algorithm takes: " << time << " seconds. " << endl; 

	// for visualization
	vector<Mat> channels;
	Mat keypoint_img;
	channels.push_back(tmp);
	channels.push_back(tmp);
	channels.push_back(tmp);
	merge(channels, keypoint_img);

	filename = filename.substr(0, filename.size()-4);
	opencv_sift(keypoint_img, filename);
	plot_keypoint(keypoint_img, kp_vec, img.rows, img.cols, img.size().height, filename);
	cout << endl;
	return 0;
}
