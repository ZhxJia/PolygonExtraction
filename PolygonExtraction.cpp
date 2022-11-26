#include <queue>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <ppl.h>

#include "PolygonExtraction.h"

#define image_show 0

namespace polygon {

	wdErrorCode BrightnessCheck(const cv::Mat& img_roi, const int& gray_threshold, const double& low_thresh, const double& high_thresh)
	{
		cv::Mat image = img_roi.clone();
		cv::threshold(image, image, gray_threshold, 255, cv::THRESH_BINARY);//根据给定阈值对图像二值化
		int sum = cv::countNonZero(image);                              //计算二值图像中非零像素数目

		//判断图像像素灰度大于阈值的数目是否位于给定值之间
		if (sum < low_thresh)
		{
			_ErrorCodeExplanation = "PolygonExtraction: part is too dark.";
			return LLPartIsTooDark;
		}
		else if (sum > high_thresh)
		{
			_ErrorCodeExplanation = "PolygonExtraction: part is too bright.";
			return LLBackgroundIsTooBright;
		}
		else
			return success;
	}
	/******************轮廓筛选*********************/

	wdErrorCode contourSelection(const std::vector<std::vector<cv::Point>>& _contours_in, std::vector<std::vector<cv::Point>>& _contours_out, bool fix_threshold = false) {
		if (_contours_in.size() == 0) {
			return LLFailToGetEnoughContours;
		}
		std::vector<int> contours_len;
		for (int i = 0; i < _contours_in.size(); ++i) {
			contours_len.push_back(_contours_in[i].size());
		}
		if (fix_threshold) {
			for (int j = 0; j < contours_len.size(); ++j) {
				if (contours_len[j] >= 60) {
					_contours_out.push_back(_contours_in[j]);
				}
			}
		}
		else {
			double contours_len_avg = common::calVecAvg<int>(contours_len);
			for (int j = 0; j < contours_len.size(); ++j) {
				if (contours_len[j] >= contours_len_avg * 0.5) {
					_contours_out.push_back(_contours_in[j]);
				}
			}
		}
		if (_contours_out.size() == 0) {
			return LLFailToGetEnoughContours;
		}


		//权宜之计:减少轮廓数量以加快示教算法速度
		//if (_contours_out.size() > 5) {
		//	std::sort(_contours_out.begin(), _contours_out.end(), [&](std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {return (contour1.size() > contour2.size()); });
		//	_contours_out.erase(_contours_out.begin() + _contours_out.size() * 0.9, _contours_out.end());
		//}

		return success;


	}

	/*********************轮廓分割********************/
	bool contoursDecomp(const std::vector<std::vector<cv::Point>>& _contours_in) {



		return true;
	}




	/*************************************************/

  // @brief: ransac fit ploygon coeff
  // @param [in]: pos_vec:
  // @param [in/out]: coeff
	//1. 随机采样K个点
	//2. 对该K个点拟合模型
	//3. 计算其他点到拟合模型的距离，小于一定阈值当作内点，统计内点数量
	//4. 重复M次，选择内点数最多的模型
	//5. 利用所有内点重新估计模型(最小二乘，可选)
	bool CPolygon::RobustPolyFit(std::vector<cv::Point>& inliers, const int max_iters, float min_radius, float min_inlier_ratio, bool fix_se) {
		int n = static_cast<int>(contours.size());
		if (n < 3) {
			return false;
		}
		int q1 = static_cast<int>(n / 4);
		int q2 = static_cast<int>(n / 2);
		int q3 = static_cast<int>(n * 3 / 4);
		std::vector<int> index(3, 0);
		float best_ratio = 0.0f;
		float best_radius = 0.0f;
		cv::Point2d best_center;

		for (int its = 0; its < max_iters; ++its) {
			index[0] = std::rand() % q2;
			index[1] = q2 + std::rand() % q1;
			index[2] = q3 + std::rand() % q1;
			if (index[0] == index[1] || index[0] == index[2] || index[1] == index[2])
				continue;

			//Create Circle from 3 Points
			cv::Point2d center;
			float radius;
			getCircle(contours[index[0]], contours[index[1]], contours[index[2]], center, radius);

			// inlier set unused at the meonet but could be used to approximate a circle from all
			std::vector<cv::Point> inliers_temp;
			float inlier_ratio = verifyCircle(contours, center, radius, inliers_temp);

			//update best circle information if necessary
			if (inlier_ratio >= best_ratio) {
				best_ratio = inlier_ratio;
				best_radius = radius;
				best_center = center;
				inliers.assign(inliers_temp.begin(), inliers_temp.end());
			}

		}

		if (best_ratio >= min_inlier_ratio) {
			arc_radius = best_radius;
			arc_ct = best_center;
			return true;
		}
		else {
			return false;
		}

	}

	int CPolygon::judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& pt) {
		cv::Point2d p12 = pt - st_pt;
		cv::Point2d p23 = ed_pt - pt;
		double outer_product = p12.x * p23.y - p23.x * p12.y;
		if (outer_product > 0) {
			return 1; //顺时针
		}
		else if (outer_product < 0) {
			return -1; //逆时针
		}
		else {
			return 0;
		}

	}

	int CPolygon::judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& ct_pt, const cv::Point2d& pt) {
		cv::Point2d p12 = st_pt - ct_pt;
		cv::Point2d p23 = pt - ct_pt;
		double outer_product = p12.x * p23.y - p23.x * p12.y;
		if (outer_product > 0) {
			return 1; //顺时针
		}
		else if (outer_product < 0) {
			return -1; //逆时针
		}
		else {
			return 0;
		}

	}

	bool CPolygon::RobustArcFit(std::vector<cv::Point>& _contours_in, int max_iters, float min_radius, float min_inlier_ratio, bool fix_se) {

#if image_show
		cv::Mat contour_show = cv::Mat::zeros(cv::Size(1500, 1500), CV_8UC3);
		common::drawCross(contour_show, _contours_in, 1, cv::Scalar(255, 255, 255));
#endif
		int n = static_cast<int>(_contours_in.size());
		if (n < 3) {
			return false;
		}
		if (max_iters == -1) {
			max_iters = n * 2;
		}
		int q1 = static_cast<int>(n / 4);
		int q2 = static_cast<int>(n / 2);
		int q3 = static_cast<int>(n * 3 / 4);
		std::vector<int> index(3, 0);
		float best_ratio = 0.0f;
		float best_radius = 0.0f;
		cv::Point2d best_center;

		for (int its = 0; its < max_iters; ++its) {
			index[0] = std::rand() % q2;
			index[1] = q2 + std::rand() % q1;
			index[2] = q3 + std::rand() % q1;
			if (index[0] == index[1] || index[0] == index[2] || index[1] == index[2])
				continue;

			//Create Circle from 3 Points
			cv::Point2d center;
			float radius;
			getCircle(_contours_in[index[0]], _contours_in[index[1]], _contours_in[index[2]], center, radius);

			// inlier set unused at the meonet but could be used to approximate a circle from all
			std::vector<cv::Point> inliers_temp;
			float inlier_ratio = verifyCircle(_contours_in, center, radius, inliers_temp);

			//update best circle information if necessary
			if (inlier_ratio >= best_ratio) {
				best_ratio = inlier_ratio;
				best_radius = radius;
				best_center = center;
				inliers.assign(inliers_temp.begin(), inliers_temp.end());
			}
			if (best_ratio >= min_inlier_ratio) {
				break;
			}

		}

		if (best_ratio >= min_inlier_ratio) {
			arc_radius = best_radius;
			arc_ct = best_center;
			// 获取起始角度和终止角度,暂时通过拟合内点值作为起点和终点
			cv::Point2d start_point(inliers.front().x, inliers.front().y);
			cv::Point2d end_point(inliers.back().x, inliers.back().y);
			double st_angle = atan2((start_point.y - arc_ct.y), (start_point.x - arc_ct.x)); //[-pi, pi]
			double ed_angle = atan2((end_point.y - arc_ct.y), (end_point.x - arc_ct.x));
			arc_st_angle = st_angle;
			arc_ed_angle = ed_angle;

			//确定圆弧的方向
			//cv::Point2d mid_point(inliers[static_cast<int>(inliers.size() / 2.0)].x, inliers[static_cast<int>(inliers.size() / 2.0)].y);
			//cv::Point2d p12 = mid_point - start_point;
			//cv::Point2d p23 = end_point - mid_point;
			//double outer_product = p12.x * p23.y - p23.x * p12.y;
			//if (outer_product > 0) {
			//	arc_dir = 1; //顺时针
			//}
			//else if (outer_product < 0) {
			//	arc_dir = -1; //逆时针
			//}
			//else {
			//	arc_dir = 0;
			//}
			int dir_last = 0;
			for (int i = 1; i < inliers.size() - 1; ++i) {
				int dir = judgeArcDir(start_point, end_point, arc_ct, inliers[i]);
				if (dir != 0) {
					if (dir_last == dir) {
						arc_dir = dir;
						break;
					}
					dir_last = dir;
				}
			}

			if (arc_dir < 0 && st_angle < 0 && ed_angle > 0) {
				st_angle += 2 * CV_PI;
			}
			if (arc_dir > 0 && st_angle > 0 && ed_angle < 0) {
				ed_angle += 2 * CV_PI;
			}
			if (arc_dir < 0 && (st_angle < ed_angle)) {
				arc_span = (2 * CV_PI - abs(st_angle - ed_angle)) * arc_dir * 180.0 / CV_PI;
			}
			else if (arc_dir > 0 && (ed_angle < st_angle)) {
				arc_span = (2 * CV_PI - abs(st_angle - ed_angle)) * arc_dir * 180.0 / CV_PI;
			}
			else if (ed_angle == st_angle) {
				arc_span = 360.0;
			}
			else {
				arc_span = abs(st_angle - ed_angle) * arc_dir * 180.0 / CV_PI;
			}

			double arc_len = abs(arc_span * CV_PI / 180.0 * arc_radius);
			if (arc_len < inliers.size() * 0.2) {
				int j = 0;
				++j;
			}
			//cv::Point2d start_point_temp;
			//cv::Point2d end_point_temp;
			//start_point_temp.x = arc_ct.x + arc_radius * cos(arc_st_angle);
			//start_point_temp.y = arc_ct.y + arc_radius * sin(arc_st_angle);

			//end_point_temp.x = arc_ct.x + arc_radius * cos(arc_ed_angle);
			//end_point_temp.y = arc_ct.y + arc_radius * sin(arc_ed_angle);
#if image_show
			drawArc(contour_show, arc_ct, arc_st_angle, arc_span, arc_radius, cv::Scalar(255, 0, 0), 5.0);
#endif
			return true;
		}
		else {
			return false;
		}

	}

	void CPolygon::drawArc(cv::Mat& src_img, const cv::Point2d& _arc_ct, const double st_angle, const double angle_span, const double _arc_radius, const cv::Scalar& color, const double precision) {

		double angle_span_rad = angle_span * CV_PI / 180.0;
		cv::Point2d start_point, end_point;
		start_point.x = _arc_ct.x + _arc_radius * cos(st_angle);
		start_point.y = _arc_ct.y + _arc_radius * sin(st_angle);

		end_point.x = _arc_ct.x + _arc_radius * cos(st_angle + angle_span_rad);
		end_point.y = _arc_ct.y + _arc_radius * sin(st_angle + angle_span_rad);

		//std::vector<cv::Point2d> arc_dots;
		//arc_dots.push_back(start_point);
		double st_angle_deg = 0.0;
		if (angle_span < 0) {
			st_angle_deg = st_angle * 180.0 / CV_PI + angle_span;
		}
		else {
			st_angle_deg = st_angle * 180.0 / CV_PI;
		}
		st_angle_deg = st_angle_deg < 0 ? st_angle_deg + 360 : st_angle_deg;
		double angle_span_abs = abs(angle_span);
		cv::ellipse(src_img, arc_ct, cv::Size(_arc_radius, _arc_radius), st_angle_deg, 0, angle_span_abs, color);

	}


	void CPolygon::drawArc(cv::Mat& src_img, const cv::Scalar& color) const {

		double st_angle_deg = 0.0;
		if (arc_span < 0) {
			st_angle_deg = arc_st_angle * 180.0 / CV_PI + arc_span;
		}
		else {
			st_angle_deg = arc_st_angle * 180.0 / CV_PI;
		}
		st_angle_deg = st_angle_deg < 0 ? st_angle_deg + 360 : st_angle_deg;
		double angle_span = abs(arc_span);
		cv::ellipse(src_img, arc_ct, cv::Size(arc_radius, arc_radius), st_angle_deg, 0, angle_span, color);

	}

	bool CPolygon::Point2LinePrj(const double A, const double B, const double C, const cv::Point& ext_pt, cv::Point2d& prj_pt) {
		cv::Point2d p1, p2; //获取直线上两点
		if (abs(A / (B + 0.00001)) < 57) {
			p1.x = ext_pt.x - 10;
			p2.x = ext_pt.x + 10;
			p1.y = (-C - A * (p1.x)) / B;
			p2.y = (-C - A * (p2.x)) / B;
		}
		else {
			p1.y = ext_pt.y - 10;
			p2.y = ext_pt.y + 10;
			p1.x = (-C - B * (p1.y)) / A;
			p2.x = (-C - B * (p2.y)) / A;
		}
		double a = p2.x - p1.x;
		double b = p2.y - p1.y;

		double moleculex = a * a * ext_pt.x + a * b * ext_pt.y - b * p2.x * p1.y + b * p1.x * p2.y;
		double moleculey = b * b * ext_pt.y + a * b * ext_pt.x - a * p1.x * p2.y + a * p2.x * p1.y;
		prj_pt.x = moleculex / (a * a + b * b);
		prj_pt.y = moleculey / (a * a + b * b);

#if image_show
		cv::Mat prj_show = cv::Mat::zeros(cv::Size(ext_pt.x + 200, ext_pt.y + 200), CV_8UC3);
		cv::line(prj_show, p1, p2, cv::Scalar(0, 255, 0));
		cv::line(prj_show, prj_pt, ext_pt, cv::Scalar(255, 0, 0));
#endif

		return true;
	}


	bool CPolygon::LineFit(const std::vector<cv::Point>& _contours_in, float min_inlier_ratio) {
		// line equation: Ax+By+C = 0
		cv::Vec4f fit_line_param;
		cv::fitLine(_contours_in, fit_line_param, CV_DIST_HUBER, 0, 0.01, 0.01);
		double A_ = -fit_line_param[1];
		double B_ = fit_line_param[0];
		double C_ = fit_line_param[1] * fit_line_param[2] - fit_line_param[0] * fit_line_param[3];
		if (A_ == 0 && B_ == 0) {
			return false;
		}
		A = A_;
		B = B_;
		C = C_;

		//calculate projection
		cv::Point2d prj_point_st;
		cv::Point2d prj_point_ed;
		Point2LinePrj(A, B, C, _contours_in.front(), prj_point_st);
		Point2LinePrj(A, B, C, _contours_in.back(), prj_point_ed);

#if image_show
		cv::Mat line_show = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
		cv::Mat contour_show = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
		common::drawCross(line_show, _contours_in, 1, cv::Scalar(255, 0, 0));
		common::drawCross(contour_show, _contours_in, 1, cv::Scalar(255, 0, 0));
		cv::line(line_show, prj_point_st, prj_point_ed, cv::Scalar(0, 255, 0));

#endif
		//判断内点(在直线上的轮廓点)数量是否达到95%
		double inliers_ratio = verifyLine(prj_point_st, prj_point_ed, _contours_in);
		if (inliers_ratio > min_inlier_ratio) {
			st = prj_point_st;
			ed = prj_point_ed;
			return true;
		}
		else {
			return false;
		}
	}

	bool CPolygon::CircleFitByPratt(const std::vector<cv::Point>& _points, double& sigma, bool err_analysis, int max_iters) {
		int npoints = _points.size();
		int iter = 0;
		if (npoints < 3)
			return false;
		cv::Moments mu = cv::moments(_points, false);
		if (mu.m00 != 0) {
			cv::Point2f centroid = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
			double Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, Cov_xy, Var_z;
			double Xi, Yi, Zi;
			double A0, A1, A2, A22;
			double Dy, xnew, x, ynew, y;
			double DET, Xcenter, Ycenter;
			Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0.;
			for (int i = 0; i < npoints; ++i) {
				Xi = static_cast<double>(_points[i].x) - centroid.x;   //  centered x-coordinates
				Yi = static_cast<double>(_points[i].y) - centroid.y;   //  centered y-coordinates
				Zi = Xi * Xi + Yi * Yi;

				Mxy += Xi * Yi;
				Mxx += Xi * Xi;
				Myy += Yi * Yi;
				Mxz += Xi * Zi;
				Myz += Yi * Zi;
				Mzz += Zi * Zi;
			}
			Mxx /= npoints;
			Myy /= npoints;
			Mxy /= npoints;
			Mxz /= npoints;
			Myz /= npoints;
			Mzz /= npoints;

			//    computing coefficients of the characteristic polynomial

			Mz = Mxx + Myy;
			Cov_xy = Mxx * Myy - Mxy * Mxy;
			Var_z = Mzz - Mz * Mz;

			A2 = 4.0 * Cov_xy - 3.0 * Mz * Mz - Mzz;
			A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz;
			A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy;
			A22 = A2 + A2;

			//    finding the root of the characteristic polynomial
			//    using Newton's method starting at x=0  
			//     (it is guaranteed to converge to the right root)
			x = 0.; y = A0;
			for (iter = 0; iter < max_iters; iter++)  // usually, 4-6 iterations are enough
			{
				Dy = A1 + x * (A22 + 16. * x * x);
				xnew = x - y / Dy;
				if ((xnew == x) || (!boost::math::isfinite(xnew))) break;
				ynew = A0 + xnew * (A1 + xnew * (A2 + 4.0 * xnew * xnew));
				if (abs(ynew) >= abs(y))  break;
				x = xnew;  y = ynew;
			}

			//    computing paramters of the fitting circle

			DET = x * x - x * Mz + Cov_xy;
			Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / DET / 2.0;
			Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / DET / 2.0;

			//       assembling the output
			arc_radius = sqrt(Xcenter * Xcenter + Ycenter * Ycenter + Mz + x + x);
			arc_ct = cv::Point2d(Xcenter + centroid.x, Ycenter + centroid.y);
		}
		else {
			return false;
		}
		if (err_analysis) {
			double dx = 0;
			double dy = 0;
			double sum = 0;
			for (int j = 0; j < npoints; ++j) {
				dx = _points[j].x - arc_ct.x;
				dy = _points[j].y - arc_ct.y;
				sum += pow(sqrt(dx * dx + dy * dy) - arc_radius, 2);
			}
			sigma = sqrt(sum / npoints);
		}
		else {
			sigma = 0.0;
		}
		return true;
	}

	bool CPolygon::CircleFitByTaubin(const std::vector<cv::Point>& _points, double& sigma, bool err_analysis, int max_iters) {
#if image_show
		cv::Mat contour_show = cv::Mat::zeros(cv::Size(1000, 1000), CV_8UC3);
		common::drawCross(contour_show, _points, 1, cv::Scalar(255, 255, 255));
#endif
		int npoints = _points.size();
		int iter = 0;
		if (npoints < 3)
			return false;
		double radius = 0;
		cv::Point2d ct;
		cv::Moments mu = cv::moments(_points, false);
		if (mu.m00 != 0) {
			cv::Point2f centroid = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
			double Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, Cov_xy, Var_z;
			double Xi, Yi, Zi;
			double A0, A1, A2, A22, A3, A33;
			double Dy, xnew, x, ynew, y;
			double DET, Xcenter, Ycenter;
			Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0.;
			for (int i = 0; i < npoints; ++i) {
				Xi = static_cast<double>(_points[i].x) - centroid.x;   //  centered x-coordinates
				Yi = static_cast<double>(_points[i].y) - centroid.y;   //  centered y-coordinates
				Zi = Xi * Xi + Yi * Yi;

				Mxy += Xi * Yi;
				Mxx += Xi * Xi;
				Myy += Yi * Yi;
				Mxz += Xi * Zi;
				Myz += Yi * Zi;
				Mzz += Zi * Zi;
			}
			Mxx /= npoints;
			Myy /= npoints;
			Mxy /= npoints;
			Mxz /= npoints;
			Myz /= npoints;
			Mzz /= npoints;

			// computing coefficients of the characteristic polynomial
			Mz = Mxx + Myy;
			Cov_xy = Mxx * Myy - Mxy * Mxy;
			Var_z = Mzz - Mz * Mz;
			A3 = 4.0 * Mz;
			A2 = -3.0 * Mz * Mz - Mzz;
			A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz;
			A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy;
			A22 = A2 + A2;
			A33 = A3 + A3 + A3;

			//  finding the root of the characteristic polynomial
			//  using Newton's method starting at x=0  
			//  (it is guaranteed to converge to the right root)
			x = 0.; y = A0;
			for (iter = 0; iter < max_iters; iter++)  // usually, 4-6 iterations are enough
			{
				Dy = A1 + x * (A22 + A33 * x);
				xnew = x - y / Dy;
				if ((xnew == x) || (!boost::math::isfinite(xnew))) break;
				ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3));
				if (abs(ynew) >= abs(y))  break;
				x = xnew;  y = ynew;
			}

			// computing paramters of the fitting circle

			DET = x * x - x * Mz + Cov_xy;
			Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / DET / 2.0;
			Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / DET / 2.0;

			// assembling the output
			radius = sqrt(Xcenter * Xcenter + Ycenter * Ycenter + Mz);
			ct = cv::Point2d(Xcenter + centroid.x, Ycenter + centroid.y);
		}
		else {
			return false;
		}

		// calculate inliers ratio
		double dx = 0;
		double dy = 0;
		double sum = 0;
		double dist = 0.0;
		for (int j = 0; j < npoints; ++j) {
			dx = _points[j].x - ct.x;
			dy = _points[j].y - ct.y;
			dist = abs(sqrt(dx * dx + dy * dy) - radius);
			sum += (dist * dist);
			if (dist < 1.5) {
				inliers.push_back(_points[j]);
			}
		}
		sigma = sqrt(sum / static_cast<double>(npoints));

		if (sigma < 0.7) {

			arc_radius = radius;
			arc_ct = ct;
			// 获取起始角度和终止角度,暂时通过拟合内点值作为起点和终点
			cv::Point2d start_point(inliers.front().x, inliers.front().y);
			cv::Point2d end_point(inliers.back().x, inliers.back().y);
			double st_angle = atan2((start_point.y - arc_ct.y), (start_point.x - arc_ct.x)); //[-pi, pi]
			double ed_angle = atan2((end_point.y - arc_ct.y), (end_point.x - arc_ct.x));
			arc_st_angle = st_angle;
			arc_ed_angle = ed_angle;

			//确定圆弧的方向
			int dir_last = 0;
			for (int i = 1; i < inliers.size() - 1; ++i) {
				int dir = judgeArcDir(start_point, end_point, arc_ct, inliers[i]);
				if (dir != 0) {
					if (dir_last == dir) {
						arc_dir = dir;
						break;
					}
					dir_last = dir;
				}
			}

			if (arc_dir < 0 && st_angle < 0 && ed_angle > 0) {
				st_angle += 2 * CV_PI;
			}
			if (arc_dir > 0 && st_angle > 0 && ed_angle < 0) {
				ed_angle += 2 * CV_PI;
			}
			if (arc_dir < 0 && (st_angle < ed_angle)) {
				arc_span = (2 * CV_PI - abs(st_angle - ed_angle)) * arc_dir * 180.0 / CV_PI;
			}
			else if (arc_dir > 0 && (ed_angle < st_angle)) {
				arc_span = (2 * CV_PI - abs(st_angle - ed_angle)) * arc_dir * 180.0 / CV_PI;
			}
			else if (ed_angle == st_angle) {
				arc_span = 360.0;
			}
			else {
				arc_span = abs(st_angle - ed_angle) * arc_dir * 180.0 / CV_PI;
			}

#if image_show
			drawArc(contour_show, arc_ct, arc_st_angle, arc_span, arc_radius, cv::Scalar(255, 0, 0), 5.0);
#endif
			return true;
		}
		else {
			return false;
		}

	}

	bool CPolygon::IRLSCircleFit(cv::InputArray _points, cv::OutputArray _circle, int distType, double param, double reps, double aeps) {

		cv::Mat points = _points.getMat();
		int npoints2 = points.checkVector(2, -1, false);
		if (npoints2 < 3)
			return false;
		if (points.depth() != CV_32F) {
			cv::Mat temp;
			points.convertTo(temp, CV_32F);
			points = temp;
		}

		return true;
	}

	bool CPolygon::IRLSCircleFit(const std::vector<cv::Point>& _points, int distType, double param, double reps, double aeps) {
		/*
			detailed description in the paper:
		N. Chernov and C. Lesort, "Least squares fitting of circles"
		  in J. Math. Imag. Vision, volume 23, (2005), pages 239-251.

		*/
		double sigma = 1000.0;
		if (_points.size() < 3)
			return false;
		CircleFitByTaubin(_points, sigma, true, 5);
		if (sigma > 2) {
			return false;
		}




		return true;
	}

	float CPolygon::verifyLine(const cv::Point2d& l_pt1, const cv::Point2d& l_pt2, const std::vector<cv::Point>& pts) {
		float n_inliers = 0;
		double l_pt_maxX = l_pt1.x > l_pt2.x ? l_pt1.x + 1.0 : l_pt2.x + 1.0;
		double l_pt_minX = l_pt1.x < l_pt2.x ? l_pt1.x - 1.0 : l_pt2.x - 1.0;
		double l_pt_maxY = l_pt1.y > l_pt2.y ? l_pt1.y + 1.0 : l_pt2.y + 1.0;
		double l_pt_minY = l_pt1.y < l_pt2.y ? l_pt1.y - 1.0 : l_pt2.y - 1.0;
		for (int i = 0; i < pts.size(); ++i) {
			if (pts[i].x >= l_pt_minX && pts[i].x <= l_pt_maxX && pts[i].y >= l_pt_minY && pts[i].y <= l_pt_maxY) {
				double dis = abs(A * pts[i].x + B * pts[i].y + C) / sqrt(A * A + B * B);
				if (dis <= 1.5) {
					n_inliers++;
					inliers.push_back(pts[i]);
				}
			}
			else {
				int k = 0;
				k++;
			}
		}
		double ratio = n_inliers / static_cast<float>(pts.size());
		return ratio;
	}


	bool CPolygon::getCircle(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, cv::Point2d& center, float& radius) {
		// Ax^2 + Ay^2 + Bx + Cy + D = 0
		float x1 = p1.x;
		float x2 = p2.x;
		float x3 = p3.x;

		float y1 = p1.y;
		float y2 = p2.y;
		float y3 = p3.y;

		//
		center.x = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2);
		center.x /= (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

		center.y = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1);
		center.y /= (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

		radius = sqrtf((center.x - x1) * (center.x - x1) + (center.y - y1) * (center.y - y1));

		return true;
	}

	float CPolygon::verifyCircle(const std::vector<cv::Point> contours, const cv::Point2d center, const float radius, std::vector<cv::Point>& inliers, bool fix_se) {
		if (contours.size() == 0)
			return 0;
		int inlier = 0;
		float dist = 0.0f;
		float minInlierDist = 1.0f;
		float maxInlierDistMax = 5.0f;
		float maxInlierDist = 1.5f;
		//if (maxInlierDist < minInlierDist) maxInlierDist = minInlierDist;
		//if (maxInlierDist > maxInlierDistMax) maxInlierDist = maxInlierDist = maxInlierDistMax;
		if (!fix_se) {
			// 按照圆计算距离
			for (int i = 0; i < contours.size(); ++i) {
				float r_dist = sqrtf((contours[i].x - center.x) * (contours[i].x - center.x) + (contours[i].y - center.y) * (contours[i].y - center.y));
				dist = fabs(r_dist - radius);
				if (dist < maxInlierDist) {
					inlier++;
					inliers.push_back(contours[i]);
				}
			}
		}
		else {
			// 固定起点和终点的圆弧计算距离

		}

		return (float)inlier / (float)(contours.size());
	}

	void getGaussianDev(int n, double sigma, std::vector<double>& kernel) {
		const int SMALL_GAUSSIAN_SIZE = 7; //预置模板

		double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
		double scale2X = -0.5 / (sigmaX * sigmaX);
		double sum = 0;

		int i;

		for (i = 0; i < n; ++i) {
			double x = i - (n - 1) * 0.5;
			double t = -x * std::exp(scale2X * x * x); // exp(-x^2 / (2 * sigma^2))
			kernel[i] = t;
			sum += kernel[i];
		}

		sum = 1. / sum;
		for (i = 0; i < n; ++i) {
			kernel[i] *= sum;
		}

	}

	void getGaussianDevDev(int n, double sigma, std::vector<double>& kernel) {

		const int SMALL_GAUSSIAN_SIZE = 7; //预置模板

		double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
		double scale2X = -0.5 / (sigmaX * sigmaX);
		double sum = 0;

		int i;

		for (i = 0; i < n; ++i) {
			double x = i - (n - 1) * 0.5;
			double t = x * x * std::exp(scale2X * x * x); // exp(-x^2 / (2 * sigma^2))
			kernel[i] = t;
			sum += kernel[i];
		}

		sum = 1. / sum;
		for (i = 0; i < n; ++i) {
			kernel[i] *= sum;
		}
	}

	cv::Mat getGaussianDev(int n, double sigma, int ktype = CV_64F) {

		const int SMALL_GAUSSIAN_SIZE = 7; //预置模板

		double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
		double scale2X = -0.5 / (sigmaX * sigmaX);
		double sum = 0;

		int i;
		cv::Mat kernel(n, 1, ktype);
		double* cd = kernel.ptr<double>();

		for (i = 0; i < n; ++i) {
			double x = i - (n - 1) * 0.5;
			double t = -x * std::exp(scale2X * x * x) / (sigmaX * sigmaX * sigmaX * 2.5066);
			cd[i] = t;
		}

		return kernel;
	}

	cv::Mat getGaussianDevDev(int n, double sigma, int ktype = CV_64F) {

		const int SMALL_GAUSSIAN_SIZE = 7; //预置模板

		double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
		double scale2X = -0.5 / (sigmaX * sigmaX);
		double sum = 0;

		int i;
		cv::Mat kernel(n, 1, ktype);
		double* cd = kernel.ptr<double>();

		for (i = 0; i < n; ++i) {
			double x = i - (n - 1) * 0.5;
			double t = x * x * std::exp(scale2X * x * x) / (std::pow(sigmaX, 5) * 2.5066);
			cd[i] = t;
		}

		return kernel;
	}

	//@description: blipFiltering: filter one pixel displacement
	void blipFilter(const std::vector<cv::Point>& _contours_in, std::vector<cv::Point>& _contours_out) {


		//int t1 = 0;
		//int t2 = 0;
		//int t3 = 0;
		//int t4 = 0;
		std::vector<double> x_projection(_contours_in.size());
		std::vector<double> y_projection(_contours_in.size());
		for (int i = 0; i < _contours_in.size(); ++i) {
			x_projection[i] = _contours_in[i].x;
			y_projection[i] = _contours_in[i].y;
		}
		std::vector<Blip> vblips_x;
		Blip blip_first_x(0, 0, x_projection[0]);
		vblips_x.push_back(blip_first_x);
		for (int j = 0; j < _contours_in.size(); ++j) {
			if (x_projection[j] == vblips_x.back().val) {
				vblips_x.back().len++;
			}
			else {
				Blip blip(j, 1, x_projection[j]);
				vblips_x.push_back(blip);
			}
		}

		std::vector<Blip> vblips_y;
		Blip blip_first_y(0, 0, y_projection[0]);
		vblips_y.push_back(blip_first_y);
		for (int j = 0; j < _contours_in.size(); ++j) {
			if (y_projection[j] == vblips_y.back().val) {
				vblips_y.back().len++;
			}
			else {
				Blip blip(j, 1, y_projection[j]);
				vblips_y.push_back(blip);
			}
		}

		for (int i = 0; i < vblips_x.size() - 2; ++i) {
			if (abs(vblips_x[i].val - vblips_x[i + 1].val) == 1 && vblips_x[i].val == vblips_x[i + 2].val) {
				if ((vblips_x[i].len + vblips_x[i + 2].len) > vblips_x[i + 1].len) {
					vblips_x[i + 1].val = vblips_x[i].val;
				}
			}
		}

		for (int i = 0; i < vblips_y.size() - 2; ++i) {
			if (abs(vblips_y[i].val - vblips_y[i + 1].val) == 1 && vblips_y[i].val == vblips_y[i + 2].val) {
				if ((vblips_y[i].len + vblips_y[i + 2].len) > vblips_y[i + 1].len) {
					vblips_y[i + 1].val = vblips_y[i].val;
				}
			}
		}

		// 恢复原有轮廓
		_contours_out.resize(_contours_in.size());
		std::vector<double> x_projection_filter(_contours_in.size());
		std::vector<double> y_projection_filter(_contours_in.size());
		for (int i = 0; i < vblips_x.size(); ++i) {
			std::vector<int> val_temp(vblips_x[i].len, vblips_x[i].val);
			std::copy(val_temp.begin(), val_temp.end(), x_projection_filter.begin() + vblips_x[i].start);
		}

		for (int i = 0; i < vblips_y.size(); ++i) {
			std::vector<int> val_temp(vblips_y[i].len, vblips_y[i].val);
			std::copy(val_temp.begin(), val_temp.end(), y_projection_filter.begin() + vblips_y[i].start);
		}

		for (int k = 0; k < _contours_out.size(); ++k) {
			_contours_out[k].x = x_projection_filter[k];
			_contours_out[k].y = y_projection_filter[k];
		}


	}

	//@description: 轮廓曲率计算:轮廓长度的离散函数
	bool _calCurvature(const cv::Mat& src_img, const std::vector<cv::Point>& _contours_in, std::vector<double> _curvature_out) {
		//1. 计算轮廓x,y坐标的投影	
		//std::vector<double> x_projection(_contours_in.size());
		//std::vector<double> y_projection(_contours_in.size());
		std::vector<cv::Point> contours_filter;
		blipFilter(_contours_in, contours_filter);
		cv::Mat x_projection = cv::Mat::zeros(cv::Size(contours_filter.size(), 1), CV_64F);
		cv::Mat y_projection = cv::Mat::zeros(cv::Size(contours_filter.size(), 1), CV_64F);
		for (int i = 0; i < contours_filter.size(); ++i) {
			x_projection.ptr<double>(0)[i] = contours_filter[i].x;
			y_projection.ptr<double>(0)[i] = contours_filter[i].y;
		}
#if image_show		
		cv::Mat projection_x_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_x_show, cv::Point(ii, contours_filter[ii].x), 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::Mat projection_y_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_y_show, cv::Point(ii, contours_filter[ii].y), 1, cv::Scalar(0, 255, 0), 1);
		}
#endif
		std::vector<double> temp_curvature(_contours_in.size());
		//std::vector<double> contours_filter(_contours_in.size());
		cv::Mat x_projection_dev, x_projection_dev2;
		cv::Mat y_projection_dev, y_projection_dev2;
		double sigma = 8.0;
		cv::Mat gaussianKernelDev = getGaussianDev(_contours_in.size(), sigma).t();
		cv::Mat gaussianKernelDevDev = getGaussianDevDev(_contours_in.size(), sigma).t();
		cv::filter2D(x_projection, x_projection_dev, CV_64F, gaussianKernelDev, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		cv::filter2D(y_projection, y_projection_dev, CV_64F, gaussianKernelDev, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		cv::filter2D(x_projection, x_projection_dev2, CV_64F, gaussianKernelDevDev, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		cv::filter2D(y_projection, y_projection_dev2, CV_64F, gaussianKernelDevDev, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

		//blipFiltering

		for (int j = 0; j < _contours_in.size(); ++j) {
			double numerator = x_projection_dev.ptr<double>(0)[j] * y_projection_dev2.ptr<double>(0)[j] - x_projection_dev2.ptr<double>(0)[j] * y_projection_dev.ptr<double>(0)[j];
			double denominator = x_projection_dev.ptr<double>(0)[j] * x_projection_dev.ptr<double>(0)[j] + (y_projection_dev.ptr<double>(0)[j] * y_projection_dev.ptr<double>(0)[j]);
			if (denominator == 0) {
				temp_curvature[j] = 0;
			}
			else {
				temp_curvature[j] = numerator / sqrt((std::pow(denominator, 3)));
			}
		}
#if image_show
		cv::Mat curvature_show = cv::Mat::zeros(cv::Size(temp_curvature.size(), 1000), CV_8UC3);
		cv::Mat curvature_show2 = cv::Mat::zeros(cv::Size(temp_curvature.size(), 1000), CV_8UC3);
		for (int i = 0; i < temp_curvature.size() - 1; ++i) {
			//common::drawCross(curvature_show, cv::Point(i, _curvature_out[i] * 200), 1, cv::Scalar(0, 0, 255), 1);
			cv::line(curvature_show, cv::Point(i, 1000 - abs(temp_curvature[i] * 10)), cv::Point(i + 1, 1000 - abs(temp_curvature[i + 1] * 10)), cv::Scalar(0, 255, 0));
			cv::line(curvature_show2, cv::Point(i, 500 - temp_curvature[i] * 10), cv::Point(i + 1, 500 - temp_curvature[i + 1] * 10), cv::Scalar(0, 255, 0));
		}
#endif
		return success;
	}

	double point2line(const cv::Point& line_pt1, const cv::Point& line_pt2, const cv::Point& pt) {
		//d = (A*x0 + B*y0 + C) / sqrt(A^2 + B^2)
		double A_, B_, C_, dis;

		A_ = line_pt2.y - line_pt1.y;
		B_ = line_pt1.x - line_pt2.x;
		C_ = line_pt2.x * line_pt1.y - line_pt1.x * line_pt2.y;
		if (A_ == 0 && B_ == 0) {
			return 0;
		}
		dis = (A_ * pt.x + B_ * pt.y + C_) / sqrt(A_ * A_ + B_ * B_);
		return dis;
	}

	double point2point(const cv::Point& pt1, const cv::Point& pt2) {
		double dis = sqrtl((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) + (pt1.y - pt2.y));
		return dis;
	}

	double calCos(const cv::Point& pt1, const cv::Point& pt2, const cv::Point ct_pt) {
		if (pt1.x == pt2.x && pt1.y == pt2.y) {
			return 0;
		}
		double cos_val = 0;
		cv::Point a = cv::Point(ct_pt.x - pt1.x, ct_pt.y - pt1.y);
		cv::Point b = cv::Point(ct_pt.x - pt2.x, ct_pt.y - pt2.y);

		cos_val = (a.x * b.x + a.y * b.y) / (sqrtf(a.x * a.x + a.y * a.y) * sqrtf(b.x * b.x + b.y * b.y));
		return cos_val;
	}

	//循环索引
	int	Index(int index, int k, int count) {
		int mod_i = (index + k) % count;
		mod_i = mod_i < 0 ? mod_i + count : mod_i;
		return mod_i;
	}


	//@description: 轮廓曲率计算:轮廓长度的离散函数
	// @brief: calculate discrete curvature of contour
	// @param [in]: src_img
	// @param [in]: _contours_in：input conotour
	// @param [in/out]: _curvature_out: output curvature
	// @param [in]: closed : whether the contour is closed


	bool calCurvature(const cv::Mat& src_img, const std::vector<cv::Point>& _contours_in, std::vector<double>& _curvature_out, std::vector<cv::Point>& contours_filter, bool colsed = true) {
		int count = _contours_in.size();
		_curvature_out.resize(_contours_in.size());
		std::vector<double> curvature_temp;
		std::vector<int> region_supports;
		contours_filter.resize(_contours_in.size());
#if image_show		
		cv::Mat projection_x_show = cv::Mat::zeros(cv::Size(_contours_in.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < _contours_in.size(); ++ii) {
			common::drawCross(projection_x_show, cv::Point(ii, _contours_in[ii].x), 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::Mat projection_y_show = cv::Mat::zeros(cv::Size(_contours_in.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < _contours_in.size(); ++ii) {
			common::drawCross(projection_y_show, cv::Point(ii, _contours_in[ii].y), 1, cv::Scalar(0, 255, 0), 1);
		}
#endif

		blipFilter(_contours_in, contours_filter);

#if image_show		
		cv::Mat projection_x_filter_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_x_filter_show, cv::Point(ii, contours_filter[ii].x), 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::Mat projection_y_filter_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_y_filter_show, cv::Point(ii, contours_filter[ii].y), 1, cv::Scalar(0, 255, 0), 1);
		}
		cv::Mat contoursfilter_img_show = cv::Mat::zeros(src_img.size(), CV_8UC3);
		cv::Mat contours_img_show = cv::Mat::zeros(src_img.size(), CV_8UC3);
		common::drawCross(contoursfilter_img_show, contours_filter, 1, cv::Scalar(0, 0, 255), 1);
		common::drawCross(contours_img_show, _contours_in, 1, cv::Scalar(0, 0, 255), 1);

#endif
		for (int i = 0; i < _contours_in.size(); ++i) {
			int k = 9;
			//while (1) {
			//	//double d_ik = point2point(_contours_in[(i - k) % count], _contours_in[(i + k) % count]);
			//	double d_ik = point2point(contours_filter[Index(i,-k, count)], contours_filter[Index(i, k, count)]);
			//	double d_ik1 = point2point(contours_filter[Index(i, -k-1, count)], contours_filter[Index(i, k+1, count)]);
			//	double l_ik = point2line(contours_filter[Index(i, -k, count)], contours_filter[Index(i, k, count)], contours_filter[i]);
			//	double l_ik1 = point2line(contours_filter[Index(i, -k-1, count)], contours_filter[Index(i, k+1, count)], contours_filter[i]);
			//
			//	if (l_ik >= l_ik1) {
			//		break;
			//	}
			//	double dl_ratio_k = d_ik / l_ik;
			//	double dl_ratio_k1 = d_ik1 / l_ik1;
			//	if (d_ik > 0 && dl_ratio_k >= dl_ratio_k1) {
			//		break;
			//	}
			//	else if (d_ik < 0 && dl_ratio_k <= dl_ratio_k1) {
			//		break;
			//	}
			//	k++;
			//	if (k > contours_filter.size() * 0.1) { break; }
			//}
			curvature_temp.resize(k);
			for (int j = 0; j < k; ++j) {
				curvature_temp[j] = calCos(contours_filter[Index(i, k, count)], contours_filter[Index(i, -k, count)], contours_filter[i]) + 1; //[0,2]
			}
			if (k % 2 == 0) {
				_curvature_out[i] = std::accumulate(curvature_temp.begin() + k / 2, curvature_temp.end(), 0.0) * 2.0 / (k + 2);
			}
			else {
				_curvature_out[i] = std::accumulate(curvature_temp.begin() + (k - 1) / 2, curvature_temp.end(), 0.0) * 2.0 / (k + 3);
			}
			region_supports.push_back(k);
		}

		// K cosine measure
#if image_show
		cv::Mat curvature_show = cv::Mat::zeros(cv::Size(_contours_in.size(), 1000), CV_8UC3);
		for (int i = 0; i < _contours_in.size() - 1; ++i) {
			//common::drawCross(curvature_show, cv::Point(i, _curvature_out[i] * 200), 1, cv::Scalar(0, 0, 255), 1);
			cv::line(curvature_show, cv::Point(i, 1000 - _curvature_out[i] * 400), cv::Point(i + 1, 1000 - _curvature_out[i + 1] * 400), cv::Scalar(0, 255, 0));
		}
#endif

		return true;
	}

	// 11点法求解曲率
	bool calCurvature11p() {
		return true;
	}

	// 3点法求解曲率
	double calPjer(const cv::Point& ct, const cv::Point& p1, const cv::Point& p2) {
		if ((ct.x == p1.x && ct.y == p1.y) || (ct.x == p2.x && ct.y == p2.y)) {
			return 0;
		}
		double t_a = sqrt(static_cast<double>((ct.x - p1.x) * (ct.x - p1.x) + (ct.y - p1.y) * (ct.y - p1.y)));
		double t_b = sqrt(static_cast<double>((ct.x - p2.x) * (ct.x - p2.x) + (ct.y - p2.y) * (ct.y - p2.y)));

		Eigen::Matrix3d M;
		M << 1, -t_a, t_a* t_a,
			1, 0, 0,
			1, t_b, t_b* t_b;
		Eigen::Matrix3d M_inv = M.inverse();
		Eigen::Vector3d X(p1.x, ct.x, p2.x);
		Eigen::Vector3d Y(p1.y, ct.y, p2.y);

		Eigen::Vector3d a = M_inv * X;
		Eigen::Vector3d b = M_inv * Y;

		double curvature = 2 * (a[2] * b[1] - b[2] * a[1]) / sqrt(pow((a[1] * a[1] + b[1] * b[1]), 3));
		Eigen::Vector2d dir(b[1], -a[1]);
		Eigen::Vector2d norm_dir = dir / sqrt(a[1] * a[1] + b[1] * b[1]); //曲率方向的单位向量
		return curvature;
	}

	bool calCurvature3p(const cv::Mat& src_img, const std::vector<cv::Point>& _contours_in, std::vector<double>& _curvature_out, bool colsed = true) {
		int count = _contours_in.size();
		_curvature_out.resize(_contours_in.size());
		std::vector<double> curvature_temp;
		std::vector<int> region_supports;
		std::vector<cv::Point> contours_filter(_contours_in.size());
#if image_show		
		cv::Mat projection_x_show = cv::Mat::zeros(cv::Size(_contours_in.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < _contours_in.size(); ++ii) {
			common::drawCross(projection_x_show, cv::Point(ii, _contours_in[ii].x), 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::Mat projection_y_show = cv::Mat::zeros(cv::Size(_contours_in.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < _contours_in.size(); ++ii) {
			common::drawCross(projection_y_show, cv::Point(ii, _contours_in[ii].y), 1, cv::Scalar(0, 255, 0), 1);
		}
#endif

		blipFilter(_contours_in, contours_filter);

#if image_show		
		cv::Mat projection_x_filter_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_x_filter_show, cv::Point(ii, contours_filter[ii].x), 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::Mat projection_y_filter_show = cv::Mat::zeros(cv::Size(contours_filter.size(), src_img.cols), CV_8UC3);
		for (int ii = 0; ii < contours_filter.size(); ++ii) {
			common::drawCross(projection_y_filter_show, cv::Point(ii, contours_filter[ii].y), 1, cv::Scalar(0, 255, 0), 1);
		}
		cv::Mat contoursfilter_img_show = cv::Mat::zeros(src_img.size(), CV_8UC3);
		cv::Mat contours_img_show = cv::Mat::zeros(src_img.size(), CV_8UC3);
		common::drawCross(contoursfilter_img_show, contours_filter, 1, cv::Scalar(0, 0, 255), 1);
		common::drawCross(contours_img_show, _contours_in, 1, cv::Scalar(0, 0, 255), 1);

#endif

		int interval = 20;
		curvature_temp.resize(interval - 1);
		for (int i = 0; i < contours_filter.size(); ++i) {
			for (int k = 1; k < interval; ++k) {
				curvature_temp[k - 1] = calPjer(contours_filter[i], contours_filter[Index(i, k, count)], contours_filter[Index(i, -k, count)]);
			}
			_curvature_out[i] = std::accumulate(curvature_temp.begin(), curvature_temp.end(), 0.0) / static_cast<double>(curvature_temp.size());
		}

#if image_show
		cv::Mat curvature_show = cv::Mat::zeros(cv::Size(_contours_in.size(), 1000), CV_8UC3);
		for (int i = 0; i < _contours_in.size() - 1; ++i) {
			//common::drawCross(curvature_show, cv::Point(i, _curvature_out[i] * 200), 1, cv::Scalar(0, 0, 255), 1);
			cv::line(curvature_show, cv::Point(i, 500 - _curvature_out[i] * 400), cv::Point(i + 1, 500 - _curvature_out[i + 1] * 400), cv::Scalar(0, 255, 0));
		}
#endif

		return true;
	}

	//@description: 提取最长连续常曲率片段
	void findLongestFrag(const std::vector<uint32_t> _hist_in, const std::vector<cv::Point> _contours_in, const std::vector<double> _curvatures_in,
		std::vector<uint32_t>& hist_out, std::vector<double>& _curvatures_out, Fragment& longest_frag_out, double step, double curvature_tolerance = 0.04) {
		std::vector<uint32_t> hist;
		hist.assign(_hist_in.begin(), _hist_in.end());
		//hist_out.assign(_hist_in.begin(), _hist_in.end());
		//_curvatures_out.assign(_curvatures_in.begin(), _curvatures_in.end());
		Fragment longest_frag;
		std::vector<std::pair<uint32_t, double>> num_curvatures;
		for (int i = 0; i < hist.size(); ++i) {
			if (hist[i] != 0) {
				num_curvatures.push_back(std::make_pair<uint32_t, double>(hist[i], i * step));
			}
		}
		std::sort(num_curvatures.begin(), num_curvatures.end(), [&](std::pair<uint32_t, double> pair1, std::pair<uint32_t, double> pair2) {return pair1.first > pair2.first; });
		int index = 0;
		while (1) {
			//std::vector<uint32_t>::iterator peak = std::max_element(hist.begin(), hist.end());
			//double peak_curvature = static_cast<double>(std::distance(hist.begin(), peak)) * step;
			if (index > num_curvatures.size())
			{
				break;
			}
			double peak_curvature = num_curvatures[index].second;
			index++;
			//*peak = 0;
			// look for the points belong to convature_peak +- curvature_tolerance
			//std::vector<Fragment> contour_frag;
			//查找满足峰值曲率的所有连续轮廓片段,并标记其中最长的片段
			//按照vote_hist的得分顺序依次各个曲率的最长连续片段
			for (int k = 0; k < _curvatures_in.size(); ++k) {
				if ((_curvatures_in[k] > peak_curvature - curvature_tolerance && _curvatures_in[k] < peak_curvature + curvature_tolerance) && _curvatures_in[k] != -1.0) {
					Fragment temp_fragment(k, k, peak_curvature);
					for (int j = k + 1; j < _curvatures_in.size(); ++j) {
						if ((_curvatures_in[j] <= peak_curvature - curvature_tolerance || _curvatures_in[j] >= peak_curvature + curvature_tolerance)) {
							temp_fragment.end_index = j;
							temp_fragment.len = temp_fragment.end_index - temp_fragment.start_index;
							if (longest_frag.len < temp_fragment.len) {
								longest_frag = temp_fragment;
							}
							k = j;
							break;
						}
					}
				}
			}
			//int succeed_bins = std::accumulate(hist.begin(), hist.end(), 0);
			int succeed_bins = std::accumulate(num_curvatures.begin() + index, num_curvatures.end(), 0, [](int sum, std::pair<uint32_t, double> pair1) {return sum + pair1.first; });
			if (succeed_bins < longest_frag.len) {
				longest_frag_out = longest_frag;
				hist_out[static_cast<int>(longest_frag.curvature / step)] -= longest_frag.len;
				for (int j = longest_frag.start_index; j < longest_frag.end_index; ++j) {
					_curvatures_out[j] = -1.0;
					//_curvatures_out.erase(_curvatures_out.begin() + longest_frag.start_index, _curvatures_out.begin() + longest_frag.end_index);
				}
				break;
			}
		}

	}

	void findLongestFragFaster(const std::vector<uint32_t> _hist_in, const std::vector<cv::Point> _contours_in, const std::vector<double> _curvatures_in,
		std::vector<uint32_t>& hist_out, std::vector<double>& _curvatures_out, Fragment& longest_frag_out, double step, double curvature_tolerance = 0.04) {
		std::vector<uint32_t> hist;
		hist.assign(_hist_in.begin(), _hist_in.end());
		Fragment longest_frag;
		while (1) {
			std::vector<uint32_t>::iterator peak = std::max_element(hist.begin(), hist.end());
			double peak_curvature = static_cast<double>(std::distance(hist.begin(), peak)) * step;
			*peak = 0;
			// look for the points belong to convature_peak +- curvature_tolerance
			//查找满足峰值曲率的所有连续轮廓片段,并标记其中最长的片段
			//按照vote_hist的得分顺序依次各个曲率的最长连续片段
			for (int k = 0; k < _curvatures_in.size(); ++k) {
				if ((_curvatures_in[k] > peak_curvature - curvature_tolerance && _curvatures_in[k] < peak_curvature + curvature_tolerance) && _curvatures_in[k] != -1.0) {
					Fragment temp_fragment(k, k, peak_curvature);
					for (int j = k + 1; j < _curvatures_in.size(); ++j) {
						if ((_curvatures_in[j] <= peak_curvature - curvature_tolerance || _curvatures_in[j] >= peak_curvature + curvature_tolerance)) {
							temp_fragment.end_index = j;
							temp_fragment.len = temp_fragment.end_index - temp_fragment.start_index;
							if (longest_frag.len < temp_fragment.len) {
								longest_frag = temp_fragment;
							}
							k = j;
							break;
						}
					}
				}
			}
			int succeed_bins = std::accumulate(hist.begin(), hist.end(), 0);
			if (succeed_bins < longest_frag.len) {
				longest_frag_out = longest_frag;
				hist_out[static_cast<int>(longest_frag.curvature / step)] -= longest_frag.len;
				for (int j = longest_frag.start_index; j < longest_frag.end_index; ++j) {
					_curvatures_out[j] = -1.0;
				}
				break;
			}
		}

	}

	// @description: 将轮廓分割为常曲率段
	// @brief: group points into contiguous segments of constant curvature
	// @param [in]: _contours_in：input contour
	// @param [in]: _curvatures_in：curvature at each point of the contour
	// @param [in/out]: _contours_segments: segments of the constant curvature
	bool voteCurvature(const std::vector<cv::Point>& _contours_in, const std::vector<double> _curvatures_in,
		std::vector<std::vector<cv::Point>>& _contours_segments, std::vector<double>& _curvatures_segments, const double curvature_tolerance = 0.04, int l_min = 8) {
		int n_bins = 80; // the step of bins: 0.01  /in [0, 2]
		double step = 0.025;
		std::vector<uint32_t> hist_(n_bins);
		for (int i = 0; i < _curvatures_in.size(); ++i) {
			int h_index = static_cast<int>(_curvatures_in[i] / step) + static_cast<int>(curvature_tolerance / step);
			h_index = h_index >= n_bins ? n_bins - 1 : h_index;
			int l_index = static_cast<int>(_curvatures_in[i] / step) - static_cast<int>(curvature_tolerance / step);
			l_index = l_index < 0 ? 0 : l_index;
			for (int j = l_index; j <= h_index; ++j) {
				++hist_[j];
			}
		}

#if image_show
		int zoom = 1.0; //缩放系数
		int hist_size = _curvatures_in.size();
		cv::Mat hist_img_show(hist_size, n_bins, CV_8U, cv::Scalar(255));
		for (int j = 0; j < hist_.size(); ++j) {
			cv::line(hist_img_show, cv::Point(j, hist_size), cv::Point(j, hist_size - hist_[j]), cv::Scalar(0));
		}
#endif
		//// get peaks culvature
		// std::vector<uint32_t>::iterator peak = std::max_element(hist_.begin(), hist_.end());
		// double peak_curvature = static_cast<double>(std::distance(hist_.begin(), peak)) * step;
		//// look for the points belong to convature_peak +- curvature_tolerance
		// std::vector<Fragment> contour_frag;
		// //查找满足峰值曲率的所有连续轮廓片段,并标记其中最长的片段
		// //按照vote_hist的得分顺序依次各个曲率的最长连续片段
		// for (int k = 0; k < _curvatures_in.size(); ++k) {
		//	 if (_curvatures_in[k] > peak_curvature - curvature_tolerance && _curvatures_in[k] < peak_curvature + curvature_tolerance) {
		//		 Fragment temp_fragment(k, k, _curvatures_in[k]);
		//		 contour_frag.push_back(temp_fragment);
		//		 for (int j = k + 1; j < _curvatures_in.size(); ++j) {
		//			 if (_curvatures_in[j] <= peak_curvature - curvature_tolerance || _curvatures_in[j] >= peak_curvature + curvature_tolerance) {
		//				contour_frag.back().end_index = j;
		//				contour_frag.back().len = contour_frag.back().end_index - contour_frag.back().start_index;
		//				k = j;
		//				break;
		//			 }
		//		 }
		//	}
		//}
		// std::sort(contour_frag.begin(), contour_frag.end());
		std::vector<Fragment> longest_frags;
		Fragment longest_frag;
		std::vector<uint32_t>hist_out;
		hist_out.assign(hist_.begin(), hist_.end());
		std::vector<double> _curvatures_out;
		_curvatures_out.assign(_curvatures_in.begin(), _curvatures_in.end());
		while (1) {
			findLongestFrag(hist_out, _contours_in, _curvatures_out, hist_out, _curvatures_out, longest_frag, step, curvature_tolerance);
			if (longest_frag.len <= l_min) {
				break;
			}
			longest_frags.push_back(longest_frag);
		}

		if (longest_frags.size() == 0) {
			return false;
		}

		for (int k = 0; k < longest_frags.size(); ++k) {
			std::vector<cv::Point> contour_temp;
			std::vector<double> curvatures_temp;
			double curvature_avg = -1.0;
			contour_temp.assign(_contours_in.begin() + longest_frags[k].start_index, _contours_in.begin() + longest_frags[k].end_index);
			_contours_segments.push_back(contour_temp);
			curvatures_temp.assign(_curvatures_in.begin() + longest_frags[k].start_index, _curvatures_in.begin() + longest_frags[k].end_index);
			curvature_avg = std::accumulate(curvatures_temp.begin(), curvatures_temp.end(), 0.0) / curvatures_temp.size();
			_curvatures_segments.push_back(curvature_avg);
			// 计算该轮廓片段的平均曲率
		}

		return true;
	}

	template<typename T> static int
		approxPolyDP_(const cv::Point_<T>* src_contour, int count0, cv::Point_<T>* dst_contour, int* approx_index,
			bool is_closed0, double eps, cv::AutoBuffer<cv::Range>& _stack)
	{
#define PUSH_SLICE(slice) \
		if( top >= stacksz ) \
		{ \
			_stack.resize(stacksz*3/2); \
			stack = (cv::Range*)_stack; \
			stacksz = _stack.size(); \
		} \
		stack[top++] = slice

#define READ_PT(pt, pos) \
		pt = src_contour[pos]; \
		if( ++pos >= count ) pos = 0

#define READ_DST_PT(pt, pos) \
		pt = dst_contour[pos]; \
		if( ++pos >= count ) pos = 0

#define WRITE_PT(pt) \
		dst_contour[new_count++] = pt

#define WRITE_INDEX(ind) \
		approx_index[index_count++] = ind

#define READ_DST_INDEX(ind, pos) \
		ind = approx_index[pos]


		typedef cv::Point_<T> PT;
		int             init_iters = 3;
		cv::Range           slice(0, 0), right_slice(0, 0);
		PT              start_pt((T)-1000000, (T)-1000000), end_pt(0, 0), pt(0, 0);
		int				ind = 0;
		int             i = 0, j, pos = 0, wpos, ipos, count = count0, new_count = 0, index_count = 0;
		int             is_closed = is_closed0;
		bool            le_eps = false;
		size_t top = 0, stacksz = _stack.size();
		cv::Range* stack = (cv::Range*)_stack;

		if (count == 0)
			return 0;

		eps *= eps;

		if (!is_closed)
		{
			right_slice.start = count;
			end_pt = src_contour[0];
			start_pt = src_contour[count - 1];

			if (start_pt.x != end_pt.x || start_pt.y != end_pt.y)
			{
				slice.start = 0;
				slice.end = count - 1;
				PUSH_SLICE(slice);
			}
			else
			{
				is_closed = 1;
				init_iters = 1;
			}
		}

		if (is_closed)
		{
			// 1. Find approximately two farthest points of the contour
			right_slice.start = 0;

			for (i = 0; i < init_iters; i++)
			{
				double dist, max_dist = 0;
				pos = (pos + right_slice.start) % count;
				ind = pos;
				READ_PT(start_pt, pos);

				for (j = 1; j < count; j++)
				{
					double dx, dy;

					READ_PT(pt, pos);
					dx = pt.x - start_pt.x;
					dy = pt.y - start_pt.y;

					dist = dx * dx + dy * dy;

					if (dist > max_dist)
					{
						max_dist = dist;
						right_slice.start = j;
					}
				}

				le_eps = max_dist <= eps;
			}

			// 2. initialize the stack
			if (!le_eps)
			{
				right_slice.end = slice.start = pos % count;
				slice.end = right_slice.start = (right_slice.start + slice.start) % count;

				PUSH_SLICE(right_slice);
				PUSH_SLICE(slice);
			}
			else {
				WRITE_PT(start_pt);
				WRITE_INDEX(ind);
			}
		}

		// 3. run recursive process
		while (top > 0)
		{
			slice = stack[--top];
			end_pt = src_contour[slice.end];
			pos = slice.start;
			ind = pos;
			READ_PT(start_pt, pos);

			if (pos != slice.end)
			{
				double dx, dy, dist, max_dist = 0;

				dx = end_pt.x - start_pt.x;
				dy = end_pt.y - start_pt.y;

				assert(dx != 0 || dy != 0);

				while (pos != slice.end)
				{
					READ_PT(pt, pos);
					dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

					if (dist > max_dist)
					{
						max_dist = dist;
						right_slice.start = (pos + count - 1) % count;
					}
				}

				le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
			}
			else
			{
				le_eps = true;
				// read starting point
				start_pt = src_contour[slice.start];
				ind = slice.start;
			}

			if (le_eps)
			{
				WRITE_PT(start_pt);
				WRITE_INDEX(ind);
			}
			else
			{
				right_slice.end = slice.end;
				slice.end = right_slice.start;
				PUSH_SLICE(right_slice);
				PUSH_SLICE(slice);
			}
		}

		if (!is_closed) {
			WRITE_PT(src_contour[count - 1]);
			ind = count - 1;
			WRITE_INDEX(ind);
		}

		// last stage: do final clean-up of the approximated contour -
		// remove extra points on the [almost] straight lines.
		is_closed = is_closed0;
		count = new_count;
		pos = is_closed ? count - 1 : 0;
		READ_DST_PT(start_pt, pos);
		wpos = pos;
		ipos = pos;
		READ_DST_PT(pt, pos);

		for (i = !is_closed; i < count - !is_closed && new_count > 2; i++)
		{
			double dx, dy, dist, successive_inner_product;
			ipos = pos;
			READ_DST_PT(end_pt, pos);

			dx = end_pt.x - start_pt.x;
			dy = end_pt.y - start_pt.y;
			dist = fabs((pt.x - start_pt.x) * dy - (pt.y - start_pt.y) * dx);
			successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
				(pt.y - start_pt.y) * (end_pt.y - pt.y);

			if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0 && dy != 0 &&
				successive_inner_product >= 0)
			{
				new_count--;
				index_count--;
				dst_contour[wpos] = start_pt = end_pt;
				READ_DST_INDEX(ind, ipos);
				approx_index[wpos] = ind;
				if (++wpos >= count) wpos = 0;
				READ_DST_PT(pt, pos);
				i++;
				continue;
			}
			dst_contour[wpos] = start_pt = pt;
			if (ipos - 1 < 0) ipos += count;
			READ_DST_INDEX(ind, ipos - 1);
			approx_index[wpos] = ind;
			if (++wpos >= count) wpos = 0;
			pt = end_pt;
		}

		if (!is_closed) {
			dst_contour[wpos] = pt;
			//if (ipos < 0) ipos += count;
			READ_DST_INDEX(ind, ipos);
			approx_index[wpos] = ind;
		}

		return new_count;
	}


	bool approxPolygon(cv::InputArray _curve, cv::OutputArray _approxCurve, cv::OutputArray _approxIndex, double epsilon, bool closed) {
		if (epsilon < 0.0 || !(epsilon < 1e30)) {
			return false;
		}
		cv::Mat curve = _curve.getMat();

		int npoints = curve.checkVector(2), depth = _curve.depth();
		if (!(npoints >= 0 && (depth == CV_32S || depth == CV_32F))) {
			return false;
		}
		//if (npoints == 0) {
		//	_approxCurve.release();
		//}
		cv::AutoBuffer<cv::Point> _buf(npoints);
		cv::AutoBuffer<cv::Range> _stack(npoints);
		cv::AutoBuffer<int> _index(npoints);

		cv::Point* buf = (cv::Point*)_buf;
		int* approx_index = (int*)_index;
		int nout = 0;

		if (depth == CV_32S)
			nout = approxPolyDP_(curve.ptr<cv::Point>(), npoints, buf, approx_index, closed, epsilon, _stack);
		else if (depth == CV_32F)
			nout = approxPolyDP_(curve.ptr<cv::Point2f>(), npoints, (cv::Point2f*)buf, approx_index, closed, epsilon, _stack);
		else
			return false;

		cv::Mat(nout, 1, CV_MAKETYPE(depth, 2), buf).copyTo(_approxCurve);
		cv::Mat(nout, 1, CV_MAKETYPE(CV_32S, 1), approx_index).copyTo(_approxIndex);

		return true;
	}


	bool recoverContours(cv::Mat& src_img, const std::vector<std::vector<polygon::CPolygon>>& _contours_polygons) {
		for (int i = 0; i < _contours_polygons.size(); ++i) {
			//一个完整轮廓
			for (int j = 0; j < _contours_polygons[i].size(); ++j) {
				if (_contours_polygons[i][j].type == polygon::LINE) {
					cv::line(src_img, _contours_polygons[i][j].st, _contours_polygons[i][j].ed, cv::Scalar(255));
				}
				else {
					_contours_polygons[i][j].drawArc(src_img, cv::Scalar(255));
				}
			}
		}
		return true;
	}


	bool Result_Whole(const std::vector<std::vector<polygon::CPolygon>>& contours_polygons, VISION_PART_POLYGON& polygon, VISION_PART_POLYGON& result,
		WireFrame& wireframe, double camera_scale_x, double camera_scale_y, double offset_x = 0.0, double offset_y = 0.0) {
		if (contours_polygons.size() == 0) {
			return false;
		}
		int poly_num = 0;
		std::vector<cv::Point> poly_inliers;
		for (int i = 0; i < contours_polygons.size(); ++i) {
			poly_num += contours_polygons[i].size();
			for (int j = 0; j < contours_polygons[i].size(); ++j) {
				//if (contours_polygons[i][j].type == LINE) {
				//	poly_vertexes.push_back(contours_polygons[i][j].st);
				//	poly_vertexes.push_back(contours_polygons[i][j].ed);
				//}
				//具有较高运算复杂度的方案
				poly_inliers.insert(poly_inliers.end(), contours_polygons[i][j].inliers.begin(), contours_polygons[i][j].inliers.end());
			}
		}
		cv::Rect body_rect = cv::boundingRect(poly_inliers);
		//wireframe.body_x = body_rect.width * camera_scale_x * 0.001;
		wireframe.body_x = body_rect.width;
		//wireframe.body_y = body_rect.height * camera_scale_y * 0.001;
		wireframe.body_y = body_rect.height;
		//wireframe.origin_x = (body_rect.x + body_rect.width / 2.0 - static_cast<float>(image_width) / 2.0) * camera_scale_x * 0.001;
		//wireframe.origin_x = (body_rect.x + body_rect.width / 2.0 + offset_x) * camera_scale_x * 0.001;
		wireframe.origin_x = (body_rect.x + body_rect.width / 2.0);
		//wireframe.origin_y = (body_rect.y + body_rect.height / 2.0 - static_cast<float>(image_height) / 2.0) * camera_scale_y * 0.001;  //变换到图像中心
		//wireframe.origin_y = (body_rect.y + body_rect.height / 2.0 + offset_y) * camera_scale_y * 0.001;  //变换到图像中心
		wireframe.origin_y = (body_rect.y + body_rect.height / 2.0);
		wireframe.angle = 0.0;
		wireframe.vertex_num = poly_num * 2;

		PART_POLYGON_PARAMETER_INTERPRETER::set_angle_margin(result, 0.0);
		PART_POLYGON_PARAMETER_INTERPRETER::set_area_margin(result, 0.0);
		PART_POLYGON_PARAMETER_INTERPRETER::set_size_type_x(result, wireframe.body_x * camera_scale_x * 0.001);
		PART_POLYGON_PARAMETER_INTERPRETER::set_size_type_y(result, wireframe.body_y * camera_scale_y * 0.001);
		PART_POLYGON_PARAMETER_INTERPRETER::set_wireframe_originX(result, wireframe.origin_x * camera_scale_x * 0.001);
		PART_POLYGON_PARAMETER_INTERPRETER::set_wireframe_originY(result, wireframe.origin_y * camera_scale_y * 0.001);
		PART_POLYGON_PARAMETER_INTERPRETER::set_wireframe_angle(result, 0.0);
		PART_POLYGON_PARAMETER_INTERPRETER::set_ext_check_bound(result, 0);
		PART_POLYGON_PARAMETER_INTERPRETER::set_ext_check_angle(result, 0);
		PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_num(result, wireframe.vertex_num);
		//result.get_VISION_PART_POLYGON_whole_det().set_VISION_POLYGON_WHOLE_DET_angle_margin(std::to_string(static_cast<long double>(0.0)));

		return true;
	}

#if 0
	bool Result_PolygonGroup(const std::vector<std::vector<polygon::CPolygon>>& contours_polygons, const WireFrame& wireframe,
		VISION_PART_POLYGON& result, double camera_scale_x, double camera_scale_y, double offset_x, double offset_y) {
		if (wireframe.vertex_num == 0) {
			return false;
		}
		//std::vector<std::vector<polygon::CPolygon>> contours_polygons(contours_select.size());
		if (contours_polygons.size() == 0) {
			return false;
		}
		int vertex_index = 0;
		double camera_scale_avg = (camera_scale_x + camera_scale_y) / 2;
		PART_POLYGON_PARAMETER_INTERPRETER::set_polygon_det_size(result, wireframe.vertex_num);
		for (int i = 0; i < contours_polygons.size(); ++i) {
			for (int j = 0; j < contours_polygons[i].size(); ++j) {
				if (contours_polygons[i][j].type == LINE) {
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointX(result, (contours_polygons[i][j].st.x + offset_x) * camera_scale_x * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointY(result, (contours_polygons[i][j].st.y + offset_y) * camera_scale_y * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_segment_angleSpan(result, 0.0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_rounding_size(result, 0.0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_control_bit(result, 0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_polygon_group_index(result, i, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_index(result, vertex_index, vertex_index);
					vertex_index++;
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointX(result, (contours_polygons[i][j].ed.x + offset_x) * camera_scale_x * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointY(result, (contours_polygons[i][j].ed.y + offset_y) * camera_scale_y * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_segment_angleSpan(result, 0.0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_rounding_size(result, 0.0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_control_bit(result, 5, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_polygon_group_index(result, i, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_index(result, vertex_index, vertex_index);
					vertex_index++;

				}
				else {
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointX(result, (contours_polygons[i][j].arc_ct.x + offset_x) * camera_scale_x * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointY(result, (contours_polygons[i][j].arc_ct.y + offset_y) * camera_scale_y * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_segment_angleSpan(result, contours_polygons[i][j].arc_st_angle, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_rounding_size(result, contours_polygons[i][j].arc_radius * camera_scale_avg * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_control_bit(result, 0, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_polygon_group_index(result, i, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_index(result, vertex_index, vertex_index);
					vertex_index++;
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointX(result, (contours_polygons[i][j].arc_ct.x + offset_x) * camera_scale_x * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_pointY(result, (contours_polygons[i][j].arc_ct.y + offset_y) * camera_scale_y * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_segment_angleSpan(result, contours_polygons[i][j].arc_span, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_rounding_size(result, contours_polygons[i][j].arc_radius * camera_scale_avg * 0.001, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_control_bit(result, 5, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_polygon_group_index(result, i, vertex_index);
					PART_POLYGON_PARAMETER_INTERPRETER::set_vertex_index(result, vertex_index, vertex_index);
					vertex_index++;
				}
			}
		}

		return true;
	}
#endif

}//polygon





/*********************************************************************/
int polygonExtraction(const cv::Mat& roi_img) {

	int errorcode = 0;
	if (roi_img.data == NULL) {
		_ErrorCodeExplanation = "polygonExtraction: the input image is Null.";
		return InputImageIdle;
	}
	cv::Mat part_img = roi_img.clone();
	int part_img_w = part_img.size().width;
	int part_img_h = part_img.size().height;

	part_timer.out("BrightnessCheck: ");
	// 2. 二值化
	cv::Mat saliency_img;
	cv::Mat binary_img;
	cv::Mat gfilter_img;
	//common::Saliency(part_img, saliency_img, 5.0);
	//cv::GaussianBlur(part_img, gfilter_img, cv::Size(3, 3), 3, 3);
	int binary_threshold = 170; //user makeup

	if (binary_threshold == 0) {
		cv::threshold(saliency_img, binary_img, 0, 255, cv::THRESH_OTSU);
	}
	else {
		cv::threshold(part_img, binary_img, binary_threshold, 255, cv::THRESH_BINARY);
	}
	// 3. blob分析处理
	cv::Mat binary_img_morph;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(binary_img, binary_img_morph, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(binary_img_morph, binary_img_morph, cv::MORPH_CLOSE, kernel);


	//4. 轮廓提取
	//cv::Mat edge_img;
	//cv::Mat laplacian_img;
	//cv::Canny(gfilter_img, edge_img, 150, 200);
	//cv::Laplacian(gfilter_img, laplacian_img, CV_16S, 3);
	//cv::Mat laplacian_img_abs;
	//cv::convertScaleAbs(laplacian_img, laplacian_img_abs);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary_img_morph, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//cv::findContours(binary_img_morph, contours_simple, hierarchy_simple, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS);
#if image_show	
	cv::Mat contours_img_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	cv::Mat contours_points_show = contours_img_show.clone();
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(contours_img_show, contours, i, cv::Scalar(255, 0, 0), 1, 8, hierarchy);
	}
	for (int ii = 0; ii < contours.size(); ++ii) {
		common::drawCross(contours_points_show, contours[ii], 1, cv::Scalar(0, 0, 255), 1);
	}

#endif

	//5. 轮廓预筛选(长度）
	std::vector<std::vector<cv::Point>> contours_select;
	//std::vector<std::vector<cv::Point>> contours_complex; //不能通过简单的曲线进行拟合
	errorcode = polygon::contourSelection(contours, contours_select, true);
#if image_show
	cv::Mat contours_select_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	cv::Mat contours_filter_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	for (int i = 0; i < contours_select.size(); ++i) {
		common::drawCross(contours_select_show, contours_select[i], 1, cv::Scalar(0, 0, 255), 1);
	}
#endif

	//6. 轮廓预拟合(先对各个连续轮廓整体进行简单图形拟合
	std::vector<int> complex_contour_index(contours_select.size()); // 1:complex contour  0: simple contour
	std::vector<std::vector<polygon::CPolygon>> contours_polygons(contours_select.size());
	for (int i = 0; i < contours_select.size(); ++i) {
		//6.1 直线拟合
		polygon::CPolygon polygon(contours_select[i], polygon::ARC);
		if (polygon.RobustArcFit(contours_select[i], 50, 1.0, 0.85)) {
			contours_polygons[i].push_back(polygon);
			complex_contour_index[i] = 0;
		}
		else {
			complex_contour_index[i] = 1;
		}
		//6.2 曲线拟合
	}
#if image_show
	cv::Mat contours_simple_fit_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	recoverContours(contours_simple_fit_show, contours_polygons);


#endif

	//7. 轮廓分段
	std::vector<std::vector<double>> curvatures(contours_select.size());
	std::vector<std::vector<cv::Point>> contours_filter(contours_select.size());
	//所有轮廓
	//for (int i = 0; i < contours_select.size(); ++i) {
	Concurrency::parallel_for(size_t(0), contours_select.size(), [&](size_t i) {
		if (complex_contour_index[i] == 1) {
			polygon::calCurvature(roi_img, contours_select[i], curvatures[i], contours_filter[i]);
			//polygon::calCurvature3p(roi_img, contours_filter[0], curvatures[0]);
			std::vector<std::vector<cv::Point>> contours_segments;
			std::vector<double> curvatures_segments; //各个分割轮廓片段的平均曲率值
			const double curvature_tolerance = 0.07;
			//需要考虑此处是采用contours_filter(经过blip_fliter), 还是contours_select(未经过blip_filter)
			polygon::voteCurvature(contours_filter[i], curvatures[i], contours_segments, curvatures_segments, curvature_tolerance);
#if image_show
			common::drawCross(contours_filter_show, contours_filter[i], 1, cv::Scalar(0, 255, 0), 1);
			cv::Mat contours_select_now_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
			common::drawCross(contours_select_now_show, contours_select[i], 1, cv::Scalar(0, 0, 255), 1);
			cv::Mat contours_vote_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
			for (int k = 0; k < contours_segments.size(); ++k) {
				common::drawCross(contours_vote_show, contours_segments[k], 1, cv::Scalar(0, 255, 0), 1);
			}
#endif
			// 8.轮廓拟合
			for (int j = 0; j < contours_segments.size(); ++j) {
				if (curvatures_segments[j] <= curvature_tolerance) {
					polygon::CPolygon polygon(contours_segments[j], polygon::LINE);
					if (polygon.LineFit(contours_segments[j])) {
						contours_polygons[i].push_back(polygon);
					}
					else {
						if (polygon.RobustArcFit(contours_segments[j], 50, 1.0, 0.85)) {
							polygon.type = polygon::ARC;
							contours_polygons[i].push_back(polygon);
						}
					}
				}
				else {
					polygon::CPolygon polygon(contours_segments[j], polygon::ARC);
					if (polygon.RobustArcFit(contours_segments[j], 50, 1.0, 0.85)) {
						contours_polygons[i].push_back(polygon);
					}
				}
			}

		}
		});
	//}



	// 9. 进行拟合基本图形单元的恢复	
	cv::Mat contours_fit_img = cv::Mat::zeros(part_img.size(), CV_8UC1);
	if (!recoverContours(contours_fit_img, contours_polygons)) {


	}

	// 10. 提取示教结果到元件类中
	// polygon::WireFrame wireframe;
	// if (!polygon::Result_Whole(contours_polygons, part, *result, wireframe, image_scalex, image_scaley)) {

	// }
#if image_show
	cv::Mat body_show = contours_fit_img.clone();
	cv::cvtColor(body_show, body_show, CV_GRAY2BGR);
	//cv::rectangle(body_show, cv::Rect(wireframe.origin_x / image_scalex * 1000 - wireframe.body_x / image_scalex * 1000 / 2.0 - offset_x, 
	//	wireframe.origin_y / image_scaley * 1000 - wireframe.body_y / image_scaley * 1000 / 2.0 - offset_y, wireframe.body_x / image_scalex * 1000, 
	//	wireframe.body_y / image_scaley * 1000), cv::Scalar(255, 0, 0));

	cv::rectangle(body_show, cv::Rect(wireframe.origin_x - wireframe.body_x / 2.0, wireframe.origin_y - wireframe.body_y / 2.0,
		wireframe.body_x, wireframe.body_y), cv::Scalar(255, 0, 0));
#endif

	// double offset_x = -(static_cast<float>(wireframe.origin_x));
	// double offset_y = -(static_cast<float>(wireframe.origin_y));
	// if (!polygon::Result_PolygonGroup(contours_polygons, wireframe, *result, image_scalex, image_scaley, offset_x, offset_y)) {

	// }
	// for test: circle fitting
	//cv::Mat circle_show = cv::Mat::zeros(contours_img_show.size(), CV_8UC1);
	//cv::circle(circle_show, cv::Point(contours_img_show.cols / 2, contours_img_show.rows / 2), 40, cv::Scalar(255), -1);
	//std::vector<std::vector<cv::Point>> contours_circle;
	//std::vector<cv::Vec4i> hierarchy_circle;
	//cv::findContours(circle_show, contours_circle, hierarchy_circle, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	////polygon::calCurvature3p(roi_img, contours_circle[0], curvatures[0]);
	////polygon::calCurvature(roi_img, contours_circle[0], curvatures[0]);
	////polygon::voteCurvature(contours_circle[0], curvatures[0], contours_segments);
	//polygon::CPolygon Circle(contours[0], polygon::ARC);
	//std::vector<cv::Point> inliers_circle;
	//if (Circle.RobustPolyFit(inliers_circle, 50, 1.0, 0.2, false)) {
	//	cv::RotatedRect fitcircle = cv::fitEllipse(inliers_circle);
	//	int jj = 0;
	//	jj++;
	//	cv::circle(contours_points_show, Circle.arc_ct, Circle.arc_radius, cv::Scalar(255, 0, 0));
	//}
	return 0;
}


int polygonExtraction2(const cv::Mat& roi_img, VISION_PART_POLYGON& part, const double image_scalex, const double image_scaley, VISION_PART_POLYGON* result, std::string& ErrorcodeExplanation) {

	int errorcode = 0;
	_ErrorCodeExplanation = "";
	ErrorcodeExplanation = _ErrorCodeExplanation;
	double scale_avg = (image_scalex + image_scaley) / 2;

	if (roi_img.data == NULL) {
		_ErrorCodeExplanation = "polygonExtraction: the input image is Null.";
		return InputImageIdle;
	}
	cv::Mat part_img = roi_img.clone();
	int part_img_w = part_img.size().width;
	int part_img_h = part_img.size().height;

	// 1. 亮度检查
	int part_img_area = part_img_w * part_img_h;
	errorcode = polygon::BrightnessCheck(part_img, 150, 0.008 * part_img_area, 0.9 * part_img_area);
	if (errorcode != success) {

		return errorcode;
	}
	part_timer.out("BrightnessCheck: ");
	// 2. 二值化
	cv::Mat saliency_img;
	cv::Mat binary_img;
	cv::Mat gfilter_img;
	//common::Saliency(part_img, saliency_img, 5.0);
	//cv::GaussianBlur(part_img, gfilter_img, cv::Size(3, 3), 3, 3);
	//double binary_threshold = part.get_VISION_PART_LL_whole_det().get_VISION_POLYGON_WHOLE_DET_threshold();
	int binary_threshold = 170;
	if (binary_threshold < 0) {
		return LLThresholdIsTooLow;
	}
	if (binary_threshold > 255) {
		return LLThresholdIsTooHigh;
	}
	if (binary_threshold == 0) {
		cv::threshold(saliency_img, binary_img, 0, 255, cv::THRESH_OTSU);
	}
	else {
		cv::threshold(part_img, binary_img, binary_threshold, 255, cv::THRESH_BINARY);
	}
	// 3. blob分析处理
	part_timer.reset();
	cv::Mat binary_img_morph;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(binary_img, binary_img_morph, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(binary_img_morph, binary_img_morph, cv::MORPH_CLOSE, kernel);

	part_timer.out("Binary and Bolb: ");
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary_img_morph, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//4. 轮廓预筛选(长度）
	std::vector<std::vector<cv::Point>> contours_select;
	errorcode = polygon::contourSelection(contours, contours_select, true);

	//5. 轮廓预拟合(先对各个连续轮廓整体进行简单图形拟合
	std::vector<int> complex_contour_index(contours_select.size()); // 1:complex contour  0: simple contour
	std::vector<std::vector<polygon::CPolygon>> contours_polygons(contours_select.size());
	for (int i = 0; i < contours_select.size(); ++i) {
#if image_show
		cv::Mat fiting_contour_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
		common::drawCross(fiting_contour_show, contours_select[i], 1, cv::Scalar(0, 0, 255), 1);
#endif
		cv::RotatedRect rect_ = cv::minAreaRect(contours_select[i]);
		// closed contours
		//float aspect_ratio = rect_.size.width / (rect_.size.height + 0.001f);
		//if (aspect_ratio > 1.3 || aspect_ratio < 0.8) 
		//	continue;
		if (rect_.size.width < 3 || rect_.size.height < 3)
			continue;
		polygon::CPolygon polygon(contours_select[i], polygon::ARC);
		double sigma;
		if (polygon.CircleFitByTaubin(contours_select[i], sigma, true, 10)) {
#if image_show
			cv::circle(fiting_contour_show, polygon.arc_ct, polygon.arc_radius, cv::Scalar(255, 0, 0));
#endif
			if (sigma < 0.7) {
				contours_polygons[i].push_back(polygon);
				complex_contour_index[i] = 0;
			}
		}
		else {
			complex_contour_index[i] = 1;
		}

	}



	//6.轮廓整体平滑后 多边形近似点提取
	std::vector<std::vector<cv::Point>> contours_approx(contours_select.size());
	std::vector<std::vector<int>> approx_indexs(contours_select.size());
	std::vector<std::vector<cv::Point>> contours_filter(contours_select.size());
	for (int i = 0; i < contours_select.size(); ++i) {
		if (complex_contour_index[i] == 1) {
			polygon::blipFilter(contours_select[i], contours_filter[i]);
			polygon::approxPolygon(contours_filter[i], contours_approx[i], approx_indexs[i], 3, true);

			// 6.1轮廓分段拟合

			std::vector<std::vector<cv::Point>> contour_segment(approx_indexs[i].size()); //单个连续闭合轮廓分割的片段
			int count = contours_approx[i].size();
			for (int j = 0; j < contours_approx[i].size(); ++j) {
				int st_index = approx_indexs[i][j];
				int ed_index = approx_indexs[i][(j + 1) % count];
				if (ed_index < st_index) {
					contour_segment[j].assign(contours_filter[i].begin() + st_index, contours_filter[i].end());
					contour_segment[j].insert(contour_segment[j].end(), contours_filter[i].begin(), contours_filter[i].begin() + ed_index);
				}
				else {
					contour_segment[j].assign(contours_filter[i].begin() + st_index, contours_filter[i].begin() + ed_index);
				}

				//6.2 轮廓片段拟合
				polygon::CPolygon polygon(contour_segment[j], polygon::LINE);
				if (polygon.LineFit(contour_segment[j])) {
					contours_polygons[i].push_back(polygon);
				}
				else {
					if (polygon.RobustArcFit(contour_segment[j], 50, 1.0, 0.85, false)) {
						polygon.type = polygon::ARC;
						contours_polygons[i].push_back(polygon);
					}
				}

			}

		}
	}

	cv::Mat contours_fit_img = cv::Mat::zeros(part_img.size(), CV_8UC1);
	if (!recoverContours(contours_fit_img, contours_polygons)) {


	}

#if image_show
	cv::Mat contours_select_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	//cv::Mat contours_approx_show = cv::Mat::zeros(part_img.size(), CV_8UC3);
	cv::Mat contours_approx_show = part_img.clone();
	cv::cvtColor(contours_approx_show, contours_approx_show, CV_GRAY2BGR);
	for (int i = 0; i < contours_select.size(); ++i) {
		common::drawCross(contours_select_show, contours_select[i], 1, cv::Scalar(0, 0, 255), 1);
	}

	for (int i = 0; i < contours_approx.size(); ++i) {
		common::drawCross(contours_approx_show, contours_filter[i], 1, cv::Scalar(0, 255, 0), 1);
	}

	for (int i = 0; i < contours_approx.size(); ++i) {
		common::drawCross(contours_approx_show, contours_approx[i], 1, cv::Scalar(0, 0, 255), 1);
	}

#endif

	//7. 轮廓片段平滑


	//8. 轮廓分段拟合


	// 10. 提取示教结果到元件类中
	polygon::WireFrame wireframe;
	if (!polygon::Result_Whole(contours_polygons, part, *result, wireframe, image_scalex, image_scaley)) {

	}
#if image_show
	cv::Mat body_show = contours_fit_img.clone();
	cv::cvtColor(body_show, body_show, CV_GRAY2BGR);
	//cv::rectangle(body_show, cv::Rect(wireframe.origin_x / image_scalex * 1000 - wireframe.body_x / image_scalex * 1000 / 2.0 - offset_x, 
	//	wireframe.origin_y / image_scaley * 1000 - wireframe.body_y / image_scaley * 1000 / 2.0 - offset_y, wireframe.body_x / image_scalex * 1000, 
	//	wireframe.body_y / image_scaley * 1000), cv::Scalar(255, 0, 0));

	cv::rectangle(body_show, cv::Rect(wireframe.origin_x - wireframe.body_x / 2.0, wireframe.origin_y - wireframe.body_y / 2.0,
		wireframe.body_x, wireframe.body_y), cv::Scalar(255, 0, 0));
#endif

	// double offset_x = -(static_cast<float>(wireframe.origin_x));
	// double offset_y = -(static_cast<float>(wireframe.origin_y));
	// if (!polygon::Result_PolygonGroup(contours_polygons, wireframe, *result, image_scalex, image_scaley, offset_x, offset_y)) {

	// }


	return 0;
}
