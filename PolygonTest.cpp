#include "PolygonTest.h"

#include <ppl.h>

#define image_show 1


namespace polygon {

	bool PolygonProcessor::Init(const VISION_PART_POLYGON* polygon, double camera_scale_x, double camera_scale_y) {
		
		_camera_scale_x = camera_scale_x;
		_camera_scale_y = camera_scale_y;

		//转换数据类型
		_vertex_num = atoi(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_vertex_num().c_str());
		_wireframe_originX = atof(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_wireframe_originX().c_str()) / camera_scale_x * 1000.0;
		_wireframe_originY = atof(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_wireframe_originY().c_str()) / camera_scale_y * 1000.0;
		_wireframe_angle = atof(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_wireframe_angle().c_str());
		_size_type_x = atof(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_size_type_x().c_str()) / camera_scale_x * 1000.0;
		_size_type_y = atof(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_size_type_y().c_str()) / camera_scale_y * 1000.0;
		_threshold = atoi(polygon->get_VISION_PART_POLYGON_whole_det().get_VISION_POLYGON_WHOLE_DET_threshold().c_str());
		std::vector<VISION_POLYGON_POLY_DET> poly_vec = polygon->get_VISION_PART_POLYGON_vector_poly_det();
		if (_vertex_num != poly_vec.size() && _vertex_num % 2 != 0) {
			return false;
		}

		for (int i = 0; i < _vertex_num; i+=2) {
			int start_index = 0;
			int end_index = 0;
			int control_bit1 = atoi(poly_vec[i].get_VISION_POLYGON_POLY_DET_control_bit().c_str());
			int control_bit2 = atoi(poly_vec[i+1].get_VISION_POLYGON_POLY_DET_control_bit().c_str());
			if (control_bit1 == 0 && control_bit2 == 5) {
				start_index = i;
				end_index = i + 1;
			}
			else {
				start_index = i + 1;
				end_index = i;
			}
			cv::Point2d st = cv::Point2d(atof(poly_vec[start_index].get_VISION_POLYGON_POLY_DET_vertex_pointX().c_str()), atof(poly_vec[start_index].get_VISION_POLYGON_POLY_DET_vertex_pointY().c_str()));
			cv::Point2d ed = cv::Point2d(atof(poly_vec[end_index].get_VISION_POLYGON_POLY_DET_vertex_pointX().c_str()), atof(poly_vec[end_index].get_VISION_POLYGON_POLY_DET_vertex_pointY().c_str()));
			if (Equal(st.x,ed.x) && Equal(st.y, ed.y)) {
				//arc	
				PolygonProcessor::Arc_Elem arc_elem;
				//arc_elem.ct = cv::Point2f(static_cast<float>(st.x) * 0.001, static_cast<float>(st.y) * 0.001);
				arc_elem.ct = cv::Point2f(st.x / camera_scale_x * 1000.0, st.y / camera_scale_y * 1000.0);
				arc_elem.angle_span = atof(poly_vec[end_index].get_VISION_POLYGON_POLY_DET_segment_angleSpan().c_str());
				arc_elem.arc_radius = atof(poly_vec[start_index].get_VISION_POLYGON_POLY_DET_rounding_size().c_str()) / ((camera_scale_x + camera_scale_y) * 0.5) * 1000;
				arc_elem.st_angle = atof(poly_vec[start_index].get_VISION_POLYGON_POLY_DET_segment_angleSpan().c_str());
				arc_elems.push_back(arc_elem);
			}
			else {
				//line
				PolygonProcessor::Line_Elem line_elem;
				line_elem.st = cv::Point(static_cast<int>(st.x / camera_scale_x * 1000.0), static_cast<int>(st.y / camera_scale_y * 1000.0));
				line_elem.ed = cv::Point(static_cast<int>(ed.x / camera_scale_x * 1000.0), static_cast<int>(ed.y / camera_scale_y * 1000.0));
				line_elems.push_back(line_elem);
			}
			
		}
		if (arc_elems.size() == 0 && line_elems.size() == 0) {
			return false;
		}

		// generate template image
		int col_ = _size_type_x;
		int row_ = _size_type_y;
		_template_image = cv::Mat::zeros(cv::Size(col_ + 6, row_ + 6), CV_32FC1);
		_template_pyrdown = cv::Mat::zeros(cv::Size(col_ / _pyr_scale + 6, row_ / _pyr_scale + 6), CV_32FC1);
		return true;
	}

	bool PolygonProcessor::Equal(double x, double target, double eps) {
		return std::abs(x - target) < eps;
	}

	bool PolygonProcessor::Equal(float x, float target, float eps) {
		return std::abs(x - target) < eps;
	}

	bool PolygonProcessor::drawTemplate(VISION_PART_POLYGON* polygon, const double& pyrdown) {
		int offset_x = _template_image.cols / 2;
		int offset_y = _template_image.rows / 2;
		cv::Point offset(offset_x, offset_y);

		int offset_pyr_x = _template_pyrdown.cols / 2;
		int offset_pyr_y = _template_pyrdown.rows / 2;
		cv::Point offset_pyr(offset_pyr_x, offset_pyr_y);
		//需要判断是否存在模板绘制超界的问题
		for (int i = 0; i < line_elems.size(); ++i) {
			cv::line(_template_image, line_elems[i].st + offset, line_elems[i].ed + offset, cv::Scalar(255));
		}

		for (int j = 0; j < arc_elems.size(); ++j) {
			drawArc(_template_image, arc_elems[j].ct + cv::Point2f(offset_x, offset_y),arc_elems[j].arc_radius ,arc_elems[j].st_angle, arc_elems[j].angle_span);
		}
		
		for (int i = 0; i < line_elems.size(); ++i) {
			cv::line(_template_pyrdown, line_elems[i].st / _pyr_scale + offset_pyr, line_elems[i].ed / _pyr_scale + offset_pyr, cv::Scalar(255));
		}

		for (int j = 0; j < arc_elems.size(); ++j) {
			drawArc(_template_pyrdown, arc_elems[j].ct  / _pyr_scale + cv::Point2f(offset_pyr_x, offset_pyr_y), arc_elems[j].arc_radius / _pyr_scale, arc_elems[j].st_angle, arc_elems[j].angle_span);
		}

		//cv::pyrDown(_template_image, _template_pyrdown);
#if image_show
		cv::Mat template_image_show = _template_image.clone();
		cv::Mat template_pyr_show = _template_pyrdown.clone();
#endif
		return true;
	}

	bool PolygonProcessor::drawArc(cv::Mat& src_img, cv::Point2f arc_ct, double arc_radius, double st_angle, double angle_span) {
		double st_angle_deg_ = 0.0;
		double angle_span_ = 0.0;
		if (angle_span < 0) {
			st_angle_deg_ = st_angle * 180.0 / CV_PI + angle_span;
		}
		else {
			st_angle_deg_ = st_angle * 180.0 / CV_PI;
		}
		st_angle_deg_ = st_angle_deg_ < 0 ? st_angle_deg_ + 360 : st_angle_deg_;
		angle_span_ = abs(angle_span);
		cv::ellipse(src_img, arc_ct, cv::Size(arc_radius, arc_radius), st_angle_deg_, 0, angle_span_, cv::Scalar(255));
		return true;
	}
	
	bool PolygonProcessor::checkBrightness(const cv::Mat& src_img, const int lower_bound, const int upper_bound, const int gray_thresh, wdErrorCode& errorcode) {
		cv::Mat binary_img;
		cv::threshold(src_img, binary_img, gray_thresh, 255, cv::THRESH_BINARY);
		int pixels_num = countNonZero(binary_img);
		if (pixels_num > upper_bound) {
			errorcode = BackgroundIsTooBright;
			return false;
		}
		else if (pixels_num < lower_bound) {
			errorcode = PartIsTooDark;
			return false;
		}
		errorcode = success;
		return true;
	}
	
	bool PolygonProcessor::calCoareseAngle(const cv::Mat& image_edge,double& angle, wdErrorCode& errorcode) {
		int dft_size = std::max(std::max(image_edge.cols, image_edge.rows), std::max(_template_pyrdown.cols, _template_pyrdown.rows));
		dft_size = cv::getOptimalDFTSize(dft_size);
		cv::Mat dft_img, dft_templ;
		dftImg(image_edge, dft_img, dft_size);
		dftImg(_template_pyrdown, dft_templ, dft_size);
#if image_show
		cv::Mat template_show = _template_pyrdown.clone();
#endif
		cv::log(dft_img, dft_img);
		cv::normalize(dft_img, dft_img, 0, 1, cv::NORM_MINMAX);
		cv::log(dft_templ, dft_templ);
		cv::normalize(dft_templ, dft_templ, 0, 1, cv::NORM_MINMAX);

		cv::Mat dft_img_polar, dft_templ_polar;
		int radius = std::min(std::min(dft_img.cols, dft_img.rows), std::min(dft_templ.cols, dft_templ.rows)) / 2;
		double angle_step = 0.5; //step: 0.5 deg
		linearPolar(dft_img, dft_img_polar, cv::Point2f((float)dft_img.cols / 2, (float)dft_img.rows / 2), radius, angle_step);
		linearPolar(dft_templ, dft_templ_polar, cv::Point2f((float)dft_templ.cols / 2, (float)dft_templ.rows / 2), radius, angle_step);

		double angle_range[2] = { -30, 30 };
		int dft_range[2] = { int(-angle_range[0] / angle_step), int(angle_range[1] / angle_step) };
		cv::Mat search_img(dft_img_polar.rows + dft_range[0] + dft_range[1], dft_img_polar.cols, dft_img_polar.type());
		dft_img_polar(cv::Rect(0, dft_img_polar.rows - dft_range[0], dft_img_polar.cols, dft_range[0])).copyTo(search_img(cv::Rect(0, 0, search_img.cols, dft_range[0])));
		dft_img_polar.copyTo(search_img(cv::Rect(0, dft_range[0], search_img.cols, dft_img_polar.rows)));
		dft_img_polar(cv::Rect(0, 0, dft_img_polar.cols, dft_range[1])).copyTo(search_img(cv::Rect(0, dft_range[0] + dft_img_polar.rows, search_img.cols, dft_range[1])));
		cv::Mat angle_corr;
		cv::matchTemplate(search_img, dft_templ_polar, angle_corr, CV_TM_CCOEFF_NORMED);
		double minVal, maxVal;
		int minPos[2], maxPos[2];
		cv::minMaxIdx(angle_corr, &minVal, &maxVal, minPos, maxPos);

		angle = maxPos[0] * angle_step - 30.0;
		
		return true;
	}

	bool PolygonProcessor::calCoareseAngle2(const cv::Mat& image_edge, const cv::Mat& image_template, double& angle, wdErrorCode& errorcode) {
		if (image_edge.cols < image_template.cols || image_edge.rows < image_template.rows) {
			return false;
		}
		int dft_size_w = cv::getOptimalDFTSize(std::max(image_edge.cols, image_edge.rows));
		int dft_size_h = dft_size_w;
		cv::Mat dft_img, dft_templ;
		ForwardFFT(image_edge, dft_img, dft_size_w, dft_size_h, true);
		ForwardFFT(image_template, dft_templ, dft_size_w, dft_size_h, true);
		cv::Mat h_img, h_temp;
		//highpass(dft_img.size(), h_img, 0.95);
		//highpass(dft_img.size(), h_temp, 0.85);
		float filter_ratio_img = 0.95;
		float filter_ratio_temp = 0.85;
		highpass(dft_img.size(), h_img, h_temp, filter_ratio_img, filter_ratio_temp);
		dft_img = dft_img.mul(h_img);
		dft_templ = dft_templ.mul(h_temp);
		//dft_templ = dft_templ.mul(h);
		cv::normalize(dft_img, dft_img, 0, 1, cv::NORM_MINMAX);
		cv::normalize(dft_templ, dft_templ, 0, 1, cv::NORM_MINMAX);
		cv::Mat dft_img_polar, dft_templ_polar;
		//int radius = static_cast<int>(std::min(std::min(image_edge.cols, image_edge.rows), std::min(_template_pyrdown.cols, _template_pyrdown.rows)) / 2.0f * 1.414);
		int radius = static_cast<int>((float)(dft_img.cols) / 2.0f * 0.8);
		double angle_step = 0.5;//0.5度步长采样
		linearPolar(dft_img, dft_img_polar, cv::Point2f((float)dft_img.cols / 2, (float)dft_img.rows / 2),
			radius, angle_step);
		linearPolar(dft_templ, dft_templ_polar, cv::Point2f((float)dft_templ.cols / 2, (float)dft_templ.rows / 2),
			radius, angle_step);
		double thearange[2] = { -30, 30 };
		int dftrange[2] = { int(-thearange[0] / angle_step), int(thearange[1] / angle_step) };
		cv::Mat search_img(dft_img_polar.rows + dftrange[0] + dftrange[1], dft_img_polar.cols, dft_img_polar.type());
		dft_img_polar(cv::Rect(0, dft_img_polar.rows - dftrange[0], dft_img_polar.cols, dftrange[0])).copyTo(search_img(cv::Rect(0, 0, search_img.cols, dftrange[0])));
		dft_img_polar.copyTo(search_img(cv::Rect(0, dftrange[0], search_img.cols, dft_img_polar.rows)));
		dft_img_polar(cv::Rect(0, 0, dft_img_polar.cols, dftrange[1])).copyTo(search_img(cv::Rect(0, dftrange[0] + dft_img_polar.rows, search_img.cols, dftrange[1])));
		cv::Mat angle_corr;
		matchTemplate(search_img, dft_templ_polar, angle_corr, CV_TM_CCOEFF_NORMED);
		double minVal, maxVal;
		int minPos[2], maxPos[2];
		cv::minMaxIdx(angle_corr, &minVal, &maxVal, minPos, maxPos);
		angle = maxPos[0] * angle_step - 30;
		return true;
	}


	void PolygonProcessor::linearPolar(const cv::Mat& src, cv::Mat& dst, const cv::Point2f& center, const int MaxRadius, double& dAngle) {
		dAngle = 90.0 / cvCeil(90 / dAngle);
		int angle_num = 4 * cvRound(90 / dAngle);
		dst.create(angle_num, MaxRadius, src.type());
		cv::Size dsize = dst.size();
		cv::Mat mapx, mapy;
		mapx.create(dsize, CV_32F);
		mapy.create(dsize, CV_32F);


		size_t dst_rows = dst.rows;
		Concurrency::parallel_for(size_t(0), dst_rows, [&](size_t i)
			{
				float ci = cos(i * dAngle * CV_PI / 180);
				float si = sin(i * dAngle * CV_PI / 180);
				float* mx = (float*)(mapx.data + i * mapx.step);
				float* my = (float*)(mapy.data + i * mapy.step);
				for (int j = 0; j < dst.cols; j++)
				{
					float x = j * ci + center.x;
					float y = j * si + center.y;
					mx[j] = x;
					my[j] = y;
				}
			});
		cv::remap(src, dst, mapx, mapy, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
	}

	cv::Mat PolygonProcessor::arrangeMat(const cv::Mat& src) {
		int cx = src.cols / 2;
		int cy = src.rows / 2;
		cv::Mat dst(cy * 2, cx * 2, src.type());
		cv::Mat src0(src, cv::Rect(0, 0, cx, cy));
		cv::Mat src1(src, cv::Rect(src.cols - cx, 0, cx, cy));
		cv::Mat src2(src, cv::Rect(0, src.rows - cy, cx, cy));
		cv::Mat src3(src, cv::Rect(src.cols - cx, src.rows - cy, cx, cy));
		cv::Mat dst0(dst, cv::Rect(0, 0, cx, cy));
		cv::Mat dst1(dst, cv::Rect(cx, 0, cx, cy));
		cv::Mat dst2(dst, cv::Rect(0, cy, cx, cy));
		cv::Mat dst3(dst, cv::Rect(cx, cy, cx, cy));
		src0.copyTo(dst3);
		src3.copyTo(dst0);
		src1.copyTo(dst2);
		src2.copyTo(dst1);
		return dst;

	}

	void PolygonProcessor::dftImg(const cv::Mat& src_img, cv::Mat& mag_dft, const int dft_size) {
		cv::Mat dft_img = cv::Mat::zeros(dft_size, dft_size, src_img.type());
		src_img.copyTo(dft_img(cv::Rect(0, 0, src_img.cols, src_img.rows)));

		dft_img.convertTo(dft_img, CV_32F);
		cv::dft(dft_img, dft_img, cv::DFT_COMPLEX_OUTPUT);
		cv::Mat planes[2];
		cv::split(dft_img, planes);
		cv::magnitude(planes[0], planes[1], mag_dft);
		mag_dft = arrangeMat(mag_dft);
	}

	void PolygonProcessor::Recomb(cv::Mat& src, cv::Mat& dst)
	{
		int cx = src.cols >> 1;
		int cy = src.rows >> 1;
		cv::Mat tmp;
		tmp.create(src.size(), src.type());
		src(cv::Rect(0, 0, cx, cy)).copyTo(tmp(cv::Rect(cx, cy, cx, cy)));
		src(cv::Rect(cx, cy, cx, cy)).copyTo(tmp(cv::Rect(0, 0, cx, cy)));
		src(cv::Rect(cx, 0, cx, cy)).copyTo(tmp(cv::Rect(0, cy, cx, cy)));
		src(cv::Rect(0, cy, cx, cy)).copyTo(tmp(cv::Rect(cx, 0, cx, cy)));
		dst = tmp;
	}

	void PolygonProcessor::highpass(cv::Size& sz, cv::Mat& dst, double ratio) {
		cv::Mat a = cv::Mat(sz.height, 1, CV_32FC1);
		cv::Mat b = cv::Mat(1, sz.width, CV_32FC1);

		float step_y = CV_PI / sz.height;
		float val = -CV_PI * 0.5;

		for (int i = 0; i < sz.height; ++i) {
			a.at<float>(i) = cos(val) * ratio;
			val += step_y;
		}

		val = -CV_PI * 0.5;
		float step_x = CV_PI / sz.width;
		for (int i = 0; i < sz.width; ++i) {
			b.at<float>(i) = cos(val) * ratio;
			val += step_x;
		}
		cv::Mat tmp = a * b;
		dst = (1.0 - tmp).mul(2.0 - tmp);
	}

	void PolygonProcessor::highpass(cv::Size& sz, cv::Mat& dst1,cv::Mat& dst2 ,double ratio1, double ratio2) {
		cv::Mat a1 = cv::Mat(sz.height, 1, CV_32FC1);
		cv::Mat a2 = cv::Mat(sz.height, 1, CV_32FC1);
		cv::Mat b1 = cv::Mat(1, sz.width, CV_32FC1);
		cv::Mat b2 = cv::Mat(1, sz.width, CV_32FC1);

		float step_y = CV_PI / sz.height;
		float val = -CV_PI * 0.5;

		for (int i = 0; i < sz.height; ++i) {
			float cos_val = cos(val);
			a1.at<float>(i) = cos_val * ratio1;
			a2.at<float>(i) = cos_val * ratio2;
			val += step_y;
		}

		val = -CV_PI * 0.5;
		float step_x = CV_PI / sz.width;
		for (int i = 0; i < sz.width; ++i) {
			float cos_val = cos(val);
			b1.at<float>(i) = cos_val * ratio1;
			b2.at<float>(i) = cos_val * ratio2;
			val += step_x;
		}
		cv::Mat tmp1 = a1 * b1;
		cv::Mat tmp2 = a2 * b2;
		dst1 = (1.0 - tmp1).mul(2.0 - tmp1);
		dst2 = (1.0 - tmp2).mul(2.0 - tmp2);
	}

	void PolygonProcessor::ForwardFFT(const cv::Mat& src_img, cv::Mat& mag_dft, int dft_size_w, int dft_size_h, bool do_recomb) {
		int M = dft_size_h;
		int N = dft_size_w;
		cv::Mat padded;

		cv::copyMakeBorder(src_img, padded, 0, M - src_img.rows, 0, N - src_img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		cv::Mat planes[] = { cv::Mat_<float>(padded),  cv::Mat::zeros(padded.size(), CV_32F) };
		cv::Mat complexImg;
		cv::merge(planes, 2, complexImg);
		cv::dft(complexImg, complexImg, cv::DFT_COMPLEX_OUTPUT);
		cv::split(complexImg, planes);
		planes[0] = planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
		planes[1] = planes[1](cv::Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2)); //向下取偶数
		if (do_recomb) {
			Recomb(planes[0], planes[0]);
			Recomb(planes[1], planes[1]);
		}
		planes[0] /= float(M * N);
		planes[1] /= float(M * N);
		cv::magnitude(planes[0], planes[1], mag_dft);
	}


	void PolygonProcessor::ImageWarpAffine(const cv::Mat& src_img, cv::Mat& dst_img, const double& angle, cv::Point2f& center, cv::Point2f& offset) {
		double rad_a = sin(angle * CV_PI / 180.0);
		double rad_b = cos(angle * CV_PI / 180.0);
		int width = src_img.size().width;
		int height = src_img.size().height;
		int rotated_width = int(height * fabs(rad_a) + width * fabs(rad_b)); //旋转后图像宽度
		int rotated_height = int(width * fabs(rad_a) + height * fabs(rad_b));

		cv::Mat trans_matrix(2, 3, CV_32FC1);
		center = cv::Point2f(float(width) / 2.0, float(height) / 2.0); //旋转中心
		trans_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		trans_matrix.at<double>(0, 2) += (rotated_width - width) / 2.0;
		trans_matrix.at<double>(1, 2) += (rotated_height - height) / 2.0;

		dst_img = cv::Mat::zeros(cv::Size(rotated_width, rotated_height), CV_8UC1);
		cv::warpAffine(src_img, dst_img, trans_matrix, dst_img.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
		offset.x = float((rotated_width - width) / 2.0);
		offset.y = float((rotated_height - height) / 2.0);
	}

	void PolygonProcessor::matchLocation(cv::Mat& match_result, std::pair<double, cv::Point2f>& location, int offset_x, int offset_y) {
		double minVal = 0;
		double maxVal = 0;
		cv::Point minLoc;
		cv::Point maxLoc;

		cv::minMaxLoc(match_result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		location = std::pair<double, cv::Point2f>(minVal, (cv::Point2f)minLoc + cv::Point2f((float)(offset_x - 1) / 2, (float)(offset_y - 1) / 2));
	}
	void PolygonProcessor::getRotatedTemp(const cv::Mat& src_temp_img, std::vector<cv::Mat>& rotated_temps, double l_thresh, double h_thresh, double step) {
		//可以采用并行
		for (double angle = l_thresh; angle <= h_thresh; angle += step) {
			cv::Mat rotated_temp;
			cv::Point2f affine_offset;
			ImageWarpAffine(src_temp_img, rotated_temp, angle, cv::Point2f(_template_pyrdown.cols / 2, _template_pyrdown.rows / 2), affine_offset);
			rotated_temps.push_back(rotated_temp);
		}
#if image_show
		for (int j = 0; j < rotated_temps.size(); ++j) {
			cv::Mat rotated_temp_show;
			rotated_temp_show = rotated_temps[j].clone();
		}
#endif

	}

	void PolygonProcessor::matchAngleTemp(const cv::Mat& src_img, const std::vector<cv::Mat>& templs, std::vector<cv::Mat>& match_result, int& best_index, cv::Point& best_loc, const int& method) {
		std::vector<std::pair<double, int>> match_val(templs.size());
		std::vector<std::pair<int, cv::Point>> match_loc(templs.size());
		int parallel_num = templs.size();

		Concurrency::parallel_for(0, parallel_num, [&templs, &src_img, method, &match_val, &match_loc](int k) {
			cv::Mat result;
			double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
			cv::matchTemplate(src_img, templs.at(k), result, method);
//#if image_show
//			cv::Mat src_show = src_img.clone();
//			cv::Mat templs_show = templs.at(k).clone();
//#endif
			cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
			match_val.at(k) = std::make_pair<double, int>(minVal, k);
			match_loc.at(k) = std::make_pair<int, cv::Point>(k, minLoc);
			});
		std::sort(match_val.begin(), match_val.end());
		best_index = match_val[0].second;
		best_loc = match_loc[best_index].second;
	}

	bool PolygonProcessor::checkResult(PartResult_Offset* result_offset, wdErrorCode& errorcode) {

		//angle range check
		if (abs(_refine_angle) > 30) {
			errorcode = PartAngleIsTooLarge;
			return false;
		}
		double offset_x = _loc.x - srcimg_ctx;
		double offset_y = _loc.y - srcimg_cty;
		// offset check

		if (abs(offset_x) + 0.5 * _template_image.cols > srcimg_ctx){
			errorcode = PartCenterXOffsetIsTooLarge;
			return false;
		}
		if (abs(offset_y) + 0.5 * _template_image.rows > srcimg_cty) {
			errorcode = PartCenterYOffsetIsTooLarge;
			return false;
		}

		result_offset->offset_X = offset_x * (_camera_scale_x) / 1000;
		result_offset->offset_Y = offset_y * (_camera_scale_y) / 1000;
		result_offset->offset_R = _refine_angle;

		errorcode = success;
		return true;
	}

	bool PolygonProcessor::Process(const cv::Mat& roi_image, PartResult_Offset* result_offset, PartResult_TimSco* result_timsco, wdErrorCode& errorcode) {
		common::Timer process_timer;
		process_timer.reset();
		cv::Mat part_img = roi_image.clone();
		srcimg_ctx = static_cast<double>(roi_image.cols) * 0.5;
		srcimg_cty = static_cast<double>(roi_image.rows) * 0.5;
		//cv::Mat part_img = cv::imread("C:\\Users\\jia_z\\Desktop\\camera2-20-06-39-127.bmp", 0);
#if image_show
		//测试用途
		double angle_test = 0;
		angle_test = 10.0;
		cv::Point2f center_test = cv::Point2f(part_img.cols / 2.0, part_img.rows / 2.0);
		cv::Mat matrix_test = cv::getRotationMatrix2D(cv::Point2f(part_img.cols / 2.0, part_img.rows / 2.0), angle_test, 1);
		cv::warpAffine(part_img, part_img, matrix_test, part_img.size(), 1, cv::BORDER_REPLICATE);
#endif

		int part_img_width = part_img.cols;
		int part_img_height = part_img.rows;
		if (!checkBrightness(part_img, part_img_width * part_img_height * 0.005, part_img_width * part_img_height * 0.9, 150, errorcode)) {
			return false;
		}

		//1. 二值化
		cv::Mat bianry_image;
		if (_threshold <= 0 || _threshold >= 255) {
			cv::threshold(part_img, bianry_image, 0, 255, cv::THRESH_OTSU);
		}
		else {
			cv::threshold(part_img, bianry_image, _threshold, 255, cv::THRESH_BINARY);
		}
		
		//2. sub-sample image matching
		//2.1 edge extraction and calculate distance transfrom image
		cv::Mat pyrdown_image;
		cv::pyrDown(part_img, pyrdown_image);
		cv::Mat pyrdown_image_blur;
		cv::blur(pyrdown_image, pyrdown_image_blur, cv::Size(5, 5));
		cv::Canny(pyrdown_image, pyr_edge, 180, 250);
		cv::Mat pyrdown_invert = ~pyr_edge;
		cv::distanceTransform(pyrdown_invert, pyr_dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
#if image_show
		cv::Mat pyr_edge_show = pyr_edge.clone();
		cv::Mat pyr_dt_show = pyr_dt.clone();
#endif
		//2.2 rough angle matching
		double angle;
		//calCoareseAngle(pyr_edge, angle, errorcode);
		calCoareseAngle2(pyr_edge,_template_pyrdown, angle, errorcode);
		//2.3 pyramid rotated template image matching
		cv::Mat rotated_template_pyrdown;
		cv::Point2f affine_offset;
		ImageWarpAffine(_template_pyrdown, rotated_template_pyrdown, -angle ,cv::Point2f(_template_pyrdown.cols/ 2.0, _template_pyrdown.rows/ 2.0), affine_offset);

		if (rotated_template_pyrdown.rows > pyr_dt.rows || rotated_template_pyrdown.cols >= pyr_dt.cols) {
			errorcode = ROIIsSmallerThanTemp;
			return false;
		}
		cv::Mat match_result;
		cv::matchTemplate(pyr_dt, rotated_template_pyrdown, match_result, CV_TM_CCORR);
		std::pair<double, cv::Point2f> precise_location;
		matchLocation(match_result, precise_location, rotated_template_pyrdown.cols, rotated_template_pyrdown.rows);

#if image_show
		//cv::cvtColor(pyr_edge_show, pyr_edge_show, CV_GRAY2BGR);
		cv::RotatedRect precise_loc_rect = cv::RotatedRect(precise_location.second, cv::Size(_size_type_x / _pyr_scale, _size_type_y / _pyr_scale), angle);
		common::drawRect(pyr_edge_show, precise_loc_rect, pyr_edge_show);
#endif
		
		//3. raw image template precise matching
		//3.1 determine the size relationship between roi image and template image
		//采用原图进行模板匹配优化(角度优化),首先根据粗匹配进行ROI图像的重新截取
		cv::Rect precise_roi;
		if (rotated_template_pyrdown.rows < 0.7 * pyr_edge.rows || rotated_template_pyrdown.cols < 0.7 * pyr_edge.cols) {
			const int expand_pixels_x = 1.5 / _camera_scale_x * 1000; //expand 3.0mm
			const int expand_pixels_y = 1.5 / _camera_scale_y * 1000;
			int left_x = (precise_location.second.x - rotated_template_pyrdown.cols * 0.5) * _pyr_scale - expand_pixels_x;
			left_x = left_x < 0 ? 0 : left_x;
			int left_y = (precise_location.second.y - rotated_template_pyrdown.rows * 0.5) * _pyr_scale - expand_pixels_y;
			left_y = left_y < 0 ? 0 : left_y;

			int roi_width = rotated_template_pyrdown.cols * _pyr_scale + 2 * expand_pixels_x;
			roi_width = left_x + roi_width >= part_img.cols ? part_img.cols - left_x - 1 : roi_width;
			int roi_height = rotated_template_pyrdown.rows * _pyr_scale + 2 * expand_pixels_y;
			roi_height = left_y + roi_height >= part_img.rows ? part_img.rows - left_y - 1 : roi_height;
			precise_roi = cv::Rect(left_x, left_y, roi_width, roi_height);

#if image_show
			cv::Mat precise_roi_show;
			common::drawRect(part_img, precise_roi, precise_roi_show);
#endif
			precise_roi_img = part_img(precise_roi).clone();
		}
		else {
			precise_roi = cv::Rect(0, 0, part_img.cols, part_img.rows);
			precise_roi_img = part_img.clone();
		}

		cv::Mat gfilter_img;
		cv::GaussianBlur(precise_roi_img, gfilter_img, cv::Size(3, 3), 3, 3);

		cv::Canny(gfilter_img, edge_img, 180, 250);
		cv::Mat invert_img = ~edge_img;
		cv::distanceTransform(invert_img, dt_img, CV_DIST_L2, CV_DIST_MASK_PRECISE);
#if image_show
		cv::Mat detect_img_show = part_img.clone();
		cv::Mat dt_img_show = dt_img.clone();
#endif

		//3.2 precise rotation angle extracted,

		// 与ROI提取的先后关系还需要进一步验证
		std::vector<cv::Mat> rotated_temps;
		std::vector<cv::Mat> match_results;
		cv::Point refine_location;
		double refine_angle = angle;
		int best_match_index = 0;
		double refine_step = 0.1;
		double refine_span = 1.0;
		getRotatedTemp(_template_image, rotated_temps, -angle - refine_span, -angle + refine_span, refine_step);
		
		
		matchAngleTemp(dt_img, rotated_temps, match_results, best_match_index, refine_location, CV_TM_CCORR);
		_refine_angle = (-angle - refine_span) + (best_match_index * refine_step);
		_loc.x = static_cast<float>(refine_location.x + precise_roi.x) + rotated_temps[best_match_index].cols / 2.0;
		_loc.y = static_cast<float>(refine_location.y + precise_roi.y) + rotated_temps[best_match_index].rows / 2.0;
#if image_show
		cv::RotatedRect pyr_rect_show = cv::RotatedRect(precise_location.second * _pyr_scale, cv::Size(_size_type_x, _size_type_y), angle);
		common::drawRect(detect_img_show, pyr_rect_show, detect_img_show);
		cv::RotatedRect raw_rect_show = cv::RotatedRect(cv::Point2f(_loc.x, _loc.y), cv::Size(_size_type_x, _size_type_y), -_refine_angle);
		common::drawRect(detect_img_show, raw_rect_show, detect_img_show,cv::Scalar(255, 0, 0));
#endif
		process_timer.out("PolytonTest Process: ");

		// angle range check
		if (!checkResult(result_offset, errorcode)) {
			return false;
		}


		return true;
	}
}//polygon
