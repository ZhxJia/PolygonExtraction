#pragma once

#include <numeric>
#include "math.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace polygon {
	class PolygonProcessor {
	public:
		PolygonProcessor() {
			_pyr_scale = 2.0;
		};
		PolygonProcessor(double camera_scale_x, double camera_scale_y, float pyr_scale = 2.0)
			:_camera_scale_x(camera_scale_x), _camera_scale_y(camera_scale_y), _pyr_scale(pyr_scale) {};
		~PolygonProcessor() {};

		bool Init(const VISION_PART_POLYGON* polygon, double camera_scale_x, double camera_scale_y);


		bool drawTemplate(VISION_PART_POLYGON* polygon, const double& pyrdown);
		//@description: gray matching based on binary image
		bool Process(const cv::Mat& roi_image, PartResult_Offset* result_offset, PartResult_TimSco* result_timsco, wdErrorCode& errorcode);

	private:
		bool drawArc(cv::Mat& src_img, cv::Point2f arc_ct, double arc_radius, double st_angle, double angle_span);
		bool checkBrightness(const cv::Mat& src_img,const int lower_bound, const int upper_bound, const int gray_thresh, wdErrorCode& errorcode);
		bool calCoareseAngle(const cv::Mat& image_edge, double& angle, wdErrorCode& errorcode);
		bool calCoareseAngle2(const cv::Mat& image_edge,const cv::Mat& image_template ,double& angle, wdErrorCode& errorcode);
		cv::Mat arrangeMat(const cv::Mat& src);
		void dftImg(const cv::Mat& src_img, cv::Mat& dft_img, const int dft_size);
		void Recomb(cv::Mat& src, cv::Mat& dst);
		void ForwardFFT(const cv::Mat& src_img, cv::Mat& mag_dft, int dft_size_w, int dft_size_h, bool do_recomb = true);
		void linearPolar(const cv::Mat& src, cv::Mat& dst, const cv::Point2f& cneter, const int MaxRadius, double& dAngle);
		void ImageWarpAffine(const cv::Mat& src_img, cv::Mat& dst_img, const double& angle, cv::Point2f& center, cv::Point2f& offset);
		void matchLocation(cv::Mat& match_result, std::pair<double, cv::Point2f>& location, int offset_x, int offset_y);
		void getRotatedTemp(const cv::Mat& src_temp_img, std::vector<cv::Mat>& rotated_temps, double l_thresh, double h_thresh, double step);
		void matchAngleTemp(const cv::Mat& src_img, const std::vector<cv::Mat>& templs, std::vector<cv::Mat>& match_result, int& best_index, cv::Point& best_loc, const int& method);
		//@description: ratio /in (0,1]
		void highpass(cv::Size& sz, cv::Mat& dst, double ratio = 0.95);
		void highpass(cv::Size& sz, cv::Mat& dst1, cv::Mat& dst2, double ratio1, double ratio2);
		bool Equal(double x, double target, double eps = 1e-6);
		bool Equal(float x, float target, float eps = 1e-6f);
		bool checkResult(PartResult_Offset* result_offset, wdErrorCode& errorcode);


	private:
		double _camera_scale_x;
		double _camera_scale_y;
		double srcimg_ctx;
		double srcimg_cty;
		cv::Mat _template_image;
		cv::Mat _template_pyrdown;
		double _pyr_scale;
		int _threshold;

		//示教所得属性
		double _size_type_x;
		double _size_type_y;
		double _wireframe_originX;
		double _wireframe_originY;
		double _wireframe_angle;
		int _vertex_num;

		cv::Mat dt_img; //distance transform image
		cv::Mat edge_img;
		
		cv::Mat pyr_edge; //edge of pyramid image
		cv::Mat pyr_dt; //distance transform image of pyramid image\

		cv::Mat precise_roi_img;
	
		
		cv::Point2f _loc; //final match result
		double _refine_angle;

		struct Line_Elem {
			Line_Elem(cv::Point _st, cv::Point _ed) : st(_st), ed(_ed) {};
			Line_Elem() {};
			cv::Point st;
			cv::Point ed;

		};

		struct Arc_Elem {
			Arc_Elem(cv::Point2f _ct, float _arc_radius, double _st_angle, double _angle_span) : ct(_ct), arc_radius(_arc_radius), st_angle(_st_angle), angle_span(_angle_span) {};
			Arc_Elem() {};
			cv::Point2f ct;
			float arc_radius;
			double st_angle;
			double angle_span;
		};

		std::vector<Line_Elem> line_elems;
		std::vector<Arc_Elem> arc_elems;


	};


} //polygon


