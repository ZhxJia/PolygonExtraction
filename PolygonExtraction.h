#pragma once
#include <opencv2/imgproc/imgproc.hpp>

namespace polygon {

	typedef enum {
		LINE = 0,
		ARC = 1,
	}FIT_TYPE;


	class CPolygon {

	public:
		CPolygon(const std::vector<cv::Point>& _contours, FIT_TYPE _type) {
			contours.assign(_contours.begin(), _contours.end());
			type = _type;
			arc_radius = 0.0;
			A = 0.0;
			B = 0.0;
			C = 0.0;
			arc_st_angle = 0.0; //圆弧起始角度
			arc_ed_angle = 0.0; //圆弧终止角度
			arc_dir = 0;
			is_closed = true;
		};
		~CPolygon() {};


		//bool CheckDeviation();
		//bool LSCircleFit();
		bool Point2LinePrj(const double A, const double B, const double C, const cv::Point& ext_pt, cv::Point2d& prj_pt);
		bool LineFit(const std::vector<cv::Point>& _contours_in, float min_inlier_ratio = 0.85);
		bool IRLSCircleFit(cv::InputArray _points, cv::OutputArray _circle, int distType, double param, double reps, double aeps);
		bool IRLSCircleFit(const std::vector<cv::Point>& _points, int distType, double param, double reps, double aeps);
		bool RobustPolyFit(std::vector<cv::Point>& inliers, int max_iters, float min_radius = 5.0f, float min_inlier_ratio = 0.2f, bool fix_se = false);
		bool CircleFitByPratt(const std::vector<cv::Point>& _points, double& sigma, bool err_analysis = false, int max_iters = 20);
		bool CircleFitByTaubin(const std::vector<cv::Point>& _points, double& sigma, bool err_analysis = false, int max_iters = 6);
		// @brief: ransac fit ploygon coeff
		// @param [in]: _contours_in: the set of points needed to fit
		// @param [in]: fix_se：fix start/end point
		bool RobustArcFit(std::vector<cv::Point>& _contours_in, const int max_iters, float min_radius = 5.0f, float min_inlier_ratio = 0.85f, bool fix_se = false);
		bool getCircle(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, cv::Point2d& center, float& radius);
		float verifyCircle(const std::vector<cv::Point> contours, const cv::Point2d center, const float radius, std::vector<cv::Point>& inliers, bool fix_se = false);

		// @description: 计算在拟合直线上的内点比例
		// @brief: group points into contiguous segments of constant curvature
		// @param [in]: l_pt1, l_pt2：two points on the line
		// @param [in]: pts：the set of points need to judge
		// @return: ratio of inliers
		float verifyLine(const cv::Point2d& l_pt1, const cv::Point2d& l_pt2, const std::vector<cv::Point>& pts);
		
		// @param [in]: precision: 圆弧绘制角度精度(步长)
		void drawArc(cv::Mat& src_img, const cv::Point2d& _arc_ct, const double st_angle, const double angle_span, const double _arc_radius, const cv::Scalar& color, const double precision = 5.0);
		void drawArc(cv::Mat& src_img, const cv::Scalar& color) const;
		
		int judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& pt);
		int judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& ct_pt ,const cv::Point2d& pt);

		
	public:
		std::vector<cv::Point> contours; //轮廓点数据
		FIT_TYPE type; //拟合类型
		cv::Point st; //起始点坐标
		cv::Point ed; //终点坐标
		cv::Point2d arc_ct; //圆心坐标
		double arc_radius; //半径
		double arc_st_angle; //圆弧起始角度(rad)
		double arc_ed_angle; //圆弧终止角度
		int arc_dir; //圆弧方向(顺正逆负)
		double arc_span; //圆弧跨度(deg,顺正逆负)
		//直线参数
		double A;
		double B;
		double C;
		bool is_closed;

		std::vector<cv::Point> inliers; //曲线拟合内点


	};

	struct Blip {
		Blip(int _start, int _len, int _val) {
			start = _start;
			len = _len;
			val = _val;
		}
		int start;
		int len;
		int val;
	};

	struct Fragment {
		Fragment(int _start, int _end, double _val) {
			start_index = _start;
			end_index = _end;
			len = 0;
			curvature = _val;
		}
		Fragment() {
			start_index = 0;
			end_index = 0;
			len = 0;
			curvature = 0.0;
		}

		bool operator <(const Fragment& frag) {
			if (len > frag.len) {
				return true;
			}
			else {
				return false;
			}
		}

		void operator =(const Fragment& frag) {
			start_index = frag.start_index;
			end_index = frag.end_index;
			curvature = frag.curvature;
			len = frag.len;
		}

		int start_index;
		int end_index;
		int len;
		double curvature;
	};
	
	// 线框(像素尺寸)
	struct WireFrame {
		float origin_x;	 //线框原点 相对于图像中心
		float origin_y;	 //线框原点
		double angle;    //线框角度
		
		double body_x;      // 本体x
		double body_y;		 // 本体y
		int vertex_num;  //顶点数量

		explicit WireFrame() {
			angle = 0.0;
			origin_x = 0;
			origin_y = 0;
			angle = 0.0;

			body_x = 0.0;
			body_y = 0.0;
		}
	};

}//polygon
