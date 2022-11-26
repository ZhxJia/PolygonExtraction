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
			arc_st_angle = 0.0; //Բ����ʼ�Ƕ�
			arc_ed_angle = 0.0; //Բ����ֹ�Ƕ�
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
		// @param [in]: fix_se��fix start/end point
		bool RobustArcFit(std::vector<cv::Point>& _contours_in, const int max_iters, float min_radius = 5.0f, float min_inlier_ratio = 0.85f, bool fix_se = false);
		bool getCircle(const cv::Point& p1, const cv::Point& p2, const cv::Point& p3, cv::Point2d& center, float& radius);
		float verifyCircle(const std::vector<cv::Point> contours, const cv::Point2d center, const float radius, std::vector<cv::Point>& inliers, bool fix_se = false);

		// @description: ���������ֱ���ϵ��ڵ����
		// @brief: group points into contiguous segments of constant curvature
		// @param [in]: l_pt1, l_pt2��two points on the line
		// @param [in]: pts��the set of points need to judge
		// @return: ratio of inliers
		float verifyLine(const cv::Point2d& l_pt1, const cv::Point2d& l_pt2, const std::vector<cv::Point>& pts);
		
		// @param [in]: precision: Բ�����ƽǶȾ���(����)
		void drawArc(cv::Mat& src_img, const cv::Point2d& _arc_ct, const double st_angle, const double angle_span, const double _arc_radius, const cv::Scalar& color, const double precision = 5.0);
		void drawArc(cv::Mat& src_img, const cv::Scalar& color) const;
		
		int judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& pt);
		int judgeArcDir(const cv::Point2d& st_pt, const cv::Point2d& ed_pt, const cv::Point2d& ct_pt ,const cv::Point2d& pt);

		
	public:
		std::vector<cv::Point> contours; //����������
		FIT_TYPE type; //�������
		cv::Point st; //��ʼ������
		cv::Point ed; //�յ�����
		cv::Point2d arc_ct; //Բ������
		double arc_radius; //�뾶
		double arc_st_angle; //Բ����ʼ�Ƕ�(rad)
		double arc_ed_angle; //Բ����ֹ�Ƕ�
		int arc_dir; //Բ������(˳���渺)
		double arc_span; //Բ�����(deg,˳���渺)
		//ֱ�߲���
		double A;
		double B;
		double C;
		bool is_closed;

		std::vector<cv::Point> inliers; //��������ڵ�


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
	
	// �߿�(���سߴ�)
	struct WireFrame {
		float origin_x;	 //�߿�ԭ�� �����ͼ������
		float origin_y;	 //�߿�ԭ��
		double angle;    //�߿�Ƕ�
		
		double body_x;      // ����x
		double body_y;		 // ����y
		int vertex_num;  //��������

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
