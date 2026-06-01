/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "keyframe.h"
#include "deep_feature.h"

#include <cmath>
template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	has_deep_features = false;
	has_good_point_deep_features = false;
	has_vprnet_descriptor = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	good_point_deep_features.resize(259, 0);
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	has_deep_features = false;
	has_good_point_deep_features = false;
	has_vprnet_descriptor = false;
	sequence = 0;
	good_point_deep_features.resize(259, 0);
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

void KeyFrame::setDeepFeatures(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
{
	deep_features = features;
	has_deep_features = deep_features.cols() > 0;
}

bool KeyFrame::hasDeepFeatures() const
{
	return has_deep_features;
}

const Eigen::Matrix<float, 259, Eigen::Dynamic> &KeyFrame::getDeepFeatures() const
{
	return deep_features;
}

void KeyFrame::setGoodPointDeepFeatures(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features)
{
	good_point_deep_features = features;
	has_good_point_deep_features = good_point_deep_features.cols() > 0;
}

bool KeyFrame::hasGoodPointDeepFeatures() const
{
	return has_good_point_deep_features;
}

const Eigen::Matrix<float, 259, Eigen::Dynamic> &KeyFrame::getGoodPointDeepFeatures() const
{
	return good_point_deep_features;
}

void KeyFrame::setVPRNetDescriptor(const std::vector<float> &descriptor)
{
	vprnet_descriptor = descriptor;
	has_vprnet_descriptor = !vprnet_descriptor.empty();
}

bool KeyFrame::hasVPRNetDescriptor() const
{
	return has_vprnet_descriptor;
}

const std::vector<float> &KeyFrame::getVPRNetDescriptor() const
{
	return vprnet_descriptor;
}


void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}


void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    PnP_T_old.setZero();
    PnP_R_old.setIdentity();

    if (matched_2d_old_norm.size() < 4 || matched_3d.size() < 4 || matched_2d_old_norm.size() != matched_3d.size())
    {
        status.assign(matched_2d_old_norm.size(), 0);
        return;
    }

	//for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	//printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    try
    {
        if (CV_MAJOR_VERSION < 3)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
        else
        {
            if (CV_MINOR_VERSION < 2)
                solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
            else
                solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
        }
    }
    catch (const cv::Exception &e)
    {
        printf("[loop_fusion][pnp] failed: %s\n", e.what());
        status.assign(matched_2d_old_norm.size(), 0);
        return;
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}


bool KeyFrame::findConnection(KeyFrame* old_kf, DeepFeature* deep_feature)
{
	TicToc tmp_t;
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<uchar> status;
	bool use_deep_match = deep_feature && hasDeepFeatures() && old_kf->hasGoodPointDeepFeatures();

	if (use_deep_match)
	{
		std::vector<cv::DMatch> deep_matches;
		if (deep_feature->matchPoints(old_kf->getGoodPointDeepFeatures(), getDeepFeatures(), deep_matches, true))
		{
			for (const auto &match : deep_matches)
			{
				const int old_idx = match.queryIdx;
				const int cur_idx = match.trainIdx;
				if (old_idx < 0 || old_idx >= old_kf->getGoodPointDeepFeatures().cols())
					continue;
				if (cur_idx < 0 || cur_idx >= getDeepFeatures().cols())
					continue;

				const cv::Point2f old_uv(
					old_kf->getGoodPointDeepFeatures()(1, old_idx),
					old_kf->getGoodPointDeepFeatures()(2, old_idx));
				const cv::Point2f cur_uv(
					getDeepFeatures()(1, cur_idx),
					getDeepFeatures()(2, cur_idx));
				Eigen::Vector3d old_norm_p, cur_norm_p;
				m_camera->liftProjective(Eigen::Vector2d(old_uv.x, old_uv.y), old_norm_p);
				m_camera->liftProjective(Eigen::Vector2d(cur_uv.x, cur_uv.y), cur_norm_p);
				const cv::Point2f old_uv_norm(old_norm_p.x() / old_norm_p.z(), old_norm_p.y() / old_norm_p.z());
				const cv::Point2f cur_uv_norm(cur_norm_p.x() / cur_norm_p.z(), cur_norm_p.y() / cur_norm_p.z());

				if (old_idx < 0 || old_idx >= static_cast<int>(old_kf->point_3d.size()))
					continue;

				matched_3d.push_back(old_kf->point_3d[old_idx]);
				matched_2d_old.push_back(old_uv);
				matched_2d_old_norm.push_back(old_uv_norm);
				matched_2d_cur.push_back(cur_uv);
				matched_2d_cur_norm.push_back(cur_uv_norm);
			}
		}
		printf("[loop_fusion][deep] cur=%d old=%d raw_deep_matches=%zu geom_matches=%zu\n",
		       index, old_kf->index, deep_matches.size(), matched_3d.size());
	}

	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if (use_deep_match)
	{
		if ((int)matched_2d_cur.size() <= MIN_LOOP_NUM)
		{
			printf("[loop_fusion][deep] cur=%d old=%d fail: matched_2d=%zu <= MIN_LOOP_NUM=%d\n",
			       index, old_kf->index, matched_2d_cur.size(), MIN_LOOP_NUM);
			return false;
		}

		Eigen::Vector3d PnP_T_cur;
		Eigen::Matrix3d PnP_R_cur;
		status.clear();
	    PnPRANSAC(matched_2d_cur_norm, matched_3d, status, PnP_T_cur, PnP_R_cur);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);

	    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	    {
	        Vector3d old_T, cur_T;
	        Matrix3d old_R, cur_R;
	        old_kf->getPose(old_T, old_R);
	        cur_T = PnP_T_cur;
	        cur_R = PnP_R_cur;
	        relative_t = old_R.transpose() * (cur_T - old_T);
	        relative_q = old_R.transpose() * cur_R;
	        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(cur_R).x() - Utility::R2ypr(old_R).x());
	        if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	        {
	        	has_loop = true;
	        	loop_index = old_kf->index;
	        	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	        	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	        	             relative_yaw;
	        	printf("[loop_fusion][deep] cur=%d old=%d success inliers=%zu rel_t_norm=%.3f rel_yaw=%.3f\n",
	        	       index, old_kf->index, matched_2d_cur.size(), relative_t.norm(), relative_yaw);
	            return true;
	        }
	    }
	    printf("[loop_fusion][deep] cur=%d old=%d fail after pnp inliers=%zu rel_t_norm=%.3f rel_yaw=%.3f, skip this loop\n",
	           index, old_kf->index, matched_2d_cur.size(), relative_t.norm(), relative_yaw);
	    return false;
	}

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);

	status.clear();
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	}

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {
	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	        printf("[loop_fusion][brief] cur=%d old=%d success inliers=%zu rel_t_norm=%.3f rel_yaw=%.3f\n",
	               index, old_kf->index, matched_2d_cur.size(), relative_t.norm(), relative_yaw);
	        return true;
	    }
	}
	printf("[loop_fusion][brief] cur=%d old=%d fail inliers=%zu rel_t_norm=%.3f rel_yaw=%.3f\n",
	       index, old_kf->index, matched_2d_cur.size(), relative_t.norm(), relative_yaw);
	return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}
