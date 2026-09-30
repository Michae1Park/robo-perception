// Plane RANSAC shared by stage 3 (table plane, spatial_node) and stage 5 (shelf supports, place_node through the
// Python binding). C++ port of vision/spatial.py fit_plane_ransac() and the sequential loop of
// vision/place.py find_planes() (D-019). Results match the numpy version within tolerances, not bit for bit:
// the random samples differ.
#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <vector>

namespace ppp_geometry
{

using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;  // N x 3
using Plane = Eigen::Vector4d;                                              // a b c d, unit normal

struct PlaneFit
{
  Plane plane;                  // normal oriented along `up`
  std::vector<uint8_t> inliers; // one flag per input point
  std::size_t count = 0;
};

struct RansacParams
{
  double dist_thresh = 0.006;   // point-to-plane distance counted as inlier (m)
  int iters = 300;
  Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
  double max_tilt_deg = 20.0;   // only planes whose normal is within this of `up` (180 = any)
  std::size_t min_inliers = 200;
  uint64_t seed = 0;
};

// Least-squares plane through points (SVD of the centred points); unit normal, sign arbitrary.
Plane refine_plane(const Points & pts, const std::vector<uint8_t> * mask = nullptr);

// Largest plane whose normal is within max_tilt_deg of `up`, refit by least squares on its inliers, then the
// inliers recomputed. std::nullopt if fewer than min_inliers support the best plane.
std::optional<PlaneFit> fit_plane_ransac(const Points & pts, const RansacParams & p);

// Sequential RANSAC: fit the biggest plane, remove its inliers, repeat (seed k for the k-th plane, as in
// place.py). Stops after max_planes or when fewer than min_inliers points remain. Inlier flags index `pts`.
std::vector<PlaneFit> find_planes(const Points & pts, const RansacParams & p, int max_planes);

// Signed distance of each point above the plane.
Eigen::VectorXd height_above(const Plane & plane, const Points & pts);

}  // namespace ppp_geometry
