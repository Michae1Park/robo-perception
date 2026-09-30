#include "ppp_geometry/ransac.hpp"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cmath>
#include <random>

namespace ppp_geometry
{

Plane refine_plane(const Points & pts, const std::vector<uint8_t> * mask)
{
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  std::size_t n = 0;
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    if (!mask || (*mask)[i]) {
      c += pts.row(i).transpose();
      ++n;
    }
  }
  c /= static_cast<double>(n);
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    if (!mask || (*mask)[i]) {
      const Eigen::Vector3d d = pts.row(i).transpose() - c;
      cov += d * d.transpose();
    }
  }
  // smallest singular vector of the centred points = eigenvector of the scatter matrix with the smallest value
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
  const Eigen::Vector3d nrm = svd.matrixU().col(2).normalized();
  Plane pl;
  pl << nrm, -nrm.dot(c);
  return pl;
}

static std::size_t mark_inliers(const Plane & pl, const Points & pts, double thresh, std::vector<uint8_t> * out)
{
  std::size_t count = 0;
  if (out) {
    out->assign(pts.rows(), 0);
  }
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    const double d = pts(i, 0) * pl[0] + pts(i, 1) * pl[1] + pts(i, 2) * pl[2] + pl[3];
    if (std::abs(d) < thresh) {
      ++count;
      if (out) {
        (*out)[i] = 1;
      }
    }
  }
  return count;
}

std::optional<PlaneFit> fit_plane_ransac(const Points & pts, const RansacParams & p)
{
  const std::size_t n = pts.rows();
  if (n < std::max<std::size_t>(3, p.min_inliers)) {
    return std::nullopt;
  }
  const Eigen::Vector3d up = p.up.normalized();
  const double cos_tilt = std::cos(p.max_tilt_deg * M_PI / 180.0);
  std::mt19937_64 rng(p.seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);

  Plane best = Plane::Zero();
  std::size_t best_count = 0;
  for (int it = 0; it < p.iters; ++it) {
    std::size_t a = pick(rng), b = pick(rng), c = pick(rng);
    if (a == b || b == c || a == c) {
      continue;
    }
    const Eigen::Vector3d p0 = pts.row(a), p1 = pts.row(b), p2 = pts.row(c);
    Eigen::Vector3d nrm = (p1 - p0).cross(p2 - p0);
    const double norm = nrm.norm();
    if (norm < 1e-9) {
      continue;
    }
    nrm /= norm;
    if (std::abs(nrm.dot(up)) < cos_tilt) {
      continue;
    }
    Plane pl;
    pl << nrm, -nrm.dot(p0);
    const std::size_t count = mark_inliers(pl, pts, p.dist_thresh, nullptr);
    if (count > best_count) {
      best = pl;
      best_count = count;
    }
  }
  if (best_count == 0 || best_count < p.min_inliers) {
    return std::nullopt;
  }
  PlaneFit fit;
  mark_inliers(best, pts, p.dist_thresh, &fit.inliers);
  fit.plane = refine_plane(pts, &fit.inliers);
  if (fit.plane.head<3>().dot(up) < 0) {
    fit.plane = -fit.plane;
  }
  fit.count = mark_inliers(fit.plane, pts, p.dist_thresh, &fit.inliers);
  return fit;
}

std::vector<PlaneFit> find_planes(const Points & pts, const RansacParams & p, int max_planes)
{
  std::vector<PlaneFit> out;
  std::vector<Eigen::Index> idx(pts.rows());
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    idx[i] = i;
  }
  for (int k = 0; k < max_planes; ++k) {
    if (idx.size() < p.min_inliers) {
      break;
    }
    Points rest(idx.size(), 3);
    for (std::size_t i = 0; i < idx.size(); ++i) {
      rest.row(i) = pts.row(idx[i]);
    }
    RansacParams pk = p;
    pk.seed = static_cast<uint64_t>(k);
    auto fit = fit_plane_ransac(rest, pk);
    if (!fit) {
      break;
    }
    PlaneFit full;
    full.plane = fit->plane;
    full.count = fit->count;
    full.inliers.assign(pts.rows(), 0);
    std::vector<Eigen::Index> keep;
    for (std::size_t i = 0; i < idx.size(); ++i) {
      if (fit->inliers[i]) {
        full.inliers[idx[i]] = 1;
      } else {
        keep.push_back(idx[i]);
      }
    }
    out.push_back(std::move(full));
    idx.swap(keep);
  }
  return out;
}

Eigen::VectorXd height_above(const Plane & plane, const Points & pts)
{
  return (pts * plane.head<3>()).array() + plane[3];
}

}  // namespace ppp_geometry
