// ppp_geometry_py: the C++ RANSAC for Python (place_node, and vision/ offline playgrounds). Same call shape as
// vision.spatial.fit_plane_ransac(): (plane[4], inlier bool mask) or (None, None).
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ppp_geometry/ransac.hpp"

namespace py = pybind11;
using ppp_geometry::Points;
using ppp_geometry::RansacParams;

static RansacParams params(double dist, int iters, const Eigen::Vector3d & up, double tilt, std::size_t min_in,
                           uint64_t seed)
{
  RansacParams p;
  p.dist_thresh = dist;
  p.iters = iters;
  p.up = up;
  p.max_tilt_deg = tilt;
  p.min_inliers = min_in;
  p.seed = seed;
  return p;
}

static py::array_t<bool> mask(const std::vector<uint8_t> & m)
{
  py::array_t<bool> out(m.size());
  auto r = out.mutable_unchecked<1>();
  for (std::size_t i = 0; i < m.size(); ++i) {
    r(i) = m[i] != 0;
  }
  return out;
}

PYBIND11_MODULE(ppp_geometry_py, m)
{
  m.doc() = "C++ plane RANSAC shared by stages 3 and 5 (ppp_geometry)";
  m.def(
    "fit_plane_ransac",
    [](const Points & pts, double dist_thresh, int iters, const Eigen::Vector3d & up, double max_tilt_deg,
       std::size_t min_inliers, uint64_t seed) -> py::tuple {
      std::optional<ppp_geometry::PlaneFit> fit;
      {
        py::gil_scoped_release release;
        fit = ppp_geometry::fit_plane_ransac(pts, params(dist_thresh, iters, up, max_tilt_deg, min_inliers, seed));
      }
      if (!fit) {
        return py::make_tuple(py::none(), py::none());
      }
      return py::make_tuple(Eigen::Vector4d(fit->plane), mask(fit->inliers));
    },
    py::arg("points"), py::arg("dist_thresh") = 0.006, py::arg("iters") = 300,
    py::arg("up") = Eigen::Vector3d::UnitZ(), py::arg("max_tilt_deg") = 20.0, py::arg("min_inliers") = 200,
    py::arg("seed") = 0,
    "Largest plane within max_tilt_deg of `up` -> (plane[4], inlier mask) or (None, None).");
  m.def(
    "find_planes",
    [](const Points & pts, double dist_thresh, int iters, const Eigen::Vector3d & up, double max_tilt_deg,
       std::size_t min_inliers, int max_planes) -> py::list {
      std::vector<ppp_geometry::PlaneFit> fits;
      {
        py::gil_scoped_release release;
        fits = ppp_geometry::find_planes(pts, params(dist_thresh, iters, up, max_tilt_deg, min_inliers, 0),
                                         max_planes);
      }
      py::list out;
      for (const auto & f : fits) {
        out.append(py::make_tuple(Eigen::Vector4d(f.plane), mask(f.inliers)));
      }
      return out;
    },
    py::arg("points"), py::arg("dist_thresh") = 0.008, py::arg("iters") = 200,
    py::arg("up") = Eigen::Vector3d::UnitZ(), py::arg("max_tilt_deg") = 10.0, py::arg("min_inliers") = 800,
    py::arg("max_planes") = 6,
    "Sequential RANSAC -> [(plane[4], inlier mask over `points`)], in the order found.");
}
