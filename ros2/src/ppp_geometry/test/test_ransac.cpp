#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "ppp_geometry/ransac.hpp"

using namespace ppp_geometry;

// Two horizontal boards (z = 0 and z = 0.32) + a vertical wall + noise, like a shelf seen from the front.
static Points shelf_cloud()
{
  std::mt19937 g(1);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  std::normal_distribution<double> noise(0.0, 0.001);
  std::vector<Eigen::Vector3d> v;
  for (int i = 0; i < 4000; ++i) v.emplace_back(u(g), u(g), noise(g));
  for (int i = 0; i < 2500; ++i) v.emplace_back(u(g), u(g), 0.32 + noise(g));
  for (int i = 0; i < 3000; ++i) v.emplace_back(u(g), 0.3 + noise(g), 0.3 + u(g));
  for (int i = 0; i < 500; ++i) v.emplace_back(u(g), u(g), 0.5 * (u(g) + 0.3));
  Points p(v.size(), 3);
  for (std::size_t i = 0; i < v.size(); ++i) p.row(i) = v[i];
  return p;
}

TEST(Ransac, LargestHorizontalPlane)
{
  RansacParams prm;
  prm.dist_thresh = 0.006;
  prm.max_tilt_deg = 10.0;
  auto fit = fit_plane_ransac(shelf_cloud(), prm);
  ASSERT_TRUE(fit.has_value());
  EXPECT_GT(fit->plane[2], 0.999);          // normal up
  EXPECT_NEAR(-fit->plane[3], 0.0, 0.002);  // z = 0 board (the bigger one), not the wall
  EXPECT_GE(fit->count, 3900u);
}

TEST(Ransac, SequentialFindsBothBoards)
{
  RansacParams prm;
  prm.dist_thresh = 0.008;
  prm.max_tilt_deg = 10.0;
  prm.min_inliers = 800;
  auto planes = find_planes(shelf_cloud(), prm, 6);
  ASSERT_EQ(planes.size(), 2u);             // the wall is vertical, the noise too sparse
  EXPECT_NEAR(-planes[0].plane[3], 0.0, 0.002);
  EXPECT_NEAR(-planes[1].plane[3], 0.32, 0.002);
}

TEST(Ransac, TooFewPoints)
{
  Points p(10, 3);
  p.setZero();
  RansacParams prm;
  EXPECT_FALSE(fit_plane_ransac(p, prm).has_value());
}
