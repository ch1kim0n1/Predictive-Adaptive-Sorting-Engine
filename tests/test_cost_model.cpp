#include <gtest/gtest.h>
#include <cost_model.h>
#include <dispatcher.h>

using namespace pase;

TEST(CostModelTest, CalibrationPositiveThroughput) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  EXPECT_GT(cm.cpu_ops_per_ms(), 0.0);
}

TEST(CostModelTest, GpuEstimateIncreasesWithN) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  double a = cm.estimate_gpu(100000, 0.5f, sizeof(int));
  double b = cm.estimate_gpu(500000, 0.5f, sizeof(int));
  EXPECT_GT(b, a);
}

TEST(CostModelTest, DispatcherUsesInsertionWhenVerySorted) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  Dispatcher d;
  Profile p{};
  p.n = 50000;
  p.sample_rate = 0.015f;
  p.sortedness = 0.98f;
  p.duplicate_ratio = 0.0f;
  p.entropy = 0.1f;
  p.avg_run_length = 200;
  p.max_run_length = 500;
  EXPECT_EQ(d.select_strategy(p, cm, sizeof(int)), Strategy::INSERTION_OPT);
}

TEST(CostModelTest, BestCpuStrategyPicksThreeWayOnDuplicates) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  Profile p{};
  p.n = 100000;
  p.sortedness = 0.4f;
  p.duplicate_ratio = 0.55f;
  p.entropy = 0.5f;
  p.avg_run_length = 5;
  Strategy s = cm.best_cpu_strategy(p, 0.90f, 64, 0.40f);
  EXPECT_EQ(s, Strategy::THREE_WAY_QS);
}
