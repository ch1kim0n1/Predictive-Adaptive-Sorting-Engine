#include <gtest/gtest.h>
#include <cost_model.h>
#include <cost_model_fit.h>
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

TEST(CostModelTest, DispatcherUsesInsertionOnlyForSmallVerySorted) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  Dispatcher d;
  Profile p{};
  p.n = 320;
  p.sample_rate = 0.015f;
  p.sortedness = 0.98f;
  p.duplicate_ratio = 0.0f;
  p.entropy = 0.1f;
  p.avg_run_length = 200;
  p.max_run_length = 500;
  EXPECT_EQ(d.select_strategy(p, cm, sizeof(int), /*gpu_available=*/true,
                              /*gpu_win_factor=*/0.85),
            Strategy::INSERTION_OPT);
}

TEST(CostModelTest, DispatcherDoesNotInsertionSortHugeSorted) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  Dispatcher d;
  Profile p{};
  p.n = 100000;
  p.sample_rate = 0.015f;
  p.sortedness = 1.0f;
  p.duplicate_ratio = 0.0f;
  p.entropy = 0.2f;
  p.avg_run_length = 2000;
  p.max_run_length = 5000;
  EXPECT_NE(d.select_strategy(p, cm, sizeof(int), /*gpu_available=*/false,
                              /*gpu_win_factor=*/0.85),
            Strategy::INSERTION_OPT);
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
  Strategy s = cm.best_cpu_strategy(p, 0.90f, 64, 0.40f, 384);
  EXPECT_EQ(s, Strategy::THREE_WAY_QS);
}

TEST(CostModelTest,
     DispatcherConservativeNearDupBorderPrefersIntroUnlessLargeMargin) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  CostModelFit fit{};
  fit.scale_three_way = 1.2;
  cm.apply_fit(fit);

  Dispatcher d;
  Profile p{};
  p.n = 100000;
  p.sortedness = 0.45f;
  p.duplicate_ratio = 0.345f;
  p.entropy = 0.55f;
  p.avg_run_length = 5;

  ASSERT_EQ(cm.best_cpu_strategy(p, d.thresholds().sorted, d.thresholds().run_merge,
                                 d.thresholds().dup, d.thresholds().max_insertion_n),
            Strategy::THREE_WAY_QS);
  EXPECT_EQ(d.select_strategy(p, cm, sizeof(int), /*gpu_available=*/false,
                              /*gpu_win_factor=*/0.85),
            Strategy::INTROSORT);
}

TEST(CostModelTest,
     DispatcherConservativeNearRunMergeBorderPrefersIntroUnlessLargeMargin) {
  CostModel cm;
  CostModel::calibrate_with_int_sort(cm);
  CostModelFit fit{};
  fit.scale_run_merge = 1.5;
  cm.apply_fit(fit);

  Dispatcher d;
  Profile p{};
  p.n = 100000;
  p.sortedness = 0.45f;
  p.duplicate_ratio = 0.05f;
  p.entropy = 0.55f;
  p.avg_run_length = 34;

  ASSERT_EQ(cm.best_cpu_strategy(p, d.thresholds().sorted, d.thresholds().run_merge,
                                 d.thresholds().dup, d.thresholds().max_insertion_n),
            Strategy::RUN_MERGE_OPT);
  EXPECT_EQ(d.select_strategy(p, cm, sizeof(int), /*gpu_available=*/false,
                              /*gpu_win_factor=*/0.85),
            Strategy::INTROSORT);
}
