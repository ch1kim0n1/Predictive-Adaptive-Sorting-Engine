#include "config_loader.h"
#include "threshold_tuner.h"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

TEST(ConfigLoader, MissingFileLeavesDefaults) {
  pase::Dispatcher::Thresholds th;
  double win = 0.85;
  const bool ok =
      pase::load_pase_config_file("/nonexistent/pase_config_xxx.json", th, win);
  EXPECT_FALSE(ok);
  EXPECT_FLOAT_EQ(th.sorted, 0.90f);
  EXPECT_EQ(th.run_merge, 32);
  EXPECT_FLOAT_EQ(th.dup, 0.32f);
  EXPECT_EQ(th.min_gpu, 250000);
  EXPECT_DOUBLE_EQ(win, 0.85);
}

TEST(ConfigLoader, PartialJsonOverrides) {
  const auto path =
      std::filesystem::temp_directory_path() / "pase_cfg_loader_test.json";
  {
    std::ofstream out(path);
    out << R"({"sorted": 0.88, "gpu_win_factor": 0.72})";
  }

  pase::Dispatcher::Thresholds th;
  double win = 0.85;
  const bool ok = pase::load_pase_config_file(path.string(), th, win);
  EXPECT_TRUE(ok);
  EXPECT_FLOAT_EQ(th.sorted, 0.88f);
  EXPECT_EQ(th.run_merge, 32);
  EXPECT_DOUBLE_EQ(win, 0.72);

  std::filesystem::remove(path);
}

TEST(ConfigLoader, ConservativeDispatchThresholdKeys) {
  const auto path =
      std::filesystem::temp_directory_path() / "pase_conservative_cfg.json";
  {
    std::ofstream out(path);
    out << R"({"dup_border_band": 0.05, "run_merge_border": 4,
              "conservative_specialist_frac": 0.91})";
  }
  pase::Dispatcher::Thresholds th;
  double win = 0.85;
  ASSERT_TRUE(pase::load_pase_config_file(path.string(), th, win));
  EXPECT_FLOAT_EQ(th.dup_border_band, 0.05f);
  EXPECT_EQ(th.run_merge_border, 4);
  EXPECT_FLOAT_EQ(th.conservative_specialist_frac, 0.91f);
  std::filesystem::remove(path);
}

TEST(ConfigLoader, CostFitPartialJson) {
  const auto path =
      std::filesystem::temp_directory_path() / "pase_cost_fit_test.json";
  {
    std::ofstream out(path);
    out << R"({"cost_fit": {"introsort": 1.15, "gpu_kernel": 0.75}})";
  }
  pase::Dispatcher::Thresholds th;
  double win = 0.85;
  pase::CostModelFit fit;
  ASSERT_TRUE(pase::load_pase_config_file(path.string(), th, win, &fit));
  EXPECT_DOUBLE_EQ(fit.scale_introsort, 1.15);
  EXPECT_DOUBLE_EQ(fit.gpu_kernel_scale, 0.75);
  EXPECT_FLOAT_EQ(th.sorted, 0.90f);
  std::filesystem::remove(path);
}

TEST(ConfigLoader, InvalidJsonReturnsFalse) {
  const auto path =
      std::filesystem::temp_directory_path() / "pase_bad_json.json";
  {
    std::ofstream out(path);
    out << "{ not valid json <<<";
  }
  pase::Dispatcher::Thresholds th;
  double win = 0.85;
  EXPECT_FALSE(pase::load_pase_config_file(path.string(), th, win));
  std::filesystem::remove(path);
}

TEST(ConfigLoader, SetGpuWinFactorClamps) {
  pase::global_threshold_tuner().set_gpu_win_factor(0.05);
  EXPECT_DOUBLE_EQ(pase::global_threshold_tuner().gpu_win_factor(), 0.55);
  pase::global_threshold_tuner().set_gpu_win_factor(2.0);
  EXPECT_DOUBLE_EQ(pase::global_threshold_tuner().gpu_win_factor(), 0.95);
  pase::global_threshold_tuner().set_gpu_win_factor(0.85);
}
