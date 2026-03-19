#include "feedback.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <filesystem>

namespace fs = std::filesystem;

namespace pase {

namespace {

const char* strategy_csv_name(Strategy s) {
  switch (s) {
    case Strategy::INSERTION_OPT:
      return "INSERTION_OPT";
    case Strategy::RUN_MERGE_OPT:
      return "RUN_MERGE_OPT";
    case Strategy::THREE_WAY_QS:
      return "THREE_WAY_QS";
    case Strategy::INTROSORT:
      return "INTROSORT";
    case Strategy::GPU_SORT:
      return "GPU_SORT";
  }
  return "UNKNOWN";
}

void ensure_env_feedback_default(FeedbackLogger& log) {
  static std::once_flag once;
  std::call_once(once, [&log] {
    const char* e = std::getenv("PASE_FEEDBACK");
    if (e && e[0] == '1' && e[1] == '\0') {
      log.set_enabled(true);
    }
  });
}

}  // namespace

FeedbackLogger& global_feedback_logger() {
  static FeedbackLogger instance;
  ensure_env_feedback_default(instance);
  return instance;
}

void set_feedback_logging(bool enabled) {
  global_feedback_logger().set_enabled(enabled);
}

bool feedback_logging_enabled() {
  return global_feedback_logger().enabled();
}

void FeedbackLogger::log(const SortLog& entry) {
  if (!enabled_) {
    return;
  }

  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  const char* home = std::getenv("HOME");
  if (!home) {
    return;
  }

  fs::path dir = fs::path(home) / ".pase";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    return;
  }

  fs::path file = dir / "sort_log.csv";
  const bool exists = fs::exists(file);
  std::ofstream out(file, std::ios::app);
  if (!out) {
    return;
  }

  if (!exists) {
    out << "sortedness,duplicate_ratio,entropy,avg_run_length,n,strategy,"
           "pred_cpu_ms,pred_gpu_ms,actual_ms,prediction_correct\n";
  }

  out << std::fixed << std::setprecision(6) << entry.sortedness << ','
      << entry.duplicate_ratio << ',' << entry.entropy << ','
      << entry.avg_run_length << ',' << entry.n << ','
      << strategy_csv_name(entry.chosen_strategy) << ','
      << entry.predicted_cpu_ms << ',' << entry.predicted_gpu_ms << ','
      << entry.actual_ms << ',' << (entry.prediction_correct ? 1 : 0) << '\n';
}

}  // namespace pase
