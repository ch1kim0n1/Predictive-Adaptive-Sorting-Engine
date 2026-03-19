#include <algorithm>
#include <random>
#include <vector>

enum class DatasetType {
  sorted,
  reverse,
  nearly_sorted_95,
  nearly_sorted_80,
  random,
  heavy_dup,
  clustered,
  pipe_organ,
  large_random_1M,
  large_random_10M
};

void generate_dataset(std::vector<int>& out, DatasetType type, size_t n,
                     unsigned seed = 42) {
  std::mt19937 rng(seed);
  out.resize(n);

  switch (type) {
    case DatasetType::sorted:
      std::iota(out.begin(), out.end(), 0);
      break;

    case DatasetType::reverse:
      std::iota(out.rbegin(), out.rend(), 0);
      break;

    case DatasetType::nearly_sorted_95: {
      std::iota(out.begin(), out.end(), 0);
      int swaps = static_cast<int>(n * 0.05);
      for (int i = 0; i < swaps; i++) {
        int a = rng() % n;
        int b = rng() % n;
        std::swap(out[a], out[b]);
      }
      break;
    }

    case DatasetType::nearly_sorted_80: {
      std::iota(out.begin(), out.end(), 0);
      int swaps = static_cast<int>(n * 0.20);
      for (int i = 0; i < swaps; i++) {
        int a = rng() % n;
        int b = rng() % n;
        std::swap(out[a], out[b]);
      }
      break;
    }

    case DatasetType::random: {
      std::iota(out.begin(), out.end(), 0);
      std::shuffle(out.begin(), out.end(), rng);
      break;
    }

    case DatasetType::heavy_dup: {
      std::uniform_int_distribution<int> dist(0, 9);
      for (size_t i = 0; i < n; i++) {
        out[i] = dist(rng);
      }
      break;
    }

    case DatasetType::clustered: {
      std::uniform_int_distribution<int> cluster(0, 4);
      std::uniform_int_distribution<int> offset(0, 100);
      for (size_t i = 0; i < n; i++) {
        int c = cluster(rng);
        out[i] = c * 1000 + offset(rng);
      }
      break;
    }

    case DatasetType::pipe_organ: {
      size_t mid = n / 2;
      for (size_t i = 0; i < mid; i++) {
        out[i] = static_cast<int>(i);
      }
      for (size_t i = mid; i < n; i++) {
        out[i] = static_cast<int>(n - i - 1);
      }
      break;
    }

    case DatasetType::large_random_1M: {
      n = 1000000;
      out.resize(n);
      std::iota(out.begin(), out.end(), 0);
      std::shuffle(out.begin(), out.end(), rng);
      break;
    }

    case DatasetType::large_random_10M: {
      n = 10000000;
      out.resize(n);
      std::iota(out.begin(), out.end(), 0);
      std::shuffle(out.begin(), out.end(), rng);
      break;
    }
  }
}

const char* dataset_name(DatasetType type) {
  switch (type) {
    case DatasetType::sorted:
      return "sorted";
    case DatasetType::reverse:
      return "reverse";
    case DatasetType::nearly_sorted_95:
      return "nearly_sorted_95";
    case DatasetType::nearly_sorted_80:
      return "nearly_sorted_80";
    case DatasetType::random:
      return "random";
    case DatasetType::heavy_dup:
      return "heavy_dup";
    case DatasetType::clustered:
      return "clustered";
    case DatasetType::pipe_organ:
      return "pipe_organ";
    case DatasetType::large_random_1M:
      return "large_random_1M";
    case DatasetType::large_random_10M:
      return "large_random_10M";
  }
  return "unknown";
}
