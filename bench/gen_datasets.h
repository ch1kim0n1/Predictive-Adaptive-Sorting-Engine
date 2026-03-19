#pragma once

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
                     unsigned seed = 42);

const char* dataset_name(DatasetType type);
