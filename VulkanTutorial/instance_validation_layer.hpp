#pragma once

#include <cstdint>
#include <vector>

struct InstanceValidationLayer
{
  const void* pNext = nullptr;

  static auto create() -> InstanceValidationLayer
  {
    return InstanceValidationLayer{};
  }

  static auto create(const void* pNext) -> InstanceValidationLayer
  {
    InstanceValidationLayer result;
    result.pNext = pNext;
    return result;
  }
};
