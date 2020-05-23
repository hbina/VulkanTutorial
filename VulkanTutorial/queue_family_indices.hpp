#pragma once

#include <optional>

// Helper struct to find queue families.
struct QueueFamilyIndices
{
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  constexpr auto isComplete() const -> bool
  {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};
