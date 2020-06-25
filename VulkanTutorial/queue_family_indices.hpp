#pragma once

#include <stdexcept>
#include <variant>

// Helper struct to find queue families.
class QueueFamilyIndices
{
  std::variant<bool, uint32_t> graphicsFamily = false;
  std::variant<bool, uint32_t> presentFamily = false;

public:
  constexpr auto isComplete() const -> bool
  {
    return std::holds_alternative<uint32_t>(graphicsFamily) &&
           std::holds_alternative<uint32_t>(presentFamily);
  }

  void assignGraphicsFamily(const uint32_t value) { graphicsFamily = value; }

  void assignPresentFamily(const uint32_t value) { presentFamily = value; }

  uint32_t getGraphicsFamily() const
  {
    try {
      return std::get<uint32_t>(graphicsFamily);
    } catch (const std::bad_variant_access&) {
      throw std::runtime_error(
        "graphics family have not been assigned a value.");
    }
  }

  uint32_t getPresentFamily() const
  {
    try {
      return std::get<uint32_t>(presentFamily);
    } catch (const std::bad_variant_access&) {
      throw std::runtime_error(
        "graphics family have not been assigned a value.");
    }
  }
};
