#pragma once

#include <stdexcept>

// Helper struct to find queue families.
class QueueFamilyIndices
{
  uint32_t* graphicsFamily = nullptr;
  uint32_t* presentFamily = nullptr;

public:
  constexpr auto isComplete() const -> bool
  {
    return graphicsFamily != nullptr && presentFamily != nullptr;
  }

  void assignGraphicsFamily(const uint32_t value)
  {
    graphicsFamily = new uint32_t(value);
  }

  void assignPresentFamily(const uint32_t value)
  {
    presentFamily = new uint32_t(value);
  }

  uint32_t getGraphicsFamily() const
  {
    if (graphicsFamily == nullptr) {
      throw std::runtime_error(
        "graphics family have not been assigned a value.");
    }

    return *graphicsFamily;
  }

  uint32_t getPresentFamily() const
  {
    if (presentFamily == nullptr) {
      throw std::runtime_error(
        "graphics family have not been assigned a value.");
    }

    return *presentFamily;
  }

  ~QueueFamilyIndices()
  {
    delete graphicsFamily;
    delete presentFamily;
  }
};
