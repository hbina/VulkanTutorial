#pragma once

#include <vulkan/vulkan.h>

#include <vector>

// Struct for querying details of swap chain support.
struct SwapChainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};