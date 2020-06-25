#pragma once

#include <vulkan/vulkan.h>

#include <vector>

// Struct for querying details of swap chain support.
// TODO: Should this be inside avk?
struct SwapChainSupportDetails
{

  SwapChainSupportDetails() = delete;

  // FIXME: Default values for this?
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;

  // TODO: We could cache the result of the devices...
  static auto get(const VkPhysicalDevice& physicalDevice,
                  const VkSurfaceKHR& surface) -> SwapChainSupportDetails
  {
    /// Query basic surface capabilities.
    VkSurfaceCapabilitiesKHR capabilities = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      physicalDevice, surface, &capabilities);

    /// Query the supported surface formats.
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
      physicalDevice, surface, &formatCount, nullptr);

    std::vector<VkSurfaceFormatKHR> formats;
    if (formatCount != 0) {
      formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
        physicalDevice, surface, &formatCount, formats.data());
    }

    /// Query supported presentations mode.
    std::vector<VkPresentModeKHR> presentModes;
    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
      physicalDevice, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, presentModes.data());
    }

    return SwapChainSupportDetails{ capabilities, formats, presentModes };
  }
};