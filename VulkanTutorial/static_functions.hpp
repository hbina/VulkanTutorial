#pragma once

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include "helper_algorithms.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static constexpr auto
hasStencilComponent(const VkFormat& format) -> bool
{
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         format == VK_FORMAT_D24_UNORM_S8_UINT;
}

static auto
checkValidationLayersSupported(const std::vector<const char*>& validationLayers)
  -> bool
{
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  return any_of_range(std::cbegin(validationLayers),
                      std::cend(validationLayers),
                      std::cbegin(availableLayers),
                      std::cend(availableLayers),
                      [](const char* lhs, const VkLayerProperties& rhs) {
                        return strcmp(lhs, rhs.layerName) == 0;
                      });
}

// Choose the optimal swap chain.
// In particular, we have to consider the following:
// 1. Surface format (color depth).
// 2. Presentation mode (conditions for "swapping" images to the screen).
// 3. Swap extent (resolution of images in swap chain).
VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
  // NOTE: Read about the actual difference betweeen color space and color
  // format.
  // Choose SRGB color space:
  // https://stackoverflow.com/questions/12524623/what-are-the-practical-differences-when-working-with-colors-in-a-linear-vs-a-no
  // Because of this, we choose SRGB color format
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  // TODO: This operation might fail.
  // Every caller to this function must supply a vector of size at least 1.
  return availableFormats.front();
}

// Choose available presentation mode.
// If possible, we would like to have `VK_PRESENT_MODE_MAILBOX_KHR` to have
// triple buffering. Defaults to `VK_PRESENT_MODE_FIFO_KHR` which is
// guaranteed to be supported.
VkPresentModeKHR
chooseSwapPresentMode(
  const std::vector<VkPresentModeKHR>& availablePresentModes)
{
  for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

// Get Vulkan extennsions required for/by:
//  1.  GLFW.
//  2.  Debugging validation layers.
static auto
getRequiredExtensions(const bool& enableValidationLayer)
  -> std::vector<const char*>
{
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions = nullptr;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (enableValidationLayer) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

static void
printDeviceProperties(const VkPhysicalDevice& device)
{
  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  std::cout << "device apiVersion:" << deviceProperties.apiVersion << "\n";
  std::cout << "device deviceID:" << deviceProperties.deviceID << "\n";
  std::cout << "device deviceName:" << deviceProperties.deviceName << "\n";
  std::cout << "device deviceType:" << deviceProperties.deviceType << "\n";
  std::cout << "device driverVersion:" << deviceProperties.driverVersion
            << "\n";
  // TODO: Implement the overload for this
  // std::cout << "device limits:" << deviceProperties.limits << "\n";
  // TODO: Figure out how to print this.
  // std::cout << "device pipelineCacheUUID:" <<
  // deviceProperties.pipelineCacheUUID
  //          << "\n";
  // TODO: Implement the overload for this
  // std::cout << "device sparseProperties:" <<
  // deviceProperties.sparseProperties
  //          << "\n";
  std::cout << "device vendorID:" << deviceProperties.vendorID << "\n";
}

static void
printDeviceFeatures(const VkPhysicalDevice& device)
{
  VkPhysicalDeviceFeatures deviceFeatures{};
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  (void)deviceFeatures; // Silence unused warning
  // TODO: Implement the print here...
}

// Loading SPIR-V shaders
static std::vector<char>
readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file! filename:" + filename);
  }

  std::size_t fileSize = static_cast<std::size_t>(file.tellg());
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();
  return buffer;
}

// Helper function to create `VkDebugUtilsMessengerEXT`.
static auto
CreateDebugUtilsMessengerEXT(
  const VkInstance& instance,
  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDebugUtilsMessengerEXT* pDebugMessenger) -> VkResult
{
  // The function `vkCreateDebugUtilsMessengerEXT` is an extension.
  // Therefore, we have to find the address of this function ourselves.
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
    instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

// Helper function to find and execute `vkDestroyDebugUtilsMessengerEXT`.
static void
DestroyDebugUtilsMessengerEXT(const VkInstance& instance,
                              const VkDebugUtilsMessengerEXT& debugMessenger,
                              const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
    instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData)
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << "\n";
  return VK_FALSE;
}

static constexpr VkDebugUtilsMessengerCreateInfoEXT
getDebugMessengerCreateInfo()
{
  VkDebugUtilsMessengerCreateInfoEXT result = {};
  result.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  result.pNext = nullptr;
  result.flags = 0;
  result.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  result.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  result.pfnUserCallback = debugCallback;
  result.pUserData = nullptr; // Optional
  return result;
}

static auto
checkDeviceSupportAllRequiredExtensions(
  VkPhysicalDevice device,
  const std::vector<const char*>& deviceExtensions) -> bool
{
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(
    device, nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(
    device, nullptr, &extensionCount, availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                           deviceExtensions.end());

  for (const VkExtensionProperties& extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}
