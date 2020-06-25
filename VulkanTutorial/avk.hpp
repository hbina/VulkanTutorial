#pragma once

#include <vulkan/vulkan.h>

#include "instance_validation_layer.hpp"

#include <vector>

namespace avk {

namespace VertexInputAttributeDescription {

static constexpr auto
create(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
  -> VkVertexInputAttributeDescription
{
  return VkVertexInputAttributeDescription{ location, binding, format, offset };
}

}

namespace InstanceCreateInfo {

static auto
create(const void* pNext,
       const VkInstanceCreateFlags flags,
       const VkApplicationInfo* pApplicationInfo,
       const std::vector<const char*>& enabledLayers,
       const std::vector<const char*>& enabledExtensions)
  -> VkInstanceCreateInfo
{
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pNext = pNext;
  createInfo.flags = flags;
  createInfo.pApplicationInfo = pApplicationInfo;
  createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
  createInfo.ppEnabledLayerNames = enabledLayers.data();
  createInfo.enabledExtensionCount =
    static_cast<uint32_t>(enabledExtensions.size());
  createInfo.ppEnabledExtensionNames = enabledExtensions.data();
  return createInfo;
};

}

namespace ApplicationInfo {

static constexpr auto
create(const void* pNext,
       const char* pApplicationName,
       const uint32_t applicationVersion,
       const char* pEngineName,
       const uint32_t engineVersion,
       const uint32_t apiVersion) -> VkApplicationInfo
{
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = pNext;
  appInfo.pApplicationName = pApplicationName;
  appInfo.applicationVersion = applicationVersion;
  appInfo.pEngineName = pEngineName;
  appInfo.engineVersion = engineVersion;
  appInfo.apiVersion = apiVersion;
  return appInfo;
}

}

}
