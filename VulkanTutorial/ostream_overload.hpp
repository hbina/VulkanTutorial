#pragma once

#include <vulkan/vulkan.h>

#include <iostream>

std::ostream&
operator<<(std::ostream& os, const VkPhysicalDeviceType& obj)
{
  switch (obj) {
    case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU: {
      os << "VK_PHYSICAL_DEVICE_TYPE_CPU";
      break;
    }
    case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: {
      os << "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
      break;
    }
    case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: {
      os << "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU";
      break;
    }
    case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: {
      os << "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU";
      break;
    }
    case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_OTHER: {
      os << "VK_PHYSICAL_DEVICE_TYPE_OTHER";
      break;
    }
    default: {
      throw std::runtime_error("Unknown VkPhysicalDeviceType:" + obj);
    }
  }
  return os;
}
