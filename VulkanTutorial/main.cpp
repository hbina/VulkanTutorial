#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

/////////////////////////////////////////////////////
//////                                         //////
//////                 GLM                     //////
//////                                         //////
/////////////////////////////////////////////////////

struct Vertex
{
  glm::vec2 pos;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription()
  {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

struct UniformBufferObject
{
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

/////////////////////////////////////////////////////
//////                                         //////
//////                DUMMY DATA               //////
//////                                         //////
/////////////////////////////////////////////////////

// Dummy data for now
const std::vector<Vertex> vertices = {
  { { -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
  { { 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
  { { 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } },
  { { -0.5f, 0.5f }, { 1.0f, 1.0f, 1.0f } }
};

using IndexTy = uint16_t;

const std::vector<IndexTy> indices = { 0, 1, 2, 2, 3, 0 };

/////////////////////////////////////////////////////
//////                                         //////
//////    HELPER TEMPLATE ALGORITHMS           //////
//////                                         //////
/////////////////////////////////////////////////////

template<typename OuterIterable,
         typename InnerIterable,
         typename BinaryPredicate>
static constexpr auto
any_of_range(const OuterIterable& outer_iterable,
             const InnerIterable& inner_iterable,
             const BinaryPredicate& pred) -> bool
{
  for (std::size_t x = 0; x < outer_iterable.size(); x++) {
    bool found = false;
    for (std::size_t y = 0; y < inner_iterable.size(); y++) {
      if (pred(outer_iterable[x], inner_iterable[y])) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

template<typename OuterIterable, typename InnerIterable>
static constexpr auto
any_of_range(const OuterIterable& outer_iterable,
             const InnerIterable& inner_iterable) -> bool
{
  return any_of_range(outer_iterable, inner_iterable, std::equal_to{});
}

//////////////////////////////////
////                          ////
////    VALIDATION LAYERS     ////
////                          ////
//////////////////////////////////

// List of validation layers we want to have.
const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYER = false;
#else
const bool ENABLE_VALIDATION_LAYER = true;
#endif

static auto
checkValidationLayerSupport() -> bool
{
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  return any_of_range(validationLayers,
                      availableLayers,
                      [](const char* lhs, const VkLayerProperties& rhs) {
                        return strcmp(lhs, rhs.layerName) == 0;
                      });
}

///////////////////////////////
//////                   //////
//////  VULKAN TUTORIAL  //////
//////                   //////
///////////////////////////////

// List of device extensions we need.
const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

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

// Get Vulkan extennsions required for/by:
//  1.  GLFW.
//  2.  Debugging validation layers.
static auto
getRequiredExtensions() -> std::vector<const char*>
{
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (ENABLE_VALIDATION_LAYER) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

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

// Struct for querying details of swap chain support.
struct SwapChainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

// Helper struct to find queue families.
struct QueueFamilyIndices
{
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  auto isComplete() const -> bool
  {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

// Loading SPIR-V shaders
static std::vector<char>
readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  std::size_t fileSize = static_cast<std::size_t>(file.tellg());
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();
  return buffer;
}

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

class HelloTriangleApplication
{

  // GLFW
  GLFWwindow* window = nullptr;

  /// Vulkan

  // General
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  // Queues
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue presentQueue = VK_NULL_HANDLE;

  // Swapchain
  VkSwapchainKHR swapChain{};
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D swapChainExtent = { 0, 0 };
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;

  // Pipeline
  VkRenderPass renderPass{};
  VkPipelineLayout pipelineLayout{};
  VkPipeline graphicsPipeline{};

  // Commands
  VkCommandPool commandPool{};
  std::vector<VkCommandBuffer> commandBuffers;

  // Buffers
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  // Synchronization
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSemaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences;
  std::vector<VkFence> imagesInFlight;
  std::size_t currentFrame = 0;

  // Explicitly handle resizing
  bool framebufferResized = false;

public:
  void run()
  {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow()
  {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window =
      glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Tutorial", nullptr, nullptr);
    // Allows us to obtain the pointer to this struct through window
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow* window,
                                        int width,
                                        int height)
  {
    auto app = reinterpret_cast<HelloTriangleApplication*>(
      glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan()
  {
    // TODO: This should be a builder pattern.
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    createCommandBuffers();
    createSyncObjects();
  }

  uint32_t findMemoryType(const uint32_t& typeFilter,
                          const VkMemoryPropertyFlags& properties)
  {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createBuffer(const VkDeviceSize& size,
                    const VkBufferUsageFlags& usage,
                    const VkMemoryPropertyFlags& properties,
                    VkBuffer& buffer,
                    VkDeviceMemory& bufferMemory)
  {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate vertex buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  void createVertexBuffer()
  {
    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);

    void* data = nullptr;
    // NOTE: Shouldn't this be memRequirements.size?
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), static_cast<std::size_t>(bufferSize));
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer,
                 vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    // Clean up staging buffers
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void createIndexBuffer()
  {
    VkDeviceSize bufferSize = sizeof(IndexTy) * indices.size();

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingBufferMemory);

    void* data = nullptr;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), static_cast<std::size_t>(bufferSize));
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 indexBuffer,
                 indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void copyBuffer(const VkBuffer& srcBuffer,
                  const VkBuffer& dstBuffer,
                  const VkDeviceSize& size)
  {
    // TODO: Its possible to create another command pool to improve performance.
    // That command  pool should use the flag
    // `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT`.
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    // Stop recording
    vkEndCommandBuffer(commandBuffer);

    // Submit command to complete transfer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }

  void cleanupSwapChain()
  {
    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
      vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
    }

    vkFreeCommandBuffers(device,
                         commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
  }

  void recreateSwapChain()
  {

    // If buffer size is 0, we just pause and no-op
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers();
  }

  void createSyncObjects()
  {
    // TODO : Should fill with `VK_NULL_HANDLE`?
    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(
            device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) !=
            VK_SUCCESS ||
          vkCreateSemaphore(
            device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) !=
            VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) !=
            VK_SUCCESS) {

        throw std::runtime_error("failed to create semaphores!");
      }
    }
  }

  void createCommandBuffers()
  {
    commandBuffers.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (std::size_t i = 0; i < commandBuffers.size(); i++) {
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0;                  // Optional
      beginInfo.pInheritanceInfo = nullptr; // Optional

      if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = { 0, 0 };
      renderPassInfo.renderArea.extent = swapChainExtent;

      VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      vkCmdBeginRenderPass(
        commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(
        commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
      VkBuffer vertexBuffers[] = { vertexBuffer };
      VkDeviceSize offsets[] = { 0 };
      vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

      vkCmdBindIndexBuffer(
        commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

      vkCmdDrawIndexed(
        commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

      vkCmdEndRenderPass(commandBuffers[i]);
      if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createCommandPool()
  {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    // TODO: There ought to be a default value for this?
    poolInfo.flags = 0; // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createFramebuffers()
  {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (std::size_t i = 0; i < swapChainImageViews.size(); i++) {
      VkImageView attachments[] = { swapChainImageViews[i] };

      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(
            device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void createRenderPass()
  {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createGraphicsPipeline()
  {
    auto vertShaderCode = readFile("bin/vert.spv");
    auto fragShaderCode = readFile("bin/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;

    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    vertShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
      vertShaderStageInfo, fragShaderStageInfo
    };

    ///////////////////////////////////////////
    //////
    //////           VERTEX INPUT
    //////
    ///////////////////////////////////////////

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    // NOTE: We are hard coding the vertex directly in the vertex because we
    // already have them in shader.vert
    // We will go back to this at some point.
    // Remember to clear this up
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions =
      &bindingDescription; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions =
      attributeDescriptions.data(); // Optional

    ///////////////////////////////////////////
    //////
    //////        INPUT ASSEMBLY
    //////
    ///////////////////////////////////////////

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    ///////////////////////////////////////////
    //////
    //////        VIEW PORT AND SCISSOR
    //////
    ///////////////////////////////////////////

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    ///////////////////////////////////////////
    //////
    //////             RASTERIZER
    //////
    ///////////////////////////////////////////

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f;          // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

    ///////////////////////////////////////////
    //////
    //////           MULTISAMPLING
    //////
    ///////////////////////////////////////////

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;          // Optional
    multisampling.pSampleMask = nullptr;            // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

    ///////////////////////////////////////////
    //////
    //////         COLOR BLENDING
    //////
    ///////////////////////////////////////////

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;            // Optional
    pipelineLayoutInfo.pSetLayouts = nullptr;         // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(
          device, &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages.data();

    // Reference all the stuff from before.
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional

    // Reference the strucutres describing the fixed-function stage
    pipelineInfo.layout = pipelineLayout;

    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1;              // Optiona

    if (vkCreateGraphicsPipelines(device,
                                  VK_NULL_HANDLE,
                                  1,
                                  &pipelineInfo,
                                  nullptr,
                                  &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
  }

  VkShaderModule createShaderModule(const std::vector<char>& code)
  {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
  }

  void createImageViews()
  {
    swapChainImageViews.resize(swapChainImages.size());

    for (std::size_t i = 0; i < swapChainImages.size(); i++) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];

      // Specify how the image should be interpreted.
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;

      // Default mapping
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      // Describe the purpose of this image and which part should be accessed.
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(
            device, &createInfo, nullptr, &swapChainImageViews[i]) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }

  void createSurface()
  {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void setupDebugMessenger()
  {
    if (!ENABLE_VALIDATION_LAYER)
      return;

    // Details about how we want to set up the debug messenger callback.
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(
          instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  void mainLoop()
  {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }

    vkDeviceWaitIdle(device);
  }

  void drawFrame()
  {
    // Synchronize CPU-GPU
    vkWaitForFences(
      device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquiring an image from the swap chain.
    uint32_t imageIndex = 0;
    VkResult result =
      vkAcquireNextImageKHR(device,
                            swapChain,
                            UINT64_MAX,
                            imageAvailableSemaphores[currentFrame],
                            VK_NULL_HANDLE,
                            &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
      vkWaitForFences(
        device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    // Mark the image as now being in use by this frame
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // Submitting the command buffer.
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    // Specify which command buffers to actually submit for execution.
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    // Specify the semaphore to signal once the command buffer finished
    // executing.
    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    if (vkQueueSubmit(
          graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr; // Optional

    VkResult queue_result = vkQueuePresentKHR(presentQueue, &presentInfo);

    // Wait for queue to prevent workload overload.
    vkQueueWaitIdle(presentQueue);

    if (queue_result == VK_ERROR_OUT_OF_DATE_KHR ||
        queue_result == VK_SUBOPTIMAL_KHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (queue_result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanup()
  {
    std::cout << "cleaning up"
              << "\n";

    cleanupSwapChain();

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);

    // Free validation layer debugger.
    if (ENABLE_VALIDATION_LAYER) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    // Free Vulkan surface and instance.
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    // Free GLFW.
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  // TODO: This should probably be integrated as a builder pattern.
  // Helper function to populate `SwapChainSupportDetails`
  SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device)
  {
    SwapChainSupportDetails details;

    // Query basic surface capabilities.
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      device, surface, &details.capabilities);

    // Query the supported surface formats.
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
      device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
        device, surface, &formatCount, details.formats.data());
    }

    // Query supported presentations mode.
    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
      device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  // The instance is the connection between your application and the Vulkan
  // library. Creating it involves specifying some details about your
  // application to the driver.
  void createInstance()
  {

    // If validation layer is enabled, check if they are all supported.
    if (ENABLE_VALIDATION_LAYER && !checkValidationLayerSupport()) {
      throw std::runtime_error(
        "validation layers requested, but not available!");
    }

    // Fill in a struct with some information about our application.
    // This data is technically optional, but it may provide some useful
    // information to the driver in order to optimize our specific application.
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // tells the Vulkan driver which global extensions and validation layers we
    // want to use. Global here means that they apply to the entire program and
    // not a specific device. Meaning it will become clear in the next few
    // chapters.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Specify the desired global extensions.
    // Vulkan is a platform agnostic API, which means that you need an extension
    // to interface with the window system. GLFW has a handy built-in function
    // that returns the extension(s) it needs to do that which we can pass to
    // the struct.
    auto glfwExtensions = getRequiredExtensions();
    auto glfwExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions.data();

    // If validation layer is enabled, we have to specify at creation.
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (ENABLE_VALIDATION_LAYER) {
      createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
      populateDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    // We've now specified everything Vulkan needs to create an instance.
    // We can finally issue the vkCreateInstance call.
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }

    // Request the number of extensions supported.
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(
      nullptr, &extensionCount, extensions.data());

    // Print all the supported extensions
    std::cout << "available extensions:"
              << "\n";
    for (const auto& x : extensions) {
      std::cout << "\t" << x.extensionName << "\n";
    }

    // Checks that extensions required by GLFW are all supported.
    auto checks_all =
      any_of_range(glfwExtensions,
                   extensions,
                   [](const std::string lhs, const VkExtensionProperties& rhs) {
                     return lhs == rhs.extensionName;
                   });
    if (!checks_all) {
      throw std::runtime_error("failed obtain all required extensions!");
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

  static void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT& createInfo)
  {
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr; // Optional
  }

  void pickPhysicalDevice()
  {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const VkPhysicalDevice& device : devices) {
      printDeviceProperties(device);
      printDeviceFeatures(device);
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  auto checkDeviceExtensionSupport(VkPhysicalDevice device) -> bool
  {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(
      device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(
      device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  auto isDeviceSuitable(const VkPhysicalDevice& device) -> bool
  {
    // TODO: Implement the checks here...
    // One idea is to calculate scores and choose only the highest scoring
    // device.
    // Some code to get started...
    // VkPhysicalDeviceProperties deviceProperties{};
    // VkPhysicalDeviceFeatures deviceFeatures{};
    // vkGetPhysicalDeviceProperties(device, &deviceProperties);
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    // We must require swap chain support.
    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  // Choose the optimal swap chain.
  // In particular, we have to consider the following:
  // 1. Surface format (color depth).
  // 2. Presentation mode (conditions for "swapping" images to the screen).
  // 3. Swap extent (resolution of images in swap chain).
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& availableFormats)
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
  VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& availablePresentModes)
  {
    for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  // Choose the swap extent.
  // The swap extent is the resolution of the swap chain images and it's almost
  // always exactly equal to the resolution of the window that we're drawing to.
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
  {
    if (capabilities.currentExtent.width != UINT32_MAX) {
      return capabilities.currentExtent;
    } else {
      int width = 0, height = 0;
      glfwGetFramebufferSize(window, &width, &height);

      VkExtent2D actualExtent = { static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height) };

      // Clamp swap extent between minImageExtend and maxImageExtend
      actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actualExtent.width));
      actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actualExtent.height));

      return actualExtent;
    }
  }

  // Create swap chain.
  void createSwapChain()
  {
    SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(),
                                      indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    // We do not want any transformation to be applied to the swap chain.
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

    // We will ignore the composite alpha channel.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode;

    // We don't care about color of pixels that are obscured.
    // This gives us better performance.
    createInfo.clipped = VK_TRUE;

    // TODO: For now, we only assume that we will only ever create 1 swap chain.
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(
      device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  auto findQueueFamilies(const VkPhysicalDevice& device) -> QueueFamilyIndices
  {
    QueueFamilyIndices indices;
    // Assign index to queue families that could be found
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
      device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
      device, &queueFamilyCount, queueFamilies.data());

    // NOTE: add logic to explicitly prefer a physical device that supports
    // drawing and presentation in the same queue for improved performance.
    int i = 0;
    for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }
    return indices;
  }

  void createLogicalDevice()
  {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    // NOTE: We do this in case the 2 queues are different.
    // That is, one queue is able to perform graphic computation and the other
    // is able to present the surface.
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
                                               indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // NOTE: For now we don't require anything special from the physical device.
    // We just let it all default to `VK_FALSE`.
    // NOTE: Not sure if that works? Because `VK_FALSE` could be 1 for all we
    // know.
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    if (ENABLE_VALIDATION_LAYER) {
      createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.ppEnabledLayerNames = nullptr;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }
};

int
main()
{
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}