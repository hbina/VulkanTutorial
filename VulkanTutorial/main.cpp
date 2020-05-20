#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

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

///////////////////////////////
//////                   //////
//////  VALIDATION LAYERS //////
//////                   //////
///////////////////////////////

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

// Helper function to create `VkDebugUtilsMessengerEXT`.
static auto
CreateDebugUtilsMessengerEXT(
  VkInstance instance,
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
DestroyDebugUtilsMessengerEXT(VkInstance instance,
                              VkDebugUtilsMessengerEXT debugMessenger,
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
printDeviceFeatures(VkPhysicalDevice device)
{
  VkPhysicalDeviceFeatures deviceFeatures{};
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  (void)deviceFeatures; // Silence unused warning
  // TODO: Implement the print here...
}

// Helper struct to find queue families.
struct QueueFamilyIndices
{
  std::optional<uint32_t> graphicsFamily;

  auto isComplete() const -> bool { return graphicsFamily.has_value(); }
};

class HelloTriangleApplication
{
  const uint32_t WIDTH = 800;
  const uint32_t HEIGHT = 600;

  // GLFW
  GLFWwindow* window = nullptr;

  // Vulkan
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;

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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan()
  {
    // TODO: This should be a builder pattern.
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice();
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
    }
  }

  void cleanup()
  {
    vkDestroyDevice(device, nullptr);
    // Free validation layer debugger.
    if (ENABLE_VALIDATION_LAYER) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    // Free Vulkan instance.
    vkDestroyInstance(instance, nullptr);

    // Free GLFW.
    glfwDestroyWindow(window);
    glfwTerminate();
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

    // Specify the desired global extensions.
    // Vulkan is a platform agnostic API, which means that you need an extension
    // to interface with the window system. GLFW has a handy built-in function
    // that returns the extension(s) it needs to do that which we can pass to
    // the struct.
    auto glfwExtensions = getRequiredExtensions();
    auto glfwExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions.data();
    createInfo.enabledLayerCount = 0;

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
      throw std::runtime_error("failed to create instance!");
    }
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                void* pUserData)
  {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

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

  static auto isDeviceSuitable(VkPhysicalDevice device) -> bool
  {
    // TODO: Implement the checks here...
    // One idea is to calculate scores and choose only the highest scoring
    // device. VkPhysicalDeviceProperties deviceProperties{};
    // VkPhysicalDeviceFeatures deviceFeatures{};
    // vkGetPhysicalDeviceProperties(device, &deviceProperties);
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    return device == VK_NULL_HANDLE ? false
                                    : findQueueFamilies(device).isComplete();
  }

  static auto findQueueFamilies(VkPhysicalDevice device) -> QueueFamilyIndices
  {
    QueueFamilyIndices indices;
    // Assign index to queue families that could be found
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
      device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
      device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
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

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    // TODO: Proper error handling.
    // This operation _may_ fail.
    queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    queueCreateInfo.queueCount = 1;
    // NOTE: Why does Vulkan decide on using float here?
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // NOTE: For now we don't require anything special from the physical device.
    // We just let it all default to `VK_FALSE`.
    // NOTE: Not sure if that works? Because `VK_FALSE` could be 1 for all we
    // know.
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;

    if (ENABLE_VALIDATION_LAYER) {
      createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
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