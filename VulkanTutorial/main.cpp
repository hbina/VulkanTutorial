#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
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
  for (auto x = 0; x < outer_iterable.size(); x++) {
    bool found = false;
    for (auto y = 0; y < inner_iterable.size(); y++) {
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

class HelloTriangleApplication
{
  const uint32_t WIDTH = 800;
  const uint32_t HEIGHT = 600;

  // GLFW
  GLFWwindow* window = nullptr;

  // Vulkan
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

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
    createInstance();
    setupDebugMessenger();
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
    if (ENABLE_VALIDATION_LAYER) {
      VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
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

  void populateDebugMessengerCreateInfo(
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