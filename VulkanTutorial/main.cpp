#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <functional>
#include <cstring>

/////////////////////////////////////////////////////
//////                                         //////
//////    HELPER TEMPLATE ALGORITHMS           //////
//////                                         //////
/////////////////////////////////////////////////////

template<typename OuterIterable, typename InnerIterable, typename BinaryPredicate>
bool any_of_range(const OuterIterable& outer_iterable, const InnerIterable& inner_iterable, const BinaryPredicate& pred) {
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
bool any_of_range(const OuterIterable& outer_iterable, const InnerIterable& inner_iterable) {
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
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

bool checkValidationLayerSupport() {
	uint32_t layerCount = 0;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	return any_of_range(validationLayers, availableLayers, [](const char* lhs, const VkLayerProperties& rhs) {
		return strcmp(lhs, rhs.layerName) == 0;
		});
}

///////////////////////////////
//////                   //////
//////  VULKAN TUTORIAL  //////
//////                   //////
///////////////////////////////

class HelloTriangleApplication {
	const uint32_t WIDTH = 800;
	const uint32_t HEIGHT = 600;
	GLFWwindow* window = nullptr;
	VkInstance instance = VK_NULL_HANDLE;

public:
	void run() {
		initWindow(); // how does it know what this is???
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan() { createInstance(); }

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		vkDestroyInstance(instance, nullptr);

		// Free GLFW
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	// The instance is the connection between your application and the Vulkan
	// library. Creating it involves specifying some details about your
	// application to the driver.
	void createInstance() {

		// If validation layer is enabled, check if they are all supported.
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
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
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions_raw;

		glfwExtensions_raw = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions_raw;
		createInfo.enabledLayerCount = 0;

		// GLFW handles all the storage by itself, so we move list of GLFW extensions 
		// to std::string for ease of use.
		std::vector<std::string> glfwExtensions;
		glfwExtensions.reserve(glfwExtensionCount);
		for (uint32_t i = 0; i < glfwExtensionCount; i++) {
			glfwExtensions.emplace_back(glfwExtensions_raw[i]);
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
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
			extensions.data());

		// Print all the supported extensions
		std::cout << "available extensions:"
			<< "\n";
		for (const auto& x : extensions) {
			std::cout << "\t" << x.extensionName << "\n";
		}

		// Checks that extensions required by GLFW are all supported.
		auto checks_all = any_of_range(glfwExtensions,
			extensions,
			[](const std::string lhs, const VkExtensionProperties& rhs) {
				return lhs == rhs.extensionName;
			});
		if (!checks_all) {
			throw std::runtime_error("failed to create instance!");
		}


	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}