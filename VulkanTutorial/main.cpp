#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>


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

		// Move list of GLFW extensions to std::string for ease of use.
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
		// FIXME: This is a hot mess...
		auto checks_all = std::accumulate(
			std::cbegin(glfwExtensions), std::cend(glfwExtensions), true,
			[&](bool checks, const std::string& x) {
				return checks &&
					std::any_of(std::cbegin(extensions), std::cend(extensions),
						[&](const VkExtensionProperties& y) {
							return x == y.extensionName;
						});
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