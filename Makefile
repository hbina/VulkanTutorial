CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/vulkan/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: VulkanTutorial/main.cpp
	mkdir -p bin
	g++ $(CFLAGS) -o bin/VulkanTutorial VulkanTutorial/main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d ./VulkanTest

clean:
	rm -rf bin
