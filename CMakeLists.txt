cmake_minimum_required(VERSION 3.0.0)
project(InstaDeferred VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 17)

include(CTest)
enable_testing()

add_executable(VulkanTutorial VulkanTutorial/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(Vulkan REQUIRED FATAL_ERROR)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(VulkanTutorial PRIVATE glfw)

target_link_libraries(VulkanTutorial PRIVATE vulkan)

target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/tinyobjloader)
target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/glm/glm)
target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/stb)
target_include_directories(VulkanTutorial PRIVATE ${VULKAN_INCLUDE_DIRS})
