cmake_minimum_required(VERSION 3.0.0)
project(InstaDeferred VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 17)

include(CTest)
enable_testing()

add_executable(VulkanTutorial VulkanTutorial/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/libraries/glfw)
add_subdirectory(${CMAKE_SOURCE_DIR}/libraries/glm)
add_subdirectory(${CMAKE_SOURCE_DIR}/libraries/Vulkan-Headers)

find_package(Vulkan REQUIRED FATAL_ERROR)

target_link_libraries(VulkanTutorial PRIVATE glm)
target_link_libraries(VulkanTutorial PRIVATE glfw)
target_link_libraries(VulkanTutorial PRIVATE Vulkan::Vulkan)

target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/Vulkan-Headers/include)
target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/tinyobjloader)
target_include_directories(VulkanTutorial PRIVATE ${CMAKE_SOURCE_DIR}/libraries/stb)
target_include_directories(VulkanTutorial PRIVATE ${VULKAN_INCLUDE_DIRS})
