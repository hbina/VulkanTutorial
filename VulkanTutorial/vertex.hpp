#pragma once

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <array>

struct Vertex
{
  // TODO : Alignment for this?
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static constexpr VkVertexInputBindingDescription getBindingDescription()
  {
    const VkVertexInputBindingDescription bindingDescription = {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };
    return bindingDescription;
  }

  static constexpr std::array<VkVertexInputAttributeDescription, 3>
  getAttributeDescriptions()
  {
    const std::array<VkVertexInputAttributeDescription, 3>
      attributeDescriptions = {
        VkVertexInputAttributeDescription{ .location = 0,
                                           .binding = 0,
                                           .format = VK_FORMAT_R32G32B32_SFLOAT,
                                           .offset = offsetof(Vertex, pos) },
        VkVertexInputAttributeDescription{ .location = 1,
                                           .binding = 0,
                                           .format = VK_FORMAT_R32G32_SFLOAT,
                                           .offset = offsetof(Vertex, color) },
        VkVertexInputAttributeDescription{ .location = 2,
                                           .binding = 0,
                                           .format = VK_FORMAT_R32G32_SFLOAT,
                                           .offset =
                                             offsetof(Vertex, texCoord) }
      };
    return attributeDescriptions;
  }

  constexpr bool operator==(const Vertex& other) const
  {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord;
  }
};

namespace std {
template<>
struct hash<Vertex>
{
  size_t operator()(Vertex const& vertex) const
  {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
}