#version 450
#extension GL_ARB_separate_shader_objects : enable

// Uniform
layout(binding = 1) uniform sampler2D texSampler;

// In
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

// Out
layout(location = 0) out vec4 outColor;

void
main()
{
  outColor = texture(texSampler, fragTexCoord);
}