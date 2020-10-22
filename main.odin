package main

import "core:fmt"
import "core:unicode/utf8"
import glfw "shared:odin-glfw/bindings"
import vk "vulkan"
import win32 "core:sys/windows"
import "core:strings"
import "core:mem"
import "core:runtime"
import "core:os"
import m "core:math/linalg"
import "util"

//glfw State
glfw_window : glfw.Window_Handle;
framebuffer_resized : bool;

// Vulkan State (some state is still declared in main() i believe)
vk_swapchain : vk.SwapchainKHR;
swapchain_images : []vk.Image;
swapchain_image_format : vk.Format;
swapchain_extent : vk.Extent2D;
physical_device : vk.PhysicalDevice;
surface : vk.SurfaceKHR;
vk_device : vk.Device;
graphics_queue : vk.Queue;
present_queue : vk.Queue;
command_pool  : vk.CommandPool;
command_buffers : []vk.CommandBuffer;
swapchain_framebuffers : []vk.Framebuffer;
swapchain_image_views : []vk.ImageView;
descriptor_set_layout : vk.DescriptorSetLayout;
pipeline_layout : vk.PipelineLayout;
vk_graphics_pipeline : vk.Pipeline;
vk_render_pass : vk.RenderPass;
vertex_buffer : vk.Buffer;
vertex_buffer_memory : vk.DeviceMemory;
index_buffer : vk.Buffer;
index_buffer_memory : vk.DeviceMemory;
uniform_buffers : []vk.Buffer;
uniform_buffers_memory : []vk.DeviceMemory;

swapchain_created : bool;

image_available_semaphores : []vk.Semaphore;
render_finished_semaphores : []vk.Semaphore;
in_flight_fences : []vk.Fence;
images_in_flight : []vk.Fence;
current_frame := 0;

// Window properties
W_WIDTH : i32 : 800;
W_HEIGHT : i32 : 600;

MAX_FRAMES_IN_FLIGHT :: 2;

vertices := [?]Vertex {
  Vertex{{-0.5, -0.5}, {1.0, 0.0, 0.0}},
  Vertex{{0.5, -0.5}, {0.0, 1.0, 0.0}},
  Vertex{{0.5, 0.5}, {0.0, 0.0, 1.0}},
  Vertex{{-0.5, 0.5}, {1.0, 1.0, 1.0}}
};

indices := [?]u16 { 0, 1, 2, 2, 3, 0};

Vertex :: struct
{
  pos : m.Vector2,
  col : m.Vector3,
};

Mat4 :: [4][4]f32;

UniformBufferObject :: struct
{
  model, view, proj : Mat4,
};

get_binding_description :: proc() -> vk.VertexInputBindingDescription
{
  binding_description : vk.VertexInputBindingDescription = {
    binding = 0,
    stride = size_of(Vertex),
    inputRate = .VERTEX
  };

  return binding_description;
}

get_attribute_descriptions :: proc() -> [2]vk.VertexInputAttributeDescription
{
  attribute_descriptions : [2]vk.VertexInputAttributeDescription = {
    {
      binding = 0,
      location = 0,
      format = .R32G32_SFLOAT,
      offset = u32(offset_of(Vertex, pos)),
    },
    {
      binding = 0,
      location = 1,
      format = .R32G32B32_SFLOAT,
      offset = u32(offset_of(Vertex, col)),
    }
  };

  return attribute_descriptions;

}

create_buffer :: proc(physical_device : vk.PhysicalDevice, vk_device : vk.Device, size : vk.DeviceSize, usage : vk.BufferUsageFlags, properties : vk.MemoryPropertyFlags, buffer : ^vk.Buffer, buffer_memory : ^vk.DeviceMemory)
{
  buffer_info : vk.BufferCreateInfo = {
    sType = .BUFFER_CREATE_INFO,
    size = size,
    usage = usage,
    sharingMode = .EXCLUSIVE,
  };

  res := vk.CreateBuffer(vk_device, &buffer_info, nil, buffer);
  if res != .SUCCESS do panic("failed to create buffer");

  mem_requirements : vk.MemoryRequirements;
  vk.GetBufferMemoryRequirements(vk_device, buffer^, &mem_requirements);

  alloc_info : vk.MemoryAllocateInfo = {
    sType = .MEMORY_ALLOCATE_INFO,
    allocationSize = mem_requirements.size,
    memoryTypeIndex = find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties),
  };

  res = vk.AllocateMemory(vk_device, &alloc_info, nil, buffer_memory);
  if res != .SUCCESS do panic("failed to allocate vertex buffer memory");

  vk.BindBufferMemory(vk_device, buffer^, buffer_memory^, 0);
}

copy_buffer :: proc(device : vk.Device, command_pool : vk.CommandPool, queue : vk.Queue, src_buffer, dst_buffer : vk.Buffer, size : vk.DeviceSize)
{
  alloc_info : vk.CommandBufferAllocateInfo = {
    sType = .COMMAND_BUFFER_ALLOCATE_INFO,
    level = .PRIMARY,
    commandPool = command_pool,
    commandBufferCount = 1
  };

  command_buffer : vk.CommandBuffer;
  vk.AllocateCommandBuffers(device, &alloc_info, &command_buffer);

  begin_info : vk.CommandBufferBeginInfo = {
    sType = .COMMAND_BUFFER_BEGIN_INFO,
    flags = {.ONE_TIME_SUBMIT}
  };

  vk.BeginCommandBuffer(command_buffer, &begin_info);

  copy_region : vk.BufferCopy = {
    srcOffset = 0,
    dstOffset = 0,
    size = size
  };

  vk.CmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);
  vk.EndCommandBuffer(command_buffer);

  submit_info : vk.SubmitInfo = {
    sType = .SUBMIT_INFO,
    commandBufferCount = 1,
    pCommandBuffers = &command_buffer
  };

  vk.QueueSubmit(queue, 1, &submit_info, 0);
  vk.QueueWaitIdle(queue);

  vk.FreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

find_memory_type :: proc(physical_device : vk.PhysicalDevice, type_filter : u32, properties : vk.MemoryPropertyFlags) -> u32
{
  mem_properties : vk.PhysicalDeviceMemoryProperties;
  vk.GetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
  // fmt.println(mem_properties);
  // fmt.println(type_filter);
  // fmt.println(properties);

  for i:u32 = 0; i < mem_properties.memoryTypeCount; i+=1
  {
    prop_found : bool;
    type_found : bool;

    if (properties & mem_properties.memoryTypes[i].propertyFlags == properties) 
    {
      prop_found = true;
    }


    if (type_filter & (1 << i) ) > 0
    {
      type_found = true;
    }

    if prop_found && type_found do return i;
  }

  panic("failed to find suitable memory type");
}

QueueFamilyIndices :: struct
{
  graphics_family : Maybe(u32),
  present_family : Maybe(u32),
};

SwapChainSupportDetails :: struct
{
  capabilities : vk.SurfaceCapabilitiesKHR,
  formats : []vk.SurfaceFormatKHR,
  present_modes : []vk.PresentModeKHR,
};

when ODIN_DEBUG 
{
  dump_proc :: proc(message:string, things_went_wrong:bool, user_data:rawptr)
  {
    if things_went_wrong do fmt.eprintln(message);
  }
}

free_swapchain_support_details :: proc(scsd : SwapChainSupportDetails)
{
  delete(scsd.formats);
  delete(scsd.present_modes);
}

//Set data structure
Set :: [dynamic]u32;
set_add :: proc(set : ^Set, value : u32)
{
  already_in_set : bool;

  for i:=0;i<len(set^);i+=1
  {
    if set^[i] == value
    {
      already_in_set = true;
      break;
    }
  }

  if !already_in_set
  {
    append(set, value);
  }
}



validation_layers := [?]string {
  "VK_LAYER_KHRONOS_validation"
};
validation_layers_cstring : []cstring;

required_device_extensions := [?]string {
  vk.KHR_SWAPCHAIN_EXTENSION_NAME
};
required_device_extensions_cstring : []cstring;

when ODIN_DEBUG 
{
  VALIDATION_LAYERS_ENABLED :: true;
}
else
{
  VALIDATION_LAYERS_ENABLED :: false;
}

StringArrayToCstringArray :: proc(str : []string, c_str: []cstring)
{
  assert(len(str) == len(c_str));
  for i:=0;i<len(str);i+=1
  {
    c_str[i] = strings.unsafe_string_to_cstring(str[i]);
  }
}

CheckValidationLayerSupport :: proc() -> bool
{
  layer_count : u32;
  vk.EnumerateInstanceLayerProperties(&layer_count, nil);
  available_layers := make([]vk.LayerProperties, layer_count);
  defer delete(available_layers);
  vk.EnumerateInstanceLayerProperties(&layer_count, &available_layers[0]);
  for i:u32=0; i<len(validation_layers); i+=1
  {
    layer_found := false;
    for j:u32=0; j<layer_count; j+=1
    {
      available_layer_str := string(cstring(&available_layers[j].layerName[0]));
      if strings.compare(available_layer_str, validation_layers[i]) == 0
      {
        layer_found = true;
        break;
      }
    }
    if !layer_found
    {
      return false;
    }
  }
  return true;
}

query_swapchain_support :: proc(device : vk.PhysicalDevice, surface : vk.SurfaceKHR) -> SwapChainSupportDetails 
{
  details : SwapChainSupportDetails;
  //querying surface capabilities
  vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  //querying surface formats
  format_count : u32;
  vk.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nil);

  if format_count != 0 
  {
    details.formats = make([]vk.SurfaceFormatKHR, format_count);
    vk.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, &details.formats[0]);
  }

  //querying supported presentation formats
  present_mode_count : u32;
  vk.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nil);

  if present_mode_count != 0
  {
    details.present_modes = make([]vk.PresentModeKHR, present_mode_count);
    vk.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, &details.present_modes[0]);
  }

  return details;
}

PopulateDebugMessengerCreateInfo :: proc(create_info : ^vk.DebugUtilsMessengerCreateInfoEXT) 
{
  mem.set(create_info, 0, size_of(type_of(create_info)));
  create_info.sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity = { .VERBOSE, .WARNING, .ERROR, .INFO };
  create_info.messageType = {.GENERAL, .VALIDATION, .PERFORMANCE };
  create_info.pfnUserCallback = VulkanDebugCallback;
  create_info.pUserData = nil;
}

framebuffer_resize_callback : glfw.Framebuffer_Size_Proc : proc "c" (window: glfw.Window_Handle, width, height: i32)
{
  framebuffer_resized = true;
}

VulkanDebugCallback : vk.ProcDebugUtilsMessengerCallbackEXT : proc "stdcall"( messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT, messageTypes: vk.DebugUtilsMessageTypeFlagsEXT, pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT, pUserData: rawptr) -> b32
{
  context = runtime.default_context();
  if .ERROR in messageSeverity
  {
      fmt.eprintln("VULKAN ERROR!!!! : ", pCallbackData.pMessage);
  }
  if .WARNING in messageSeverity
  {
      fmt.eprintln("VULKAN WARNING!!!! : ", pCallbackData.pMessage);
  }
  else
  {
    // fmt.eprintln("Vk validation layer: ", pCallbackData.pMessage);
  }
  return false;
}

update_uniform_buffer :: proc(current_image : u32)
{
  //u were here...
  #assert(false);

}

find_queue_families :: proc(device : vk.PhysicalDevice, surface : vk.SurfaceKHR) -> QueueFamilyIndices 
{
  indices : QueueFamilyIndices;
  // Logic to find queue family indices to populate struct with
  queue_family_count : u32;
  vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nil);
  queue_families := make([]vk.QueueFamilyProperties, queue_family_count);
  defer delete(queue_families);
  // can you make this function name longer?
  vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, &queue_families[0]);

  for i:u32=0; int(i)<len(queue_families); i+=1
  {
    if .GRAPHICS in queue_families[i].queueFlags
    {
      indices.graphics_family = i; 
    }

    present_support : b32;
    vk.GetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
    if present_support
    {
      indices.present_family = i;
    }
  }

  return indices;
}

has_all_queue_families :: proc(indices : QueueFamilyIndices) -> bool
{
  return indices.graphics_family != nil && indices.present_family != nil;
}

is_device_suitable :: proc(device:vk.PhysicalDevice, surface : vk.SurfaceKHR) -> bool
{
  // a device is suitable when it has support for all the queue families we need
  //   and it supports all the required extensions
  //   and swap chain support is adequate
  indices := find_queue_families(device, surface);


  //checking for extensions
  extension_count : u32;
  vk.EnumerateDeviceExtensionProperties(device, nil, &extension_count, nil);
  available_extensions := make([]vk.ExtensionProperties, extension_count);
  defer delete(available_extensions);
  vk.EnumerateDeviceExtensionProperties(device, nil, &extension_count, &available_extensions[0]);

  required_extensions_left : int = len(required_device_extensions);

  loop: for i:u32=0;i<extension_count;i+=1
  {
    extension_name := string(cstring(&available_extensions[i].extensionName[0]));

    for j:=0;j<len(required_device_extensions);j+=1
    {
      if strings.compare(extension_name, required_device_extensions[j]) == 0
      {
        required_extensions_left -= 1;
        if required_extensions_left == 0
        {
          break loop;
        }
      }
    }
  }

  //checking for swap chain support
  swapchain_adequate : bool;
  if required_extensions_left == 0
  {
    swapchain_support := query_swapchain_support(device, surface);
    defer free_swapchain_support_details(swapchain_support);
    swapchain_adequate = len(swapchain_support.formats) > 0 &&
    len(swapchain_support.present_modes) > 0;
  }

  return (has_all_queue_families(indices) 
  && required_extensions_left == 0 
  && swapchain_adequate);
}

cleanup_swapchain :: proc()
{
  for framebuffer in swapchain_framebuffers
  {
    vk.DestroyFramebuffer(vk_device, framebuffer, nil);
  }

  if len(command_buffers) > 0
  {
    vk.FreeCommandBuffers(vk_device, command_pool, u32(len(command_buffers)), &command_buffers[0]);
  }

  vk.DestroyPipeline(vk_device, vk_graphics_pipeline, nil);
  vk.DestroyPipelineLayout(vk_device, pipeline_layout, nil);
  vk.DestroyRenderPass(vk_device, vk_render_pass, nil);

  for _,i in swapchain_image_views
  {
    vk.DestroyBuffer(vk_device, uniform_buffers[i], nil);
    vk.FreeMemory(vk_device, uniform_buffers_memory[i], nil);
  }


  for iv in swapchain_image_views
  {
    vk.DestroyImageView(vk_device, iv, nil);
  }

  vk.DestroySwapchainKHR(vk_device, vk_swapchain, nil);

  // delete all the "make"s in recreate_swapchain:

  delete(swapchain_images);
  delete(swapchain_image_views);
  delete(swapchain_framebuffers);
  delete(uniform_buffers);
  delete(uniform_buffers_memory);
  delete(command_buffers);
}

//this is
// create swap chain
// create image views
// create render pass
// graphics pipeline
// framebuffers
// this is to be reused in init_vulkan (main()) and in recreate_swapchan
init_swapchain_dependant_stuff :: proc()
{
  choose_swapchain_surface_format :: proc(available_formats : []vk.SurfaceFormatKHR) -> vk.SurfaceFormatKHR
  {
    for i:int=0; i<len(available_formats); i+=1
    {
      format := available_formats[i];
      if format.format == .B8G8R8A8_SRGB && format.colorSpace == .SRGB_NONLINEAR
      {
        return format;
      }
    }

    return available_formats[0];
  }

  choose_swap_present_mode :: proc(available_present_modes : []vk.PresentModeKHR) -> vk.PresentModeKHR
  {
    for i:int=0; i<len(available_present_modes); i+=1
    {
      if available_present_modes[i] == .MAILBOX
      {
        return .MAILBOX;
      }
    }

    return .FIFO;
  }

  choose_swap_extent :: proc(capabilities : vk.SurfaceCapabilitiesKHR) -> vk.Extent2D
  {
    if capabilities.currentExtent.width != max(u32)
    {
      return capabilities.currentExtent;
    }
    else
    {
      width, height : i32;
      glfw.GetFramebufferSize(glfw_window, &width, &height);

      actual_extent : vk.Extent2D = {u32(width), u32(height)};
      actual_extent.width = max(capabilities.minImageExtent.width,
      min(capabilities.maxImageExtent.width, actual_extent.width));
      actual_extent.height = max(capabilities.minImageExtent.height,
      min(capabilities.maxImageExtent.height, actual_extent.height));
      return actual_extent;
    }
  }

  // ----------- Creating Swap Chain -----------
  {
    // Creating swap chain
    swapchain_support := query_swapchain_support(physical_device, surface);
    defer free_swapchain_support_details(swapchain_support);

    surface_format := choose_swapchain_surface_format(swapchain_support.formats);
    swapchain_image_format = surface_format.format;

    present_mode := choose_swap_present_mode(swapchain_support.present_modes);
    extent := choose_swap_extent(swapchain_support.capabilities);
    swapchain_extent = extent;

    image_count := swapchain_support.capabilities.minImageCount + 1;

    if swapchain_support.capabilities.maxImageCount > 0 && image_count > swapchain_support.capabilities.maxImageCount 
    {
      image_count = swapchain_support.capabilities.maxImageCount;
    }

    sc_create_info : vk.SwapchainCreateInfoKHR = {
      sType = .SWAPCHAIN_CREATE_INFO_KHR,
      surface = surface,
      minImageCount = image_count,
      imageFormat = surface_format.format,
      imageColorSpace = surface_format.colorSpace,
      imageExtent = extent,
      imageArrayLayers = 1,
      imageUsage = {.COLOR_ATTACHMENT},
    };

    indices := find_queue_families(physical_device, surface);
    queue_family_indices : [2]u32 = { indices.graphics_family.(u32),
    indices.present_family.(u32) };

    if indices.graphics_family.(u32) != indices.present_family.(u32)
    {
      sc_create_info.imageSharingMode = .CONCURRENT;
      sc_create_info.queueFamilyIndexCount = 2;
      sc_create_info.pQueueFamilyIndices = &queue_family_indices[0];
    }
    else
    {
      sc_create_info.imageSharingMode = .EXCLUSIVE;
      sc_create_info.queueFamilyIndexCount = 0;
      sc_create_info.pQueueFamilyIndices = nil;
    }

    sc_create_info.preTransform = swapchain_support.capabilities.currentTransform;
    sc_create_info.compositeAlpha = { .OPAQUE };

    sc_create_info.presentMode = present_mode;
    sc_create_info.clipped = true;
    // sc_create_info.oldSwapchain = nil;

    res := vk.CreateSwapchainKHR(vk_device, &sc_create_info, nil, &vk_swapchain);
    if res != .SUCCESS
    {
      panic("error creating swap chain");
    }


    sc_image_count : u32;
    vk.GetSwapchainImagesKHR(vk_device, vk_swapchain, &sc_image_count, nil);
    swapchain_images = make([]vk.Image, sc_image_count);
    vk.GetSwapchainImagesKHR(vk_device, vk_swapchain, &sc_image_count, &swapchain_images[0]);
  }

  // ----------- Image Views -----------

  swapchain_image_views =  make([]vk.ImageView, len(swapchain_images));

  for _, i in swapchain_images
  {
    iv_create_info : vk.ImageViewCreateInfo = {
      sType = .IMAGE_VIEW_CREATE_INFO,
      image = swapchain_images[i],
      viewType = .D2,
      format = swapchain_image_format,
      components = {.IDENTITY, .IDENTITY, .IDENTITY, .IDENTITY},
      subresourceRange = {{.COLOR}, 0, 1, 0, 1},
    };
    res := vk.CreateImageView(vk_device, &iv_create_info, nil, &swapchain_image_views[i]);
    if res != .SUCCESS
    {
      panic("could not create image view");
    }
  }

  // ----------- Render Pass -----------
  {

    // attachment description

    // we will only use one color attachment

    color_attachment : vk.AttachmentDescription = {
      format = swapchain_image_format,
      samples = {._1},
      loadOp = .CLEAR,
      storeOp = .STORE,
      initialLayout = .UNDEFINED,
      finalLayout = .PRESENT_SRC_KHR,
    };

    //subpasses and attachment references
    color_attachment_ref : vk.AttachmentReference = {
      attachment = 0,
      layout = .COLOR_ATTACHMENT_OPTIMAL
    };

    subpass : vk.SubpassDescription = {
      pipelineBindPoint = .GRAPHICS,
      colorAttachmentCount = 1,
      pColorAttachments = &color_attachment_ref,
    };

    //subpass dependencies (wtf even is this)
    dependency : vk.SubpassDependency = {
      srcSubpass = vk.SUBPASS_EXTERNAL,
      dstSubpass = 0,
      srcStageMask = { .COLOR_ATTACHMENT_OUTPUT },
      srcAccessMask = {},
      dstStageMask = { .COLOR_ATTACHMENT_OUTPUT },
      dstAccessMask = { .COLOR_ATTACHMENT_WRITE },
    };

    render_pass_create_info : vk.RenderPassCreateInfo = {
      sType = .RENDER_PASS_CREATE_INFO,
      attachmentCount = 1,
      pAttachments = &color_attachment,
      subpassCount = 1,
      pSubpasses = &subpass,
      dependencyCount = 1,
      pDependencies = &dependency,
    };

    res := vk.CreateRenderPass(vk_device, &render_pass_create_info, nil, &vk_render_pass);
    if res != .SUCCESS do panic("failed to create render pass");
  }

  // ----------- Descriptor Set Layout -----------
  {

    ubo_layout_binding : vk.DescriptorSetLayoutBinding = {
      binding = 0,
      descriptorType = .UNIFORM_BUFFER,
      descriptorCount = 1,
      stageFlags = {.VERTEX},
      pImmutableSamplers = nil
    };

    layout_info : vk.DescriptorSetLayoutCreateInfo = {
      sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount = 1,
      pBindings = &ubo_layout_binding,
    };

    res := vk.CreateDescriptorSetLayout(vk_device, &layout_info, nil , &descriptor_set_layout);
    if res != .SUCCESS do panic("failed to create descriptor set layout");
  }

  // ----------- Graphics Pipeline -----------

  create_shader_module :: proc(vk_device : vk.Device, code : []byte) -> vk.ShaderModule
  {
    shader_module : vk.ShaderModule;

    create_info : vk.ShaderModuleCreateInfo = {
      sType = .SHADER_MODULE_CREATE_INFO,
      codeSize = len(code),
      pCode = cast(^u32)&code[0],
    };

    res := vk.CreateShaderModule(vk_device, &create_info, nil, &shader_module);
    if res != .SUCCESS
    {
      panic("could not create shader module");
    }

    return shader_module;
  }

  {
    // loading shader files
    frag_file, vert_file : []byte;

    err : bool;

    frag_file, err = os.read_entire_file("frag.spv");
    defer delete(frag_file);
    if !err
    {
      panic("could not load frag spv file");
    }

    vert_file, err = os.read_entire_file("vert.spv");
    defer delete(vert_file);
    if !err
    {
      panic("could not load vert spv file");
    }

    // making shader modules
    vert_shader_module := create_shader_module(vk_device, vert_file);
    defer vk.DestroyShaderModule(vk_device, vert_shader_module, nil);
    frag_shader_module := create_shader_module(vk_device, frag_file);
    defer vk.DestroyShaderModule(vk_device, frag_shader_module, nil);

    // pipeline shader stages
    vert_shader_stage_info : vk.PipelineShaderStageCreateInfo = {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.VERTEX},
      module = vert_shader_module,
      pName = "main"
    };

    frag_shader_stage_info : vk.PipelineShaderStageCreateInfo = {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.FRAGMENT},
      module = frag_shader_module,
      pName = "main"
    };

    shader_stages : [2]vk.PipelineShaderStageCreateInfo = {vert_shader_stage_info, frag_shader_stage_info};

    //fixed functions

    // vertex input

    binding_description := get_binding_description();
    attribute_descriptions := get_attribute_descriptions();


    vertex_input_create_info : vk.PipelineVertexInputStateCreateInfo = {
      sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      vertexBindingDescriptionCount = 1,
      pVertexBindingDescriptions = &binding_description,
      vertexAttributeDescriptionCount = u32(len(attribute_descriptions)),
      pVertexAttributeDescriptions = &attribute_descriptions[0],
    };

    //input assembly
    input_assembly_create_info : vk.PipelineInputAssemblyStateCreateInfo = {
      sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      topology = .TRIANGLE_LIST,
      primitiveRestartEnable = false
    };

    //viewport
    viewport : vk.Viewport = {
      x = 0.0,
      y = 0.0,
      width = cast(f32)swapchain_extent.width,
      height = cast(f32)swapchain_extent.height,
      minDepth = 0.0,
      maxDepth = 1.0
    };

    //scissor
    scissor : vk.Rect2D = {
      offset = {0, 0},
      extent = swapchain_extent,
    };

    viewport_state_create_info : vk.PipelineViewportStateCreateInfo = {
      sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      viewportCount = 1,
      pViewports = &viewport,
      scissorCount = 1,
      pScissors = &scissor,
    };

    //rasterizer

    rasterizer : vk.PipelineRasterizationStateCreateInfo = {
      sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      depthClampEnable = false,
      rasterizerDiscardEnable = false,
      polygonMode = .FILL,
      lineWidth = 1.0,
      cullMode = {.BACK},
      frontFace = .CLOCKWISE,
      depthBiasEnable = false,
    };


    // multisampling
    // disabled for now
    multisampling : vk.PipelineMultisampleStateCreateInfo = {
      sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      sampleShadingEnable = false,
      rasterizationSamples = {._1},
      minSampleShading = 1.0,
      pSampleMask = nil,
      alphaToCoverageEnable = false,
      alphaToOneEnable = false,
    };

    // color blending
    color_blend_attachment : vk.PipelineColorBlendAttachmentState = {
      colorWriteMask = {.R, .G, .B, .A},
      blendEnable = true,
      srcColorBlendFactor = .SRC_ALPHA,
      dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
      colorBlendOp = .ADD,
      srcAlphaBlendFactor = .ONE,
      dstAlphaBlendFactor = .ZERO,
      alphaBlendOp = .ADD
    };

    color_blending : vk.PipelineColorBlendStateCreateInfo = {
      sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      logicOpEnable = false,
      logicOp = .COPY,
      attachmentCount = 1,
      pAttachments = &color_blend_attachment,
      blendConstants = {0.0,0.0,0.0,0.0},
    };

    // dynamic state

    dynamic_states : [2]vk.DynamicState = {
      .VIEWPORT, .LINE_WIDTH,
    };

    dynamic_state_create_info : vk.PipelineDynamicStateCreateInfo = {
      sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      dynamicStateCount = 2,
      pDynamicStates = &dynamic_states[0],
    };

    // pipeline layout
    pipeline_layout_create_info : vk.PipelineLayoutCreateInfo = {
      sType = .PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount = 1,
      pSetLayouts = &descriptor_set_layout,
      pushConstantRangeCount = 0,
      pPushConstantRanges = nil
    };

    res := vk.CreatePipelineLayout(vk_device, &pipeline_layout_create_info, nil, &pipeline_layout);
    if res != .SUCCESS do panic("error at creating pipeline layout");

    // finally creating pipeline
    pipeline_create_info : vk.GraphicsPipelineCreateInfo = {
      sType = .GRAPHICS_PIPELINE_CREATE_INFO,
      stageCount = 2,
      pStages = &shader_stages[0],
      pVertexInputState = &vertex_input_create_info,
      pInputAssemblyState = &input_assembly_create_info,
      pViewportState = &viewport_state_create_info,
      pRasterizationState = &rasterizer,
      pMultisampleState = &multisampling,
      pDepthStencilState = nil,
      pColorBlendState = &color_blending,
      pDynamicState = nil,
      layout = pipeline_layout,
      renderPass = vk_render_pass,
      subpass = 0,
      // basePipelineHandle = nil,
      // basePipelineIndex = -1,
    };

    res = vk.CreateGraphicsPipelines(vk_device, 0, 1, &pipeline_create_info, nil, &vk_graphics_pipeline);
    if res != .SUCCESS do panic("failed to create pipeline");

  }

  // ----------- Framebuffers -----------

  swapchain_framebuffers = make([]vk.Framebuffer, len(swapchain_image_views));

  for _, i in swapchain_image_views
  {
    attachments : [1]vk.ImageView = {
      swapchain_image_views[i]
    };

    framebuffer_info : vk.FramebufferCreateInfo = {
      sType = .FRAMEBUFFER_CREATE_INFO,
      renderPass = vk_render_pass,
      attachmentCount = 1,
      pAttachments = &attachments[0],
      width = swapchain_extent.width,
      height = swapchain_extent.height,
      layers = 1,
    };

    res := vk.CreateFramebuffer(vk_device, &framebuffer_info, nil, &swapchain_framebuffers[i]);
    if res != .SUCCESS do panic("failed to create framebuffer");

  }

  swapchain_created = true;
}

create_uniform_buffers :: proc()
{
  buffer_size : vk.DeviceSize = size_of(UniformBufferObject);
  uniform_buffers = make([]vk.Buffer, len(swapchain_images));
  uniform_buffers_memory = make([]vk.DeviceMemory, len(swapchain_images));

  for i:=0;i<len(swapchain_images);i+=1
  {
    create_buffer(physical_device, vk_device, buffer_size, {.UNIFORM_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, &uniform_buffers[i], &uniform_buffers_memory[i]);
  }
}

create_command_buffers :: proc()
{
  // ----------- Command Buffers -----------

  command_buffers = make([]vk.CommandBuffer, len(swapchain_framebuffers));
  {

    // command buffer allocation

    alloc_info : vk.CommandBufferAllocateInfo = {
      sType = .COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool = command_pool,
      level = .PRIMARY,
      commandBufferCount = cast(u32)len(command_buffers),
    };

    res := vk.AllocateCommandBuffers(vk_device, &alloc_info, &command_buffers[0]);
    if res != .SUCCESS do panic("failed to allocate command buffers");

    // recording command buffers

    for _, i in command_buffers
    {
      begin_info : vk.CommandBufferBeginInfo = {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = {},
        pInheritanceInfo = nil,
      };

      res = vk.BeginCommandBuffer(command_buffers[i], &begin_info);
      if res != .SUCCESS do panic("failed to begin recording command buffer");

      render_pass_info : vk.RenderPassBeginInfo = {
        sType = .RENDER_PASS_BEGIN_INFO,
        renderPass = vk_render_pass,
        framebuffer = swapchain_framebuffers[i],
        renderArea = {
          offset = {0, 0},
          extent = swapchain_extent,
        },
      };

      clear_color : vk.ClearValue;
      clear_color.color.float32 = {0.854, 0.078, 0.901,1.0};

      render_pass_info.clearValueCount = 1;
      render_pass_info.pClearValues = &clear_color;

      vk.CmdBeginRenderPass(command_buffers[i], &render_pass_info, .INLINE);
      vk.CmdBindPipeline(command_buffers[i], .GRAPHICS, vk_graphics_pipeline);

      vertex_buffers : [1]vk.Buffer = {vertex_buffer};
      offsets : [1]vk.DeviceSize = {0};
      vk.CmdBindVertexBuffers(command_buffers[i], 0, 1, &vertex_buffers[0], &offsets[0]);
      vk.CmdBindIndexBuffer(command_buffers[i], index_buffer, 0, .UINT16);

      vk.CmdDrawIndexed(command_buffers[i], u32(len(indices)), 1, 0, 0, 0);
      vk.CmdEndRenderPass(command_buffers[i]);
      res = vk.EndCommandBuffer(command_buffers[i]);
      if res != .SUCCESS do panic("failed to record command buffer");

    }

  }
}

recreate_swapchain :: proc()
{
  //checking if window is minimized
  width, height : i32;
  glfw.GetFramebufferSize(glfw_window, &width, &height);
  for width == 0 || height == 0
  {
    glfw.GetFramebufferSize(glfw_window, &width, &height);
    glfw.WaitEvents();
  }

  vk.DeviceWaitIdle(vk_device);

  // Cleaning up previous swap chain
  if swapchain_created do cleanup_swapchain();
  
  init_swapchain_dependant_stuff();
  create_uniform_buffers();
  create_command_buffers();
}

draw_frame :: proc()
{
  // waiting for fence
  vk.WaitForFences(vk_device, 1, &in_flight_fences[current_frame], true, max(u64));

  // acquiring an image from the swap chain..

  image_index : u32;
  res := vk.AcquireNextImageKHR(vk_device, vk_swapchain, max(u64), image_available_semaphores[current_frame], 0, &image_index);
  if res == .ERROR_OUT_OF_DATE_KHR
  {
    recreate_swapchain();
    //not sure what to do here, this may be wrong
    // current_frame = 0;
    return;
  }
  else if res != .SUCCESS && res != .SUBOPTIMAL_KHR
  {
    panic("failed to acquire swap chain image");
  }


  // Check if a previous frame is using this image (i.e. there is its fence to wait on)
  if images_in_flight[image_index] != 0
  {
    vk.WaitForFences(vk_device, 1, &images_in_flight[image_index], true, max(u64));
  }

  // Mark the image as now being in use by this frame
  images_in_flight[image_index] = in_flight_fences[current_frame];

  // submitting the right command buffer

  wait_semaphores : [1]vk.Semaphore = {image_available_semaphores[current_frame]};
  wait_stages : vk.PipelineStageFlags = {.COLOR_ATTACHMENT_OUTPUT};
  signal_semaphores : [1]vk.Semaphore = {render_finished_semaphores[current_frame]};

  update_uniform_buffer(image_index);

  submit_info : vk.SubmitInfo = {
    sType = .SUBMIT_INFO,
    waitSemaphoreCount = 1,
    pWaitSemaphores = &wait_semaphores[0],
    pWaitDstStageMask = &wait_stages,
    commandBufferCount = 1,
    pCommandBuffers = &command_buffers[image_index],
    signalSemaphoreCount = 1,
    pSignalSemaphores  = &signal_semaphores[0],
  };

  vk.ResetFences(vk_device, 1, &in_flight_fences[current_frame]);

  res = vk.QueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]);
  if res != .SUCCESS do panic("failed to submit draw command buffer");

  // presentation. submitting the rendered image back to the swapchain

  swapchains : [1]vk.SwapchainKHR = {vk_swapchain};

  present_info : vk.PresentInfoKHR = {
    sType = .PRESENT_INFO_KHR,
    waitSemaphoreCount = 1,
    pWaitSemaphores = &signal_semaphores[0],
    swapchainCount = 1,
    pSwapchains = &swapchains[0],
    pImageIndices = &image_index,
    pResults = nil,
  };

  res = vk.QueuePresentKHR(present_queue, &present_info);
  if res == .ERROR_OUT_OF_DATE_KHR || res == .SUBOPTIMAL_KHR || framebuffer_resized
  {
    framebuffer_resized = false;  
    recreate_swapchain();
  }
  else if res != .SUCCESS
  {
    panic("failed to present swap chain image");
  }
  // if res != .SUCCESS do panic("failed to present the image");

  current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

main :: proc()
{
  when ODIN_DEBUG do context.allocator = util.memleak_allocator(true);
  when ODIN_DEBUG do defer util.memleak_dump(context.allocator, dump_proc, nil);

  validation_layers_cstring = make([]cstring, len(validation_layers));
  defer delete(validation_layers_cstring);
  StringArrayToCstringArray(validation_layers[:], validation_layers_cstring);
  required_device_extensions_cstring = make([]cstring, len(required_device_extensions));
  defer delete(required_device_extensions_cstring);
  StringArrayToCstringArray(required_device_extensions[:], required_device_extensions_cstring);
  // Glfw Init
  glfw_err := glfw.Init();
  if glfw_err == 0 do panic("glfw could not init");
  glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API);
  glfw.WindowHint(glfw.RESIZABLE, glfw.TRUE);
  glfw_window = glfw.CreateWindow(W_WIDTH, W_HEIGHT, "Vulkan in Odin", nil, nil);
  glfw.SetFramebufferSizeCallback(glfw_window, framebuffer_resize_callback);

  // Loading Vulkan Functions...
  vk_dll := win32.LoadLibraryW(win32.utf8_to_wstring("vulkan-1.dll"));

  //here you load the functions that do not require an instance
  // after you create the instance you load all the functions that require an instance
  vk.GetInstanceProcAddr = auto_cast win32.GetProcAddress(vk_dll, "vkGetInstanceProcAddr");
  if vk.GetInstanceProcAddr == nil do panic("getinstanceprocaddr not loaded");
  vk.load_proc_addresses_no_instance();

  // ----------- Creating Vulkan Instanace -----------

  vk_instance : vk.Instance;

  {

    app_info : vk.ApplicationInfo = {
      sType = .APPLICATION_INFO,
      pApplicationName = "Hello Triangle",
      applicationVersion = vk.MAKE_VERSION(1, 0, 0),
      pEngineName = "No Engine",
      engineVersion = vk.MAKE_VERSION(1, 0, 0),
    };

    create_info : vk.InstanceCreateInfo = {
      sType = vk.StructureType.INSTANCE_CREATE_INFO,
      pApplicationInfo = &app_info,
    };


    glfw_extension_count : u32;
    glfw_extensions := glfw.GetRequiredInstanceExtensions(&glfw_extension_count);

    when VALIDATION_LAYERS_ENABLED
    {
      vk_extension_count := glfw_extension_count + 1;
    }
    else
    {
      vk_extension_count := glfw_extension_count;
    }

    glfw_extensions_slice := mem.slice_ptr(glfw_extensions,int(glfw_extension_count));
    vk_extensions := make([]cstring, vk_extension_count);
    defer delete(vk_extensions);
    copy(vk_extensions, glfw_extensions_slice);
    // adding the debug extension
    when VALIDATION_LAYERS_ENABLED
    {
      vk_extensions[glfw_extension_count] = vk.EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    create_info.enabledExtensionCount = vk_extension_count;
    create_info.ppEnabledExtensionNames = &vk_extensions[0];

    //Vulkan validation layers
    if VALIDATION_LAYERS_ENABLED && !CheckValidationLayerSupport()
    {
      panic("not all validation layers are available");
    }

    //Listing all available instance extensions
    /*
    {
      vk_extension_property_count : u32;
      vk.EnumerateInstanceExtensionProperties(nil, &vk_extension_property_count, nil);
      vk_extension_properties := make([]vk.ExtensionProperties, vk_extension_property_count);
      defer delete(vk_extension_properties);
      vk.EnumerateInstanceExtensionProperties(nil, &vk_extension_property_count, &vk_extension_properties[0]);

      fmt.println("Available extensions:");

      for i:u32= 0; i < vk_extension_property_count; i+=1
      {
        fmt.println("\t", cstring(&(vk_extension_properties[i].extensionName[0])));
      }
    }
    */

    debug_utils_msg_create_info_for_vk_instance : vk.DebugUtilsMessengerCreateInfoEXT;

    //Specifying validation layers to create_info
    when VALIDATION_LAYERS_ENABLED
    {
      create_info.enabledLayerCount = len(validation_layers);
      create_info.ppEnabledLayerNames = &validation_layers_cstring[0];

      PopulateDebugMessengerCreateInfo(&debug_utils_msg_create_info_for_vk_instance);
      create_info.pNext = auto_cast &debug_utils_msg_create_info_for_vk_instance;
    }

    vk_res := vk.CreateInstance(&create_info, nil, &vk_instance);
    if vk_res != vk.Result.SUCCESS
    {
      panic("error at vk.CreateInstance");
    }

    //Loading all the vulkan functions that require a vulkan instance
    vk.load_proc_addresses_with_instance(&vk_instance);

    //Outputting vulkan version
    pApiVersion: u32;
    vk_res = vk.EnumerateInstanceVersion(&pApiVersion);
    if vk_res != vk.Result.SUCCESS
    {
      panic("error at vk.enumerateinstanceversion");
    }
    fmt.printf("Vulkan Api Version: major: %v minor: %v patch %v\n", pApiVersion >> 22, 
    (pApiVersion >> 12) & 0x3ff,
    pApiVersion & 0xfff);
  }

  // ----------- Window Surface Creation -----------


  {
    //this is win32 specific code :o
    create_info : vk.Win32SurfaceCreateInfoKHR = {
      sType = .WIN32_SURFACE_CREATE_INFO_KHR,
      hwnd = win32.HWND(glfw.GetWin32Window(glfw_window)),
      hinstance = win32.HANDLE(win32.GetModuleHandleW(nil)),
    };

    res := vk.CreateWin32SurfaceKHR(vk_instance, &create_info, nil, &surface);
    if res != .SUCCESS
    {
      panic("failed to create window surface");
    }

    // instead of doing that, we will use glfw to create a surface
    //  doing it this way is platform agnostic
    // NOTE(lucypero): actually we won't because the glfw bindings i use lack this function..
    // res := glfw.CreateWindowSurface()
    
    // Querying for presentation support
  }


  // -----------  Setting up the debug messenger -----------

  when VALIDATION_LAYERS_ENABLED
  {
    vk_debug_messenger : vk.DebugUtilsMessengerEXT;

    {
      debug_utils_msg_create_info : vk.DebugUtilsMessengerCreateInfoEXT;
      PopulateDebugMessengerCreateInfo(&debug_utils_msg_create_info);

      res := vk.CreateDebugUtilsMessengerEXT(vk_instance, &debug_utils_msg_create_info, nil, &vk_debug_messenger);
      if res != .SUCCESS
      {
        panic("error at vk.createdebugutilsmessengerext");
      }

      //send debug message
      /*
      msg_callback_data : vk.DebugUtilsMessengerCallbackDataEXT = {
        .DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
        nil, {}, nil, 0, "test message", 0, nil, 0, nil, 0, nil
      };
      vk.SubmitDebugUtilsMessageEXT(vk_instance, {.WARNING}, {.GENERAL}, &msg_callback_data);
      */

    }
  }

  // ----------- Selecting Physical Device -----------



  {
    device_count : u32;
    vk.EnumeratePhysicalDevices(vk_instance, &device_count, nil);
    if device_count == 0
    {
      panic("no compatible device for vulkan found");
    }

    devices := make([]vk.PhysicalDevice, device_count);
    defer delete(devices);
    vk.EnumeratePhysicalDevices(vk_instance, &device_count, &devices[0]);

    // pick a suitable device
    for i:int=0; i<len(devices); i+=1
    {
      if(is_device_suitable(devices[i], surface))
      {
        //Found the device we'll use
        physical_device = devices[i];
        break;
      }
    }

    //get some info about the picked device, just for fun
    device_properties : vk.PhysicalDeviceProperties;
    vk.GetPhysicalDeviceProperties(physical_device, &device_properties);
    fmt.println("Vulkan will run on device: ", cstring(&device_properties.deviceName[0]));

  }

  // ----------- Setting up a logical device -----------

  {
    // specifying the queues to be created

    indices := find_queue_families(physical_device, surface);

    unique_queue_families : Set;
    defer delete(unique_queue_families);
    set_add(&unique_queue_families, indices.graphics_family.(u32));
    set_add(&unique_queue_families, indices.present_family.(u32));

    queue_create_infos := make([]vk.DeviceQueueCreateInfo, len(unique_queue_families));
    defer delete(queue_create_infos);

    for i:=0;i<len(unique_queue_families);i+=1
    {
      queue_priority : f32 = 1.0;
      queue_create_info : vk.DeviceQueueCreateInfo = {
        sType = .DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex = unique_queue_families[i],
        queueCount = 1,
        pQueuePriorities = &queue_priority,
      };
      queue_create_infos[i] = queue_create_info;
    }

    // Specifying used device features

    // for now no special features are required
    device_features : vk.PhysicalDeviceFeatures;

    // creating the logical device
    vk_device_create_info : vk.DeviceCreateInfo = {
      sType = .DEVICE_CREATE_INFO,
      pQueueCreateInfos = &queue_create_infos[0],
      queueCreateInfoCount = u32(len(unique_queue_families)),
      pEnabledFeatures = &device_features,
      enabledExtensionCount = len(required_device_extensions),
      ppEnabledExtensionNames = &required_device_extensions_cstring[0],
    };

    when VALIDATION_LAYERS_ENABLED 
    {
      vk_device_create_info.enabledLayerCount = len(validation_layers);
      vk_device_create_info.ppEnabledLayerNames = &validation_layers_cstring[0];
    }
    else
    {
      vk_device_create_info.enabledLayerCount = 0;
    }

    res := vk.CreateDevice(physical_device, &vk_device_create_info, nil, &vk_device);
    if res != .SUCCESS do panic("failed to create logical device");

    vk.GetDeviceQueue(vk_device, indices.graphics_family.(u32), 0, &graphics_queue);
    vk.GetDeviceQueue(vk_device, indices.present_family.(u32), 0, &present_queue);
  }

  init_swapchain_dependant_stuff();

  // ----------- Command Pool -----------
  {
    queue_family_indices := find_queue_families(physical_device, surface);
    pool_info : vk.CommandPoolCreateInfo = {
      sType = .COMMAND_POOL_CREATE_INFO,
      queueFamilyIndex = queue_family_indices.graphics_family.(u32),
      flags = {},
    };

    res := vk.CreateCommandPool(vk_device, &pool_info, nil, &command_pool);
    if res != .SUCCESS do panic("failed to create command pool");
  }

  // ----------- Vertex Buffers -----------
  {

    //Here we create two buffers: one that is cpu visible that we'll use to store the vertex data, and one that is not.. then we will transfer the data from one to the other

    // this is because the final buffer (device_local) is faster, so the vertex data is loaded from high performance memory

    buffer_size : vk.DeviceSize = size_of(vertices[0]) * len(vertices);

    staging_buffer : vk.Buffer;
    staging_buffer_memory : vk.DeviceMemory;

    create_buffer(physical_device, vk_device, buffer_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT}, &staging_buffer, &staging_buffer_memory);

    data : rawptr;
    vk.MapMemory(vk_device, staging_buffer_memory, 0, buffer_size, nil, &data);
    mem.copy(data, &vertices[0], cast(int)buffer_size);
    vk.UnmapMemory(vk_device, staging_buffer_memory);

    create_buffer(physical_device, vk_device, buffer_size, {.TRANSFER_DST, .VERTEX_BUFFER}, {.DEVICE_LOCAL}, &vertex_buffer, &vertex_buffer_memory);

    copy_buffer(vk_device, command_pool, graphics_queue, staging_buffer, vertex_buffer, buffer_size);

    vk.DestroyBuffer(vk_device, staging_buffer, nil);
    vk.FreeMemory(vk_device, staging_buffer_memory, nil);
  }

  // ----------- Index Buffers -----------

  {
    buffer_size : vk.DeviceSize = size_of(indices[0]) * len(indices);

    staging_buffer : vk.Buffer;
    staging_buffer_memory : vk.DeviceMemory;

    create_buffer(physical_device, vk_device, buffer_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT}, &staging_buffer, &staging_buffer_memory);

    data : rawptr;
    vk.MapMemory(vk_device, staging_buffer_memory, 0, buffer_size, nil, &data);
    mem.copy(data, &indices[0], cast(int)buffer_size);
    vk.UnmapMemory(vk_device, staging_buffer_memory);

    create_buffer(physical_device, vk_device, buffer_size, {.TRANSFER_DST, .INDEX_BUFFER}, {.DEVICE_LOCAL}, &index_buffer, &index_buffer_memory);

    copy_buffer(vk_device, command_pool, graphics_queue, staging_buffer, index_buffer, buffer_size);

    vk.DestroyBuffer(vk_device, staging_buffer, nil);
    vk.FreeMemory(vk_device, staging_buffer_memory, nil);
  }

  // ----------- Uniform Buffers -----------

  create_uniform_buffers();

  create_command_buffers();

  // ----------- Semaphores and fences -----------

  //creating semaphores used to sync queue operations of draw commnads and presentation

  image_available_semaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT);
  defer delete(image_available_semaphores);
  render_finished_semaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT);
  defer delete(render_finished_semaphores);

  in_flight_fences = make([]vk.Fence, MAX_FRAMES_IN_FLIGHT);
  defer delete(in_flight_fences);

  images_in_flight = make([]vk.Fence, len(swapchain_images));
  defer delete(images_in_flight);

  {
    semaphore_info : vk.SemaphoreCreateInfo = {
      sType = .SEMAPHORE_CREATE_INFO,
    };

    fence_info : vk.FenceCreateInfo = {
      sType = .FENCE_CREATE_INFO,
      flags = {.SIGNALED},
    };

    for i:= 0 ; i < MAX_FRAMES_IN_FLIGHT ; i+=1
    {
      res := vk.CreateSemaphore(vk_device, &semaphore_info, nil, &image_available_semaphores[i]);
      if res != .SUCCESS do panic("failed to create image available semaphore");
      res = vk.CreateSemaphore(vk_device, &semaphore_info, nil, &render_finished_semaphores[i]);
      if res != .SUCCESS do panic("failed to create render finished semaphore");
      res = vk.CreateFence(vk_device, &fence_info, nil, &in_flight_fences[i]);
      if res != .SUCCESS do panic("failed to create fence");
    }

  }


  // ----------- Main Loop -----------


  for glfw.WindowShouldClose(glfw_window) == 0
  {
    glfw.PollEvents();
    
    // drawing triangle...
    draw_frame();
  }

  vk.DeviceWaitIdle(vk_device);

  // ----------- Cleaning up -----------


  when VALIDATION_LAYERS_ENABLED
  {
    // if u comment this, the debug callback should output something but it doesn't.. i am scared
    vk.DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nil);
  }

  cleanup_swapchain();

  vk.DestroyDescriptorSetLayout(vk_device, descriptor_set_layout, nil);

  vk.DestroyBuffer(vk_device, vertex_buffer, nil);
  vk.FreeMemory(vk_device, vertex_buffer_memory, nil);

  vk.DestroyBuffer(vk_device, index_buffer, nil);
  vk.FreeMemory(vk_device, index_buffer_memory, nil);

  vk.DestroySurfaceKHR(vk_instance, surface, nil);
  vk.DestroyDevice(vk_device, nil);
  vk.DestroyCommandPool(vk_device, command_pool, nil);

  for i:= 0 ; i < MAX_FRAMES_IN_FLIGHT ; i+=1
  {
    vk.DestroySemaphore(vk_device, render_finished_semaphores[i], nil);
    vk.DestroySemaphore(vk_device, image_available_semaphores[i], nil);
    vk.DestroyFence(vk_device, in_flight_fences[i], nil);
  }

  vk.DestroyInstance(vk_instance, nil);

  glfw.DestroyWindow(glfw_window);
  glfw.Terminate();


}
