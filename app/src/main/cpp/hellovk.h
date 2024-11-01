/*
 * Based on https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f and
 * https://github.com/android/ndk-samples/tree/main/hello-vulkan
 */

#include <android/asset_manager.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <assert.h>
#include <vulkan/vulkan.h>

#include <array>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace vkt {
#define LOG_TAG "hellovkjni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define VK_CHECK(x)                           \
  do {                                        \
    VkResult err = x;                         \
    if (err) {                                \
      LOGE("Detected Vulkan error: %d", err); \
      abort();                                \
    }                                         \
  } while (0)

struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    bool isComplete() {
        return computeFamily.has_value();
    }
};

struct ANativeWindowDeleter {
    void operator()(ANativeWindow *window) { ANativeWindow_release(window); }
};

std::vector<uint8_t> LoadBinaryFileToVector(const char *file_path, AAssetManager *assetManager) {
    std::vector<uint8_t> file_content;
    assert(assetManager);
    AAsset *file = AAssetManager_open(assetManager, file_path, AASSET_MODE_BUFFER);
    size_t file_length = AAsset_getLength(file);

    file_content.resize(file_length);

    AAsset_read(file, file_content.data(), file_length);
    AAsset_close(file);
    return file_content;
}

const char *toStringMessageSeverity(VkDebugUtilsMessageSeverityFlagBitsEXT s) {
    switch (s) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            return "VERBOSE";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            return "ERROR";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            return "WARNING";
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            return "INFO";
        default:
            return "UNKNOWN";
    }
}

const char *toStringMessageType(VkDebugUtilsMessageTypeFlagsEXT s) {
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "General | Validation | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "Validation | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "General | Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT))
        return "Performance";
    if (s == (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        return "General | Validation";
    if (s == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) return "Validation";
    if (s == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return "General";
    return "Unknown";
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void * /* pUserData */) {
    auto ms = toStringMessageSeverity(messageSeverity);
    auto mt = toStringMessageType(messageType);
    printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);
    return VK_FALSE;
}

static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            nullptr,
            0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            debugCallback
    };
}

static VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(
        VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks *pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloVK {
public:
    void initVulkanCompute();
    void compute();
    void cleanupCompute();
    void reset(ANativeWindow *newWindow, AAssetManager *newManager);
    bool initialized = false;

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDeviceAndQueue();
    void createDescriptorSetLayout();
    void createCommandPool();
    void createCommandBuffer();
    static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkValidationLayerSupport();
    static std::vector<const char *> getRequiredExtensions(bool enableValidation);
    VkShaderModule createShaderModule(const std::vector<uint8_t> &code);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkDeviceSize memorySize);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer &buffer,
                      VkDeviceMemory &bufferMemory);
    void createStorageBuffers();
    void createComputePipeline();
    void createDescriptorPool();
    void createDescriptorSets();

    /*
     * In order to enable validation layer toggle this to true and
     * follow the README.md instructions concerning the validation
     * layers. You will be required to add separate vulkan validation
     * '*.so' files in order to enable this.
     *
     * The validation layers are not shipped with the APK as they are sizeable.
     */
    bool enableValidationLayers = false;

    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> deviceExtensions = {};
    std::unique_ptr<ANativeWindow, ANativeWindowDeleter> window;
    AAssetManager *assetManager;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t queueFamilyIndex;
    VkQueue computeQueue;

    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;

    std::vector<VkBuffer> storageBuffers;
    std::vector<VkDeviceMemory> storageBuffersMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorSet descriptorSet;
};

#define BUFFER_LENGTH 16384
#define BUFFER_SIZE (BUFFER_LENGTH * sizeof(int32_t))
#define BUFFER_COUNT 2

void HelloVK::initVulkanCompute() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDeviceAndQueue();
    setupDebugMessenger();
    createDescriptorSetLayout();
    createStorageBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createComputePipeline();
    createCommandPool();
    createCommandBuffer();
    initialized = true;
}

/*
 *	Create a buffer with specified usage and memory properties
 *  Upon creation, these buffers will list memory requirements which need to be
 *  satisfied by the device in use in order to be created.
 */
void HelloVK::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags properties, VkBuffer &buffer,
                           VkDeviceMemory &bufferMemory) {
    const VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            size,
            usage,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueFamilyIndex
    };

    VK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, size);

    const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            memRequirements.size,
            memoryTypeIndex
    };

    VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &bufferMemory));

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

/*
 * Finds the index of the memory heap which matches a particular buffer's memory
 * requirements. Vulkan manages these requirements as a bitset, in this case
 * expressed through a uint32_t.
 */
uint32_t HelloVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkDeviceSize memorySize) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties &&
            (memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size)) {
            return i;
        }
    }

    return memoryTypeIndex;
}

void HelloVK::createStorageBuffers() {
    storageBuffers.resize(BUFFER_COUNT);
    storageBuffersMemory.resize(BUFFER_COUNT);

    for (size_t i = 0; i < BUFFER_COUNT; i++) {
        createBuffer(BUFFER_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     storageBuffers[i], storageBuffersMemory[i]);
    }
}

void HelloVK::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2] = {
            {
                    0,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    nullptr
            },
            {
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    nullptr
            }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            2,
            descriptorSetLayoutBindings
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout));
}

void HelloVK::reset(ANativeWindow *newWindow, AAssetManager *newManager) {
  window.reset(newWindow);
  assetManager = newManager;
}

void HelloVK::compute() {
    int32_t *dataIn, *dataOut;
    VK_CHECK(vkMapMemory(device, storageBuffersMemory[0], 0, BUFFER_SIZE, 0, (void**)&dataIn));
    for (uint32_t k = 0; k < BUFFER_LENGTH; k++) {
        dataIn[k] = rand();
    }
    vkUnmapMemory(device, storageBuffersMemory[0]);

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
    };
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, BUFFER_LENGTH, 1, 1);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &commandBuffer,
            0,
            nullptr
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, 0));
    VK_CHECK(vkQueueWaitIdle(queue));

    VK_CHECK(vkMapMemory(device, storageBuffersMemory[1], 0, BUFFER_SIZE, 0, (void **)&dataOut));
    bool error = false;
    for (uint32_t k = 0; k < BUFFER_LENGTH; k++) {
        if (dataOut[k] != dataIn[k]) {
            LOGE("Error occurred in compute shader");
            error = true;
            break;
        }
    }
    if (!error) {
        LOGI("Compute shader executed successfully");
    }
    vkUnmapMemory(device, storageBuffersMemory[1]);
    vkResetCommandBuffer(commandBuffer, 0);
}

void HelloVK::createDescriptorPool() {
    VkDescriptorPoolSize descriptorPoolSize = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            2
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            1,
            &descriptorPoolSize
    };

    VK_CHECK(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));
}

void HelloVK::createDescriptorSets() {
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptorPool,
            1,
            &descriptorSetLayout
    };

    VK_CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    VkDescriptorBufferInfo in_descriptorBufferInfo = {
            storageBuffers[0],
            0,
            VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo out_descriptorBufferInfo = {
            storageBuffers[1],
            0,
            VK_WHOLE_SIZE
    };

    VkWriteDescriptorSet writeDescriptorSet[2] = {
            {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptorSet,
                    0,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &in_descriptorBufferInfo,
                    nullptr
            },
            {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptorSet,
                    1,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &out_descriptorBufferInfo,
                    nullptr
            }
    };
    vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, nullptr);
}

void HelloVK::cleanupCompute() {
    vkDeviceWaitIdle(device);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    for (size_t i = 0; i < BUFFER_COUNT; i++) {
        vkDestroyBuffer(device, storageBuffers[i], nullptr);
        vkFreeMemory(device, storageBuffersMemory[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    initialized = false;
    LOGI("Cleanup completed successfully");
}

void HelloVK::setupDebugMessenger() {
    if (!enableValidationLayers) {
        return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    VK_CHECK(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                          &debugMessenger));
}

bool HelloVK::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
        bool layerFound = false;
        for (const auto &layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }
    return true;
}

std::vector<const char *> HelloVK::getRequiredExtensions(bool enableValidationLayers) {
    std::vector<const char *> extensions;
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

void HelloVK::createInstance() {
    assert(!enableValidationLayers || checkValidationLayerSupport());
    auto requiredExtensions = getRequiredExtensions(enableValidationLayers);
    const VkApplicationInfo applicationInfo = {
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "VKComputeSample",
            0,
            "",
            0,
            VK_MAKE_VERSION(1, 0, 9)
    };

    VkInstanceCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &applicationInfo,
            0,
            nullptr,
            0,
            nullptr
    };

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    if (enableValidationLayers) {
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,extensions.data());
    LOGI("available extensions");
    for (const auto &extension : extensions) {
        LOGI("\t %s", extension.extensionName);
    }
}

// BEGIN DEVICE SUITABILITY
// Functions to find a suitable physical device to execute Vulkan commands.

QueueFamilyIndices HelloVK::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }
    return indices;
}

bool HelloVK::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

bool HelloVK::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    return indices.isComplete() && extensionsSupported;
}

void HelloVK::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    assert(deviceCount > 0);  // failed to find GPUs with Vulkan support!

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    assert(physicalDevice != VK_NULL_HANDLE);  // failed to find a suitable GPU!
}
// END DEVICE SUITABILITY

void HelloVK::createLogicalDeviceAndQueue() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    const float queuePrioritory = 1.0f;
    const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            indices.computeFamily.value(),
            1,
            &queuePrioritory
    };

    const VkDeviceCreateInfo deviceCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            1,
            &deviceQueueCreateInfo,
            0,
            nullptr,
            0,
            nullptr,
            nullptr
    };

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    queueFamilyIndex = indices.computeFamily.value();
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
}

void HelloVK::createComputePipeline() {
    auto compShaderCode = LoadBinaryFileToVector("shaders/shader.comp.spv", assetManager);
    VkShaderModule compShaderModule = createShaderModule(compShaderCode);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayout,
            0,
            nullptr
    };

    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    compShaderModule,
                    "main",
                    nullptr
            },
            pipelineLayout,
            0,
            0
    };

    VK_CHECK(vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, nullptr, &computePipeline));
}

VkShaderModule HelloVK::createShaderModule(const std::vector<uint8_t> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();

    // Satisifies alignment requirements since the allocator
    // in vector ensures worst case requirements
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}

void HelloVK::createCommandPool() {
    VkCommandPoolCreateInfo commandPoolCreateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex
    };
    VK_CHECK(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool));
}

void HelloVK::createCommandBuffer() {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
    };

    VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));
}

}  // namespace vkt