#!/usr/bin/python3 -i
#
# Copyright (c) 2015-2016 The Khronos Group Inc.
# Copyright (c) 2015-2016 Valve Corporation
# Copyright (c) 2015-2016 LunarG, Inc.
# Copyright (c) 2015-2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Mike Stroyan <stroyan@google.com>
# Author: Mark Lobodzinski <mark@lunarg.com>

import os,re,sys
from generator import *
from common_codegen import *

# ThreadGeneratorOptions - subclass of GeneratorOptions.
#
# Adds options used by ThreadOutputGenerator objects during threading
# layer generation.
#
# Additional members
#   prefixText - list of strings to prefix generated header with
#     (usually a copyright statement + calling convention macros).
#   protectFile - True if multiple inclusion protection should be
#     generated (based on the filename) around the entire header.
#   protectFeature - True if #ifndef..#endif protection should be
#     generated around a feature interface in the header file.
#   genFuncPointers - True if function pointer typedefs should be
#     generated
#   protectProto - If conditional protection should be generated
#     around prototype declarations, set to either '#ifdef'
#     to require opt-in (#ifdef protectProtoStr) or '#ifndef'
#     to require opt-out (#ifndef protectProtoStr). Otherwise
#     set to None.
#   protectProtoStr - #ifdef/#ifndef symbol to use around prototype
#     declarations, if protectProto is set
#   apicall - string to use for the function declaration prefix,
#     such as APICALL on Windows.
#   apientry - string to use for the calling convention macro,
#     in typedefs, such as APIENTRY.
#   apientryp - string to use for the calling convention macro
#     in function pointer typedefs, such as APIENTRYP.
#   indentFuncProto - True if prototype declarations should put each
#     parameter on a separate line
#   indentFuncPointer - True if typedefed function pointers should put each
#     parameter on a separate line
#   alignFuncParam - if nonzero and parameters are being put on a
#     separate line, align parameter names at the specified column
class ThreadGeneratorOptions(GeneratorOptions):
    def __init__(self,
                 filename = None,
                 directory = '.',
                 apiname = None,
                 profile = None,
                 versions = '.*',
                 emitversions = '.*',
                 defaultExtensions = None,
                 addExtensions = None,
                 removeExtensions = None,
                 emitExtensions = None,
                 sortProcedure = regSortFeatures,
                 prefixText = "",
                 genFuncPointers = True,
                 protectFile = True,
                 protectFeature = True,
                 apicall = '',
                 apientry = '',
                 apientryp = '',
                 indentFuncProto = True,
                 indentFuncPointer = False,
                 alignFuncParam = 0,
                 expandEnumerants = True):
        GeneratorOptions.__init__(self, filename, directory, apiname, profile,
                                  versions, emitversions, defaultExtensions,
                                  addExtensions, removeExtensions, emitExtensions, sortProcedure)
        self.prefixText      = prefixText
        self.genFuncPointers = genFuncPointers
        self.protectFile     = protectFile
        self.protectFeature  = protectFeature
        self.apicall         = apicall
        self.apientry        = apientry
        self.apientryp       = apientryp
        self.indentFuncProto = indentFuncProto
        self.indentFuncPointer = indentFuncPointer
        self.alignFuncParam  = alignFuncParam
        self.expandEnumerants = expandEnumerants


# ThreadOutputGenerator - subclass of OutputGenerator.
# Generates Thread checking framework
#
# ---- methods ----
# ThreadOutputGenerator(errFile, warnFile, diagFile) - args as for
#   OutputGenerator. Defines additional internal state.
# ---- methods overriding base class ----
# beginFile(genOpts)
# endFile()
# beginFeature(interface, emit)
# endFeature()
# genType(typeinfo,name)
# genStruct(typeinfo,name)
# genGroup(groupinfo,name)
# genEnum(enuminfo, name)
# genCmd(cmdinfo)
class ThreadOutputGenerator(OutputGenerator):
    """Generate specified API interfaces in a specific style, such as a C header"""

    inline_copyright_message = """
// This file is ***GENERATED***.  Do Not Edit.
// See layer_chassis_dispatch_generator.py for modifications.

/* Copyright (c) 2015-2018 The Khronos Group Inc.
 * Copyright (c) 2015-2018 Valve Corporation
 * Copyright (c) 2015-2018 LunarG, Inc.
 * Copyright (c) 2015-2018 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Mark Lobodzinski <mark@lunarg.com>
 */"""

    inline_custom_header_preamble = """
#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <string>

VK_DEFINE_NON_DISPATCHABLE_HANDLE(DISTINCT_NONDISPATCHABLE_PHONY_HANDLE)
// The following line must match the vulkan_core.h condition guarding VK_DEFINE_NON_DISPATCHABLE_HANDLE
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || defined(__ia64) || \
    defined(_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
// If pointers are 64-bit, then there can be separate counters for each
// NONDISPATCHABLE_HANDLE type.  Otherwise they are all typedef uint64_t.
#define DISTINCT_NONDISPATCHABLE_HANDLES
// Make sure we catch any disagreement between us and the vulkan definition
static_assert(std::is_pointer<DISTINCT_NONDISPATCHABLE_PHONY_HANDLE>::value,
              "Mismatched non-dispatchable handle handle, expected pointer type.");
#else
// Make sure we catch any disagreement between us and the vulkan definition
static_assert(std::is_same<uint64_t, DISTINCT_NONDISPATCHABLE_PHONY_HANDLE>::value,
              "Mismatched non-dispatchable handle handle, expected uint64_t.");
#endif

// Suppress unused warning on Linux
#if defined(__GNUC__)
#define DECORATE_UNUSED __attribute__((unused))
#else
#define DECORATE_UNUSED
#endif

// clang-format off
static const char DECORATE_UNUSED *kVUID_Threading_Info = "UNASSIGNED-Threading-Info";
static const char DECORATE_UNUSED *kVUID_Threading_MultipleThreads = "UNASSIGNED-Threading-MultipleThreads";
static const char DECORATE_UNUSED *kVUID_Threading_SingleThreadReuse = "UNASSIGNED-Threading-SingleThreadReuse";
// clang-format on

#undef DECORATE_UNUSED

struct object_use_data {
    loader_platform_thread_id thread;
    int reader_count;
    int writer_count;
};

template <typename T>
class counter {
public:
    const char *typeName;
    VkDebugReportObjectTypeEXT objectType;
    debug_report_data **report_data;
    std::unordered_map<T, object_use_data> uses;
    std::mutex counter_lock;
    std::condition_variable counter_condition;


    void StartWrite(T object) {
        if (object == VK_NULL_HANDLE) {
            return;
        }
        bool skip = false;
        loader_platform_thread_id tid = loader_platform_get_thread_id();
        std::unique_lock<std::mutex> lock(counter_lock);
        if (uses.find(object) == uses.end()) {
            // There is no current use of the object.  Record writer thread.
            struct object_use_data *use_data = &uses[object];
            use_data->reader_count = 0;
            use_data->writer_count = 1;
            use_data->thread = tid;
        } else {
            struct object_use_data *use_data = &uses[object];
            if (use_data->reader_count == 0) {
                // There are no readers.  Two writers just collided.
                if (use_data->thread != tid) {
                    skip |= log_msg(*report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, objectType, (uint64_t)(object),
                        kVUID_Threading_MultipleThreads,
                        "THREADING ERROR : object of type %s is simultaneously used in "
                        "thread 0x%" PRIx64 " and thread 0x%" PRIx64,
                        typeName, (uint64_t)use_data->thread, (uint64_t)tid);
                    if (skip) {
                        // Wait for thread-safe access to object instead of skipping call.
                        while (uses.find(object) != uses.end()) {
                            counter_condition.wait(lock);
                        }
                        // There is now no current use of the object.  Record writer thread.
                        struct object_use_data *new_use_data = &uses[object];
                        new_use_data->thread = tid;
                        new_use_data->reader_count = 0;
                        new_use_data->writer_count = 1;
                    } else {
                        // Continue with an unsafe use of the object.
                        use_data->thread = tid;
                        use_data->writer_count += 1;
                    }
                } else {
                    // This is either safe multiple use in one call, or recursive use.
                    // There is no way to make recursion safe.  Just forge ahead.
                    use_data->writer_count += 1;
                }
            } else {
                // There are readers.  This writer collided with them.
                if (use_data->thread != tid) {
                    skip |= log_msg(*report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, objectType, (uint64_t)(object),
                        kVUID_Threading_MultipleThreads,
                        "THREADING ERROR : object of type %s is simultaneously used in "
                        "thread 0x%" PRIx64 " and thread 0x%" PRIx64,
                        typeName, (uint64_t)use_data->thread, (uint64_t)tid);
                    if (skip) {
                        // Wait for thread-safe access to object instead of skipping call.
                        while (uses.find(object) != uses.end()) {
                            counter_condition.wait(lock);
                        }
                        // There is now no current use of the object.  Record writer thread.
                        struct object_use_data *new_use_data = &uses[object];
                        new_use_data->thread = tid;
                        new_use_data->reader_count = 0;
                        new_use_data->writer_count = 1;
                    } else {
                        // Continue with an unsafe use of the object.
                        use_data->thread = tid;
                        use_data->writer_count += 1;
                    }
                } else {
                    // This is either safe multiple use in one call, or recursive use.
                    // There is no way to make recursion safe.  Just forge ahead.
                    use_data->writer_count += 1;
                }
            }
        }
    }

    void FinishWrite(T object) {
        if (object == VK_NULL_HANDLE) {
            return;
        }
        // Object is no longer in use
        std::unique_lock<std::mutex> lock(counter_lock);
        uses[object].writer_count -= 1;
        if ((uses[object].reader_count == 0) && (uses[object].writer_count == 0)) {
            uses.erase(object);
        }
        // Notify any waiting threads that this object may be safe to use
        lock.unlock();
        counter_condition.notify_all();
    }

    void StartRead(T object) {
        if (object == VK_NULL_HANDLE) {
            return;
        }
        bool skip = false;
        loader_platform_thread_id tid = loader_platform_get_thread_id();
        std::unique_lock<std::mutex> lock(counter_lock);
        if (uses.find(object) == uses.end()) {
            // There is no current use of the object.  Record reader count
            struct object_use_data *use_data = &uses[object];
            use_data->reader_count = 1;
            use_data->writer_count = 0;
            use_data->thread = tid;
        } else if (uses[object].writer_count > 0 && uses[object].thread != tid) {
            // There is a writer of the object.
            skip |= false;
            log_msg(*report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, objectType, (uint64_t)(object), kVUID_Threading_MultipleThreads,
                "THREADING ERROR : object of type %s is simultaneously used in "
                "thread 0x%" PRIx64 " and thread 0x%" PRIx64,
                typeName, (uint64_t)uses[object].thread, (uint64_t)tid);
            if (skip) {
                // Wait for thread-safe access to object instead of skipping call.
                while (uses.find(object) != uses.end()) {
                    counter_condition.wait(lock);
                }
                // There is no current use of the object.  Record reader count
                struct object_use_data *use_data = &uses[object];
                use_data->reader_count = 1;
                use_data->writer_count = 0;
                use_data->thread = tid;
            } else {
                uses[object].reader_count += 1;
            }
        } else {
            // There are other readers of the object.  Increase reader count
            uses[object].reader_count += 1;
        }
    }
    void FinishRead(T object) {
        if (object == VK_NULL_HANDLE) {
            return;
        }
        std::unique_lock<std::mutex> lock(counter_lock);
        uses[object].reader_count -= 1;
        if ((uses[object].reader_count == 0) && (uses[object].writer_count == 0)) {
            uses.erase(object);
        }
        // Notify any waiting threads that this object may be safe to use
        lock.unlock();
        counter_condition.notify_all();
    }
    counter(const char *name = "", VkDebugReportObjectTypeEXT type = VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, debug_report_data **rep_data = nullptr) {
        typeName = name;
        objectType = type;
        report_data = rep_data;
    }
};



class ThreadSafety : public ValidationObject {
public:

    // Override chassis read/write locks for this validation object
    void write_lock() {}
    void write_unlock() {}

    std::mutex command_pool_lock;
    std::unordered_map<VkCommandBuffer, VkCommandPool> command_pool_map;

    counter<VkCommandBuffer> c_VkCommandBuffer;
    counter<VkDevice> c_VkDevice;
    counter<VkInstance> c_VkInstance;
    counter<VkQueue> c_VkQueue;
#ifdef DISTINCT_NONDISPATCHABLE_HANDLES

    // Special entry to allow tracking of command pool Reset and Destroy
    counter<VkCommandPool> c_VkCommandPoolContents;
    counter<VkBuffer> c_VkBuffer;
    counter<VkBufferView> c_VkBufferView;
    counter<VkCommandPool> c_VkCommandPool;
    counter<VkDescriptorPool> c_VkDescriptorPool;
    counter<VkDescriptorSet> c_VkDescriptorSet;
    counter<VkDescriptorSetLayout> c_VkDescriptorSetLayout;
    counter<VkDeviceMemory> c_VkDeviceMemory;
    counter<VkEvent> c_VkEvent;
    counter<VkFence> c_VkFence;
    counter<VkFramebuffer> c_VkFramebuffer;
    counter<VkImage> c_VkImage;
    counter<VkImageView> c_VkImageView;
    counter<VkPipeline> c_VkPipeline;
    counter<VkPipelineCache> c_VkPipelineCache;
    counter<VkPipelineLayout> c_VkPipelineLayout;
    counter<VkQueryPool> c_VkQueryPool;
    counter<VkRenderPass> c_VkRenderPass;
    counter<VkSampler> c_VkSampler;
    counter<VkSemaphore> c_VkSemaphore;
    counter<VkShaderModule> c_VkShaderModule;
    counter<VkDebugReportCallbackEXT> c_VkDebugReportCallbackEXT;
    counter<VkObjectTableNVX> c_VkObjectTableNVX;
    counter<VkIndirectCommandsLayoutNVX> c_VkIndirectCommandsLayoutNVX;
    counter<VkDisplayKHR> c_VkDisplayKHR;
    counter<VkDisplayModeKHR> c_VkDisplayModeKHR;
    counter<VkSurfaceKHR> c_VkSurfaceKHR;
    counter<VkSwapchainKHR> c_VkSwapchainKHR;
    counter<VkDescriptorUpdateTemplateKHR> c_VkDescriptorUpdateTemplateKHR;
    counter<VkValidationCacheEXT> c_VkValidationCacheEXT;
    counter<VkSamplerYcbcrConversionKHR> c_VkSamplerYcbcrConversionKHR;
    counter<VkDebugUtilsMessengerEXT> c_VkDebugUtilsMessengerEXT;
    counter<VkAccelerationStructureNV> c_VkAccelerationStructureNV;

#else   // DISTINCT_NONDISPATCHABLE_HANDLES
    // Special entry to allow tracking of command pool Reset and Destroy
    counter<uint64_t> c_VkCommandPoolContents;

    counter<uint64_t> c_uint64_t;
#endif  // DISTINCT_NONDISPATCHABLE_HANDLES

    ThreadSafety()
        : c_VkCommandBuffer("VkCommandBuffer", VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, &report_data),
          c_VkDevice("VkDevice", VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT, &report_data),
          c_VkInstance("VkInstance", VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT, &report_data),
          c_VkQueue("VkQueue", VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT, &report_data),
          c_VkCommandPoolContents("VkCommandPool", VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT, &report_data),

#ifdef DISTINCT_NONDISPATCHABLE_HANDLES
          c_VkBuffer("VkBuffer", VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, &report_data),
          c_VkBufferView("VkBufferView", VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT, &report_data),
          c_VkCommandPool("VkCommandPool", VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT, &report_data),
          c_VkDescriptorPool("VkDescriptorPool", VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT, &report_data),
          c_VkDescriptorSet("VkDescriptorSet", VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT, &report_data),
          c_VkDescriptorSetLayout("VkDescriptorSetLayout", VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT, &report_data),
          c_VkDeviceMemory("VkDeviceMemory", VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT, &report_data),
          c_VkEvent("VkEvent", VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT, &report_data),
          c_VkFence("VkFence", VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT, &report_data),
          c_VkFramebuffer("VkFramebuffer", VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT, &report_data),
          c_VkImage("VkImage", VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, &report_data),
          c_VkImageView("VkImageView", VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT, &report_data),
          c_VkPipeline("VkPipeline", VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, &report_data),
          c_VkPipelineCache("VkPipelineCache", VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT, &report_data),
          c_VkPipelineLayout("VkPipelineLayout", VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT, &report_data),
          c_VkQueryPool("VkQueryPool", VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT, &report_data),
          c_VkRenderPass("VkRenderPass", VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT, &report_data),
          c_VkSampler("VkSampler", VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT, &report_data),
          c_VkSemaphore("VkSemaphore", VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT, &report_data),
          c_VkShaderModule("VkShaderModule", VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, &report_data),
          c_VkDebugReportCallbackEXT("VkDebugReportCallbackEXT", VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT, &report_data),
          c_VkObjectTableNVX("VkObjectTableNVX", VK_DEBUG_REPORT_OBJECT_TYPE_OBJECT_TABLE_NVX_EXT, &report_data),
          c_VkIndirectCommandsLayoutNVX("VkIndirectCommandsLayoutNVX",
                                        VK_DEBUG_REPORT_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX_EXT, &report_data),
          c_VkDisplayKHR("VkDisplayKHR", VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT, &report_data),
          c_VkDisplayModeKHR("VkDisplayModeKHR", VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT, &report_data),
          c_VkSurfaceKHR("VkSurfaceKHR", VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT, &report_data),
          c_VkSwapchainKHR("VkSwapchainKHR", VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT, &report_data),
          c_VkDescriptorUpdateTemplateKHR("VkDescriptorUpdateTemplateKHR",
                                          VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT, &report_data),
          c_VkSamplerYcbcrConversionKHR("VkSamplerYcbcrConversionKHR",
                                        VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT, &report_data),
          c_VkDebugUtilsMessengerEXT("VkDebugUtilsMessengerEXT", VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, &report_data),
          c_VkAccelerationStructureNV("VkAccelerationStructureNV", VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT, &report_data)

#else   // DISTINCT_NONDISPATCHABLE_HANDLES
          c_uint64_t("NON_DISPATCHABLE_HANDLE", VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, &report_data)
#endif  // DISTINCT_NONDISPATCHABLE_HANDLES
              {};

#define WRAPPER(type)                                                \
    void StartWriteObject(type object) {                             \
        c_##type.StartWrite(object);                                 \
    }                                                                \
    void FinishWriteObject(type object) {                            \
        c_##type.FinishWrite(object);                                \
    }                                                                \
    void StartReadObject(type object) {                              \
        c_##type.StartRead(object);                                  \
    }                                                                \
    void FinishReadObject(type object) {                             \
        c_##type.FinishRead(object);                                 \
    }

WRAPPER(VkDevice)
WRAPPER(VkInstance)
WRAPPER(VkQueue)
#ifdef DISTINCT_NONDISPATCHABLE_HANDLES

WRAPPER(VkBuffer)
WRAPPER(VkBufferView)
WRAPPER(VkCommandPool)
WRAPPER(VkDescriptorPool)
WRAPPER(VkDescriptorSet)
WRAPPER(VkDescriptorSetLayout)
WRAPPER(VkDeviceMemory)
WRAPPER(VkEvent)
WRAPPER(VkFence)
WRAPPER(VkFramebuffer)
WRAPPER(VkImage)
WRAPPER(VkImageView)
WRAPPER(VkPipeline)
WRAPPER(VkPipelineCache)
WRAPPER(VkPipelineLayout)
WRAPPER(VkQueryPool)
WRAPPER(VkRenderPass)
WRAPPER(VkSampler)
WRAPPER(VkSemaphore)
WRAPPER(VkShaderModule)
WRAPPER(VkDebugReportCallbackEXT)
WRAPPER(VkObjectTableNVX)
WRAPPER(VkIndirectCommandsLayoutNVX)
WRAPPER(VkDisplayKHR)
WRAPPER(VkDisplayModeKHR)
WRAPPER(VkSurfaceKHR)
WRAPPER(VkSwapchainKHR)
WRAPPER(VkDescriptorUpdateTemplateKHR)
WRAPPER(VkValidationCacheEXT)
WRAPPER(VkSamplerYcbcrConversionKHR)
WRAPPER(VkDebugUtilsMessengerEXT)
WRAPPER(VkAccelerationStructureNV)

#else   // DISTINCT_NONDISPATCHABLE_HANDLES
WRAPPER(uint64_t)
#endif  // DISTINCT_NONDISPATCHABLE_HANDLES

    // VkCommandBuffer needs check for implicit use of command pool
    void StartWriteObject(VkCommandBuffer object, bool lockPool = true) {
        if (lockPool) {
            std::unique_lock<std::mutex> lock(command_pool_lock);
            VkCommandPool pool = command_pool_map[object];
            lock.unlock();
            StartWriteObject(pool);
        }
        c_VkCommandBuffer.StartWrite(object);
    }
    void FinishWriteObject(VkCommandBuffer object, bool lockPool = true) {
        c_VkCommandBuffer.FinishWrite(object);
        if (lockPool) {
            std::unique_lock<std::mutex> lock(command_pool_lock);
            VkCommandPool pool = command_pool_map[object];
            lock.unlock();
            FinishWriteObject(pool);
        }
    }
    void StartReadObject(VkCommandBuffer object) {
        std::unique_lock<std::mutex> lock(command_pool_lock);
        VkCommandPool pool = command_pool_map[object];
        lock.unlock();
        // We set up a read guard against the "Contents" counter to catch conflict vs. vkResetCommandPool and vkDestroyCommandPool
        // while *not* establishing a read guard against the command pool counter itself to avoid false postives for
        // non-externally sync'd command buffers
        c_VkCommandPoolContents.StartRead(pool);
        c_VkCommandBuffer.StartRead(object);
    }
    void FinishReadObject(VkCommandBuffer object) {
        c_VkCommandBuffer.FinishRead(object);
        std::unique_lock<std::mutex> lock(command_pool_lock);
        VkCommandPool pool = command_pool_map[object];
        lock.unlock();
        c_VkCommandPoolContents.FinishRead(pool);
    }"""


    inline_custom_source_preamble = """
void ThreadSafety::PreCallRecordAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                       VkCommandBuffer *pCommandBuffers) {
    StartReadObject(device);
    StartWriteObject(pAllocateInfo->commandPool);
}

void ThreadSafety::PostCallRecordAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                        VkCommandBuffer *pCommandBuffers) {
    FinishReadObject(device);
    FinishWriteObject(pAllocateInfo->commandPool);

    // Record mapping from command buffer to command pool
    for (uint32_t index = 0; index < pAllocateInfo->commandBufferCount; index++) {
        std::lock_guard<std::mutex> lock(command_pool_lock);
        command_pool_map[pCommandBuffers[index]] = pAllocateInfo->commandPool;
    }
}

void ThreadSafety::PreCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                       VkDescriptorSet *pDescriptorSets) {
    StartReadObject(device);
    StartWriteObject(pAllocateInfo->descriptorPool);
    // Host access to pAllocateInfo::descriptorPool must be externally synchronized
}

void ThreadSafety::PostCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                        VkDescriptorSet *pDescriptorSets) {
    FinishReadObject(device);
    FinishWriteObject(pAllocateInfo->descriptorPool);
    // Host access to pAllocateInfo::descriptorPool must be externally synchronized
}

void ThreadSafety::PreCallRecordFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                                   const VkCommandBuffer *pCommandBuffers) {
    const bool lockCommandPool = false;  // pool is already directly locked
    StartReadObject(device);
    StartWriteObject(commandPool);
    for (uint32_t index = 0; index < commandBufferCount; index++) {
        StartWriteObject(pCommandBuffers[index], lockCommandPool);
    }
    // The driver may immediately reuse command buffers in another thread.
    // These updates need to be done before calling down to the driver.
    for (uint32_t index = 0; index < commandBufferCount; index++) {
        FinishWriteObject(pCommandBuffers[index], lockCommandPool);
        std::lock_guard<std::mutex> lock(command_pool_lock);
        command_pool_map.erase(pCommandBuffers[index]);
    }
}

void ThreadSafety::PostCallRecordFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                                    const VkCommandBuffer *pCommandBuffers) {
    FinishReadObject(device);
    FinishWriteObject(commandPool);
}

void ThreadSafety::PreCallRecordResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags) {
    StartReadObject(device);
    StartWriteObject(commandPool);
    // Check for any uses of non-externally sync'd command buffers (for example from vkCmdExecuteCommands)
    c_VkCommandPoolContents.StartWrite(commandPool);
    // Host access to commandPool must be externally synchronized
}

void ThreadSafety::PostCallRecordResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags) {
    FinishReadObject(device);
    FinishWriteObject(commandPool);
    c_VkCommandPoolContents.FinishWrite(commandPool);
    // Host access to commandPool must be externally synchronized
}

void ThreadSafety::PreCallRecordDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    StartReadObject(device);
    StartWriteObject(commandPool);
    // Check for any uses of non-externally sync'd command buffers (for example from vkCmdExecuteCommands)
    c_VkCommandPoolContents.StartWrite(commandPool);
    // Host access to commandPool must be externally synchronized
}

void ThreadSafety::PostCallRecordDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks *pAllocator) {
    FinishReadObject(device);
    FinishWriteObject(commandPool);
    c_VkCommandPoolContents.FinishWrite(commandPool);
}

"""


    # This is an ordered list of sections in the header file.
    TYPE_SECTIONS = ['include', 'define', 'basetype', 'handle', 'enum',
                     'group', 'bitmask', 'funcpointer', 'struct']
    ALL_SECTIONS = TYPE_SECTIONS + ['command']
    def __init__(self,
                 errFile = sys.stderr,
                 warnFile = sys.stderr,
                 diagFile = sys.stdout):
        OutputGenerator.__init__(self, errFile, warnFile, diagFile)
        # Internal state - accumulators for different inner block text
        self.sections = dict([(section, []) for section in self.ALL_SECTIONS])

    # Check if the parameter passed in is a pointer to an array
    def paramIsArray(self, param):
        return param.attrib.get('len') is not None

    # Check if the parameter passed in is a pointer
    def paramIsPointer(self, param):
        ispointer = False
        for elem in param:
            if ((elem.tag is not 'type') and (elem.tail is not None)) and '*' in elem.tail:
                ispointer = True
        return ispointer

    # Check if an object is a non-dispatchable handle
    def isHandleTypeNonDispatchable(self, handletype):
        handle = self.registry.tree.find("types/type/[name='" + handletype + "'][@category='handle']")
        if handle is not None and handle.find('type').text == 'VK_DEFINE_NON_DISPATCHABLE_HANDLE':
            return True
        else:
            return False

    # Check if an object is a dispatchable handle
    def isHandleTypeDispatchable(self, handletype):
        handle = self.registry.tree.find("types/type/[name='" + handletype + "'][@category='handle']")
        if handle is not None and handle.find('type').text == 'VK_DEFINE_HANDLE':
            return True
        else:
            return False

    def makeThreadUseBlock(self, cmd, functionprefix):
        """Generate C function pointer typedef for <command> Element"""
        paramdecl = ''
        # Find and add any parameters that are thread unsafe
        params = cmd.findall('param')
        for param in params:
            paramname = param.find('name')
            if False: # self.paramIsPointer(param):
                paramdecl += '    // not watching use of pointer ' + paramname.text + '\n'
            else:
                externsync = param.attrib.get('externsync')
                if externsync == 'true':
                    if self.paramIsArray(param):
                        paramdecl += 'for (uint32_t index=0;index<' + param.attrib.get('len') + ';index++) {\n'
                        paramdecl += '    ' + functionprefix + 'WriteObject(' + paramname.text + '[index]);\n'
                        paramdecl += '}\n'
                    else:
                        paramdecl += functionprefix + 'WriteObject(' + paramname.text + ');\n'
                elif (param.attrib.get('externsync')):
                    if self.paramIsArray(param):
                        # Externsync can list pointers to arrays of members to synchronize
                        paramdecl += 'for (uint32_t index=0;index<' + param.attrib.get('len') + ';index++) {\n'
                        second_indent = ''
                        for member in externsync.split(","):
                            # Replace first empty [] in member name with index
                            element = member.replace('[]','[index]',1)
                            if '[]' in element:
                                # Replace any second empty [] in element name with
                                # inner array index based on mapping array names like
                                # "pSomeThings[]" to "someThingCount" array size.
                                # This could be more robust by mapping a param member
                                # name to a struct type and "len" attribute.
                                limit = element[0:element.find('s[]')] + 'Count'
                                dotp = limit.rfind('.p')
                                limit = limit[0:dotp+1] + limit[dotp+2:dotp+3].lower() + limit[dotp+3:]
                                paramdecl += '    for(uint32_t index2=0;index2<'+limit+';index2++)\n'
                                element = element.replace('[]','[index2]')
                                second_indent = '   '
                            paramdecl += '    ' + second_indent + functionprefix + 'WriteObject(' + element + ');\n'
                        paramdecl += '}\n'
                    else:
                        # externsync can list members to synchronize
                        for member in externsync.split(","):
                            member = str(member).replace("::", "->")
                            member = str(member).replace(".", "->")
                            paramdecl += '    ' + functionprefix + 'WriteObject(' + member + ');\n'
                else:
                    paramtype = param.find('type')
                    if paramtype is not None:
                        paramtype = paramtype.text
                    else:
                        paramtype = 'None'
                    if (self.isHandleTypeDispatchable(paramtype) or self.isHandleTypeNonDispatchable(paramtype)) and paramtype != 'VkPhysicalDevice':
                        if self.paramIsArray(param) and ('pPipelines' != paramname.text):
                            # Add pointer dereference for array counts that are pointer values
                            dereference = ''
                            for candidate in params:
                                if param.attrib.get('len') == candidate.find('name').text:
                                    if self.paramIsPointer(candidate):
                                        dereference = '*'
                            param_len = str(param.attrib.get('len')).replace("::", "->")
                            paramdecl += 'for (uint32_t index = 0; index < ' + dereference + param_len + '; index++) {\n'
                            paramdecl += '    ' + functionprefix + 'ReadObject(' + paramname.text + '[index]);\n'
                            paramdecl += '}\n'
                        elif not self.paramIsPointer(param):
                            # Pointer params are often being created.
                            # They are not being read from.
                            paramdecl += functionprefix + 'ReadObject(' + paramname.text + ');\n'
        explicitexternsyncparams = cmd.findall("param[@externsync]")
        if (explicitexternsyncparams is not None):
            for param in explicitexternsyncparams:
                externsyncattrib = param.attrib.get('externsync')
                paramname = param.find('name')
                paramdecl += '// Host access to '
                if externsyncattrib == 'true':
                    if self.paramIsArray(param):
                        paramdecl += 'each member of ' + paramname.text
                    elif self.paramIsPointer(param):
                        paramdecl += 'the object referenced by ' + paramname.text
                    else:
                        paramdecl += paramname.text
                else:
                    paramdecl += externsyncattrib
                paramdecl += ' must be externally synchronized\n'

        # Find and add any "implicit" parameters that are thread unsafe
        implicitexternsyncparams = cmd.find('implicitexternsyncparams')
        if (implicitexternsyncparams is not None):
            for elem in implicitexternsyncparams:
                paramdecl += '// '
                paramdecl += elem.text
                paramdecl += ' must be externally synchronized between host accesses\n'

        if (paramdecl == ''):
            return None
        else:
            return paramdecl
    def beginFile(self, genOpts):
        OutputGenerator.beginFile(self, genOpts)
        #
        # TODO: LUGMAL -- remove this and add our copyright
        # User-supplied prefix text, if any (list of strings)
        write(self.inline_copyright_message, file=self.outFile)

        self.header_file = (genOpts.filename == 'thread_safety.h')
        self.source_file = (genOpts.filename == 'thread_safety.cpp')

        if not self.header_file and not self.source_file:
            print("Error: Output Filenames have changed, update generator source.\n")
            sys.exit(1)

        if self.source_file:
            write('#include "chassis.h"', file=self.outFile)
            write('#include "thread_safety.h"', file=self.outFile)
            self.newline()
            write(self.inline_custom_source_preamble, file=self.outFile)
        else:
            write(self.inline_custom_header_preamble, file=self.outFile)


    def endFile(self):
        if self.header_file:
            write('};', file=self.outFile)

        # Finish processing in superclass
        OutputGenerator.endFile(self)

    def beginFeature(self, interface, emit):
        #write('// starting beginFeature', file=self.outFile)
        # Start processing in superclass
        OutputGenerator.beginFeature(self, interface, emit)
        # C-specific
        # Accumulate includes, defines, types, enums, function pointer typedefs,
        # end function prototypes separately for this feature. They're only
        # printed in endFeature().
        self.featureExtraProtect = GetFeatureProtect(interface)
        self.sections = dict([(section, []) for section in self.ALL_SECTIONS])
        #write('// ending beginFeature', file=self.outFile)
    def endFeature(self):
        # C-specific
        # Actually write the interface to the output file.
        #write('// starting endFeature', file=self.outFile)
        if (self.emit):
            self.newline()
            if (self.genOpts.protectFeature):
                write('#ifndef', self.featureName, file=self.outFile)
            # If type declarations are needed by other features based on
            # this one, it may be necessary to suppress the ExtraProtect,
            # or move it below the 'for section...' loop.
            #write('// endFeature looking at self.featureExtraProtect', file=self.outFile)
            if (self.featureExtraProtect is not None):
                write('#ifdef', self.featureExtraProtect, file=self.outFile)
            #write('#define', self.featureName, '1', file=self.outFile)
            for section in self.TYPE_SECTIONS:
                #write('// endFeature writing section'+section, file=self.outFile)
                contents = self.sections[section]
                if contents:
                    write('\n'.join(contents), file=self.outFile)
                    self.newline()
            #write('// endFeature looking at self.sections[command]', file=self.outFile)
            if (self.sections['command']):
                write('\n'.join(self.sections['command']), end=u'', file=self.outFile)
                self.newline()
            if (self.featureExtraProtect is not None):
                write('#endif /*', self.featureExtraProtect, '*/', file=self.outFile)
            if (self.genOpts.protectFeature):
                write('#endif /*', self.featureName, '*/', file=self.outFile)
        # Finish processing in superclass
        OutputGenerator.endFeature(self)
        #write('// ending endFeature', file=self.outFile)
    #
    # Append a definition to the specified section
    def appendSection(self, section, text):
        # self.sections[section].append('SECTION: ' + section + '\n')
        self.sections[section].append(text)
    #
    # Type generation
    def genType(self, typeinfo, name, alias):
        pass
    #
    # Struct (e.g. C "struct" type) generation.
    # This is a special case of the <type> tag where the contents are
    # interpreted as a set of <member> tags instead of freeform C
    # C type declarations. The <member> tags are just like <param>
    # tags - they are a declaration of a struct or union member.
    # Only simple member declarations are supported (no nested
    # structs etc.)
    def genStruct(self, typeinfo, typeName, alias):
        OutputGenerator.genStruct(self, typeinfo, typeName, alias)
        body = 'typedef ' + typeinfo.elem.get('category') + ' ' + typeName + ' {\n'
        # paramdecl = self.makeCParamDecl(typeinfo.elem, self.genOpts.alignFuncParam)
        for member in typeinfo.elem.findall('.//member'):
            body += self.makeCParamDecl(member, self.genOpts.alignFuncParam)
            body += ';\n'
        body += '} ' + typeName + ';\n'
        self.appendSection('struct', body)
    #
    # Group (e.g. C "enum" type) generation.
    # These are concatenated together with other types.
    def genGroup(self, groupinfo, groupName, alias):
        pass
    # Enumerant generation
    # <enum> tags may specify their values in several ways, but are usually
    # just integers.
    def genEnum(self, enuminfo, name, alias):
        pass
    #
    # Command generation
    def genCmd(self, cmdinfo, name, alias):
        # Commands shadowed by interface functions and are not implemented
        # TODO:  Many of these no longer need to be manually written routines.  Winnow list.
        special_functions = [
            'vkCreateDevice',
            'vkCreateInstance',
            'vkAllocateCommandBuffers',
            'vkFreeCommandBuffers',
            'vkResetCommandPool',
            'vkDestroyCommandPool',
            'vkAllocateDescriptorSets',
            'vkQueuePresentKHR',
        ]
        if name == 'vkQueuePresentKHR' or (name in special_functions and self.source_file):
            return

        if (("DebugMarker" in name or "DebugUtilsObject" in name) and "EXT" in name):
            self.appendSection('command', '// TODO - not wrapping EXT function ' + name)
            return

        # Determine first if this function needs to be intercepted
        startthreadsafety = self.makeThreadUseBlock(cmdinfo.elem, 'Start')
        if startthreadsafety is None:
            return
        finishthreadsafety = self.makeThreadUseBlock(cmdinfo.elem, 'Finish')

        OutputGenerator.genCmd(self, cmdinfo, name, alias)

        # setup common to call wrappers
        # first parameter is always dispatchable
        dispatchable_type = cmdinfo.elem.find('param/type').text
        dispatchable_name = cmdinfo.elem.find('param/name').text

        decls = self.makeCDecls(cmdinfo.elem)

        if self.source_file:
            pre_decl = decls[0][:-1]
            pre_decl = pre_decl.split("VKAPI_CALL ")[1]
            pre_decl = 'void ThreadSafety::PreCallRecord' + pre_decl + ' {'

            # PreCallRecord
            self.appendSection('command', '')
            self.appendSection('command', pre_decl)
            self.appendSection('command', "    " + "\n    ".join(str(startthreadsafety).rstrip().split("\n")))
            self.appendSection('command', '}')

            post_decl = pre_decl.replace('PreCallRecord', 'PostCallRecord')

            # PostCallRecord
            self.appendSection('command', '')
            self.appendSection('command', post_decl)
            self.appendSection('command', "    " + "\n    ".join(str(finishthreadsafety).rstrip().split("\n")))
            self.appendSection('command', '}')

        if self.header_file:
            pre_decl = decls[0][:-1]
            pre_decl = pre_decl.split("VKAPI_CALL ")[1]
            pre_decl = 'void PreCallRecord' + pre_decl + ';'

            # PreCallRecord
            self.appendSection('command', '')
            self.appendSection('command', pre_decl)

            post_decl = pre_decl.replace('PreCallRecord', 'PostCallRecord')

            # PostCallRecord
            self.appendSection('command', '')
            self.appendSection('command', post_decl)

    #
    # override makeProtoName to drop the "vk" prefix
    def makeProtoName(self, name, tail):
        return self.genOpts.apientry + name[2:] + tail
