/* Copyright (c) 2015-2019 The Khronos Group Inc.
 * Copyright (c) 2015-2019 Valve Corporation
 * Copyright (c) 2015-2019 LunarG, Inc.
 * Copyright (C) 2015-2019 Google Inc.
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
 * Author: Karl Schultz <karl@lunarg.com>
 * Author: Tony Barbour <tony@LunarG.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include "vk_loader_platform.h"
#include "vk_dispatch_table_helper.h"
#include "vk_enum_string_helper.h"
#include "chassis.h"
#include "gpu_validation.h"
#include "vk_layer_utils.h"

uint32_t GpuVal::GetQueueState(VkQueue queue) {
    auto it = queueMap.find(queue);
    if (it == queueMap.end()) {
        return 0;
    }
    return it->second;
}

COMMAND_POOL_STATE *GpuVal::GetCommandPoolState(VkCommandPool pool) {
    auto it = commandPoolMap.find(pool);
    if (it == commandPoolMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

PHYSICAL_DEVICE_STATE *GpuVal::GetPhysicalDeviceState(VkPhysicalDevice phys) {
    auto *phys_dev_map = ((physical_device_map.size() > 0) ? &physical_device_map : &instance_state->physical_device_map);
    auto it = phys_dev_map->find(phys);
    if (it == phys_dev_map->end()) {
        return nullptr;
    }
    return &it->second;
}

PHYSICAL_DEVICE_STATE *GpuVal::GetPhysicalDeviceState() { return physical_device_state; }

std::string FormatDebugLabel(const char *prefix, const LoggingLabel &label) {
    if (label.Empty()) return std::string();
    std::string out;
    string_sprintf(&out, "%sVkDebugUtilsLabel(name='%s' color=[%g, %g %g, %g])", prefix, label.name.c_str(), label.color[0],
                   label.color[1], label.color[2], label.color[3]);
    return out;
}

// Retrieve pipeline node ptr for given pipeline object
PIPELINE_STATE *GpuVal::GetPipelineState(VkPipeline pipeline) {
    auto it = pipelineMap.find(pipeline);
    if (it == pipelineMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

PIPELINE_LAYOUT_STATE const *GpuVal::GetPipelineLayout(VkPipelineLayout pipeLayout) {
    auto it = pipelineLayoutMap.find(pipeLayout);
    if (it == pipelineLayoutMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::shared_ptr<cvdescriptorset::DescriptorSetLayout const> const GetDescriptorSetLayout(GpuVal const *dev_data,
                                                                                         VkDescriptorSetLayout dsLayout) {
    auto it = dev_data->descriptorSetLayoutMap.find(dsLayout);
    if (it == dev_data->descriptorSetLayoutMap.end()) {
        return nullptr;
    }
    return it->second;
}

SHADER_MODULE_STATE const *GpuVal::GetShaderModuleState(VkShaderModule module) {
    auto it = shaderModuleMap.find(module);
    if (it == shaderModuleMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

// Return Set node ptr for specified set or else NULL
cvdescriptorset::DescriptorSet *GpuVal::GetSetNode(VkDescriptorSet set) {
    auto set_it = setMap.find(set);
    if (set_it == setMap.end()) {
        return NULL;
    }
    return set_it->second.get();
}

// Return Pool node ptr for specified pool or else NULL
DESCRIPTOR_POOL_STATE *GpuVal::GetDescriptorPoolState(const VkDescriptorPool pool) {
    auto pool_it = descriptorPoolMap.find(pool);
    if (pool_it == descriptorPoolMap.end()) {
        return NULL;
    }
    return pool_it->second.get();
}

// Remove set from setMap and delete the set
void GpuVal::FreeDescriptorSet(cvdescriptorset::DescriptorSet *descriptor_set) { setMap.erase(descriptor_set->GetSet()); }

// Free all DS Pools including their Sets & related sub-structs
// NOTE : Calls to this function should be wrapped in mutex
void GpuVal::DeletePools() {
    for (auto ii = descriptorPoolMap.begin(); ii != descriptorPoolMap.end();) {
        // Remove this pools' sets from setMap and delete them
        for (auto ds : ii->second->sets) {
            FreeDescriptorSet(ds);
        }
        ii->second->sets.clear();
        ii = descriptorPoolMap.erase(ii);
    }
}

// For given CB object, fetch associated CB Node from map
CMD_BUFFER_STATE *GpuVal::GetCBState(const VkCommandBuffer cb) {
    auto it = commandBufferMap.find(cb);
    if (it == commandBufferMap.end()) {
        return NULL;
    }
    return it->second.get();
}

static void AddCommandBufferBinding(std::unordered_set<CMD_BUFFER_STATE *> *cb_bindings, VK_OBJECT obj, CMD_BUFFER_STATE *cb_node) {
    cb_bindings->insert(cb_node);
}
// Reset the command buffer state
void GpuVal::ResetCommandBufferState(const VkCommandBuffer cb) {
    CMD_BUFFER_STATE *pCB = GetCBState(cb);
    if (pCB) {
        // Reset CB state (note that createInfo is not cleared)
        pCB->commandBuffer = cb;
        pCB->hasDrawCmd = false;
        for (auto &item : pCB->lastBound) {
            item.second.reset();
        }

        pCB->activeRenderPass = nullptr;
        pCB->primaryCommandBuffer = VK_NULL_HANDLE;

        // Remove reverse command buffer links.
        for (auto pSubCB : pCB->linkedCommandBuffers) {
            pSubCB->linkedCommandBuffers.erase(pCB);
        }
        pCB->linkedCommandBuffers.clear();

        // Clean up the label data
        ResetCmdDebugUtilsLabel(report_data, pCB->commandBuffer);
        pCB->debug_label.Reset();
    }

    GpuResetCommandBuffer(cb);
}
void GpuVal::InitGpuValidation() {
    // Process the layer settings file.
    enum CoreValidationGpuFlagBits {
        CORE_VALIDATION_GPU_VALIDATION_ALL_BIT = 0x00000001,
        CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT = 0x00000002,
    };
    typedef VkFlags CoreGPUFlags;
    static const std::unordered_map<std::string, VkFlags> gpu_flags_option_definitions = {
        {std::string("all"), CORE_VALIDATION_GPU_VALIDATION_ALL_BIT},
        {std::string("reserve_binding_slot"), CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT},
    };
    std::string gpu_flags_key = "lunarg_core_validation.gpu_validation";
    CoreGPUFlags gpu_flags = GetLayerOptionFlags(gpu_flags_key, gpu_flags_option_definitions, 0);
    gpu_flags_key = "khronos_validation.gpu_validation";
    gpu_flags |= GetLayerOptionFlags(gpu_flags_key, gpu_flags_option_definitions, 0);
    instance_state->enabled.gpu_validation = true;
    if (gpu_flags & CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT) {
        instance_state->enabled.gpu_validation_reserve_binding_slot = true;
    }
}

void GpuVal::PostCallRecordCreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance, VkResult result) {
    if (VK_SUCCESS != result) return;
    InitGpuValidation();
}

void GpuVal::PreCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkDevice *pDevice,
                                           std::unique_ptr<safe_VkDeviceCreateInfo> &modified_create_info) {
    // GPU Validation can possibly turn on device features, so give it a chance to change the create info.
    VkPhysicalDeviceFeatures supported_features;
    DispatchGetPhysicalDeviceFeatures(gpu, &supported_features);
    GpuPreCallRecordCreateDevice(gpu, modified_create_info, &supported_features);
}

void GpuVal::PostCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice, VkResult result) {
    if (VK_SUCCESS != result) return;

    ValidationObject *device_object = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);
    ValidationObject *validation_data = GetValidationObject(device_object->object_dispatch, LayerObjectTypeCoreValidation);
    GpuVal *gpu_val = static_cast<GpuVal *>(validation_data);

    // Make sure that queue_family_properties are obtained for this device's physical_device, even if the app has not
    // previously set them through an explicit API call.
    uint32_t count;
    auto pd_state = GetPhysicalDeviceState(gpu);
    DispatchGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    pd_state->queue_family_count = count;
    pd_state->queue_family_properties.resize(std::max(static_cast<uint32_t>(pd_state->queue_family_properties.size()), count));
    DispatchGetPhysicalDeviceQueueFamilyProperties(gpu, &count, &pd_state->queue_family_properties[0]);
    // Save local link to this device's physical device state
    gpu_val->physical_device_state = pd_state;
    DispatchGetPhysicalDeviceProperties(gpu, &gpu_val->phys_dev_props);

    const auto &dev_ext = gpu_val->device_extensions;

    gpu_val->GpuPostCallRecordCreateDevice(&enabled);
}

void GpuVal::PreCallRecordDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    if (!device) return;
    GpuPreCallRecordDestroyDevice();
    pipelineMap.clear();
    commandBufferMap.clear();
    renderPassMap.clear();
    // This will also delete all sets in the pool & remove them from setMap
    // All sets should be removed
    queueMap.clear();
    layer_debug_utils_destroy_device(device);
}

void GpuVal::PreCallRecordDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks *pAllocator) {
    if (!renderPass) return;
    RENDER_PASS_STATE *rp_state = GetRenderPassState(renderPass);
    VK_OBJECT obj_struct = {HandleToUint64(renderPass), kVulkanObjectTypeRenderPass};
    renderPassMap.erase(renderPass);
}

void GpuVal::RecordCreateRenderPassState(RenderPassCreateVersion rp_version, std::shared_ptr<RENDER_PASS_STATE> &render_pass,
                                             VkRenderPass *pRenderPass) {
    render_pass->renderPass = *pRenderPass;
    // Even though render_pass is an rvalue-ref parameter, still must move s.t. move assignment is invoked.
    renderPassMap[*pRenderPass] = std::move(render_pass);
}

void GpuVal::PostCallRecordCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass,
                                                VkResult result) {
    if (VK_SUCCESS != result) return;
    auto render_pass_state = std::make_shared<RENDER_PASS_STATE>(pCreateInfo);
    RecordCreateRenderPassState(RENDER_PASS_VERSION_1, render_pass_state, pRenderPass);
}

void GpuVal::PostCallRecordCreateRenderPass2KHR(VkDevice device, const VkRenderPassCreateInfo2KHR *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass,
                                                    VkResult result) {
    if (VK_SUCCESS != result) return;
    auto render_pass_state = std::make_shared<RENDER_PASS_STATE>(pCreateInfo);
    RecordCreateRenderPassState(RENDER_PASS_VERSION_2, render_pass_state, pRenderPass);
}

void GpuVal::PostCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence,
                                           VkResult result) {
    GpuPostCallQueueSubmit(queue, submitCount, pSubmits, fence);
}

void GpuVal::PreCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence) {
    if (device_extensions.vk_ext_descriptor_indexing) {
        GpuPreCallRecordQueueSubmit(queue, submitCount, pSubmits, fence);
    }
}

void GpuVal::PreCallRecordDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                  const VkAllocationCallbacks *pAllocator) {
    if (!shaderModule) return;
    shaderModuleMap.erase(shaderModule);
}

void GpuVal::PreCallRecordDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator) {
    if (!pipeline) return;
    PIPELINE_STATE *pipeline_state = GetPipelineState(pipeline);
    VK_OBJECT obj_struct = {HandleToUint64(pipeline), kVulkanObjectTypePipeline};
    // Any bound cmd buffers are now invalid
    GpuPreCallRecordDestroyPipeline(pipeline);
    pipelineMap.erase(pipeline);
}

void GpuVal::PreCallRecordDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                         const VkAllocationCallbacks *pAllocator) {
    if (!descriptorSetLayout) return;
    auto layout_it = descriptorSetLayoutMap.find(descriptorSetLayout);
    if (layout_it != descriptorSetLayoutMap.end()) {
        layout_it->second.get()->MarkDestroyed();
        descriptorSetLayoutMap.erase(layout_it);
    }
}

void GpuVal::PreCallRecordDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                    const VkAllocationCallbacks *pAllocator) {
    if (!descriptorPool) return;
    DESCRIPTOR_POOL_STATE *desc_pool_state = GetDescriptorPoolState(descriptorPool);
    VK_OBJECT obj_struct = {HandleToUint64(descriptorPool), kVulkanObjectTypeDescriptorPool};
    if (desc_pool_state) {
        // Free sets that were in this pool
        for (auto ds : desc_pool_state->sets) {
            FreeDescriptorSet(ds);
        }
        descriptorPoolMap.erase(descriptorPool);
    }
}

// Free all command buffers in given list, removing all references/links to them using ResetCommandBufferState
void GpuVal::FreeCommandBufferStates(COMMAND_POOL_STATE *pool_state, const uint32_t command_buffer_count,
                                         const VkCommandBuffer *command_buffers) {
    for (uint32_t i = 0; i < command_buffer_count; i++) {
        auto cb_state = GetCBState(command_buffers[i]);
        // Remove references to command buffer's state and delete
        if (cb_state) {
            // reset prior to delete, removing various references to it.
            // TODO: fix this, it's insane.
            ResetCommandBufferState(cb_state->commandBuffer);
            // Remove CBState from CB map
            commandBufferMap.erase(cb_state->commandBuffer);
            // Remove the cb debug labels
            EraseCmdDebugUtilsLabel(report_data, cb_state->commandBuffer);
            // Remove CBState from CB map
            commandBufferMap.erase(cb_state->commandBuffer);
        }
    }
}

void GpuVal::PreCallRecordGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                          VkPhysicalDeviceProperties *pPhysicalDeviceProperties) {
    if (enabled.gpu_validation_reserve_binding_slot) {
        if (pPhysicalDeviceProperties->limits.maxBoundDescriptorSets > 1) {
            pPhysicalDeviceProperties->limits.maxBoundDescriptorSets -= 1;
        } else {
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,
                    HandleToUint64(physicalDevice), "UNASSIGNED-GPU-Assisted Validation Setup Error.",
                    "Unable to reserve descriptor binding slot on a device with only one slot.");
        }
    }
}

void GpuVal::PostCallRecordEnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                        VkPhysicalDevice *pPhysicalDevices, VkResult result) {
    if ((NULL != pPhysicalDevices) && ((result == VK_SUCCESS || result == VK_INCOMPLETE))) {
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
            auto &phys_device_state = physical_device_map[pPhysicalDevices[i]];
            phys_device_state.phys_device = pPhysicalDevices[i];
        }
    }
}

void GpuVal::PreCallRecordFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                                 const VkCommandBuffer *pCommandBuffers) {
    FreeCommandBufferStates(nullptr, commandBufferCount, pCommandBuffers);
}

void GpuVal::UpdateDrawState(CMD_BUFFER_STATE *cb_state, const VkPipelineBindPoint bind_point) {
    auto const &state = cb_state->lastBound[bind_point];
    PIPELINE_STATE *pPipe = state.pipeline_state;
    if (VK_NULL_HANDLE != state.pipeline_layout) {
        for (const auto &set_binding_pair : pPipe->active_slots) {
            uint32_t setIndex = set_binding_pair.first;
            // Pull the set node
            cvdescriptorset::DescriptorSet *descriptor_set = state.boundDescriptorSets[setIndex];
            if (!descriptor_set->IsPushDescriptor()) {
                // Bind this set and its active descriptor resources to the command buffer
                descriptor_set->BindCommandBuffer(cb_state, set_binding_pair.second);
                // descriptor_set->UpdateDrawState(this, cb_state, binding_req_map);
            }
        }
    }
}
RENDER_PASS_STATE *GpuVal::GetRenderPassState(VkRenderPass renderpass) {
    auto it = renderPassMap.find(renderpass);
    if (it == renderPassMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::shared_ptr<RENDER_PASS_STATE> GpuVal::GetRenderPassStateSharedPtr(VkRenderPass renderpass) {
    auto it = renderPassMap.find(renderpass);
    if (it == renderPassMap.end()) {
        return nullptr;
    }
    return it->second;
}

bool GpuVal::PreCallValidateCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                        const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                        const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                        void *cgpl_state_data) {
    bool skip = false;
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    cgpl_state->pipe_state.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        cgpl_state->pipe_state.push_back(std::unique_ptr<PIPELINE_STATE>(new PIPELINE_STATE));
        (cgpl_state->pipe_state)[i]->initGraphicsPipeline(&pCreateInfos[i],
                                                          GetRenderPassStateSharedPtr(pCreateInfos[i].renderPass));
        (cgpl_state->pipe_state)[i]->pipeline_layout = *GetPipelineLayout(pCreateInfos[i].layout);
    }

    return skip;
}

// GPU validation may replace pCreateInfos for the down-chain call
void GpuVal::PreCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                      const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                      void *cgpl_state_data) {
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    cgpl_state->pCreateInfos = pCreateInfos;
    // GPU Validation may replace instrumented shaders with non-instrumented ones, so allow it to modify the createinfos.
    cgpl_state->gpu_create_infos =
        GpuPreCallRecordCreateGraphicsPipelines(pipelineCache, count, pCreateInfos, pAllocator, pPipelines, cgpl_state->pipe_state);
    cgpl_state->pCreateInfos = reinterpret_cast<VkGraphicsPipelineCreateInfo *>(cgpl_state->gpu_create_infos.data());
}

void GpuVal::PostCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                       const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                       VkResult result, void *cgpl_state_data) {
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (cgpl_state->pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((cgpl_state->pipe_state)[i]);
        }
    }
    // GPU val needs clean up regardless of result
    GpuPostCallRecordCreateGraphicsPipelines(count, pCreateInfos, pAllocator, pPipelines);
    cgpl_state->gpu_create_infos.clear();
    cgpl_state->pipe_state.clear();
}

void GpuVal::PostCallRecordCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                      VkResult result, void *pipe_state_data) {
    std::vector<std::unique_ptr<PIPELINE_STATE>> *pipe_state =
        reinterpret_cast<std::vector<std::unique_ptr<PIPELINE_STATE>> *>(pipe_state_data);

    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (*pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((*pipe_state)[i]);
        }
    }
}

void GpuVal::PostCallRecordCreateRayTracingPipelinesNV(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                           const VkRayTracingPipelineCreateInfoNV *pCreateInfos,
                                                           const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                           VkResult result, void *pipe_state_data) {
    std::vector<std::unique_ptr<PIPELINE_STATE>> *pipe_state =
        reinterpret_cast<std::vector<std::unique_ptr<PIPELINE_STATE>> *>(pipe_state_data);
    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (*pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((*pipe_state)[i]);
        }
    }
}

void GpuVal::PostCallRecordCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                         const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout,
                                                         VkResult result) {
    if (VK_SUCCESS != result) return;
    descriptorSetLayoutMap[*pSetLayout] = std::make_shared<cvdescriptorset::DescriptorSetLayout>(pCreateInfo, *pSetLayout);
}

void GpuVal::PreCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout,
                                                   void *cpl_state_data) {
    create_pipeline_layout_api_state *cpl_state = reinterpret_cast<create_pipeline_layout_api_state *>(cpl_state_data);
    GpuPreCallCreatePipelineLayout(pCreateInfo, pAllocator, pPipelineLayout, &cpl_state->new_layouts,
                                   &cpl_state->modified_create_info);
}

void GpuVal::PostCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout,
                                                    VkResult result) {
    // Clean up GPU validation
    GpuPostCallCreatePipelineLayout(result);
    if (VK_SUCCESS != result) return;
    std::unique_ptr<PIPELINE_LAYOUT_STATE> pipeline_layout_state(new PIPELINE_LAYOUT_STATE{});
    pipeline_layout_state->layout = *pPipelineLayout;
    pipeline_layout_state->set_layouts.resize(pCreateInfo->setLayoutCount);
    pipelineLayoutMap[*pPipelineLayout] = std::move(pipeline_layout_state);
}

// Ensure the pool contains enough descriptors and descriptor sets to satisfy
// an allocation request. Fills common_data with the total number of descriptors of each type required,
// as well as DescriptorSetLayout ptrs used for later update.
bool GpuVal::PreCallValidateAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                       VkDescriptorSet *pDescriptorSets, void *ads_state_data) {
    // Always update common data
    cvdescriptorset::AllocateDescriptorSetsData *ads_state =
        reinterpret_cast<cvdescriptorset::AllocateDescriptorSetsData *>(ads_state_data);
    UpdateAllocateDescriptorSetsData(pAllocateInfo, ads_state);
    // All state checks for AllocateDescriptorSets is done in single function
    return VK_SUCCESS;
}

// Allocation state was good and call down chain was made so update state based on allocating descriptor sets
void GpuVal::PostCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                      VkDescriptorSet *pDescriptorSets, VkResult result, void *ads_state_data) {
    if (VK_SUCCESS != result) return;
    // All the updates are contained in a single cvdescriptorset function
    cvdescriptorset::AllocateDescriptorSetsData *ads_state =
        reinterpret_cast<cvdescriptorset::AllocateDescriptorSetsData *>(ads_state_data);
    PerformAllocateDescriptorSets(pAllocateInfo, pDescriptorSets, ads_state);
}

void GpuVal::PostCallRecordAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pCreateInfo,
                                                      VkCommandBuffer *pCommandBuffer, VkResult result) {
    if (VK_SUCCESS != result) return;
    for (uint32_t i = 0; i < pCreateInfo->commandBufferCount; i++) {
        // Add command buffer to its commandPool map
        std::unique_ptr<CMD_BUFFER_STATE> pCB(new CMD_BUFFER_STATE{});
        pCB->device = device;
        // Add command buffer to map
        commandBufferMap[pCommandBuffer[i]] = std::move(pCB);
        ResetCommandBufferState(pCommandBuffer[i]);
    }
}

void GpuVal::PostCallRecordResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags, VkResult result) {
    if (VK_SUCCESS == result) {
        ResetCommandBufferState(commandBuffer);
    }
}

void GpuVal::PreCallRecordCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                              VkPipeline pipeline) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    assert(cb_state);

    auto pipe_state = GetPipelineState(pipeline);
    cb_state->lastBound[pipelineBindPoint].pipeline_state = pipe_state;
    AddCommandBufferBinding(&pipe_state->cb_bindings, {HandleToUint64(pipeline), kVulkanObjectTypePipeline}, cb_state);
}

void GpuVal::PreCallRecordCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                                            VkPipelineStageFlags sourceStageMask, VkPipelineStageFlags dstStageMask,
                                            uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                            uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                            uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers) {
    GpuPreCallValidateCmdWaitEvents(sourceStageMask);
}

void GpuVal::PreCallRecordCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                      VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                      const VkWriteDescriptorSet *pDescriptorWrites) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    RecordCmdPushDescriptorSetState(cb_state, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
}

// Update pipeline_layout bind points applying the "Pipeline Layout Compatibility" rules
void GpuVal::UpdateLastBoundDescriptorSets(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint pipeline_bind_point,
                                               const PIPELINE_LAYOUT_STATE *pipeline_layout, uint32_t first_set, uint32_t set_count,
                                               const std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets,
                                               uint32_t dynamic_offset_count, const uint32_t *p_dynamic_offsets) {
    // Defensive
    assert(set_count);
    if (0 == set_count) return;
    assert(pipeline_layout);
    if (!pipeline_layout) return;

    uint32_t required_size = first_set + set_count;
    const uint32_t last_binding_index = required_size - 1;

    // Some useful shorthand
    auto &last_bound = cb_state->lastBound[pipeline_bind_point];

    auto &bound_sets = last_bound.boundDescriptorSets;
    auto &dynamic_offsets = last_bound.dynamicOffsets;

    const uint32_t current_size = static_cast<uint32_t>(bound_sets.size());
    assert(current_size == dynamic_offsets.size());

    // We need this three times in this function, but nowhere else
    auto push_descriptor_cleanup = [&last_bound](const cvdescriptorset::DescriptorSet *ds) -> bool {
        if (ds && ds->IsPushDescriptor()) {
            assert(ds == last_bound.push_descriptor_set.get());
            last_bound.push_descriptor_set = nullptr;
            return true;
        }
        return false;
    };

    // Clean up the "disturbed" before and after the range to be set
    if (required_size < current_size) {
        // We're not disturbing past last, so leave the upper binding data alone.
        required_size = current_size;
    }

    // We resize if we need more set entries or if those past "last" are disturbed
    if (required_size != current_size) {
        // TODO: put these size tied things in a struct (touches many lines)
        bound_sets.resize(required_size);
        dynamic_offsets.resize(required_size);
    }

    // Now update the bound sets with the input sets
    const uint32_t *input_dynamic_offsets = p_dynamic_offsets;  // "read" pointer for dynamic offset data
    for (uint32_t input_idx = 0; input_idx < set_count; input_idx++) {
        auto set_idx = input_idx + first_set;  // set_idx is index within layout, input_idx is index within input descriptor sets
        cvdescriptorset::DescriptorSet *descriptor_set = descriptor_sets[input_idx];

        // Record binding (or push)
        if (descriptor_set != last_bound.push_descriptor_set.get()) {
            // Only cleanup the push descriptors if they aren't the currently used set.
            push_descriptor_cleanup(bound_sets[set_idx]);
        }
        bound_sets[set_idx] = descriptor_set;

        if (descriptor_set) {
            auto set_dynamic_descriptor_count = descriptor_set->GetDynamicDescriptorCount();
            // TODO: Add logic for tracking push_descriptor offsets (here or in caller)
            if (set_dynamic_descriptor_count && input_dynamic_offsets) {
                const uint32_t *end_offset = input_dynamic_offsets + set_dynamic_descriptor_count;
                dynamic_offsets[set_idx] = std::vector<uint32_t>(input_dynamic_offsets, end_offset);
                input_dynamic_offsets = end_offset;
                assert(input_dynamic_offsets <= (p_dynamic_offsets + dynamic_offset_count));
            } else {
                dynamic_offsets[set_idx].clear();
            }
        }
    }
}

// Update the bound state for the bind point, including the effects of incompatible pipeline layouts
void GpuVal::PreCallRecordCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                    VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
                                                    const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                                                    const uint32_t *pDynamicOffsets) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    auto pipeline_layout = GetPipelineLayout(layout);
    std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets;
    descriptor_sets.reserve(setCount);

    // Construct a list of the descriptors
    bool found_non_null = false;
    for (uint32_t i = 0; i < setCount; i++) {
        cvdescriptorset::DescriptorSet *descriptor_set = GetSetNode(pDescriptorSets[i]);
        descriptor_sets.emplace_back(descriptor_set);
        found_non_null |= descriptor_set != nullptr;
    }
    if (found_non_null) {  // which implies setCount > 0
        UpdateLastBoundDescriptorSets(cb_state, pipelineBindPoint, pipeline_layout, firstSet, setCount, descriptor_sets,
                                      dynamicOffsetCount, pDynamicOffsets);
        cb_state->lastBound[pipelineBindPoint].pipeline_layout = layout;
    }
}

void GpuVal::RecordCmdPushDescriptorSetState(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint pipelineBindPoint,
                                                 VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                 const VkWriteDescriptorSet *pDescriptorWrites) {
    const auto &pipeline_layout = GetPipelineLayout(layout);
    // Short circuit invalid updates
    if (!pipeline_layout || (set >= pipeline_layout->set_layouts.size()) || !pipeline_layout->set_layouts[set] ||
        !pipeline_layout->set_layouts[set]->IsPushDescriptor())
        return;

    // We need a descriptor set to update the bindings with, compatible with the passed layout
    const auto dsl = pipeline_layout->set_layouts[set];
    auto &last_bound = cb_state->lastBound[pipelineBindPoint];
    auto &push_descriptor_set = last_bound.push_descriptor_set;
    // If we are disturbing the current push_desriptor_set clear it
    if (!push_descriptor_set) {
        push_descriptor_set.reset(new cvdescriptorset::DescriptorSet(0, 0, dsl, 0, this));
    }

    std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets = {push_descriptor_set.get()};
    UpdateLastBoundDescriptorSets(cb_state, pipelineBindPoint, pipeline_layout, set, 1, descriptor_sets, 0, nullptr);
    last_bound.pipeline_layout = layout;
}

// Generic function to handle state update for all CmdDraw* and CmdDispatch* type functions
void GpuVal::UpdateStateCmdDrawDispatchType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point) {
    UpdateDrawState(cb_state, bind_point);
}

// Generic function to handle state update for all CmdDraw* type functions
void GpuVal::UpdateStateCmdDrawType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point) {
    UpdateStateCmdDrawDispatchType(cb_state, bind_point);
    cb_state->hasDrawCmd = true;
}
void GpuVal::PostCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawDispatchType(cb_state, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void GpuVal::PostCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawDispatchType(cb_state, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void GpuVal::PostCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                       uint32_t firstVertex, uint32_t firstInstance) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PostCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                              uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PostCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                               uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PostCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                      uint32_t count, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                      VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                      uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                             VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                             uint32_t maxDrawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawMeshTasksNV(VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawMeshTasksIndirectNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                         uint32_t drawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawMeshTasksIndirectCountNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                              VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                              uint32_t maxDrawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                      uint32_t firstVertex, uint32_t firstInstance) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                             uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                              uint32_t stride) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                     uint32_t count, uint32_t stride) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void GpuVal::PreCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void GpuVal::PreCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void GpuVal::PreCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                 const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule,
                                                 void *csm_state_data) {
    create_shader_module_api_state *csm_state = reinterpret_cast<create_shader_module_api_state *>(csm_state_data);
    GpuPreCallCreateShaderModule(pCreateInfo, pAllocator, pShaderModule, &csm_state->unique_shader_id,
                                 &csm_state->instrumented_create_info, &csm_state->instrumented_pgm);
}

void GpuVal::PostCallRecordCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                  const VkAllocationCallbacks *pAllocator, VkShaderModule *pShaderModule,
                                                  VkResult result, void *csm_state_data) {
    if (VK_SUCCESS != result) return;
    create_shader_module_api_state *csm_state = reinterpret_cast<create_shader_module_api_state *>(csm_state_data);

    bool is_spirv = (pCreateInfo->pCode[0] == spv::MagicNumber);
    std::unique_ptr<SHADER_MODULE_STATE> new_shader_module(
        is_spirv ? new SHADER_MODULE_STATE(pCreateInfo, *pShaderModule, csm_state->unique_shader_id) : new SHADER_MODULE_STATE());
    shaderModuleMap[*pShaderModule] = std::move(new_shader_module);
}

// DescriptorSet Tracking Source Code
// ExtendedBinding collects a VkDescriptorSetLayoutBinding and any extended
// state that comes from a different array/structure so they can stay together
// while being sorted by binding number.
struct ExtendedBinding {
    ExtendedBinding(const VkDescriptorSetLayoutBinding *l, VkDescriptorBindingFlagsEXT f) : layout_binding(l), binding_flags(f) {}

    const VkDescriptorSetLayoutBinding *layout_binding;
    VkDescriptorBindingFlagsEXT binding_flags;
};

struct BindingNumCmp {
    bool operator()(const ExtendedBinding &a, const ExtendedBinding &b) const {
        return a.layout_binding->binding < b.layout_binding->binding;
    }
};

using DescriptorSetLayoutDef = cvdescriptorset::DescriptorSetLayoutDef;
using DescriptorSetLayoutId = cvdescriptorset::DescriptorSetLayoutId;

// Canonical dictionary of DescriptorSetLayoutDef (without any handle/device specific information)
cvdescriptorset::DescriptorSetLayoutDict descriptor_set_layout_dict;

DescriptorSetLayoutId GetCanonicalId(const VkDescriptorSetLayoutCreateInfo *p_create_info) {
    return descriptor_set_layout_dict.look_up(DescriptorSetLayoutDef(p_create_info));
}

// Construct DescriptorSetLayout instance from given create info
cvdescriptorset::DescriptorSetLayoutDef::DescriptorSetLayoutDef(const VkDescriptorSetLayoutCreateInfo *p_create_info)
    : flags_(p_create_info->flags), binding_count_(0), descriptor_count_(0), dynamic_descriptor_count_(0) {
    const auto *flags_create_info = lvl_find_in_chain<VkDescriptorSetLayoutBindingFlagsCreateInfoEXT>(p_create_info->pNext);

    std::set<ExtendedBinding, BindingNumCmp> sorted_bindings;
    const uint32_t input_bindings_count = p_create_info->bindingCount;
    // Sort the input bindings in binding number order, eliminating duplicates
    for (uint32_t i = 0; i < input_bindings_count; i++) {
        VkDescriptorBindingFlagsEXT flags = 0;
        if (flags_create_info && flags_create_info->bindingCount == p_create_info->bindingCount) {
            flags = flags_create_info->pBindingFlags[i];
        }
        sorted_bindings.insert(ExtendedBinding(p_create_info->pBindings + i, flags));
    }

    // Store the create info in the sorted order from above
    std::map<uint32_t, uint32_t> binding_to_dyn_count;
    uint32_t index = 0;
    binding_count_ = static_cast<uint32_t>(sorted_bindings.size());
    bindings_.reserve(binding_count_);
    binding_flags_.reserve(binding_count_);
    binding_to_index_map_.reserve(binding_count_);
    for (auto input_binding : sorted_bindings) {
        // Add to binding and map, s.t. it is robust to invalid duplication of binding_num
        const auto binding_num = input_binding.layout_binding->binding;
        binding_to_index_map_[binding_num] = index++;
        bindings_.emplace_back(input_binding.layout_binding);
        auto &binding_info = bindings_.back();
        binding_flags_.emplace_back(input_binding.binding_flags);

        descriptor_count_ += binding_info.descriptorCount;
        if (binding_info.descriptorCount > 0) {
            non_empty_bindings_.insert(binding_num);
        }

        if (binding_info.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            binding_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            binding_to_dyn_count[binding_num] = binding_info.descriptorCount;
            dynamic_descriptor_count_ += binding_info.descriptorCount;
        }
    }
    assert(bindings_.size() == binding_count_);
    assert(binding_flags_.size() == binding_count_);
    uint32_t global_index = 0;
    binding_to_global_index_range_map_.reserve(binding_count_);
    // Vector order is finalized so create maps of bindings to descriptors and descriptors to indices
    for (uint32_t i = 0; i < binding_count_; ++i) {
        auto binding_num = bindings_[i].binding;
        auto final_index = global_index + bindings_[i].descriptorCount;
        binding_to_global_index_range_map_[binding_num] = IndexRange(global_index, final_index);
        if (final_index != global_index) {
            global_start_to_index_map_[global_index] = i;
        }
        global_index = final_index;
    }

    // Now create dyn offset array mapping for any dynamic descriptors
    uint32_t dyn_array_idx = 0;
    binding_to_dynamic_array_idx_map_.reserve(binding_to_dyn_count.size());
    for (const auto &bc_pair : binding_to_dyn_count) {
        binding_to_dynamic_array_idx_map_[bc_pair.first] = dyn_array_idx;
        dyn_array_idx += bc_pair.second;
    }
}

size_t cvdescriptorset::DescriptorSetLayoutDef::hash() const {
    hash_util::HashCombiner hc;
    hc << flags_;
    hc.Combine(bindings_);
    hc.Combine(binding_flags_);
    return hc.Value();
}

// Return valid index or "end" i.e. binding_count_;
// The asserts in "Get" are reduced to the set where no valid answer(like null or 0) could be given
// Common code for all binding lookups.
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetIndexFromBinding(uint32_t binding) const {
    const auto &bi_itr = binding_to_index_map_.find(binding);
    if (bi_itr != binding_to_index_map_.cend()) return bi_itr->second;
    return GetBindingCount();
}
VkDescriptorSetLayoutBinding const *cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorSetLayoutBindingPtrFromIndex(
    const uint32_t index) const {
    if (index >= bindings_.size()) return nullptr;
    return bindings_[index].ptr();
}
// Return descriptorCount for given index, 0 if index is unavailable
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorCountFromIndex(const uint32_t index) const {
    if (index >= bindings_.size()) return 0;
    return bindings_[index].descriptorCount;
}
// For the given index, return descriptorType
VkDescriptorType cvdescriptorset::DescriptorSetLayoutDef::GetTypeFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    if (index < bindings_.size()) return bindings_[index].descriptorType;
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}
// For the given index, return stageFlags
VkShaderStageFlags cvdescriptorset::DescriptorSetLayoutDef::GetStageFlagsFromIndex(const uint32_t index) const {
    assert(index < bindings_.size());
    if (index < bindings_.size()) return bindings_[index].stageFlags;
    return VkShaderStageFlags(0);
}
// Return binding flags for given index, 0 if index is unavailable
VkDescriptorBindingFlagsEXT cvdescriptorset::DescriptorSetLayoutDef::GetDescriptorBindingFlagsFromIndex(
    const uint32_t index) const {
    if (index >= binding_flags_.size()) return 0;
    return binding_flags_[index];
}

// For the given binding, return the global index range
// As start and end are often needed in pairs, get both with a single hash lookup.
const cvdescriptorset::IndexRange &cvdescriptorset::DescriptorSetLayoutDef::GetGlobalIndexRangeFromBinding(
    const uint32_t binding) const {
    assert(binding_to_global_index_range_map_.count(binding));
    // In error case max uint32_t so index is out of bounds to break ASAP
    const static IndexRange kInvalidRange = {0xFFFFFFFF, 0xFFFFFFFF};
    const auto &range_it = binding_to_global_index_range_map_.find(binding);
    if (range_it != binding_to_global_index_range_map_.end()) {
        return range_it->second;
    }
    return kInvalidRange;
}

// Move to next valid binding having a non-zero binding count
uint32_t cvdescriptorset::DescriptorSetLayoutDef::GetNextValidBinding(const uint32_t binding) const {
    auto it = non_empty_bindings_.upper_bound(binding);
    assert(it != non_empty_bindings_.cend());
    if (it != non_empty_bindings_.cend()) return *it;
    return GetMaxBinding() + 1;
}
// For given index, return ptr to ImmutableSampler array
VkSampler const *cvdescriptorset::DescriptorSetLayoutDef::GetImmutableSamplerPtrFromIndex(const uint32_t index) const {
    if (index < bindings_.size()) {
        return bindings_[index].pImmutableSamplers;
    }
    return nullptr;
}

// The DescriptorSetLayout stores the per handle data for a descriptor set layout, and references the common defintion for the
// handle invariant portion
cvdescriptorset::DescriptorSetLayout::DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo *p_create_info,
                                                          const VkDescriptorSetLayout layout)
    : layout_(layout), layout_destroyed_(false), layout_id_(GetCanonicalId(p_create_info)) {}

cvdescriptorset::AllocateDescriptorSetsData::AllocateDescriptorSetsData(uint32_t count)
    : required_descriptors_by_type{}, layout_nodes(count, nullptr) {}

cvdescriptorset::DescriptorSet::DescriptorSet(const VkDescriptorSet set, const VkDescriptorPool pool,
                                              const std::shared_ptr<DescriptorSetLayout const> &layout, uint32_t variable_count,
                                              GpuVal *dev_data)
    : set_(set),
      pool_state_(nullptr),
      p_layout_(layout),
      device_data_(dev_data),
      limits_(dev_data->phys_dev_props.limits),
      variable_count_(variable_count) {
    pool_state_ = dev_data->GetDescriptorPoolState(pool);
    // Foreach binding, create default descriptors of given type
    descriptors_.reserve(p_layout_->GetTotalDescriptorCount());
    for (uint32_t i = 0; i < p_layout_->GetBindingCount(); ++i) {
        for (uint32_t di = 0; di < p_layout_->GetDescriptorCountFromIndex(i); ++di) descriptors_.emplace_back(new Descriptor);
    }
}

cvdescriptorset::DescriptorSet::~DescriptorSet() {}

// Bind cb_node to this set and this set to cb_node.
// Prereq: This should be called for a set that has been confirmed to be active for the given cb_node, meaning it's going
//   to be used in a draw by the given cb_node
void cvdescriptorset::DescriptorSet::BindCommandBuffer(CMD_BUFFER_STATE *cb_node,
                                                       const std::map<uint32_t, descriptor_req> &binding_req_map) {
    // bind cb to this descriptor set
    cb_bindings.insert(cb_node);
    // For the active slots, use set# to look up descriptorSet from boundDescriptorSets, and bind all of that descriptor set's
    // resources
    for (auto binding_req_pair : binding_req_map) {
        auto binding = binding_req_pair.first;
        auto range = p_layout_->GetGlobalIndexRangeFromBinding(binding);
        for (uint32_t i = range.start; i < range.end; ++i) {
            descriptors_[i]->BindCommandBuffer(cb_node);
        }
    }
}

// Update the drawing state for the affected descriptors.
// Set cb_node to this set and this set to cb_node.
// Add the bindings of the descriptor
// Set the layout based on the current descriptor layout (will mask subsequent layer mismatch errors)
void cvdescriptorset::DescriptorSet::UpdateDrawState(GpuVal *device_data, CMD_BUFFER_STATE *cb_node,
                                                     const std::map<uint32_t, descriptor_req> &binding_req_map) {
    // bind cb to this descriptor set
    cb_bindings.insert(cb_node);
    // For the active slots, use set# to look up descriptorSet from boundDescriptorSets, and bind all of that descriptor set's
    // resources
    for (auto binding_req_pair : binding_req_map) {
        auto binding = binding_req_pair.first;
        // We aren't validating descriptors created with PARTIALLY_BOUND or UPDATE_AFTER_BIND, so don't record state
        if (p_layout_->GetDescriptorBindingFlagsFromBinding(binding) &
            (VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT)) {
            continue;
        }
        auto range = p_layout_->GetGlobalIndexRangeFromBinding(binding);
        for (uint32_t i = range.start; i < range.end; ++i) {
            descriptors_[i]->UpdateDrawState(device_data, cb_node);
        }
    }
}

void cvdescriptorset::Descriptor::BindCommandBuffer(CMD_BUFFER_STATE *cb_node) {}

void cvdescriptorset::Descriptor::UpdateDrawState(GpuVal *dev_data, CMD_BUFFER_STATE *cb_node) {}

cvdescriptorset::Descriptor::Descriptor() { updated = false; }

// Update the common AllocateDescriptorSetsData
void GpuVal::UpdateAllocateDescriptorSetsData(const VkDescriptorSetAllocateInfo *p_alloc_info,
                                                  cvdescriptorset::AllocateDescriptorSetsData *ds_data) {
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        auto layout = GetDescriptorSetLayout(this, p_alloc_info->pSetLayouts[i]);
        if (layout) {
            ds_data->layout_nodes[i] = layout;
        }
    }
}

// Decrement allocated sets from the pool and insert new sets into set_map
void GpuVal::PerformAllocateDescriptorSets(const VkDescriptorSetAllocateInfo *p_alloc_info,
                                               const VkDescriptorSet *descriptor_sets,
                                               const cvdescriptorset::AllocateDescriptorSetsData *ds_data) {
    const auto *variable_count_info = lvl_find_in_chain<VkDescriptorSetVariableDescriptorCountAllocateInfoEXT>(p_alloc_info->pNext);
    bool variable_count_valid = variable_count_info && variable_count_info->descriptorSetCount == p_alloc_info->descriptorSetCount;

    // Create tracking object for each descriptor set; insert into global map and the pool's set.
    for (uint32_t i = 0; i < p_alloc_info->descriptorSetCount; i++) {
        uint32_t variable_count = variable_count_valid ? variable_count_info->pDescriptorCounts[i] : 0;

        std::unique_ptr<cvdescriptorset::DescriptorSet> new_ds(new cvdescriptorset::DescriptorSet(
            descriptor_sets[i], p_alloc_info->descriptorPool, ds_data->layout_nodes[i], variable_count, this));
        setMap[descriptor_sets[i]] = std::move(new_ds);
    }
}
