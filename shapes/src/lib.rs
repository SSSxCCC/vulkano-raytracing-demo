use ash::vk;
use glam::{Affine3A, Vec3};
use raytracing_util::{
    ash::{AshBuffer, AshPipeline, SbtRegion, ShaderGroup},
    RenderContext,
};
use std::sync::Arc;
use vulkano::{
    acceleration_structure::AabbPositions,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract},
    descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    image::{view::ImageView, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{layout::PipelineDescriptorSetLayoutCreateInfo, PipelineLayout},
    sync::GpuFuture,
    VulkanObject,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
};

#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Trace),
    );
    use winit::platform::android::EventLoopBuilderExtAndroid;
    let event_loop = EventLoopBuilder::new().with_android_app(app).build();
    _main(event_loop);
}

#[cfg(not(target_os = "android"))]
#[allow(dead_code)]
fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .parse_default_env()
        .init();
    let event_loop = EventLoopBuilder::new().build();
    _main(event_loop);
}

fn _main(event_loop: EventLoop<()>) {
    let mut config = VulkanoConfig::default();
    config.device_extensions.khr_deferred_host_operations = true;
    config.device_extensions.khr_acceleration_structure = true;
    config.device_extensions.khr_ray_tracing_pipeline = true;
    config.device_features.acceleration_structure = true;
    config.device_features.ray_tracing_pipeline = true;
    config.device_features.buffer_device_address = true;
    let vulkano = VulkanoContext::new(config);
    let context = RenderContext::new(vulkano);
    let draw_context = DrawContext::new(&context);
    let mut windows = VulkanoWindows::default();

    event_loop.run(move |event, event_loop, control_flow| match event {
        Event::Resumed => {
            log::debug!("Event::Resumed");
            windows.create_window(
                &event_loop,
                context.vulkano(),
                &WindowDescriptor::default(),
                |info| {
                    //info.image_format = Some(Format::R32G32B32A32_SFLOAT);
                    info.image_usage |= ImageUsage::STORAGE;
                },
            );
        }
        Event::Suspended => {
            log::debug!("Event::Suspended");
            windows.remove_renderer(windows.primary_window_id().unwrap());
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => {
                log::debug!("WindowEvent::CloseRequested");
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::Resized(_) => {
                log::debug!("WindowEvent::Resized");
                if let Some(renderer) = windows.get_primary_renderer_mut() {
                    renderer.resize()
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                log::debug!("WindowEvent::ScaleFactorChanged");
                if let Some(renderer) = windows.get_primary_renderer_mut() {
                    renderer.resize()
                }
            }
            _ => (),
        },
        Event::RedrawRequested(_) => {
            if let Some(renderer) = windows.get_primary_renderer_mut() {
                let acquire_future = renderer.acquire().unwrap();
                let draw_future = draw_image(
                    &context,
                    &draw_context,
                    renderer.swapchain_image_view(),
                    acquire_future,
                );
                renderer.present(draw_future, true);
            }
        }
        Event::MainEventsCleared => {
            if let Some(renderer) = windows.get_primary_renderer() {
                renderer.window().request_redraw()
            }
        }
        _ => (),
    });
}

struct DrawContext {
    pipeline: AshPipeline,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    #[allow(unused)]
    sbt_buffer: AshBuffer,
    sbt_region: SbtRegion,
    #[allow(unused)]
    device: Arc<Device>, // device must be destroyed after vk buffer
}

impl DrawContext {
    fn new(context: &RenderContext) -> Self {
        let raygen_shader_module = raygen::load(context.vulkano().device().clone()).unwrap();
        let miss_shader_module = miss::load(context.vulkano().device().clone()).unwrap();
        let closesthit_shader_module =
            closesthit::load(context.vulkano().device().clone()).unwrap();
        let sphere_intersection_shader_module =
            sphere_intersection::load(context.vulkano().device().clone()).unwrap();
        let sphere_closesthit_shader_module =
            sphere_closesthit::load(context.vulkano().device().clone()).unwrap();

        let (shader_stages, stages) = raytracing_util::create_shader_stages([
            raygen_shader_module,
            miss_shader_module,
            closesthit_shader_module,
            sphere_intersection_shader_module,
            sphere_closesthit_shader_module,
        ]);

        let shader_groups = raytracing_util::ash::create_shader_groups([
            ShaderGroup::General(0),
            ShaderGroup::General(1),
            ShaderGroup::TrianglesHitGroup {
                closest_hit_shader: 2,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
            },
            ShaderGroup::ProceduralHitGroup {
                closest_hit_shader: 4,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: 3,
            },
        ]);

        let pipeline_layout = PipelineLayout::new(
            context.vulkano().device().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(context.vulkano().device().clone())
                .unwrap(),
        )
        .unwrap();
        let descriptor_set_layout = pipeline_layout.set_layouts()[0].clone();

        let pipeline = AshPipeline::new(
            unsafe {
                context.ash().rt_pipeline().create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::builder()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_pipeline_ray_recursion_depth(1)
                        .layout(pipeline_layout.handle())
                        .build()],
                    None,
                )
            }
            .unwrap()[0],
            context.ash().device().clone(),
        );

        let (sbt_buffer, sbt_region) = raytracing_util::ash::create_sbt_buffer_and_region(
            context.ash(),
            *pipeline,
            shader_groups.len(),
        );

        DrawContext {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            sbt_buffer,
            sbt_region,
            device: context.vulkano().device().clone(),
        }
    }
}

fn draw_image(
    context: &RenderContext,
    draw_context: &DrawContext,
    image_view: Arc<ImageView>,
    before_future: Box<dyn GpuFuture>,
) -> Box<dyn GpuFuture> {
    // acceleration structures
    let vertices = vec![
        [-0.5, -0.5, 0.0].into(),
        [0.0, 0.5, 0.0].into(),
        [0.5, -0.5, 0.0].into(),
    ];
    let (blas, blas_future) =
        raytracing_util::vulkano::create_triangle_bottom_level_acceleration_structure(
            context.vulkano().memory_allocator().clone(),
            context.vulkano_ext().command_buffer_allocator(),
            context.vulkano().graphics_queue().clone(),
            vec![(vertices, None)],
        );

    let (blas_sphere, blas_sphere_future) =
        raytracing_util::vulkano::create_aabb_bottom_level_acceleration_structure(
            context.vulkano().memory_allocator().clone(),
            context.vulkano_ext().command_buffer_allocator(),
            context.vulkano().graphics_queue().clone(),
            vec![AabbPositions {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            }],
        );

    let instances = vec![
        (
            blas,
            0,
            vec![
                Affine3A::from_translation(Vec3::new(-1.5, 1.1, 0.0)),
                Affine3A::from_translation(Vec3::new(0.0, -1.1, 0.0)),
                Affine3A::from_translation(Vec3::new(1.5, 1.1, 0.0)),
            ],
        ),
        (
            blas_sphere,
            1,
            vec![Affine3A::from_translation(Vec3::new(0.0, 0.0, 0.0))],
        ),
    ];
    let (tlas, tlas_future) = raytracing_util::vulkano::create_top_level_acceleration_structure(
        context.vulkano().memory_allocator().clone(),
        context.vulkano_ext().command_buffer_allocator(),
        context.vulkano().graphics_queue().clone(),
        instances,
    );

    let before_future = before_future
        .join(blas_future)
        .join(blas_sphere_future)
        .join(tlas_future);

    let color_buffer = Buffer::from_iter(
        context.vulkano().memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [
            1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        context.vulkano_ext().descriptor_set_allocator(),
        draw_context.descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::acceleration_structure(0, tlas),
            WriteDescriptorSet::image_view(1, image_view.clone()),
            WriteDescriptorSet::buffer(2, color_buffer),
        ],
        [],
    )
    .unwrap();

    let command_buffer = AutoCommandBufferBuilder::primary(
        context.vulkano_ext().command_buffer_allocator(),
        context.vulkano().graphics_queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap()
    .build()
    .unwrap();

    let command_buffer_handle = command_buffer.handle();
    unsafe {
        context
            .ash()
            .device()
            .begin_command_buffer(
                command_buffer_handle,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            )
            .expect("Failed to begin recording Command Buffer at beginning!");
        context.ash().device().cmd_bind_pipeline(
            command_buffer_handle,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            *draw_context.pipeline,
        );
        context.ash().device().cmd_bind_descriptor_sets(
            command_buffer_handle,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            draw_context.pipeline_layout.handle(),
            0,
            &[descriptor_set.handle()],
            &[],
        );
        context.ash().device().cmd_push_constants(
            command_buffer_handle,
            draw_context.pipeline_layout.handle(),
            vk::ShaderStageFlags::RAYGEN_KHR,
            0,
            &1.0f32.to_le_bytes(),
        );
        context.ash().rt_pipeline().cmd_trace_rays(
            command_buffer_handle,
            &draw_context.sbt_region.raygen,
            &draw_context.sbt_region.miss,
            &draw_context.sbt_region.hit,
            &draw_context.sbt_region.call,
            image_view.image().extent()[0],
            image_view.image().extent()[1],
            1,
        );
        context
            .ash()
            .device()
            .end_command_buffer(command_buffer_handle)
            .unwrap();
    }

    command_buffer
        .execute_after(before_future, context.vulkano().graphics_queue().clone())
        .unwrap()
        .boxed()
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
            layout(set = 0, binding = 1, rgba8) uniform image2D out_image;
            layout(push_constant) uniform PushConstants {
                float x;
            } pcs;

            layout(location = 0) rayPayloadEXT vec3 payload;

            void main() {
                uvec3 launch_id = gl_LaunchIDEXT;
                uvec3 launch_size = gl_LaunchSizeEXT;

                vec2 pixel_center = vec2(launch_id.x, launch_id.y) + vec2(0.5, 0.5);
                vec2 in_uv = pixel_center / vec2(launch_size.x, launch_size.y);

                vec2 d = in_uv * 2.0 - vec2(1.0, 1.0);
                float aspect_ratio = float(launch_size.x) / float(launch_size.y);

                vec3 origin = vec3(0.0, 0.0, -2.0);
                vec3 direction = normalize(vec3(d.x * aspect_ratio, -d.y, 1.0));

                payload = vec3(0.0);

                traceRayEXT(top_level_as, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin, 0.001, direction, 1000.0, 0);

                ivec2 pos = ivec2(launch_id.x, launch_id.y);
                imageStore(out_image, pos, vec4(payload * pcs.x, 1.0));
            }
        ",
    }
}

mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            layout(location = 0) rayPayloadInEXT vec3 payload;

            void main() {
                payload = vec3(0.5, 0.5, 0.5);
            }
        ",
    }
}

mod closesthit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            layout(set = 0, binding = 2) buffer ColorBuffer {
                vec3 colors[];
            };

            layout(location = 0) rayPayloadInEXT vec3 payload;

            void main() {
                payload = colors[gl_InstanceID];
            }
        ",
    }
}

mod sphere_intersection {
    vulkano_shaders::shader! {
        ty: "intersection",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            hitAttributeEXT vec3 hit_pos;

            void main() {
                vec3 ray_origin = gl_ObjectRayOriginEXT;
                vec3 ray_direction = gl_ObjectRayDirectionEXT;
                vec3 world_ray_origin = gl_WorldRayOriginEXT;
                vec3 world_ray_direction = gl_WorldRayDirectionEXT;

                float t_min = gl_RayTminEXT;
                float t_max = gl_RayTmaxEXT;

                vec3 oc = ray_origin;
                float a = dot(ray_direction, ray_direction);
                float half_b = dot(oc, ray_direction);
                float c = dot(oc, oc) - 1.0;

                float discriminant = half_b * half_b - a * c;
                if (discriminant < 0.0) {
                    return;
                }

                float sqrtd = sqrt(discriminant);

                float root0 = (-half_b - sqrtd) / a;
                float root1 = (-half_b + sqrtd) / a;

                if (root0 >= t_min && root0 <= t_max) {
                    hit_pos = world_ray_origin + root0 * world_ray_direction;
                    reportIntersectionEXT(root0, 0);
                }

                if (root1 >= t_min && root1 <= t_max) {
                    hit_pos = world_ray_origin + root1 * world_ray_direction;
                    reportIntersectionEXT(root1, 0);
                }
            }
        ",
    }
}

mod sphere_closesthit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            layout(location = 0) rayPayloadInEXT vec3 outColor;

            void main() {
                outColor = gl_ObjectToWorldEXT[3];
            }
        ",
    }
}
