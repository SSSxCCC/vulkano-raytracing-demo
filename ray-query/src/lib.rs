use glam::Affine3A;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        DescriptorSet, WriteDescriptorSet,
    },
    image::{view::ImageView, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::AttachmentStoreOp,
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl From<[f32; 2]> for MyVertex {
    fn from(value: [f32; 2]) -> Self {
        MyVertex { position: value }
    }
}

#[cfg(target_os = "android")]
use winit::platform::android::{activity::AndroidApp, EventLoopBuilderExtAndroid};

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Trace),
    );
    let event_loop = EventLoop::builder().with_android_app(app).build().unwrap();
    _main(event_loop);
}

#[cfg(not(target_os = "android"))]
#[allow(dead_code)]
fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .parse_default_env()
        .init();
    let event_loop = EventLoop::new().unwrap();
    _main(event_loop);
}

fn _main(event_loop: EventLoop<()>) {
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut config = VulkanoConfig::default();
    config.device_extensions.khr_deferred_host_operations = true;
    config.device_extensions.khr_acceleration_structure = true;
    config.device_extensions.khr_ray_query = true;
    config.device_features.acceleration_structure = true;
    config.device_features.ray_query = true;
    config.device_features.dynamic_rendering = true;
    config.device_features.buffer_device_address = true;
    let context = VulkanoContext::new(config);
    let windows = VulkanoWindows::default();
    let mut application = Application { context, windows };

    log::warn!("Vulkano start main loop!");
    event_loop.run_app(&mut application).unwrap();
}

struct Application {
    context: VulkanoContext,
    windows: VulkanoWindows,
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        log::debug!("Resumed");
        self.windows.create_window(
            &event_loop,
            &self.context,
            &WindowDescriptor::default(),
            |info| {
                //info.image_format = Some(Format::R8G8B8A8_UNORM);
                info.image_usage |= ImageUsage::STORAGE;
            },
        );
    }

    fn suspended(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        log::debug!("Suspended");
        self.windows
            .remove_renderer(self.windows.primary_window_id().unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::debug!("WindowEvent::CloseRequested");
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                log::debug!("WindowEvent::Resized");
                if let Some(renderer) = self.windows.get_primary_renderer_mut() {
                    renderer.resize();
                    renderer.window().request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                log::debug!("WindowEvent::ScaleFactorChanged");
                if let Some(renderer) = self.windows.get_primary_renderer_mut() {
                    renderer.resize();
                    renderer.window().request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                log::trace!("WindowEvent::RedrawRequested");
                if let Some(renderer) = self.windows.get_primary_renderer_mut() {
                    let acquire_future = renderer.acquire(None, |_| {}).unwrap();
                    let draw_future = draw_image(
                        &self.context,
                        renderer.swapchain_image_view(),
                        acquire_future,
                    );
                    renderer.present(draw_future, true);
                }
            }
            _ => (),
        }
    }
}

fn draw_image(
    context: &VulkanoContext,
    image_view: Arc<ImageView>,
    before_future: Box<dyn GpuFuture>,
) -> Box<dyn GpuFuture> {
    // The quad buffer that covers the entire surface
    let quad: [MyVertex; 6] = [
        [-1.0, -1.0].into(),
        [-1.0, 1.0].into(),
        [1.0, -1.0].into(),
        [1.0, 1.0].into(),
        [1.0, -1.0].into(),
        [-1.0, 1.0].into(),
    ];
    let quad_buffer = Buffer::from_iter(
        context.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        quad,
    )
    .unwrap();

    let pipeline = {
        let vs = vs::load(context.device().clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(context.device().clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            context.device().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(context.device().clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(image_view.format())],
            ..Default::default()
        };
        GraphicsPipeline::new(
            context.device().clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.color_attachment_formats.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                viewport_state: Some(ViewportState::default()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let image_extent = image_view.image().extent();
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [image_extent[0] as f32, image_extent[1] as f32],
        depth_range: 0.0..=1.0,
    };

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        context.device().clone(),
        Default::default(),
    ));

    let (tlas, before_future) = {
        let vertices = vec![
            [-0.5, -0.25, 1.0].into(),
            [0.0, 0.5, 1.0].into(),
            [0.25, -0.1, 1.0].into(),
        ];
        let (blas, blas_future) =
            raytracing_util::create_triangle_bottom_level_acceleration_structure(
                context.memory_allocator().clone(),
                command_buffer_allocator.clone(),
                context.graphics_queue().clone(),
                vec![(vertices, None)],
            );

        let instances = vec![(blas, 0, vec![Affine3A::IDENTITY])];
        let (tlas, tlas_future) = raytracing_util::create_top_level_acceleration_structure(
            context.memory_allocator().clone(),
            command_buffer_allocator.clone(),
            context.graphics_queue().clone(),
            instances,
        );

        (tlas, before_future.join(blas_future).join(tlas_future))
    };

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        context.device().clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));

    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts().get(0).unwrap().clone(),
        [WriteDescriptorSet::acceleration_structure(0, tlas)],
        [],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        context.graphics_queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .begin_rendering(RenderingInfo {
            color_attachments: vec![Some(RenderingAttachmentInfo {
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(image_view.clone())
            })],
            ..Default::default()
        })
        .unwrap()
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .bind_vertex_buffers(0, quad_buffer.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            descriptor_set.clone(),
        )
        .unwrap();

    unsafe { builder.draw(quad_buffer.len() as u32, 1, 0, 0) }.unwrap();

    builder.end_rendering().unwrap();

    let command_buffer = builder.build().unwrap();
    command_buffer
        .execute_after(before_future, context.graphics_queue().clone())
        .unwrap()
        .boxed()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460
            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 out_uv;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                out_uv = position;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            #extension GL_EXT_ray_query : enable
            layout(location = 0) in vec2 in_uv;
            layout(location = 0) out vec4 f_color;
            layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;
            void main() {
                float t_min = 0.01;
                float t_max = 1000.0;
                vec3 origin = vec3(0.0, 0.0, 0.0);
                vec3 direction = normalize(vec3(in_uv * 1.0, 1.0));
                rayQueryEXT ray_query;
                rayQueryInitializeEXT(
                    ray_query,
                    top_level_acceleration_structure,
                    gl_RayFlagsTerminateOnFirstHitEXT,
                    0xFF,
                    origin,
                    t_min,
                    direction,
                    t_max
                );
                rayQueryProceedEXT(ray_query);
                if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                    // miss
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    // hit
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            }
        ",
    }
}
