use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{vec3, Affine3A, Quat, Vec3};
use rand::prelude::*;
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

#[derive(Clone, Copy, Default, Zeroable, Pod)]
#[repr(C)]
pub struct EnumMaterialPod {
    data: [f32; 4],
    t: u32,
    _pad: [f32; 3],
}

impl EnumMaterialPod {
    pub fn new_lambertian(albedo: Vec3) -> Self {
        Self {
            data: [albedo.x, albedo.y, albedo.z, 0.0],
            t: 0,
            _pad: [0.0, 0.0, 0.0],
        }
    }

    pub fn new_metal(albedo: Vec3, fuzz: f32) -> Self {
        Self {
            data: [albedo.x, albedo.y, albedo.z, fuzz],
            t: 1,
            _pad: [0.0, 0.0, 0.0],
        }
    }

    pub fn new_dielectric(ir: f32) -> Self {
        Self {
            data: [ir, 0.0, 0.0, 0.0],
            t: 2,
            _pad: [0.0, 0.0, 0.0],
        }
    }
}

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
    let scene = sample_scene();

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
                    scene.clone(),
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
        let sphere_intersection_shader_module =
            sphere_intersection::load(context.vulkano().device().clone()).unwrap();
        let sphere_closesthit_shader_module =
            sphere_closesthit::load(context.vulkano().device().clone()).unwrap();

        let (shader_stages, stages) = raytracing_util::create_shader_stages([
            raygen_shader_module,
            miss_shader_module,
            sphere_intersection_shader_module,
            sphere_closesthit_shader_module,
        ]);

        let shader_groups = raytracing_util::ash::create_shader_groups([
            ShaderGroup::General(0),
            ShaderGroup::General(1),
            ShaderGroup::ProceduralHitGroup {
                closest_hit_shader: 3,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: 2,
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
    scene: (Vec<Affine3A>, Vec<EnumMaterialPod>),
) -> Box<dyn GpuFuture> {
    // acceleration structures
    let (blas, blas_future) =
        raytracing_util::vulkano::create_aabb_bottom_level_acceleration_structure(
            context.vulkano().memory_allocator().clone(),
            context.vulkano_ext().command_buffer_allocator(),
            context.vulkano().graphics_queue().clone(),
            vec![AabbPositions {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            }],
        );

    let (sphere_instances, materials) = scene;
    let (tlas, tlas_future) = raytracing_util::vulkano::create_top_level_acceleration_structure(
        context.vulkano().memory_allocator().clone(),
        context.vulkano_ext().command_buffer_allocator(),
        context.vulkano().graphics_queue().clone(),
        vec![(blas, 0, sphere_instances)],
    );

    let before_future = before_future.join(blas_future).join(tlas_future);

    let material_buffer = Buffer::from_iter(
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
        materials,
    )
    .unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        context.vulkano_ext().descriptor_set_allocator(),
        draw_context.descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::acceleration_structure(0, tlas),
            WriteDescriptorSet::image_view(1, image_view.clone()),
            WriteDescriptorSet::buffer(2, material_buffer),
        ],
        [],
    )
    .unwrap();

    let mut rng = StdRng::from_entropy();

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
            &rng.next_u32().to_le_bytes(),
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

fn sample_scene() -> (Vec<Affine3A>, Vec<EnumMaterialPod>) {
    let mut rng = StdRng::from_entropy();
    let mut world = Vec::new();

    world.push((
        create_sphere_instance(vec3(0.0, -1000.0, 0.0), 1000.0),
        EnumMaterialPod::new_lambertian(vec3(0.5, 0.5, 0.5)),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let center = vec3(
                a as f32 + 0.9 * rng.gen::<f32>(),
                0.2,
                b as f32 + 0.9 * rng.gen::<f32>(),
            );

            let choose_mat: f32 = rng.gen();

            if (center - vec3(4.0, 0.2, 0.0)).length() > 0.9 {
                match choose_mat {
                    x if x < 0.8 => {
                        let albedo = vec3(rng.gen(), rng.gen(), rng.gen())
                            * vec3(rng.gen(), rng.gen(), rng.gen());

                        world.push((
                            create_sphere_instance(center, 0.2),
                            EnumMaterialPod::new_lambertian(albedo),
                        ));
                    }
                    x if x < 0.95 => {
                        let albedo = vec3(
                            rng.gen_range(0.5..1.0),
                            rng.gen_range(0.5..1.0),
                            rng.gen_range(0.5..1.0),
                        );
                        let fuzz = rng.gen_range(0.0..0.5);

                        world.push((
                            create_sphere_instance(center, 0.2),
                            EnumMaterialPod::new_metal(albedo, fuzz),
                        ));
                    }
                    _ => world.push((
                        create_sphere_instance(center, 0.2),
                        EnumMaterialPod::new_dielectric(1.5),
                    )),
                }
            }
        }
    }

    world.push((
        create_sphere_instance(vec3(0.0, 1.0, 0.0), 1.0),
        EnumMaterialPod::new_dielectric(1.5),
    ));

    world.push((
        create_sphere_instance(vec3(-4.0, 1.0, 0.0), 1.0),
        EnumMaterialPod::new_lambertian(vec3(0.4, 0.2, 0.1)),
    ));

    world.push((
        create_sphere_instance(vec3(4.0, 1.0, 0.0), 1.0),
        EnumMaterialPod::new_metal(vec3(0.7, 0.6, 0.5), 0.0),
    ));

    let mut spheres = Vec::new();
    let mut materials = Vec::new();

    for (sphere, material) in world.into_iter() {
        spheres.push(sphere);
        materials.push(material);
    }

    (spheres, materials)
}

fn create_sphere_instance(pos: Vec3, size: f32) -> Affine3A {
    Affine3A::from_scale_rotation_translation(Vec3::splat(size), Quat::IDENTITY, pos)
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require

            // =============== Random ===============

            struct PCG32si {
                uint state;
            };

            // Step function for PCG32
            void pcg_oneseq_32_step_r(inout PCG32si rng) {
                const uint PCG_DEFAULT_MULTIPLIER_32 = 747796405u;
                const uint PCG_DEFAULT_INCREMENT_32 = 2891336453u;
                rng.state = (rng.state * PCG_DEFAULT_MULTIPLIER_32 + PCG_DEFAULT_INCREMENT_32);
            }

            // PCG output function
            uint pcg_output_rxs_m_xs_32_32(uint state) {
                uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                return (word >> 22u) ^ word;
            }

            // Create a new RNG with a seed
            PCG32si pcg_new(uint seed) {
                PCG32si rng;
                rng.state = seed;
                pcg_oneseq_32_step_r(rng);
                rng.state += seed;  // equivalent to wrapping_add
                pcg_oneseq_32_step_r(rng);
                return rng;
            }

            // Generate a random uint
            uint next_u32(inout PCG32si rng) {
                uint old_state = rng.state;
                pcg_oneseq_32_step_r(rng);
                return pcg_output_rxs_m_xs_32_32(old_state);
            }

            // Generate a random float [0.0, 1.0)
            float next_f32(inout PCG32si rng) {
                const uint float_size = 32u;  // Number of bits in a float
                const uint float_precision = 24u;  // Precision for floating point numbers (23 bits + 1 sign bit)
                const float scale = 1.0 / float(1 << float_precision);

                uint value = next_u32(rng);
                value >>= (float_size - float_precision);  // Shift to get the desired precision
                return scale * float(value);
            }

            // Generate a random float in the range [min, max]
            float next_f32_range(inout PCG32si rng, float min, float max) {
                return min + (max - min) * next_f32(rng);
            }

            // =============== Math ===============

            #define PI 3.1415926538

            vec3 random_in_unit_sphere(inout PCG32si rng) {
                // Generate random spherical coordinates (direction)
                float theta = next_f32_range(rng, 0.0, 2.0 * PI);  // Uniform azimuthal angle
                float phi = next_f32_range(rng, -1.0, 1.0);        // Uniform cosine of polar angle

                // Sample radius as the cube root of a uniform random value to ensure uniform distribution in volume
                float r = pow(next_f32(rng), 1.0 / 3.0);  // Cube root of a uniform random number in [0, 1]

                // Convert spherical coordinates (r, theta, phi) to Cartesian coordinates
                float x = r * sqrt(1.0 - phi * phi) * cos(theta);
                float y = r * sqrt(1.0 - phi * phi) * sin(theta);
                float z = r * phi;

                return vec3(x, y, z);
            }

            vec3 random_in_hemisphere(vec3 normal, inout PCG32si rng) {
                vec3 v = normalize(random_in_unit_sphere(rng));
                if (dot(normal, v) > 0.0) {
                    return v;
                } else {
                    return -v;
                }
            }

            vec3 random_in_unit_disk(inout PCG32si rng) {
                // Generate random angle between 0 and 2π
                float theta = next_f32_range(rng, 0.0, 2.0 * PI);

                // Generate random radius squared between 0 and 1, then take the square root to make it uniform in area
                float r2 = next_f32(rng);  // Uniformly sample r^2 (radius squared)
                float r = sqrt(r2);        // Take square root to get radius

                // Convert polar coordinates to Cartesian coordinates
                float x = r * cos(theta);
                float y = r * sin(theta);

                return vec3(x, y, 0.0);
            }

            // =============== Ray and Payload structs ===============

            struct Ray {
                vec3 origin;
                vec3 direction;
            };

            Ray default_Ray() {
                return Ray(vec3(0.0), vec3(0.0));
            }

            struct RayPayload {
                vec3 position;
                vec3 normal;
                bool is_miss;
                uint material;
                bool front_face;
            };

            RayPayload default_RayPayload() {
                RayPayload payload;
                payload.position = vec3(0.0);
                payload.normal = vec3(0.0);
                payload.is_miss = false;
                payload.material = 0u;
                payload.front_face = false;
                return payload;
            }

            // =============== Materials ===============

            float reflectance(float cosine, float ref_idx) {
                float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
                r0 = r0 * r0;
                return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
            }

            struct Scatter {
                vec3 color;
                Ray ray;
            };

            Scatter default_Scatter() {
                return Scatter(vec3(0.0), default_Ray());
            }

            // Materials
            struct Lambertian {
                vec3 albedo;
            };

            struct Metal {
                vec3 albedo;
                float fuzz;
            };

            struct Dielectric {
                float ir;
            };

            // Scatter functions for different materials
            bool scatter_Lambertian(Lambertian material, Ray ray, RayPayload ray_payload, inout PCG32si rng, inout Scatter scatter) {
                vec3 scatter_direction = ray_payload.normal + normalize(random_in_unit_sphere(rng));
                scatter_direction = (length(scatter_direction) < 1e-8) ? ray_payload.normal : scatter_direction;

                scatter.ray.origin = ray_payload.position;
                scatter.ray.direction = scatter_direction;
                scatter.color = material.albedo;

                return true;
            }

            bool scatter_Metal(Metal material, Ray ray, RayPayload ray_payload, inout PCG32si rng, inout Scatter scatter) {
                vec3 reflected = reflect(normalize(ray.direction), ray_payload.normal);
                vec3 scatter_direction = reflected + material.fuzz * random_in_unit_sphere(rng);

                if (dot(scatter_direction, ray_payload.normal) > 0.0) {
                    scatter.ray.origin = ray_payload.position;
                    scatter.ray.direction = scatter_direction;
                    scatter.color = material.albedo;
                    return true;
                }
                return false;
            }

            bool scatter_Dielectric(Dielectric material, Ray ray, RayPayload ray_payload, inout PCG32si rng, inout Scatter scatter) {
                float refraction_ratio = ray_payload.front_face ? (1.0 / material.ir) : material.ir;
                vec3 unit_direction = normalize(ray.direction);
                float cos_theta = min(dot(-unit_direction, ray_payload.normal), 1.0);
                float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                bool cannot_refract = refraction_ratio * sin_theta > 1.0;

                vec3 direction = (cannot_refract || reflectance(cos_theta, refraction_ratio) > next_f32(rng))
                    ? reflect(unit_direction, ray_payload.normal)
                    : refract(unit_direction, ray_payload.normal, refraction_ratio);

                scatter.ray.origin = ray_payload.position;
                scatter.ray.direction = direction;
                scatter.color = vec3(1.0, 1.0, 1.0); // White for Dielectric

                return true;
            }

            // Scatter function for EnumMaterial
            struct EnumMaterial {
                vec4 data;
                uint t;
            };

            bool scatter_EnumMaterial(EnumMaterial material, Ray ray, RayPayload ray_payload, inout PCG32si rng, inout Scatter scatter) {
                if (material.t == 0u) {
                    Lambertian material = Lambertian(material.data.xyz);
                    return scatter_Lambertian(material, ray, ray_payload, rng, scatter);
                } else if (material.t == 1u) {
                    Metal material = Metal(material.data.xyz, material.data.w);
                    return scatter_Metal(material, ray, ray_payload, rng, scatter);
                } else if (material.t == 2u) {
                    Dielectric material = Dielectric(material.data.x);
                    return scatter_Dielectric(material, ray, ray_payload, rng, scatter);
                } else {
                    return false;
                }
            }

            // =============== Camera ===============

            // Camera structure
            struct Camera {
                vec3 origin;
                vec3 lower_left_corner;
                vec3 horizontal;
                vec3 vertical;
                vec3 u;
                vec3 v;
                float lens_radius;
            };

            // Camera creation function
            Camera create_camera(vec3 look_from, vec3 look_at, vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist) {
                float theta = vfov;
                float h = tan(theta / 2.0);
                float viewport_height = 2.0 * h;
                float viewport_width = aspect_ratio * viewport_height;

                vec3 w = normalize(look_from - look_at);
                vec3 u = normalize(cross(vup, w));
                vec3 v = cross(w, u);

                vec3 origin = look_from;
                vec3 horizontal = focus_dist * viewport_width * u;
                vec3 vertical = focus_dist * viewport_height * v;
                vec3 lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

                Camera cam;
                cam.origin = origin;
                cam.lower_left_corner = lower_left_corner;
                cam.horizontal = horizontal;
                cam.vertical = vertical;
                cam.u = u;
                cam.v = v;
                cam.lens_radius = aperture / 2.0;

                return cam;
            }

            // Function to generate a ray from the camera
            Ray get_ray(Camera cam, float s, float t, inout PCG32si rng) {
                vec3 rd = cam.lens_radius * random_in_unit_disk(rng);
                vec3 offset = cam.u * rd.x + cam.v * rd.y;

                Ray r;
                r.origin = cam.origin + offset;
                r.direction = normalize(cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin - offset);

                return r;
            }

            // =============== Shader ===============

            layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
            layout(set = 0, binding = 1, rgba8) uniform image2D out_image;
            layout(set = 0, binding = 2) buffer Materials {
                EnumMaterial materials[];
            };

            layout(location = 0) rayPayloadEXT RayPayload payload;

            layout(push_constant) uniform PushConstants {
                uint seed;
            };

            void main() {
                // Launch ID and size (inbuilt variables in GLSL)
                uvec3 launch_id = gl_LaunchIDEXT;
                uvec3 launch_size = gl_LaunchSizeEXT;

                // Random seed initialization
                uint rand_seed = (launch_id.y * launch_size.x + launch_id.x) ^ seed;
                PCG32si rng = pcg_new(rand_seed);

                // Camera setup
                Camera camera = create_camera(
                    vec3(13.0, 2.0, 3.0),
                    vec3(0.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    radians(20.0),
                    float(launch_size.x) / float(launch_size.y),
                    0.1,
                    10.0
                );

                uint cull_mask = 0xff;
                float tmin = 0.001;
                float tmax = 100000.0;

                vec3 final_color = vec3(0.0);

                const uint N_SAMPLES = 30;

                for (uint i = 0; i < N_SAMPLES; i++) {
                    float u = (float(launch_id.x) + next_f32(rng)) / float(launch_size.x - 1);
                    float v = (float(launch_id.y) + next_f32(rng)) / float(launch_size.y - 1);

                    vec3 color = vec3(1.0);
                    Ray ray = get_ray(camera, u, v, rng);

                    for (int j = 0; j < 30; j++) {
                        payload = default_RayPayload();
                        traceRayEXT(
                            top_level_as,
                            gl_RayFlagsOpaqueEXT,
                            cull_mask,
                            0, 0, 0,
                            ray.origin, tmin, ray.direction, tmax,
                            0
                        );

                        if (payload.is_miss) {
                            color *= payload.position;
                            break;
                        } else {
                            Scatter scatter = default_Scatter();
                            if (scatter_EnumMaterial(materials[payload.material], ray, payload, rng, scatter)) {
                                color *= scatter.color;
                                ray = scatter.ray;
                            } else {
                                break;
                            }
                        }
                    }

                    final_color += color;
                }

                final_color = pow(final_color / float(N_SAMPLES), vec3(0.5));

                ivec2 pos = ivec2(launch_id.xy);
                pos.y = int(launch_size.y) - 1 - pos.y;

                imageStore(out_image, pos, vec4(final_color, 1.0));
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

            layout(location = 0) rayPayloadInEXT RayPayload {
                vec3 position;
                vec3 normal;
                bool is_miss;
                uint material;
                bool front_face;
            } payload;

            void main() {
                vec3 world_ray_direction = normalize(gl_WorldRayDirectionEXT);
                float t = 0.5 * (world_ray_direction.y + 1.0);
                vec3 color = mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);

                payload.is_miss = true;
                payload.position = color;
                payload.normal = vec3(0.0, 0.0, 0.0);
                payload.material = 0;
                payload.front_face = false;
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

            hitAttributeEXT float t;

            void main() {
                vec3 ray_origin = gl_ObjectRayOriginEXT;
                vec3 ray_direction = gl_ObjectRayDirectionEXT;
                float t_min = gl_RayTminEXT;
                float t_max = gl_RayTmaxEXT;

                vec3 oc = ray_origin;
                float a = dot(ray_direction, ray_direction);
                float half_b = dot(oc, ray_direction);
                float c = dot(oc, oc) - 1.0;

                float discriminant = half_b * half_b - a * c;
                if (discriminant < 0.0) {
                    return;  // No intersection
                }

                float sqrtd = sqrt(discriminant);
                float root0 = (-half_b - sqrtd) / a;
                float root1 = (-half_b + sqrtd) / a;

                if (root0 >= t_min && root0 <= t_max) {
                    t = root0;
                    reportIntersectionEXT(root0, 0);  // Report intersection
                }

                if (root1 >= t_min && root1 <= t_max) {
                    t = root1;
                    reportIntersectionEXT(root1, 0);  // Report intersection
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

            struct RayPayload {
                vec3 position;
                vec3 normal;
                bool is_miss;
                uint material;
                bool front_face;
            };

            RayPayload new_RayPayload(vec3 position, vec3 outward_normal, vec3 ray_direction, uint material) {
                bool front_face = dot(ray_direction, outward_normal) < 0.0;
                vec3 normal = front_face ? outward_normal : -outward_normal;

                return RayPayload(
                    position,
                    normal,
                    false, // is_miss initialized to false
                    material,
                    front_face
                );
            }

            hitAttributeEXT float t;
            layout(location = 0) rayPayloadInEXT RayPayload payload;

            void main() {
                vec3 hit_pos = gl_WorldRayOriginEXT + t * gl_WorldRayDirectionEXT;
                vec3 normal = normalize(hit_pos - gl_ObjectToWorldEXT[3]);
                payload = new_RayPayload(hit_pos, normal, gl_WorldRayDirectionEXT, gl_InstanceCustomIndexEXT);
            }
        ",
    }
}
