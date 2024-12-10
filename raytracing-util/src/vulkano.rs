use glam::{Affine3A, Vec3};
use std::sync::Arc;
use vulkano::{
    acceleration_structure::{
        AabbPositions, AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryAabbsData,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
        GeometryFlags, GeometryInstanceFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::graphics::vertex_input::Vertex,
    sync::GpuFuture,
    DeviceSize, Packed24_8,
};
use vulkano_util::context::VulkanoContext;

pub struct VulkanoContextExtension {
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
}

impl VulkanoContextExtension {
    pub fn new(vulkano: &VulkanoContext) -> Self {
        VulkanoContextExtension {
            command_buffer_allocator: StandardCommandBufferAllocator::new(
                vulkano.device().clone(),
                Default::default(),
            ),
            descriptor_set_allocator: StandardDescriptorSetAllocator::new(
                vulkano.device().clone(),
                Default::default(),
            ),
        }
    }

    pub fn command_buffer_allocator(&self) -> &StandardCommandBufferAllocator {
        &self.command_buffer_allocator
    }

    pub fn descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator {
        &self.descriptor_set_allocator
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TriangleVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

impl From<[f32; 3]> for TriangleVertex {
    fn from(value: [f32; 3]) -> Self {
        TriangleVertex { position: value }
    }
}

impl From<Vec3> for TriangleVertex {
    fn from(value: Vec3) -> Self {
        TriangleVertex {
            position: value.to_array(),
        }
    }
}

pub fn create_triangle_bottom_level_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    vertices_and_indices: Vec<(Vec<TriangleVertex>, Option<Vec<u16>>)>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let description = TriangleVertex::per_vertex();
    assert_eq!(
        description.stride,
        std::mem::size_of::<TriangleVertex>() as u32
    );

    let mut triangles = vec![];
    let mut max_primitive_counts = vec![];
    let mut build_range_infos = vec![];

    for (vertices, indices) in vertices_and_indices {
        let buffer_create_info = BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        };
        let allocation_create_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            buffer_create_info.clone(),
            allocation_create_info.clone(),
            vertices,
        )
        .unwrap();

        let index_buffer = indices.map(|indices| {
            IndexBuffer::U16(
                Buffer::from_iter(
                    memory_allocator.clone(),
                    buffer_create_info,
                    allocation_create_info,
                    indices,
                )
                .unwrap(),
            )
        });

        let vertex_count = vertex_buffer.len() as u32;
        let primitive_count = vertex_count / 3;
        triangles.push(AccelerationStructureGeometryTrianglesData {
            flags: GeometryFlags::OPAQUE,
            vertex_data: Some(vertex_buffer.into_bytes()),
            vertex_stride: description.stride,
            max_vertex: vertex_count,
            index_data: index_buffer,
            transform_data: None,
            ..AccelerationStructureGeometryTrianglesData::new(
                description.members.get("position").unwrap().format,
            )
        });
        max_primitive_counts.push(primitive_count);
        build_range_infos.push(AccelerationStructureBuildRangeInfo {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        })
    }

    let geometries = AccelerationStructureGeometries::Triangles(triangles);
    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::BottomLevel,
        build_info,
        &max_primitive_counts,
        build_range_infos,
    )
}

pub fn create_aabb_bottom_level_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    aabb_positions: Vec<AabbPositions>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let mut aabbs = vec![];
    let mut max_primitive_counts = vec![];
    let mut build_range_infos = vec![];

    for aabb_positions in aabb_positions {
        let buffer_create_info = BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        };
        let allocation_create_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let aabb_positions_buffer = Buffer::from_data(
            memory_allocator.clone(),
            buffer_create_info,
            allocation_create_info,
            aabb_positions,
        )
        .unwrap();

        aabbs.push(AccelerationStructureGeometryAabbsData {
            flags: GeometryFlags::OPAQUE,
            data: Some(aabb_positions_buffer.into_bytes()),
            stride: std::mem::size_of::<AabbPositions>() as u32,
            ..Default::default()
        });
        max_primitive_counts.push(1);
        build_range_infos.push(AccelerationStructureBuildRangeInfo {
            primitive_count: 1,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        })
    }

    let geometries = AccelerationStructureGeometries::Aabbs(aabbs);
    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::BottomLevel,
        build_info,
        &max_primitive_counts,
        build_range_infos,
    )
}

pub fn create_top_level_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    instances: Vec<(Arc<AccelerationStructure>, u32, Vec<Affine3A>)>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let instances = instances
        .into_iter()
        .map(|(blas, sbt_index, transforms)| {
            let blas_ref = blas.device_address().get();
            transforms
                .into_iter()
                .map(move |transform| (transform, sbt_index, blas_ref))
        })
        .flatten()
        .enumerate()
        .map(
            |(i, (transform, sbt_index, blas_ref))| AccelerationStructureInstance {
                transform: super::affine3a_to_rows_array_2d(transform),
                instance_custom_index_and_mask: Packed24_8::new(i as _, 0xff),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    sbt_index,
                    GeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE.into(),
                ),
                acceleration_structure_reference: blas_ref,
            },
        )
        .collect::<Vec<_>>();

    let instance_count = instances.len();

    let values = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        instances,
    )
    .unwrap();

    let geometries =
        AccelerationStructureGeometries::Instances(AccelerationStructureGeometryInstancesData {
            flags: GeometryFlags::OPAQUE,
            ..AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(values)),
            )
        });

    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let build_range_infos = [AccelerationStructureBuildRangeInfo {
        primitive_count: instance_count as _,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    }];

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::TopLevel,
        build_info,
        &[instance_count as _],
        build_range_infos,
    )
}

fn build_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    ty: AccelerationStructureType,
    mut build_info: AccelerationStructureBuildGeometryInfo,
    max_primitive_counts: &[u32],
    build_range_infos: impl IntoIterator<Item = AccelerationStructureBuildRangeInfo>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let device = memory_allocator.device();

    let AccelerationStructureBuildSizesInfo {
        acceleration_structure_size,
        build_scratch_size,
        ..
    } = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &build_info,
            max_primitive_counts,
        )
        .unwrap();

    let acceleration_structure =
        create_acceleration_structure(memory_allocator.clone(), ty, acceleration_structure_size);
    let scratch_buffer = create_scratch_buffer(memory_allocator, build_scratch_size);

    build_info.dst_acceleration_structure = Some(acceleration_structure.clone());
    build_info.scratch_data = Some(scratch_buffer);

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder
            .build_acceleration_structure(build_info, build_range_infos.into_iter().collect())
            .unwrap();
    }

    let command_buffer = builder.build().unwrap();
    let gpu_future = command_buffer.execute(queue).unwrap().boxed();

    (acceleration_structure, gpu_future)
}

fn create_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    ty: AccelerationStructureType,
    size: DeviceSize,
) -> Arc<AccelerationStructure> {
    let buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap();

    unsafe {
        AccelerationStructure::new(
            memory_allocator.device().clone(),
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(buffer)
            },
        )
        .unwrap()
    }
}

fn create_scratch_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    size: DeviceSize,
) -> Subbuffer<[u8]> {
    Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap()
}
