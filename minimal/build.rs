use std::error::Error;

use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    SpirvBuilder::new(
        "../minimal-shader",
        "spirv-unknown-vulkan1.2",
    )
    .capability(Capability::RayTracingKHR)
    .extension("SPV_KHR_ray_tracing")
    .print_metadata(MetadataPrintout::Full)
    .build()?;

    Ok(())
}
