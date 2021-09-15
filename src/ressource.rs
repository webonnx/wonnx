use wgpu::util::DeviceExt;

// Get a device and a queue
pub async fn request_device_queue() -> (wgpu::Device, wgpu::Queue) {
    // `()` indicates that the macro takes no argument.
    // The macro will expand into the contents of this block.

    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
        })
        .await
        .expect("No GPU Found for referenced preference");

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("Could not create adapter for GPU device")
}

pub fn create_buffer_init(device: &wgpu::Device, array: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(array),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
    })
}

pub fn create_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let slice_size = size as usize * std::mem::size_of::<f32>();
    let size = slice_size as wgpu::BufferAddress;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

pub fn read_only_buffer(device: &wgpu::Device, array: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(array),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_request_device_queue() {
        pollster::block_on(crate::ressource::request_device_queue());
    }

    #[test]
    fn test_create_buffer_init() {
        let (device, _) = pollster::block_on(crate::ressource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let _ = crate::ressource::create_buffer_init(&device, &data);
    }
}
