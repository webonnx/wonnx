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

pub fn create_buffer_init<T: Clone + bytemuck::Pod>(
    device: &wgpu::Device,
    array: &[T],
    name: &str,
) -> wgpu::Buffer {
    let size = array.len();
    if size % 4 != 0 && size != 0 {
        let mut array = array.to_vec();
        array.resize(size + 4 - size % 4, array[0]);

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&array),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(array),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
        })
    }
}

pub fn create_buffer(device: &wgpu::Device, size: u64, name: &str) -> wgpu::Buffer {
    let slacked_size = if size % 4 != 0 {
        size + (4 - size % 4)
    } else {
        size
    };

    let slice_size = usize::max(16, slacked_size as usize * std::mem::size_of::<f32>());
    let size = slice_size as wgpu::BufferAddress;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    })
}

pub fn read_only_buffer(device: &wgpu::Device, array: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(array),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

pub fn output_buffer(device: &wgpu::Device, size: u64, name: &str) -> wgpu::Buffer {
    let slacked_size = if size % 4 != 0 {
        size + (4 - size % 4)
    } else {
        size
    };

    let slice_size = usize::max(16, slacked_size as usize * std::mem::size_of::<f32>());
    let size = slice_size as wgpu::BufferAddress;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    })
}

pub fn size(dims: &[i64]) -> i64 {
    dims.iter().product::<i64>()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_request_device_queue() {
        pollster::block_on(crate::resource::request_device_queue());
    }

    #[test]
    fn test_create_buffer_init() {
        let (device, _) = pollster::block_on(crate::resource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let _ = crate::resource::create_buffer_init(&device, &data, "test");
    }
}
