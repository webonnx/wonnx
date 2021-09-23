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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    })
}

pub fn create_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let slice_size = size as usize * std::mem::size_of::<f32>();
    let size = slice_size as wgpu::BufferAddress;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    })
}

pub fn read_only_buffer(device: &wgpu::Device, array: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(array),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
    })
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
        let _ = crate::resource::create_buffer_init(&device, &data);
    }
}

pub fn size(tensor: &crate::onnx::ValueInfoProto) -> i64 {
    i64::max(
        tensor
            .get_field_type()
            .get_tensor_type()
            .get_shape()
            .get_dim()
            .iter()
            .fold(1, |acc, dim| acc * dim.get_dim_value())
            * std::mem::size_of::<f32>() as i64,
        16,
    )
}

pub fn len(tensor: &crate::onnx::ValueInfoProto) -> i64 {
    tensor
        .get_field_type()
        .get_tensor_type()
        .get_shape()
        .get_dim()
        .iter()
        .fold(1, |acc, dim| acc * dim.get_dim_value())
}

pub fn len_index(tensor: &crate::onnx::ValueInfoProto, index: usize) -> Option<i64> {
    let shape = tensor
        .get_field_type()
        .get_tensor_type()
        .get_shape()
        .get_dim();

    let len = shape.len();
    if index < len {
        Some(shape[index].get_dim_value())
    } else {
        None
    }
}
