use ndarray as nd17;
use ndarray016 as nd16;

fn convert_16_17(a: nd16::Array2<f32>) -> nd17::Array2<f32> {
    let shape = a.raw_dim();

    // safest cross-version conversion
    let data: Vec<f32> = a.iter().copied().collect();

    nd17::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
}

fn convert_17_16(a: nd17::Array2<f32>) -> nd16::Array2<f32> {
    let shape = a.raw_dim();

    let data: Vec<f32> = a.iter().copied().collect();

    nd16::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
}
