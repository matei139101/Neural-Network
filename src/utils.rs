pub fn sygmoid(input: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-input));
}

pub fn dot_product<T1: Into<f32> + Copy, T2: Into<f32> + Copy> (vector1: &[T1], vector2: &[T2]) -> f32 {
    let mut  product: f32 = 0.0;

    for index in 0..vector1.len(){
        product += vector1[index].into() * vector2[index].into();
    }

    return product;
}