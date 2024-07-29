pub fn sygmoid(input: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-input));
}

pub fn dot_product(vector1: Vec<f32>, vector2: Vec<f32>) -> f32 {
    let mut  product: f32 = 0.0;

    for index in 1..vector1.len(){
        product += vector1[index] * vector2[index];
    }

    return product;
}