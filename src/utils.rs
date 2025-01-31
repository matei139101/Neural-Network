use rand::Rng;

pub fn sygmoid(input: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-input));
}

pub fn dot_product(vector1: &[f32], vector2: &[f32]) -> f32 {
    let mut  product: f32 = 0.0;

    for index in 0..vector1.len(){
        product += vector1[index] * vector2[index];
    }

    return product;
}

pub fn compare(input: f32, correction: f32) -> f32 {
    let mut answer: f32;

    answer = correction - input;
    answer = answer.abs();

    return answer;
}

pub fn random_number(min_range: f32, max_range: f32 ) -> f32 {
    return rand::thread_rng().gen_range(min_range..max_range);
}