// pub fn sygmoid(input: f32) -> f32 {
//     return 1.0 / (1.0 + f32::exp(-input));
// }

use rand::Rng;

pub fn dot_product(vector1: &Vec<f32>, vector2: &Vec<f32>) -> f32 {
    vector1.iter()
        .zip(vector2.iter())
        .map(|(a, b)| a * b)
        .sum()
}

// pub fn compare(input: f32, correction: f32) -> f32 {
//     let mut answer: f32;

//     answer = correction - input;
//     answer = answer.abs();

//     return answer;
// }

pub fn random_number(min_range: f32, max_range: f32 ) -> f32 {
    return rand::thread_rng().gen_range(min_range..max_range);
}