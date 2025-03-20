// pub fn sygmoid(input: f32) -> f32 {
//     return 1.0 / (1.0 + f32::exp(-input));
// }
use rand::Rng;

pub fn dot_product(vec1: &Vec<f32>, mat: &Vec<Vec<f32>>) -> Vec<f32> {
    let cols = mat[0].len();
    let rows = mat.len();

    let mut result = vec![0.0; cols];

    for j in 0..cols {
        for i in 0..rows {
            result[j] += mat[i][j] * vec1[i];
        }
    }

    return result
}

pub fn random_number(min_range: f32, max_range: f32 ) -> f32 {
    return rand::thread_rng().gen_range(min_range..max_range);
}