pub trait LossFunction {
    fn multi_calculate(&self, inputs: Vec<f32>, correct_values: Vec<f32>) -> Vec<f32>;
    fn single_calculate(&self, inputs: Vec<f32>, correct_values: Vec<f32>) -> f32;
}