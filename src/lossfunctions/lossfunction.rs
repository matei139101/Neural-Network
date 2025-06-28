pub trait LossFunction: Send + Sync {
    fn calculate(&self, input_set: &Vec<f32>, target_set: &Vec<f32>) -> f32;
    fn derivative(&self, output: &f32, target: &f32, mean_size: usize) -> f32;
}