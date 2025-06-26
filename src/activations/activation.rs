pub trait Activation {
    fn calculate(&self, inputs: &Vec<f32>) -> Vec<f32>;
    fn derivative(&self, net_input: &f32) -> f32;
}