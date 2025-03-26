pub trait LossFunction {
    fn calculate(&self, input: &f32, target: &f32) -> f32;
    fn derivative(&self, output: &f32, target: &f32) -> f32;
}