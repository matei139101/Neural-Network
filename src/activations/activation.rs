pub trait Activation {
    fn calculate(inputs: Vec<f32>) -> Vec<f32>;
}