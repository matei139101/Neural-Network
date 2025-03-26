pub trait Layer {
    fn get_inputs(&self) -> usize;
    fn get_neurons(&self) -> usize;
    fn get_weights(&self) -> &Vec<Vec<f32>>;

    fn process (&mut self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> &Vec<f32>;
    fn make_weights (&mut self);
    fn allign(&self, alligner: &usize) -> bool;
    fn get_derivatives(&self, inputs: &Vec<f32>, weights: &Vec<Vec<f32>>, backwards_derivative: &Vec<f32>) -> Vec<f32>;
}