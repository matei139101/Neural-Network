pub trait Layer {
    fn get_inputs(&self) -> usize;
    fn get_neurons(&self) -> usize;
    fn get_weights(&self) -> &Vec<Vec<f32>>;

    fn clear_layer(&mut self);
    fn process (&mut self, input: &Vec<f32>) -> &Vec<f32>;
    fn make_weights (&mut self);
    fn allign(&self, alligner: &usize) -> bool;
    fn back_propagate(&mut self, backwards_derivative: &Vec<f32>) -> Vec<f32>;
    fn train(&mut self, learning_rate: f32);
}