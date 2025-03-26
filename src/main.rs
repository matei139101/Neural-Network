use activations::{sigmoid::Sigmoid, tanh::Tanh};
use layers::dense::Dense;
use lossfunctions::meansquarederror::MeanSquaredError;
use model::Model;
use utils::{logger, math};

mod layers;
mod activations;
mod lossfunctions;
mod utils;
mod model;

fn main() {
    let input: Vec<f32> = vec![1f32, 0f32];
    let targets: Vec<f32> = vec![1f32];

    let mut model: Model = Model::new(input, Box::new(MeanSquaredError {}));
    model.add_layer(Box::new(Dense::new(2, 5, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(5, 1, Box::new(Sigmoid {}))));
    model.fit();
    model.predict();

    model.train(&targets, 0.001f32);
}