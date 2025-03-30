use activations::sigmoid::Sigmoid;
use layers::dense::Dense;
use lossfunctions::meansquarederror::MeanSquaredError;
use model::Model;

mod layers;
mod activations;
mod lossfunctions;
mod utils;
mod model;

fn main() {
    let input: Vec<Vec<f32>> = vec![
        vec![0f32, 1f32], vec![1f32, 0f32]
        ];
    let targets: Vec<Vec<f32>> = vec![
        vec![1f32], vec![0f32]
        ];

    let mut model: Model = Model::new( Box::new(MeanSquaredError {}));
    model.add_layer(Box::new(Dense::new(2, 5, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(5, 5, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(5, 5, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(5, 1, Box::new(Sigmoid {}))));
    model.prepare();
    model.train(&input, &targets, 0.01f32, 1000);
}