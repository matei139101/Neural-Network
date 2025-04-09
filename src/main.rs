use activations::sigmoid::Sigmoid;
use layers::dense::Dense;
use lossfunctions::meansquarederror::MeanSquaredError;
use model::Model;
use trainer::Trainer;

mod layers;
mod activations;
mod lossfunctions;
mod utils;
mod model;
mod trainer;

fn main() {
    let input: Vec<Vec<f32>> = vec![
        vec![1f32, 0f32, 0f32], 
        vec![1f32, 1f32, 0f32],
        vec![0f32, 0f32, 1f32],
        vec![0f32, 1f32, 1f32],
        vec![1f32, 1f32, 1f32],
        ];
    let targets: Vec<Vec<f32>> = vec![
        vec![1f32], vec![1f32], vec![0f32], vec![0f32], vec![1f32]
        ];

    let mut model: Model = Model::new( Box::new(MeanSquaredError {}));
    model.add_layer(Box::new(Dense::new(3, 1, Box::new(Sigmoid {}))));
    model.prepare();

    let mut trainer = Trainer::new(model, input, targets);
    trainer.train(100, 0.1f32);
}