use activations::sigmoid::Sigmoid;
use layers::dense::Dense;
use lossfunctions::meansquarederror::MeanSquaredError;
use model::Model;
use trainer::Trainer;

use crate::utils::dataset::Dataset;

mod layers;
mod activations;
mod lossfunctions;
mod utils;
mod model;
mod trainer;

fn main() {
    let dataset = Dataset::new();
    
    let mut model: Model = Model::new( Box::new(MeanSquaredError {}));
    model.add_layer(Box::new(Dense::new(28*28, 256, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(256, 128, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(128, 64, Box::new(Sigmoid {}))));
    model.add_layer(Box::new(Dense::new(64, 10, Box::new(Sigmoid {}))));
    model.prepare();

    let mut trainer = Trainer::new(model, dataset.get_training_set(), dataset.get_target_set());
    trainer.train(30, 0.01f32);
}