use crate::{model::Model, utils::logger::{self, DebugTier}};

pub struct Trainer {
    model: Model,
    input: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>
}

impl Trainer {
    pub fn new(model: Model, input: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        Trainer {
            model,
            input,
            targets
        }
    }

    pub fn train(&mut self, epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            // Run model epochs
            let mut epoch_output: Vec<Vec<f32>> = vec![];
            for (index, _) in self.input.iter().enumerate() {
                let output = self.model.predict(&self.input[index]);
                epoch_output.push(output.clone());
                self.model.back_propagate(&output, &self.targets[index]);
            }

            
            for layer in &mut self.model.layers {
                layer.train(learning_rate);
            }

            logger::log(DebugTier::IMPORTANT, format!("Epoch: {}, Loss: {}", epoch+1, self.model.loss(&epoch_output, &self.targets)));
        }
    }
}