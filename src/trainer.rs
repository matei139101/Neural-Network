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
            let zipped_input_targets = self.input.iter().zip(self.targets.iter());
            let mut epoch_output: Vec<Vec<f32>> = vec![];
            for (input_vec, target_vec) in zipped_input_targets {
                let output = self.model.predict(input_vec);
                epoch_output.push(output.clone());
                self.model.back_propagate(&output, target_vec);
            }

            for layer in &mut self.model.layers {
                layer.train(learning_rate);
            }

            logger::log(DebugTier::IMPORTANT, format!("Epoch: {}, Loss: {}", epoch+1, self.model.loss(&epoch_output, &self.targets)));
        }
    }
}