pub struct LayerOutput {
    pub net_output: Vec<f32>,
    pub activated_output: Vec<f32>
}

impl LayerOutput {
    pub fn new (net_output: Vec<f32>, activated_output: Vec<f32> ) -> Self {
        LayerOutput {
            net_output,
            activated_output
        }
    }
}

pub struct ModelOutput {
    pub output: Vec<LayerOutput>,
}

impl ModelOutput {
    pub fn new() -> Self {
        ModelOutput {
            output: vec![],
        }
    }

    pub fn push(&mut self, layer_output: LayerOutput) {
        self.output.push(layer_output);
    }

    pub fn last(&self) -> &LayerOutput {
        self.output.last().expect("No output in ModelOutput")
    }

    pub fn len(&self) -> usize {
        self.output.len()
    }

    pub fn get_by_layer(&self, layer_index: usize) -> &LayerOutput {
        self.output.get(layer_index).expect("No output for the specified layer")
    }
}