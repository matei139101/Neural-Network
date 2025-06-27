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