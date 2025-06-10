use mnist::{Mnist, MnistBuilder};

pub struct Dataset {
    training_set: Vec<Vec<f32>>,
    target_set: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn new() -> Dataset {
        let Mnist {
            trn_img,
            trn_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(1_000)
            .finalize();

        let num_images = 1_000;
        let img_size = 28 * 28;
        let train_data_vec: Vec<Vec<f32>> = (0..num_images)
            .map(|i| {
                let start = i * img_size;
                let end = start + img_size;
                trn_img[start..end]
                    .iter()
                    .map(|&x| x as f32 / 256.0)
                    .collect()
            })
            .collect();
 
        let one_hot_labels: Vec<Vec<f32>> = trn_lbl.iter()
            .map(|&label| {
                let mut one_hot = vec![0.0; 10];
                let idx = label as usize;
                one_hot[idx] = 1.0;
                one_hot
            })
            .collect();

        Dataset {
            training_set: train_data_vec,
            target_set: one_hot_labels,
        }
    }

    pub fn get_training_set(&self) -> Vec<Vec<f32>> {
        self.training_set.clone()
    }

    pub fn get_target_set(&self) -> Vec<Vec<f32>> {
        self.target_set.clone()
    }
}