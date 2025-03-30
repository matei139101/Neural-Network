use rand::Rng;

pub fn dot_product(vec1: &Vec<f32>, mat: &Vec<Vec<f32>>) -> Vec<f32> {
    let cols = mat[0].len();
    let rows = mat.len();

    let mut result = vec![0.0; cols];

    for j in 0..cols {
        for i in 0..rows {
            result[j] += mat[i][j] * vec1[i];
        }
    }

    return result
}

pub fn avg_third_dimension(data: Vec<Vec<Vec<f32>>>) -> Vec<Vec<f32>> {
    let mut sums = Vec::new();
    let mut avgs = Vec::new();

    for mat in &data {
        if mat.is_empty() {
            sums.push(vec![]);
            avgs.push(vec![]);
            continue;
        }

        let cols = mat[0].len();
        let mut sum_vec = vec![0.0; cols];

        // Sum up elements
        for row in mat {
            for (i, &val) in row.iter().enumerate() {
                sum_vec[i] += val;
            }
        }

        // Compute averages
        let row_count = mat.len() as f32;
        let avg_vec: Vec<f32> = sum_vec.iter().map(|&sum| sum / row_count).collect();

        sums.push(sum_vec);
        avgs.push(avg_vec);
    }

    return avgs;
}

pub fn random_number(min_range: f32, max_range: f32 ) -> f32 {
    return rand::thread_rng().gen_range(min_range..max_range);
}

pub fn sigmoid(input: &f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-input));
}

pub fn meansquarederror(value: &f32, correct_value: &f32) -> f32 {
    return (value - correct_value) * (value - correct_value);
}