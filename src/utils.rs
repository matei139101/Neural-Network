use rand::Rng;

pub fn sygmoid(input: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-input));
}

pub fn dot_product<T1: Into<f32> + Copy, T2: Into<f32> + Copy> (vector1: &[T1], vector2: &[T2]) -> f32 {
    let mut  product: f32 = 0.0;

    for index in 0..vector1.len(){
        product += vector1[index].into() * vector2[index].into();
    }

    return product;
}

pub fn compare(input: f32, correction: f32) -> f32 {
    let mut answer: f32;

    answer = correction - input;
    answer = answer.abs();

    return answer;
}

pub fn random_number(min_range: f32, max_range: f32 ) -> f32 {
    return rand::thread_rng().gen_range(min_range..max_range);
}

/*
pub struct CastedF32<T>(T);

impl CastedF32<bool> {
    fn get(&self) -> f32 {
        if self.0 { 1.0 } else { 0.0 }
    }
}

impl CastedF32<i32> {
    fn get(&self) -> f32 {
        self.0 as f32
    }
}

impl CastedF32<f32> {
    fn get(&self) -> f32 {
        self.0
    }
}
*/