pub enum debug_tier {
    HIGH,
    MEDIUM,
    LOW
}

pub fn log(tier: debug_tier, message: &str) {
    println!("{message}");
}