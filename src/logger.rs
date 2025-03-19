//The tag is to suppress warnings while still working out the functions
#[allow(dead_code)]
pub enum DebugTier {
    HIGH = 10,
    MEDIUM = 5,
    LOW = 1
}

pub fn log(tier: DebugTier, message: String) {
    match tier {
        DebugTier::LOW | DebugTier::MEDIUM | DebugTier::HIGH => { println!("{message}") }
    }
}