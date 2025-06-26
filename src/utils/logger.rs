//The tag is to suppress warnings while still working out the functions
#[derive(Copy, Clone)]
pub enum DebugTier {
    IMPORTANT = 15,
    HIGH = 10,
    MEDIUM = 5,
    LOW = 1
}

impl PartialEq for DebugTier {
    fn eq(&self, other: &Self) -> bool {
        *self as i32 == *other as i32
    }
}

impl PartialOrd for DebugTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (*self as i32).partial_cmp(&(*other as i32))
    }
}

const GLOBAL_TIER: DebugTier = DebugTier::IMPORTANT;

pub fn log(tier: DebugTier, message: String) {
    if tier >= GLOBAL_TIER {
        print!("\n{message}")
    }
}

pub fn logln(tier: DebugTier, message: String) {
    if tier >= GLOBAL_TIER {
        print!("{message}")
    }
}