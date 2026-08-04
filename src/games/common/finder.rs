// use crate::error::Result;
// use crate::memory::ProcessMemory;

/// Discovered memory addresses.
#[derive(Debug, Clone, Default)]
pub struct DiscoveredAddresses {
    pub resident: Option<usize>,
    pub player_pos: Option<usize>,
    pub bullets: Option<usize>,
    pub enemies: Option<usize>,
    pub items: Option<usize>,
    pub score: Option<usize>,
    pub power: Option<usize>,
    pub midboss: Option<usize>,
    pub midboss_hp: Option<usize>,
    pub boss: Option<usize>,
    pub boss_hp: Option<usize>,
    pub boss_2: Option<usize>,
    pub boss_2_hp: Option<usize>,
    pub stage_point: Option<usize>,
    /// Not present in th02.
    pub stage_graze: Option<usize>,
    /// Have different meaning in different games.
    pub dream_score: Option<usize>,
    pub key_det: Option<usize>,
}
