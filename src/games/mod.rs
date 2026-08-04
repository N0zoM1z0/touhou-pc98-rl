pub mod common;
//pub mod th04;
pub mod th05_c;

// use crate::error::Result;
// use self::common::finder::GameFinder;
// It is called th05_c because I used anniversery branch of ReC98
// project. For now, only th05 is present (the custom version), so
// I just map th05 to th05_c.
//
// Create a finder for the specified game.
//pub fn create_finder(game: &str, pid: i32) -> Result<Box<dyn GameFinder>> {
//    match game {
//        //"th04" => Ok(Box::new(th04::finder::TH04Finder::new(pid)?)),
//        "th05" => Ok(Box::new(th05_c::finder::TH05Finder::new(pid)?)),
//        _ => Err(crate::error::Error::InvalidConfig {
//            reason: format!("Unknown game: {}", game),
//        }),
//    }
//}
