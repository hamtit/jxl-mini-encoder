use std::fmt;

/// Encoding errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimensions {
        width: u32,
        height: u32,
    },
    BufferSize {
        expected: usize,
        actual: usize,
    },
    Unsupported(&'static str),
    /// An animation without frames.
    NoFrames,
    /// Frames differ in size or in having alpha.
    FrameMismatch,
    /// Ticks per second numerator or denominator is zero.
    InvalidAnimation,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height } => {
                write!(f, "invalid image dimensions {width}x{height}")
            }
            Error::BufferSize { expected, actual } => {
                write!(f, "pixel buffer has {actual} bytes, expected {expected}")
            }
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::NoFrames => write!(f, "animation has no frames"),
            Error::FrameMismatch => write!(f, "animation frames differ in size or alpha"),
            Error::InvalidAnimation => write!(f, "animation tick rate must be non-zero"),
        }
    }
}

impl std::error::Error for Error {}
