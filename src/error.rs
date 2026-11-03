use std::fmt;

/// Encoding errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimensions { width: u32, height: u32 },
    BufferSize { expected: usize, actual: usize },
    Unsupported(&'static str),
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
        }
    }
}

impl std::error::Error for Error {}
