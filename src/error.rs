#[derive(thiserror::Error, Debug)]
pub enum PaddleOcrError {
    #[error("ort error:`{0}`")]
    Ort(#[from] ort::Error),
    #[error("io error:`{0}`")]
    Io(#[from] std::io::Error),
    #[error("custom error:`{0}`")]
    Custom(String)
}

impl PaddleOcrError {
    pub fn custom(s: &str) -> Self {
        Self::Custom(s.to_string())
    }
}

pub type PaddleOcrResult<T> = Result<T, PaddleOcrError>;
