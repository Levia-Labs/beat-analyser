//! WAV input. ffmpeg has already decoded whatever the user uploaded into PCM WAV
//! (`analysis.to_wav`), so this reads WAV only.

use std::path::Path;

pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Read a WAV file and mix it down to mono, samples in [-1, 1].
pub fn read_wav(path: &Path) -> Result<Audio, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };

    let samples = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(Audio {
        samples,
        sample_rate: spec.sample_rate,
    })
}
