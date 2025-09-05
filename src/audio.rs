use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};
use std::sync::{Arc, Mutex, mpsc};

pub enum AudioCommand {
    SetVolume(f32),
    SetTrack(Track),
}

pub struct AudioSystem {
    _stream: cpal::Stream,
    channel: mpsc::Sender<AudioCommand>,
}

pub enum Track {
    Xpansive,
    Bear,
    YouLost,
    Bassline,
}

// TODO support sound effects that play for a brief period and then finish.
// For that we need to keep state of current sound effect(s). The state: which
// track, and when it started playing (in sample_clock time. That means we
// need to change the sample clock mod behavior otherwise we'll glitch. Oh well
// whatever.) Or we can track where we are in the track right now. I ... think
// this still glitches when the sample clock modulo happens. Okay. Then we need
// to know the volume for each track. (Background music is just another track).
// We also need global volume. Then we do a weighted sum of the output from each
// track and scale the whole thing by global volume. Also we need to know the
// length of the track (or infinity). Once the track reaches its end, we remove
// it from currently playing. Decsision: do we allow multiple instances of the
// same track to be played? If not, then we can just have one slot in the vector
// for each track we support and simply track whether it's playing or not.
// Simple, dumb and no allocation in the hot path.
struct AudioState {
    sample_clock: f32,
    volume: f32,
    track: Track,
}

impl AudioSystem {
    pub fn new() -> Self {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .expect("no output device available");
        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_max_sample_rate();

        let sample_format = supported_config.sample_format();
        let config: cpal::StreamConfig = supported_config.into();
        let sample_rate = config.sample_rate.0 as f32;
        let channels = config.channels as usize;

        println!("sample format {:?}, rate {:?}", sample_format, sample_rate);

        let (sender, receiver) = mpsc::channel::<AudioCommand>();

        let audio_state = Arc::new(Mutex::new(AudioState {
            sample_clock: 0.,
            volume: 1.,
            track: Track::Bassline,
        }));

        let mut next_value = move || {
            while let Ok(cmd) = receiver.try_recv() {
                if let Ok(mut state) = audio_state.try_lock() {
                    match cmd {
                        AudioCommand::SetVolume(v) => state.volume = v.clamp(0., 1.),
                        AudioCommand::SetTrack(t) => state.track = t,
                    }
                }
            }

            if let Ok(mut state) = audio_state.try_lock() {
                state.sample_clock = (state.sample_clock + 1.0) % (10. * 60. * sample_rate);
                let t = 8000. * state.sample_clock / sample_rate;
                let t = t as u32;

                // let expr = t * (42 & t >> 10);
                // let expr = t*((t&4096?6:16)+(1&t>>14))>>(3&t>>8)|t>>(t&4096?3:4);
                // let expr = t*((t&4096?6:16)+(1&t>>14))>>(3&t>>8)|t>>(t&4096?3:4);
                // let expr = (t * 5 & t >> 7) | (t * 3 & t >> 10);

                let expr = match state.track {
                    Track::Xpansive => (t >> 7 | t | t >> 6) * 10 + 4 * (t & t >> 13 | t >> 6),
                    Track::YouLost => {
                        let q = t >> 9 | t >> 13;
                        if q != 0 { t % (t / q) } else { 0 }
                    }
                    Track::Bassline => {
                        (!t >> 2)
                            * (if (127 & t * (7 & t >> 10)) < (245 & t * (2 + (5 & t >> 14))) {
                                1
                            } else {
                                0
                            })
                    }
                    Track::Bear => {
                        let c = if t % 16 != 0 { 2 } else { 6 };
                        t + (t & t ^ t >> 6) - t * ((t >> 9) & (c) & t >> 9)
                    }
                };
                let expr_char = expr as u8;

                expr_char as f32 * state.volume / 255.
            } else {
                0.
            }
        };

        let err_fn = |err| eprintln!("an error occurred on stream: {err}");

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    write_data(data, channels, &mut next_value)
                },
                err_fn,
                None,
            )
            .unwrap();
        stream.play().unwrap();

        Self {
            _stream: stream,
            channel: sender,
        }
    }

    fn send_command(&self, command: AudioCommand) {
        self.channel.send(command).unwrap()
    }

    pub fn set_volume(&self, volume: f32) {
        self.send_command(AudioCommand::SetVolume(volume))
    }
    pub fn set_track(&self, track: Track) {
        self.send_command(AudioCommand::SetTrack(track))
    }
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}
