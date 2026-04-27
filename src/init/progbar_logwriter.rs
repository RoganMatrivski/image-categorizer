use indicatif::MultiProgress;
use std::io::Write;
pub struct ProgressBarLogWriter<W: Write> {
    writer: W,
    mpb: MultiProgress,
}
impl<W: Write> ProgressBarLogWriter<W> {
    pub fn new(writer: W, mpb: MultiProgress) -> Self {
        Self { writer, mpb }
    }
}
impl<W: Write> Write for ProgressBarLogWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.mpb.suspend(|| self.writer.write(buf))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.mpb.suspend(|| self.writer.flush())
    }
}
