//! Video playback delegated to libvlc, rendered natively. libvlc draws straight
//! into an `NSView` we add over the eframe/wgpu window (`set_nsobject`), so the
//! video lives on a CALayer that macOS animates smoothly — native fullscreen
//! behaves like IINA, and snapshots come out at full source resolution. The view
//! is mouse-transparent (`hitTest:` -> nil), so egui keeps owning all input.

use anyhow::{Result, anyhow};
use eframe::egui;
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly, class, define_class, msg_send};
use objc2_app_kit::{NSApplication, NSAutoresizingMaskOptions, NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::Duration;

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "3gp",
];
pub const SEEK_STEP: f64 = 5.0;
/// Minimum gap between seek commands handed to libvlc.
const SEEK_FLUSH_MS: u64 = 200;
/// How long a sent seek keeps serving as the base/readback position while
/// libvlc catches up.
const SEEK_SETTLE_MS: u64 = 800;

pub fn is_video_file(path: &Path) -> bool {
    crate::archive::has_ext(path, VIDEO_EXTENSIONS)
}

/// The libvlc core, shareable across threads (libvlc is thread-safe).
struct Instance(*mut libvlc_instance_t);
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

/// Warm up the shared libvlc core on a background thread so the plugin scan
/// runs during app startup, not when the first video opens. The plugin path
/// is resolved here, on the main thread, before anything else reads the
/// environment.
pub fn preload() {
    ensure_plugin_path();
    std::thread::spawn(|| {
        let _ = VideoPlayer::shared_instance();
    });
}

define_class!(
    // A layer-backed, mouse-transparent host for libvlc's vout. Returning nil from
    // `hitTest:` lets clicks fall through to the egui layer (play/seek/fullscreen).
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiVideoView"]
    struct VideoView;

    impl VideoView {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, _point: NSPoint) -> *mut NSView {
            core::ptr::null_mut()
        }
    }
);

/// libvlc loads its codecs/demuxers/output as plugins at runtime. Point it at
/// the copy vendored next to the executable (see scripts/bundle-macos.sh) so the
/// shipped .app is self-contained; fall back to a system VLC for local dev.
fn ensure_plugin_path() {
    if std::env::var_os("VLC_PLUGIN_PATH").is_some() {
        return;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("../Frameworks/plugins"));
        candidates.push(dir.join("Frameworks/plugins"));
        candidates.push(dir.join("plugins"));
    }
    // Dev runs (raw target/ binary) use the libs extracted into vendor/vlc.
    candidates.push(PathBuf::from(format!(
        "{}/vendor/vlc/plugins",
        env!("CARGO_MANIFEST_DIR")
    )));
    candidates.push(PathBuf::from("/opt/homebrew/lib/vlc/plugins"));
    candidates.push(PathBuf::from("/usr/local/lib/vlc/plugins"));
    if let Some(dir) = candidates.into_iter().find(|c| c.is_dir()) {
        // SAFETY: set once at startup, before any VLC/thread reads the environment.
        unsafe { std::env::set_var("VLC_PLUGIN_PATH", dir) };
    }
}

pub struct VideoPlayer {
    mp: *mut libvlc_media_player_t,
    // The native render surface; kept alive while playing and released, on the
    // main thread, once the background teardown has stopped the vout.
    view: ManuallyDrop<Retained<VideoView>>,
    /// Stream source serving libvlc's read callbacks (`open_bytes` /
    /// `open_range`); null for plain file media. Freed by the teardown worker,
    /// after the player has stopped.
    src: *mut StreamSrc,
    paused: bool,
    muted: bool,
    volume: f32,
    scrubbing: bool,
    /// Seek target not yet handed to libvlc. Rapid seeks (key mashing, scrub
    /// drags) accumulate here and flush at most once per `SEEK_FLUSH_MS`, so
    /// libvlc never queues a backlog of demux/decode restarts.
    pending_seek: Option<f64>,
    /// Last target sent to libvlc and when; the base for chained relative
    /// seeks and the throttle clock for the next flush.
    sent_seek: Option<(f64, std::time::Instant)>,
    /// Position to restore once libvlc's input is live — a `set_time` issued
    /// right after `play` lands before the input thread exists and is dropped.
    resume_at: Option<f64>,
    /// View rotation in clockwise quarter-turns (0..4): R turns right, E turns
    /// left, T resets.
    rotation: u8,
    /// Last applied (rotation, width, height); skips redundant relayout.
    last_layout: Option<(u8, u32, u32)>,
}

impl VideoPlayer {
    pub fn open(path: &Path) -> Result<Self> {
        let mtm = MainThreadMarker::new().ok_or_else(|| anyhow!("video must open on main thread"))?;
        let cpath = CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| anyhow!("path contains a NUL byte"))?;
        let instance = Self::shared_instance()?;
        let media = unsafe { libvlc_media_new_path(instance, cpath.as_ptr()) };
        if media.is_null() {
            return Err(anyhow!("libvlc_media_new_path failed"));
        }
        Self::start(mtm, media, std::ptr::null_mut())
    }

    /// Play a video held entirely in memory (a decompressed archive entry).
    /// The buffer is served to libvlc through the `src_*` read/seek callbacks,
    /// so nothing is extracted to disk.
    pub fn open_bytes(data: Vec<u8>) -> Result<Self> {
        Self::open_stream(StreamSrc::Mem { data, pos: 0 })
    }

    /// Play `len` bytes of `path` starting at `start` — a stored
    /// (uncompressed) archive entry, pread straight out of the archive file.
    /// No decompression, no buffer, no temp file.
    pub fn open_range(path: &Path, start: u64, len: u64) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::open_stream(StreamSrc::File { file, start, len, pos: 0 })
    }

    fn open_stream(src: StreamSrc) -> Result<Self> {
        let mtm = MainThreadMarker::new().ok_or_else(|| anyhow!("video must open on main thread"))?;
        let instance = Self::shared_instance()?;
        let src = Box::into_raw(Box::new(src));
        let media = unsafe {
            libvlc_media_new_callbacks(
                instance,
                Some(src_open),
                Some(src_read),
                Some(src_seek),
                Some(src_close),
                src.cast(),
            )
        };
        if media.is_null() {
            unsafe { drop(Box::from_raw(src)) };
            return Err(anyhow!("libvlc_media_new_callbacks failed"));
        }
        Self::start(mtm, media, src)
    }

    /// One libvlc core shared by every player. `libvlc_new` loads the whole
    /// plugin bank — a multi-second scan — so it happens exactly once for the
    /// process life, kicked off early by `preload` so the cost overlaps app
    /// startup instead of stalling the first video open.
    fn shared_instance() -> Result<*mut libvlc_instance_t> {
        static INSTANCE: std::sync::OnceLock<Instance> = std::sync::OnceLock::new();
        let instance = INSTANCE
            .get_or_init(|| {
                ensure_plugin_path();
                let args = [
                    CString::new("--no-video-title-show").unwrap(),
                    CString::new("--quiet").unwrap(),
                ];
                let argv: Vec<*const c_char> = args.iter().map(|a| a.as_ptr()).collect();
                Instance(unsafe { libvlc_new(argv.len() as c_int, argv.as_ptr()) })
            })
            .0;
        if instance.is_null() {
            return Err(anyhow!("libvlc_new failed (is VLC installed?)"));
        }
        Ok(instance)
    }

    /// Shared tail of the `open*` constructors: loop option, player, native
    /// render view, play. Releases everything (including `src`) if a step fails.
    fn start(
        mtm: MainThreadMarker,
        media: *mut libvlc_media_t,
        src: *mut StreamSrc,
    ) -> Result<Self> {
        let release_all = |mp: *mut libvlc_media_player_t| unsafe {
            if !mp.is_null() {
                libvlc_media_player_release(mp);
            }
            if !src.is_null() {
                drop(Box::from_raw(src));
            }
        };

        let loop_opt = CString::new(":input-repeat=65535").unwrap();
        unsafe { libvlc_media_add_option(media, loop_opt.as_ptr()) };
        let fast_seek = CString::new(":input-fast-seek").unwrap();
        unsafe { libvlc_media_add_option(media, fast_seek.as_ptr()) };

        let mp = unsafe { libvlc_media_player_new_from_media(media) };
        unsafe { libvlc_media_release(media) };
        if mp.is_null() {
            release_all(std::ptr::null_mut());
            return Err(anyhow!("libvlc_media_player_new_from_media failed"));
        }

        // Build the native render view and slot it behind the AppKit control bar
        // but above the wgpu layer, sized to (and auto-resizing with) the window.
        let view = match Self::make_view(mtm) {
            Some(v) => v,
            None => {
                release_all(mp);
                return Err(anyhow!("no window to attach the video view to"));
            }
        };
        unsafe { libvlc_media_player_set_nsobject(mp, Retained::as_ptr(&view) as *mut c_void) };

        if unsafe { libvlc_media_player_play(mp) } != 0 {
            view.removeFromSuperview();
            release_all(mp);
            return Err(anyhow!("libvlc_media_player_play failed"));
        }

        Ok(Self {
            mp,
            view: ManuallyDrop::new(view),
            src,
            paused: false,
            muted: false,
            volume: 1.0,
            scrubbing: false,
            pending_seek: None,
            sent_seek: None,
            resume_at: None,
            rotation: 0,
            last_layout: None,
        })
    }

    fn make_view(mtm: MainThreadMarker) -> Option<Retained<VideoView>> {
        let app = NSApplication::sharedApplication(mtm);
        let window = app.keyWindow().or_else(|| app.mainWindow())?;
        let content = window.contentView()?;
        let bounds = content.bounds();
        let view: Retained<VideoView> = {
            let this = VideoView::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };
        view.setWantsLayer(true);
        view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        // Positioned `Below` (relative to nil) puts it at the back of the subview
        // list, so the control bar added later stays on top of the video.
        content.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
        Some(view)
    }

    /// Keep the UI loop ticking while playing so the seek bar / clock advance —
    /// the video itself is driven by libvlc's own vout, not by egui repaints.
    pub fn update(&mut self, egui_ctx: &egui::Context) {
        if !self.paused {
            egui_ctx.request_repaint_after(Duration::from_millis(100));
        }
        if let Some(t) = self.resume_at {
            if unsafe { libvlc_media_player_get_time(self.mp) } >= 0 {
                self.resume_at = None;
                self.seek_to(t);
            }
            egui_ctx.request_repaint_after(Duration::from_millis(50));
        }
        if self.pending_seek.is_some() {
            self.flush_seek(false);
            egui_ctx.request_repaint_after(Duration::from_millis(50));
        }
        self.layout();
    }

    /// Turn the view by `quarters` clockwise quarter-turns (3 = one turn
    /// counterclockwise), cycling through 0/90/180/270.
    pub fn rotate(&mut self, quarters: u8) {
        self.rotation = (self.rotation + quarters) % 4;
    }

    pub fn reset_rotation(&mut self) {
        self.rotation = 0;
    }

    /// Size and orient the native video view for the current rotation. For a
    /// quarter turn we hand libvlc a viewport with width/height swapped and spin
    /// the view about its centre, so libvlc re-fits the frame and the rotated
    /// result lands back inside the window — windowed or fullscreen alike, since
    /// the superview bounds are re-read every frame.
    fn layout(&mut self) {
        let (w, h) = {
            let Some(sv) = (unsafe { self.view.superview() }) else { return };
            let b = sv.bounds();
            (b.size.width, b.size.height)
        };
        let key = (self.rotation, w.round() as u32, h.round() as u32);
        if self.last_layout == Some(key) {
            return;
        }
        self.last_layout = Some(key);

        let frame = if self.rotation % 2 == 1 {
            NSRect {
                origin: NSPoint::new((w - h) / 2.0, (h - w) / 2.0),
                size: NSSize::new(h, w),
            }
        } else {
            NSRect { origin: NSPoint::new(0.0, 0.0), size: NSSize::new(w, h) }
        };
        let angle = -(self.rotation as f64) * 90.0;
        let view: &VideoView = &self.view;
        // Apply frame + rotation together with implicit animations off, so live
        // resizes and fullscreen transitions track the window without a lag frame.
        unsafe {
            let _: () = msg_send![class!(CATransaction), begin];
            let _: () = msg_send![class!(CATransaction), setDisableActions: true];
            view.setFrameCenterRotation(0.0);
            view.setFrame(frame);
            view.setFrameCenterRotation(angle);
            let _: () = msg_send![class!(CATransaction), commit];
        }
    }

    /// Write the current frame to `path` (PNG by extension) at full source
    /// resolution — exactly what's on screen, decoded by libvlc's vout.
    pub fn save_snapshot(&self, path: &Path) -> Result<()> {
        let cpath = CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| anyhow!("path contains a NUL byte"))?;
        let r = unsafe { libvlc_video_take_snapshot(self.mp, 0, cpath.as_ptr(), 0, 0) };
        if r == 0 {
            Ok(())
        } else {
            Err(anyhow!("libvlc_video_take_snapshot failed ({r})"))
        }
    }

    pub fn toggle_pause(&mut self) {
        self.set_paused(!self.paused);
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        unsafe { libvlc_media_player_set_pause(self.mp, paused as c_int) };
    }

    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
        unsafe { libvlc_audio_set_mute(self.mp, self.muted as c_int) };
    }

    /// Set volume to `v`, clamped to [0, 1]. A positive volume unmutes,
    /// matching what most players do.
    pub fn set_volume(&mut self, v: f32) {
        self.volume = v.clamp(0.0, 1.0);
        if self.volume > 0.0 {
            self.muted = false;
            unsafe { libvlc_audio_set_mute(self.mp, 0) };
        }
        unsafe { libvlc_audio_set_volume(self.mp, (self.volume * 100.0) as c_int) };
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Seek relative to the newest requested target (not the possibly stale
    /// playback clock), so mashed presses chain into one accumulated jump.
    pub fn seek_by(&mut self, delta: f64) {
        let base = self.pending_seek.or_else(|| self.recent_seek_target());
        let t = (base.unwrap_or_else(|| self.vlc_position()) + delta).max(0.0);
        self.seek_to(t);
    }

    pub fn seek_to(&mut self, target: f64) {
        self.pending_seek = Some(target.max(0.0));
        self.flush_seek(false);
    }

    /// Hand the pending target to libvlc, at most once per `SEEK_FLUSH_MS`
    /// unless forced. Last value wins; superseded targets are never sent.
    fn flush_seek(&mut self, force: bool) {
        let Some(target) = self.pending_seek else { return };
        let throttled = self
            .sent_seek
            .is_some_and(|(_, at)| at.elapsed() < Duration::from_millis(SEEK_FLUSH_MS));
        if throttled && !force {
            return;
        }
        self.pending_seek = None;
        self.sent_seek = Some((target, std::time::Instant::now()));
        unsafe { libvlc_media_player_set_time(self.mp, (target * 1000.0) as i64) };
    }

    /// The last sent target, while libvlc is still likely working on it.
    fn recent_seek_target(&self) -> Option<f64> {
        self.sent_seek
            .filter(|(_, at)| at.elapsed() < Duration::from_millis(SEEK_SETTLE_MS))
            .map(|(t, _)| t)
    }

    /// libvlc has no separate keyframe seek, so scrubbing reuses the plain seek.
    /// Each intermediate seek restarts the decoder and spits out a burst of audio,
    /// so mute the output for the duration of the drag and restore it in `end_scrub`.
    pub fn scrub_to(&mut self, target: f64) {
        if !self.scrubbing {
            self.scrubbing = true;
            unsafe { libvlc_audio_set_mute(self.mp, 1) };
        }
        self.seek_to(target);
    }

    /// Finish a scrub: land on the final drag position, then undo the
    /// scrub-time mute, leaving the user's mute intact.
    pub fn end_scrub(&mut self) {
        if self.scrubbing {
            self.scrubbing = false;
            self.flush_seek(true);
            unsafe { libvlc_audio_set_mute(self.mp, self.muted as c_int) };
        }
    }

    /// The position the player is at or headed to: an unflushed or in-flight
    /// seek target reads back immediately, so the seek bar tracks rapid
    /// presses without waiting on libvlc.
    pub fn position(&self) -> f64 {
        self.resume_at
            .or(self.pending_seek)
            .or_else(|| self.recent_seek_target())
            .unwrap_or_else(|| self.vlc_position())
    }

    /// Continue from `t` as soon as playback has started.
    pub fn resume_from(&mut self, t: f64) {
        self.resume_at = Some(t.max(0.0));
    }

    fn vlc_position(&self) -> f64 {
        let ms = unsafe { libvlc_media_player_get_time(self.mp) };
        if ms < 0 { 0.0 } else { ms as f64 / 1000.0 }
    }
    pub fn duration(&self) -> f64 {
        let ms = unsafe { libvlc_media_player_get_length(self.mp) };
        if ms < 0 { 0.0 } else { ms as f64 / 1000.0 }
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn is_muted(&self) -> bool {
        self.muted
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        // Detach the view now (main thread) so a newly opened video layers over a
        // clean window; the blocking stop happens off-thread so switching videos
        // doesn't stall the UI. VLC keeps the view alive through the raw pointer
        // until it has stopped, then releases it back on the main thread.
        self.view.removeFromSuperview();
        let job = Teardown {
            mp: self.mp,
            src: self.src,
            view: Retained::into_raw(unsafe { ManuallyDrop::take(&mut self.view) }),
        };
        std::thread::spawn(move || job.run());
    }
}

/// The blocking half of tearing a player down. `libvlc_media_player_stop` joins
/// VLC's decoder/output threads, so it runs on a worker thread instead of the
/// main one. Only reached after the owning `VideoPlayer` is gone.
struct Teardown {
    mp: *mut libvlc_media_player_t,
    src: *mut StreamSrc,
    view: *mut VideoView,
}

// The player handle and stream source are handed over exclusively; the view is
// only carried through to its main-thread release below, never used off-main.
unsafe impl Send for Teardown {}

impl Teardown {
    fn run(self) {
        unsafe {
            libvlc_media_player_stop(self.mp);
            libvlc_media_player_release(self.mp);
            if !self.src.is_null() {
                drop(Box::from_raw(self.src));
            }
            // The vout has quit and no longer touches the view; release it back on
            // the main thread, where AppKit requires NSView deallocation to happen.
            dispatch_async_f(main_queue(), self.view.cast(), release_view);
        }
    }
}

unsafe extern "C" fn release_view(view: *mut c_void) {
    drop(unsafe { Retained::from_raw(view.cast::<VideoView>()) });
}

#[repr(C)]
struct dispatch_queue_s {
    _private: [u8; 0],
}

fn main_queue() -> *mut dispatch_queue_s {
    (&raw const _dispatch_main_q).cast_mut()
}

unsafe extern "C" {
    static _dispatch_main_q: dispatch_queue_s;
    fn dispatch_async_f(
        queue: *mut dispatch_queue_s,
        context: *mut c_void,
        work: unsafe extern "C" fn(*mut c_void),
    );
}

/// Backing store for archive playback: libvlc pulls the media through the
/// `src_*` callbacks below instead of opening a path. The state is only
/// touched from libvlc's input thread, never from Rust while the stream lives.
enum StreamSrc {
    /// A decompressed archive entry held in memory.
    Mem { data: Vec<u8>, pos: usize },
    /// A byte range of a file on disk (a stored zip entry), served with pread
    /// — no decompression and no buffering beyond libvlc's own.
    File { file: std::fs::File, start: u64, len: u64, pos: u64 },
}

impl StreamSrc {
    fn len(&self) -> u64 {
        match self {
            StreamSrc::Mem { data, .. } => data.len() as u64,
            StreamSrc::File { len, .. } => *len,
        }
    }

    fn set_pos(&mut self, offset: u64) {
        match self {
            StreamSrc::Mem { pos, .. } => *pos = offset as usize,
            StreamSrc::File { pos, .. } => *pos = offset,
        }
    }
}

unsafe extern "C" fn src_open(opaque: *mut c_void, datap: *mut *mut c_void, sizep: *mut u64) -> c_int {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    // The `:input-repeat` loop reopens the stream: rewind, don't reallocate.
    s.set_pos(0);
    unsafe {
        *datap = opaque;
        *sizep = s.len();
    }
    0
}

unsafe extern "C" fn src_read(opaque: *mut c_void, buf: *mut u8, len: usize) -> isize {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    match s {
        StreamSrc::Mem { data, pos } => {
            let n = len.min(data.len().saturating_sub(*pos));
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(*pos), buf, n) };
            *pos += n;
            n as isize
        }
        StreamSrc::File { file, start, len: total, pos } => {
            use std::os::unix::fs::FileExt;
            let n = len.min(total.saturating_sub(*pos) as usize);
            let out = unsafe { std::slice::from_raw_parts_mut(buf, n) };
            match file.read_at(out, *start + *pos) {
                Ok(n) => {
                    *pos += n as u64;
                    n as isize
                }
                Err(_) => -1,
            }
        }
    }
}

unsafe extern "C" fn src_seek(opaque: *mut c_void, offset: u64) -> c_int {
    let s = unsafe { &mut *opaque.cast::<StreamSrc>() };
    if offset > s.len() {
        return -1;
    }
    s.set_pos(offset);
    0
}

/// The source outlives the stream (freed by the teardown worker, after the
/// player has fully stopped), so closing the stream is a no-op.
unsafe extern "C" fn src_close(_opaque: *mut c_void) {}

#[allow(non_camel_case_types)]
enum libvlc_instance_t {}
#[allow(non_camel_case_types)]
enum libvlc_media_t {}
#[allow(non_camel_case_types)]
enum libvlc_media_player_t {}

type MediaOpenCb = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut u64) -> c_int;
type MediaReadCb = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> isize;
type MediaSeekCb = unsafe extern "C" fn(*mut c_void, u64) -> c_int;
type MediaCloseCb = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" {
    fn libvlc_new(argc: c_int, argv: *const *const c_char) -> *mut libvlc_instance_t;
    fn libvlc_media_new_path(inst: *mut libvlc_instance_t, path: *const c_char) -> *mut libvlc_media_t;
    fn libvlc_media_new_callbacks(
        inst: *mut libvlc_instance_t,
        open_cb: Option<MediaOpenCb>,
        read_cb: Option<MediaReadCb>,
        seek_cb: Option<MediaSeekCb>,
        close_cb: Option<MediaCloseCb>,
        opaque: *mut c_void,
    ) -> *mut libvlc_media_t;
    fn libvlc_media_add_option(md: *mut libvlc_media_t, opt: *const c_char);
    fn libvlc_media_release(md: *mut libvlc_media_t);
    fn libvlc_media_player_new_from_media(md: *mut libvlc_media_t) -> *mut libvlc_media_player_t;
    fn libvlc_media_player_release(mp: *mut libvlc_media_player_t);
    fn libvlc_media_player_play(mp: *mut libvlc_media_player_t) -> c_int;
    fn libvlc_media_player_stop(mp: *mut libvlc_media_player_t);
    fn libvlc_media_player_set_pause(mp: *mut libvlc_media_player_t, do_pause: c_int);
    fn libvlc_media_player_set_nsobject(mp: *mut libvlc_media_player_t, drawable: *mut c_void);
    fn libvlc_media_player_get_time(mp: *mut libvlc_media_player_t) -> i64;
    fn libvlc_media_player_set_time(mp: *mut libvlc_media_player_t, t: i64);
    fn libvlc_media_player_get_length(mp: *mut libvlc_media_player_t) -> i64;
    fn libvlc_video_take_snapshot(
        mp: *mut libvlc_media_player_t,
        num: c_uint,
        path: *const c_char,
        width: c_uint,
        height: c_uint,
    ) -> c_int;
    fn libvlc_audio_set_volume(mp: *mut libvlc_media_player_t, volume: c_int) -> c_int;
    fn libvlc_audio_set_mute(mp: *mut libvlc_media_player_t, status: c_int);
}
