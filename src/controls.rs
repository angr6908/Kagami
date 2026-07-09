//! A native AppKit video control bar layered over the eframe/wgpu window: a
//! frosted `NSVisualEffectView` carrying real `NSSlider`/`NSImageView`/
//! `NSTextField` widgets, styled like QuickTime/IINA. The overlay is purely
//! visual — its container view returns nil from `hitTest:` so every click falls
//! through to the egui layer underneath, which keeps owning seek/play input.

use std::cell::Cell;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSColor, NSControlSize, NSFont, NSImage, NSImageScaling, NSImageView, NSSlider,
    NSTextField, NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState,
    NSVisualEffectView,
};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[name = "KagamiPassThroughView"]
    struct PassThroughView;

    impl PassThroughView {
        #[unsafe(method(hitTest:))]
        fn hit_test(&self, _point: NSPoint) -> *mut NSView {
            core::ptr::null_mut()
        }
    }
);

pub struct VideoState {
    pub position: f64,
    pub duration: f64,
    pub paused: bool,
    pub muted: bool,
    pub volume: f32,
    pub visible: bool,
}

/// A widget's geometry in egui points (top-left origin), so the caller can map a
/// click/drag onto it. The native bar is mouse-transparent, so egui drives both
/// the seek and volume sliders over these rects.
pub struct Hit {
    pub min_x: f32,
    pub min_y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct Bar {
    pub seek: Hit,
    pub volume: Hit,
}

pub struct NativeControls {
    mtm: MainThreadMarker,
    container: Retained<PassThroughView>,
    bar: Retained<NSVisualEffectView>,
    play: Retained<NSImageView>,
    elapsed: Retained<NSTextField>,
    seek: Retained<NSSlider>,
    duration: Retained<NSTextField>,
    speaker: Retained<NSImageView>,
    volume: Retained<NSSlider>,
    last_paused: Cell<Option<bool>>,
    last_muted: Cell<Option<bool>>,
}

const BAR_H: f64 = 56.0;
const BAR_MAX_W: f64 = 720.0;
const BAR_MARGIN: f64 = 24.0;
const PAD: f64 = 16.0;
const ICON: f64 = 22.0;
const SPK: f64 = 16.0;
const LABEL_W: f64 = 46.0;
const LABEL_H: f64 = 14.0;
const VOL_W: f64 = 70.0;
const SLIDER_H: f64 = 15.0;

fn rect(x: f64, y: f64, w: f64, h: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(w, h))
}

fn symbol(name: &str) -> Option<Retained<NSImage>> {
    let s = NSString::from_str(name);
    NSImage::imageWithSystemSymbolName_accessibilityDescription(&s, None)
}

impl NativeControls {
    pub fn new(mtm: MainThreadMarker) -> Option<Self> {
        let app = NSApplication::sharedApplication(mtm);
        let window = app.keyWindow().or_else(|| app.mainWindow())?;
        let content = window.contentView()?;
        let bounds = content.bounds();

        let container: Retained<PassThroughView> = {
            let this = PassThroughView::alloc(mtm);
            unsafe { msg_send![this, initWithFrame: bounds] }
        };

        let bar = {
            let v = NSVisualEffectView::initWithFrame(NSVisualEffectView::alloc(mtm), bounds);
            v.setMaterial(NSVisualEffectMaterial::HUDWindow);
            v.setBlendingMode(NSVisualEffectBlendingMode::WithinWindow);
            v.setState(NSVisualEffectState::Active);
            v.setWantsLayer(true);
            unsafe {
                let layer: *mut AnyObject = msg_send![&*v, layer];
                if !layer.is_null() {
                    let _: () = msg_send![layer, setCornerRadius: 11.0_f64];
                    let _: () = msg_send![layer, setMasksToBounds: true];
                }
            }
            v
        };

        let white = NSColor::whiteColor();

        let make_icon = |name: &str| {
            let iv = NSImageView::new(mtm);
            if let Some(img) = symbol(name) {
                iv.setImage(Some(&img));
            }
            iv.setImageScaling(NSImageScaling::ScaleProportionallyUpOrDown);
            iv.setContentTintColor(Some(&white));
            iv
        };
        let play = make_icon("play.fill");
        let speaker = make_icon("speaker.wave.2.fill");

        let font = NSFont::systemFontOfSize(11.0);
        let make_label = || {
            let l = NSTextField::labelWithString(&NSString::from_str("0:00"), mtm);
            l.setTextColor(Some(&white));
            l.setFont(Some(&font));
            l
        };
        let elapsed = make_label();
        let duration = make_label();

        let make_slider = || {
            let s = NSSlider::new(mtm);
            s.setMinValue(0.0);
            s.setMaxValue(1.0);
            s.setControlSize(NSControlSize::Small);
            s
        };
        let seek = make_slider();
        let volume = make_slider();

        container.addSubview(&bar);
        container.addSubview(&play);
        container.addSubview(&elapsed);
        container.addSubview(&seek);
        container.addSubview(&duration);
        container.addSubview(&speaker);
        container.addSubview(&volume);
        content.addSubview(&container);

        Some(Self {
            mtm,
            container,
            bar,
            play,
            elapsed,
            seek,
            duration,
            speaker,
            volume,
            last_paused: Cell::new(None),
            last_muted: Cell::new(None),
        })
    }

    pub fn hide(&self) {
        self.container.setHidden(true);
    }

    /// Lay the bar out for a `w`x`h` (points) content view and push the current
    /// playback state into the widgets. Returns the seek/volume hitboxes in egui
    /// points.
    pub fn update(
        &self,
        w: f64,
        h: f64,
        st: &VideoState,
        elapsed: &str,
        duration: &str,
    ) -> Option<Bar> {
        self.container.setFrame(rect(0.0, 0.0, w, h));
        self.container.setHidden(!st.visible);
        if !st.visible {
            return None;
        }

        let bw = (w - 2.0 * BAR_MARGIN).min(BAR_MAX_W);
        let bx = (w - bw) / 2.0;
        let by = BAR_MARGIN;
        let cy = by + BAR_H / 2.0;
        self.bar.setFrame(rect(bx, by, bw, BAR_H));

        let play_x = bx + PAD;
        self.play.setFrame(rect(play_x, cy - ICON / 2.0, ICON, ICON));

        let elapsed_x = play_x + ICON + 12.0;
        self.elapsed
            .setFrame(rect(elapsed_x, cy - LABEL_H / 2.0, LABEL_W, LABEL_H));

        let vol_x = bx + bw - PAD - VOL_W;
        self.volume
            .setFrame(rect(vol_x, cy - SLIDER_H / 2.0, VOL_W, SLIDER_H));
        let spk_x = vol_x - 8.0 - SPK;
        self.speaker
            .setFrame(rect(spk_x, cy - SPK / 2.0, SPK, SPK));
        let dur_x = spk_x - 12.0 - LABEL_W;
        self.duration
            .setFrame(rect(dur_x, cy - LABEL_H / 2.0, LABEL_W, LABEL_H));

        let seek_x = elapsed_x + LABEL_W + 12.0;
        let seek_w = (dur_x - 12.0 - seek_x).max(40.0);
        self.seek
            .setFrame(rect(seek_x, cy - SLIDER_H / 2.0, seek_w, SLIDER_H));

        let frac = if st.duration > 0.0 {
            (st.position / st.duration).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.seek.setDoubleValue(frac);
        self.volume.setDoubleValue(st.volume.clamp(0.0, 1.0) as f64);

        self.elapsed.setStringValue(&NSString::from_str(elapsed));
        self.duration.setStringValue(&NSString::from_str(duration));

        if self.last_paused.get() != Some(st.paused) {
            self.last_paused.set(Some(st.paused));
            if let Some(img) = symbol(if st.paused { "play.fill" } else { "pause.fill" }) {
                self.play.setImage(Some(&img));
            }
        }
        if self.last_muted.get() != Some(st.muted) {
            self.last_muted.set(Some(st.muted));
            let name = if st.muted {
                "speaker.slash.fill"
            } else {
                "speaker.wave.2.fill"
            };
            if let Some(img) = symbol(name) {
                self.speaker.setImage(Some(&img));
            }
        }

        let _ = self.mtm;
        let row_top = (h - (cy - SLIDER_H / 2.0) - SLIDER_H) as f32;
        Some(Bar {
            seek: Hit {
                min_x: seek_x as f32,
                min_y: row_top,
                width: seek_w as f32,
                height: SLIDER_H as f32,
            },
            volume: Hit {
                min_x: vol_x as f32,
                min_y: row_top,
                width: VOL_W as f32,
                height: SLIDER_H as f32,
            },
        })
    }
}
