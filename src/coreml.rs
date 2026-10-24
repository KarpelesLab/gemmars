//! Apple Core ML, reached through the Objective-C runtime with `dlsym` (like `metal.rs`): no
//! framework is linked and nothing is needed at build time. This is the small subset the ANEMLL
//! runner needs (see `anemll.rs`): compiled models (`.mlmodelc`, optionally one function of a
//! multifunction model), their feature descriptions, stateful predictions (`MLState`, macOS 15 /
//! iOS 18) and `MLMultiArray`s (fp16 / fp32 / int32), plain or backed by an IOSurface
//! `CVPixelBuffer`, the layout the Neural Engine reads and writes without copies.
//!
//! Ownership follows Cocoa: [`Obj`] holds one reference and releases it on drop. Every call that
//! may autorelease runs inside an autorelease pool popped before returning, so long generations do
//! not accumulate objects.

use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;
use std::sync::OnceLock;

type Id = *mut c_void;
type Sel = *mut c_void;

/// Where Core ML may run a model (`MLComputeUnits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeUnits {
    CpuOnly = 0,
    CpuAndGpu = 1,
    All = 2,
    CpuAndNeuralEngine = 3,
}

impl ComputeUnits {
    /// `ane` (CPU + Neural Engine), `all`, `gpu` (CPU + GPU) or `cpu`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "ane" | "neural-engine" | "cpu-ane" => Self::CpuAndNeuralEngine,
            "all" => Self::All,
            "gpu" | "cpu-gpu" => Self::CpuAndGpu,
            "cpu" => Self::CpuOnly,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpu",
            Self::CpuAndGpu => "gpu",
            Self::All => "all",
            Self::CpuAndNeuralEngine => "ane",
        }
    }
}

/// Element type of a multi-array (`MLMultiArrayDataType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    F16,
    F32,
    F64,
    I32,
}

impl DataType {
    fn raw(self) -> isize {
        match self {
            Self::F16 => 0x10000 | 16,
            Self::F32 => 0x10000 | 32,
            Self::F64 => 0x10000 | 64,
            Self::I32 => 0x20000 | 32,
        }
    }
    fn from_raw(v: isize) -> Option<Self> {
        Some(match v {
            0x10010 => Self::F16,
            0x10020 => Self::F32,
            0x10040 => Self::F64,
            0x20020 => Self::I32,
            _ => return None,
        })
    }
    pub fn size(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 => 8,
        }
    }
}

// ---- Runtime -------------------------------------------------------------------------------------

struct Rt {
    msg_send: usize,
    sel_register: unsafe extern "C" fn(*const c_char) -> Sel,
    get_class: unsafe extern "C" fn(*const c_char) -> Id,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
    cv_create: unsafe extern "C" fn(*const c_void, usize, usize, u32, Id, *mut Id) -> i32,
    cv_release: unsafe extern "C" fn(Id),
    cv_lock: unsafe extern "C" fn(Id, u64) -> i32,
    cv_unlock: unsafe extern "C" fn(Id, u64) -> i32,
    cv_base: unsafe extern "C" fn(Id) -> *mut c_void,
    cv_row_bytes: unsafe extern "C" fn(Id) -> usize,
}

/// A symbol as the C function pointer type `F` (the caller names its signature).
fn sym<F: Copy>(lib: &dlsym::Library, name: &CStr) -> Result<F, String> {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
    let p = unsafe { lib.get::<unsafe extern "C" fn()>(name) }
        .map(|s| s.as_ptr())
        .map_err(|e| format!("{}: {e}", name.to_string_lossy()))?;
    // SAFETY: F is a function pointer type matching the symbol's documented C signature.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

fn open_rt() -> Result<Rt, String> {
    let open = |p: &str| {
        unsafe { dlsym::Library::open(p) }
            .map(dlsym::Library::leak)
            .map_err(|e| format!("{p}: {e}"))
    };
    let objc = open("/usr/lib/libobjc.A.dylib")?;
    // Loading the frameworks registers their classes with the runtime.
    open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
    open("/System/Library/Frameworks/CoreML.framework/CoreML")?;
    let cv = open("/System/Library/Frameworks/CoreVideo.framework/CoreVideo")?;
    Ok(Rt {
        msg_send: sym::<*mut c_void>(objc, c"objc_msgSend")? as usize,
        sel_register: sym(objc, c"sel_registerName")?,
        get_class: sym(objc, c"objc_getClass")?,
        pool_push: sym(objc, c"objc_autoreleasePoolPush")?,
        pool_pop: sym(objc, c"objc_autoreleasePoolPop")?,
        cv_create: sym(cv, c"CVPixelBufferCreate")?,
        cv_release: sym(cv, c"CVPixelBufferRelease")?,
        cv_lock: sym(cv, c"CVPixelBufferLockBaseAddress")?,
        cv_unlock: sym(cv, c"CVPixelBufferUnlockBaseAddress")?,
        cv_base: sym(cv, c"CVPixelBufferGetBaseAddress")?,
        cv_row_bytes: sym(cv, c"CVPixelBufferGetBytesPerRow")?,
    })
}

static RT: OnceLock<Result<Rt, String>> = OnceLock::new();

fn rt() -> Result<&'static Rt> {
    RT.get_or_init(open_rt).as_ref().map_err(|e| anyhow!("Core ML unavailable: {e}"))
}

/// The runtime once [`available`] (or any loader) succeeded.
fn rtu() -> &'static Rt {
    rt().expect("Core ML runtime")
}

/// Whether Core ML can be used (the frameworks load); the error says why not.
pub fn available() -> Result<()> {
    rt().map(|_| ())
}

/// `[obj sel args...]` with a typed `objc_msgSend`; the selector is registered once per call site.
macro_rules! msg {
    ($obj:expr, $sel:literal $(, $a:expr => $t:ty)* ; $ret:ty) => {{
        static SEL: OnceLock<usize> = OnceLock::new();
        let r = rtu();
        let sel = *SEL.get_or_init(|| unsafe { (r.sel_register)(concat!($sel, "\0").as_ptr().cast()) } as usize) as Sel;
        // SAFETY: arm64 objc_msgSend takes the method's own C signature; callers pick it.
        let f: unsafe extern "C" fn(Id, Sel $(, $t)*) -> $ret = unsafe { std::mem::transmute(r.msg_send) };
        // Arguments may themselves be `msg!` calls (nested unsafe blocks).
        #[allow(unused_unsafe)]
        let v = unsafe { f($obj, sel $(, $a)*) };
        v
    }};
}

/// A class object, looked up once per call site.
macro_rules! class {
    ($name:literal) => {{
        static CLS: OnceLock<usize> = OnceLock::new();
        #[allow(unused_unsafe)]
        let c = *CLS.get_or_init(|| unsafe { (rtu().get_class)(concat!($name, "\0").as_ptr().cast()) } as usize) as Id;
        c
    }};
}

/// Runs `f` inside an autorelease pool.
pub fn autoreleasepool<R>(f: impl FnOnce() -> R) -> R {
    struct Pool(*mut c_void);
    impl Drop for Pool {
        fn drop(&mut self) {
            unsafe { (rtu().pool_pop)(self.0) }
        }
    }
    let _p = Pool(unsafe { (rtu().pool_push)() });
    f()
}

/// One owned reference to an Objective-C object.
pub struct Obj(NonNull<c_void>);

impl Obj {
    /// Takes ownership of a +1 reference (`alloc`/`init`, `new...`, `copy`).
    fn owned(id: Id) -> Option<Obj> {
        NonNull::new(id).map(Obj)
    }
    /// Retains an autoreleased (+0) object.
    fn retained(id: Id) -> Option<Obj> {
        let id = NonNull::new(id)?;
        msg!(id.as_ptr(), "retain"; Id);
        Some(Obj(id))
    }
    fn id(&self) -> Id {
        self.0.as_ptr()
    }
}

impl Clone for Obj {
    fn clone(&self) -> Self {
        Obj::retained(self.id()).expect("non-null")
    }
}

impl Drop for Obj {
    fn drop(&mut self) {
        msg!(self.id(), "release"; ());
    }
}

// Core ML objects may be retained, released and used from any thread (one at a time here).
unsafe impl Send for Obj {}

fn nsstring(s: &str) -> Obj {
    let a = msg!(class!("NSString"), "alloc"; Id);
    let id = msg!(a, "initWithBytes:length:encoding:", s.as_ptr() => *const u8, s.len() => usize, 4usize => usize; Id);
    Obj::owned(id).expect("NSString")
}

fn to_string(s: Id) -> String {
    if s.is_null() {
        return String::new();
    }
    let p = msg!(s, "UTF8String"; *const c_char);
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// An `NSError` out-parameter turned into an error ("what: description").
fn ns_error(err: Id, what: &str) -> anyhow::Error {
    if err.is_null() {
        return anyhow!("{what}: failed (no error details)");
    }
    let d = to_string(msg!(err, "localizedDescription"; Id));
    anyhow!("{what}: {d}")
}

/// Autoreleased `NSArray<NSNumber>` of `v`.
fn ns_numbers(v: &[usize]) -> Id {
    let nums: Vec<Id> = v
        .iter()
        .map(|&x| msg!(class!("NSNumber"), "numberWithInteger:", x as isize => isize; Id))
        .collect();
    msg!(class!("NSArray"), "arrayWithObjects:count:", nums.as_ptr() => *const Id, nums.len() => usize; Id)
}

/// The `NSNumber`s of an `NSArray` as integers.
fn ns_number_list(arr: Id) -> Vec<usize> {
    if arr.is_null() {
        return Vec::new();
    }
    let n = msg!(arr, "count"; usize);
    (0..n)
        .map(|i| {
            let o = msg!(arr, "objectAtIndex:", i => usize; Id);
            msg!(o, "integerValue"; isize).max(0) as usize
        })
        .collect()
}

/// Autoreleased `NSDictionary` from (key, object) pairs.
fn ns_dict(pairs: &[(Id, Id)]) -> Id {
    let keys: Vec<Id> = pairs.iter().map(|p| p.0).collect();
    let vals: Vec<Id> = pairs.iter().map(|p| p.1).collect();
    msg!(class!("NSDictionary"), "dictionaryWithObjects:forKeys:count:",
        vals.as_ptr() => *const Id, keys.as_ptr() => *const Id, keys.len() => usize; Id)
}

// ---- Feature descriptions ----------------------------------------------------------------------

/// What a model input, output or state is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    MultiArray,
    State,
    /// Another `MLFeatureType` (images, strings, ...), unused here.
    Other(isize),
}

/// One input / output / state of a model function.
#[derive(Clone, Debug)]
pub struct FeatureDesc {
    pub name: String,
    pub kind: FeatureKind,
    /// The declared shape (the default one for flexible shapes); empty when unknown.
    pub shape: Vec<usize>,
    pub dtype: Option<DataType>,
}

fn describe(dict: Id) -> Vec<FeatureDesc> {
    if dict.is_null() {
        return Vec::new();
    }
    let keys = msg!(dict, "allKeys"; Id);
    let n = msg!(keys, "count"; usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let k = msg!(keys, "objectAtIndex:", i => usize; Id);
        let d = msg!(dict, "objectForKey:", k => Id; Id);
        let ty = msg!(d, "type"; isize);
        let (kind, shape, dtype) = match ty {
            5 => {
                let c = msg!(d, "multiArrayConstraint"; Id);
                let shape = ns_number_list(msg!(c, "shape"; Id));
                (FeatureKind::MultiArray, shape, DataType::from_raw(msg!(c, "dataType"; isize)))
            }
            8 => {
                let c = msg!(d, "stateConstraint"; Id);
                if c.is_null() {
                    (FeatureKind::State, Vec::new(), None)
                } else {
                    let shape = ns_number_list(msg!(c, "bufferShape"; Id));
                    (FeatureKind::State, shape, DataType::from_raw(msg!(c, "dataType"; isize)))
                }
            }
            t => (FeatureKind::Other(t), Vec::new(), None),
        };
        out.push(FeatureDesc {
            name: to_string(k),
            kind,
            shape,
            dtype,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

// ---- Models ------------------------------------------------------------------------------------

/// A loaded compiled model (one function of it, for multifunction models).
pub struct Model {
    obj: Obj,
    pub function: Option<String>,
    pub inputs: Vec<FeatureDesc>,
    pub outputs: Vec<FeatureDesc>,
    pub states: Vec<FeatureDesc>,
}

impl Model {
    /// Loads a compiled `.mlmodelc` directory. `function` selects one function of a
    /// multifunction model (macOS 15 / iOS 18).
    pub fn load(path: &std::path::Path, units: ComputeUnits, function: Option<&str>) -> Result<Model> {
        rt()?;
        let path_s = path.to_str().ok_or_else(|| anyhow!("{}: path is not UTF-8", path.display()))?;
        if !path.is_dir() {
            bail!("{}: no such compiled model (.mlmodelc directory)", path.display());
        }
        autoreleasepool(|| {
            let p = nsstring(path_s);
            let url = msg!(class!("NSURL"), "fileURLWithPath:", p.id() => Id; Id);
            let cfg = Obj::owned(msg!(msg!(class!("MLModelConfiguration"), "alloc"; Id), "init"; Id))
                .ok_or_else(|| anyhow!("MLModelConfiguration"))?;
            msg!(cfg.id(), "setComputeUnits:", units as isize => isize; ());
            if let Some(f) = function {
                let sel = unsafe { (rtu().sel_register)(c"setFunctionName:".as_ptr()) };
                if msg!(cfg.id(), "respondsToSelector:", sel => Sel; i8) == 0 {
                    bail!("this OS has no multifunction Core ML models (needs macOS 15 / iOS 18)");
                }
                let name = nsstring(f);
                msg!(cfg.id(), "setFunctionName:", name.id() => Id; ());
            }
            let mut err: Id = std::ptr::null_mut();
            let m = msg!(class!("MLModel"), "modelWithContentsOfURL:configuration:error:",
                url => Id, cfg.id() => Id, &mut err as *mut Id => *mut Id; Id);
            let what = match function {
                Some(f) => format!("load {} (function {f})", path.display()),
                None => format!("load {}", path.display()),
            };
            let obj = Obj::retained(m).ok_or_else(|| ns_error(err, &what))?;
            let desc = msg!(obj.id(), "modelDescription"; Id);
            let inputs = describe(msg!(desc, "inputDescriptionsByName"; Id));
            let outputs = describe(msg!(desc, "outputDescriptionsByName"; Id));
            let sel = unsafe { (rtu().sel_register)(c"stateDescriptionsByName".as_ptr()) };
            let states = if msg!(desc, "respondsToSelector:", sel => Sel; i8) != 0 {
                describe(msg!(desc, "stateDescriptionsByName"; Id))
            } else {
                Vec::new()
            };
            Ok(Model {
                obj,
                function: function.map(str::to_string),
                inputs,
                outputs,
                states,
            })
        })
    }

    pub fn input(&self, name: &str) -> Option<&FeatureDesc> {
        self.inputs.iter().find(|d| d.name == name)
    }
    pub fn output(&self, name: &str) -> Option<&FeatureDesc> {
        self.outputs.iter().find(|d| d.name == name)
    }

    /// A fresh state (zeroed KV caches) for this model's stateful functions.
    pub fn new_state(&self) -> Result<State> {
        autoreleasepool(|| {
            let sel = unsafe { (rtu().sel_register)(c"newState".as_ptr()) };
            if msg!(self.obj.id(), "respondsToSelector:", sel => Sel; i8) == 0 {
                bail!("this OS has no stateful Core ML models (needs macOS 15 / iOS 18)");
            }
            Obj::owned(msg!(self.obj.id(), "newState"; Id))
                .map(State)
                .ok_or_else(|| anyhow!("MLModel newState returned nil"))
        })
    }

    /// Binds input arrays and output backings (preallocated arrays Core ML writes the named
    /// outputs into; they must match the outputs' shapes and types) for repeated predictions:
    /// write new values into the same arrays and call [`run`](Self::run).
    pub fn prepare(&self, inputs: &[(&str, &MultiArray)], backings: &[(&str, &MultiArray)]) -> Result<Prepared> {
        autoreleasepool(|| {
            let keys: Vec<Obj> = inputs.iter().map(|(n, _)| nsstring(n)).collect();
            let pairs: Vec<(Id, Id)> = keys.iter().zip(inputs).map(|(k, (_, a))| (k.id(), a.obj.id())).collect();
            let dict = ns_dict(&pairs);
            let mut err: Id = std::ptr::null_mut();
            let fp = msg!(msg!(class!("MLDictionaryFeatureProvider"), "alloc"; Id), "initWithDictionary:error:",
                dict => Id, &mut err as *mut Id => *mut Id; Id);
            let provider = Obj::owned(fp).ok_or_else(|| ns_error(err, "feature provider"))?;
            let options = Obj::owned(msg!(msg!(class!("MLPredictionOptions"), "alloc"; Id), "init"; Id))
                .ok_or_else(|| anyhow!("MLPredictionOptions"))?;
            if !backings.is_empty() {
                let bkeys: Vec<Obj> = backings.iter().map(|(n, _)| nsstring(n)).collect();
                let bpairs: Vec<(Id, Id)> = bkeys.iter().zip(backings).map(|(k, (_, a))| (k.id(), a.obj.id())).collect();
                msg!(options.id(), "setOutputBackings:", ns_dict(&bpairs) => Id; ());
            }
            Ok(Prepared { provider, options })
        })
    }

    /// Runs a prepared prediction; outputs without a backing are returned in fresh arrays.
    pub fn run(&self, prep: &Prepared, state: Option<&State>) -> Result<Prediction> {
        autoreleasepool(|| {
            let mut err: Id = std::ptr::null_mut();
            let (fp, opts) = (prep.provider.id(), prep.options.id());
            let out = match state {
                Some(s) => msg!(self.obj.id(), "predictionFromFeatures:usingState:options:error:",
                    fp => Id, s.0.id() => Id, opts => Id, &mut err as *mut Id => *mut Id; Id),
                None => msg!(self.obj.id(), "predictionFromFeatures:options:error:",
                    fp => Id, opts => Id, &mut err as *mut Id => *mut Id; Id),
            };
            let what = match &self.function {
                Some(f) => format!("prediction ({f})"),
                None => "prediction".to_string(),
            };
            Obj::retained(out).map(Prediction).ok_or_else(|| ns_error(err, &what))
        })
    }

    /// One prediction ([`prepare`](Self::prepare) + [`run`](Self::run)).
    pub fn predict(&self, inputs: &[(&str, &MultiArray)], state: Option<&State>, backings: &[(&str, &MultiArray)]) -> Result<Prediction> {
        self.run(&self.prepare(inputs, backings)?, state)
    }
}

/// Inputs and output backings bound for repeated predictions ([`Model::prepare`]).
pub struct Prepared {
    provider: Obj,
    options: Obj,
}

/// The KV caches (and any other states) of a stateful model, shared by its functions.
pub struct State(Obj);

/// The outputs of one prediction.
pub struct Prediction(Obj);

impl Prediction {
    /// The named multi-array output.
    pub fn get(&self, name: &str) -> Result<MultiArray> {
        autoreleasepool(|| {
            let k = nsstring(name);
            let v = msg!(self.0.id(), "featureValueForName:", k.id() => Id; Id);
            if v.is_null() {
                bail!("model has no output {name}");
            }
            let a = msg!(v, "multiArrayValue"; Id);
            let obj = Obj::retained(a).ok_or_else(|| anyhow!("output {name} is not a multi-array"))?;
            Ok(MultiArray::wrap(obj, None))
        })
    }
}

// ---- Multi-arrays ------------------------------------------------------------------------------

/// A `CVPixelBuffer` reference (IOSurface-backed fp16 storage).
struct PixelBuffer(Id);

impl Drop for PixelBuffer {
    fn drop(&mut self) {
        unsafe { (rtu().cv_release)(self.0) }
    }
}

unsafe impl Send for PixelBuffer {}

/// An `MLMultiArray`.
pub struct MultiArray {
    obj: Obj,
    /// Our own pixel buffer when the array was made by [`MultiArray::surface`].
    _pixel: Option<PixelBuffer>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

/// Locked view of a multi-array's memory: element strides per dimension.
struct View {
    base: *mut u8,
    strides: Vec<usize>,
    unlock: Option<Id>,
}

impl Drop for View {
    fn drop(&mut self) {
        if let Some(pb) = self.unlock {
            unsafe { (rtu().cv_unlock)(pb, 0) };
        }
    }
}

const PIXEL_FORMAT_F16: u32 = u32::from_be_bytes(*b"L00h"); // kCVPixelFormatType_OneComponent16Half

impl MultiArray {
    fn wrap(obj: Obj, pixel: Option<PixelBuffer>) -> MultiArray {
        let (shape, dtype) = autoreleasepool(|| {
            (
                ns_number_list(msg!(obj.id(), "shape"; Id)),
                DataType::from_raw(msg!(obj.id(), "dataType"; isize)).unwrap_or(DataType::F32),
            )
        });
        MultiArray {
            obj,
            _pixel: pixel,
            shape,
            dtype,
        }
    }

    /// A zeroed array with Core ML-owned contiguous memory.
    pub fn new(shape: &[usize], dtype: DataType) -> Result<MultiArray> {
        rt()?;
        let a = autoreleasepool(|| {
            let mut err: Id = std::ptr::null_mut();
            let a = msg!(msg!(class!("MLMultiArray"), "alloc"; Id), "initWithShape:dataType:error:",
                ns_numbers(shape) => Id, dtype.raw() => isize, &mut err as *mut Id => *mut Id; Id);
            Obj::owned(a).ok_or_else(|| ns_error(err, &format!("MLMultiArray {shape:?}")))
        })?;
        let mut a = MultiArray::wrap(a, None);
        a.fill_zero();
        Ok(a)
    }

    /// A zeroed fp16 array backed by an IOSurface pixel buffer (width = last dimension, height =
    /// the product of the others), as the Neural Engine reads and writes directly.
    pub fn surface(shape: &[usize]) -> Result<MultiArray> {
        rt()?;
        let w = *shape.last().ok_or_else(|| anyhow!("empty shape"))?;
        let h: usize = shape[..shape.len() - 1].iter().product();
        let r = rtu();
        let (obj, pb) = autoreleasepool(|| {
            let iosurf = ns_dict(&[]);
            let yes = msg!(class!("NSNumber"), "numberWithBool:", 1i8 => i8; Id);
            let (k1, k2) = (nsstring("IOSurfaceProperties"), nsstring("MetalCompatibility"));
            let attrs = ns_dict(&[(k1.id(), iosurf), (k2.id(), yes)]);
            let mut pb: Id = std::ptr::null_mut();
            let st = unsafe { (r.cv_create)(std::ptr::null(), w, h.max(1), PIXEL_FORMAT_F16, attrs, &mut pb) };
            if st != 0 || pb.is_null() {
                bail!("CVPixelBufferCreate {w}x{h}: error {st}");
            }
            let pb = PixelBuffer(pb);
            let a = msg!(msg!(class!("MLMultiArray"), "alloc"; Id), "initWithPixelBuffer:shape:",
                pb.0 => Id, ns_numbers(shape) => Id; Id);
            let a = Obj::owned(a).ok_or_else(|| anyhow!("MLMultiArray from pixel buffer {shape:?}"))?;
            Ok((a, pb))
        })?;
        let mut a = MultiArray::wrap(obj, Some(pb));
        a.fill_zero();
        Ok(a)
    }

    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }

    fn view(&self) -> View {
        autoreleasepool(|| {
            let r = rtu();
            let pb = msg!(self.obj.id(), "pixelBuffer"; Id);
            if !pb.is_null() {
                unsafe { (r.cv_lock)(pb, 0) };
                let base = unsafe { (r.cv_base)(pb) }.cast::<u8>();
                let row = unsafe { (r.cv_row_bytes)(pb) } / self.dtype.size();
                // Rows are the last dimension; every other dimension is packed over rows.
                let n = self.shape.len();
                let mut strides = vec![1usize; n];
                if n >= 2 {
                    strides[n - 2] = row;
                    for i in (0..n - 2).rev() {
                        strides[i] = strides[i + 1] * self.shape[i + 1];
                    }
                }
                return View {
                    base,
                    strides,
                    unlock: Some(pb),
                };
            }
            let base = msg!(self.obj.id(), "dataPointer"; *mut c_void).cast::<u8>();
            let strides = ns_number_list(msg!(self.obj.id(), "strides"; Id));
            View {
                base,
                strides,
                unlock: None,
            }
        })
    }

    /// Calls `f(row_ptr, row_len)` for each last-dimension row, in logical order.
    fn for_rows(&self, mut f: impl FnMut(*mut u8, usize)) {
        let v = self.view();
        let n = self.shape.len();
        if n == 0 || self.count() == 0 {
            return;
        }
        let es = self.dtype.size();
        let last = self.shape[n - 1];
        // Rows must be contiguous in their last dimension (true for everything Core ML makes).
        assert!(v.strides[n - 1] == 1 || last == 1, "non-unit innermost stride");
        let rows: usize = self.shape[..n - 1].iter().product();
        let mut idx = vec![0usize; n.saturating_sub(1)];
        for _ in 0..rows {
            let off: usize = idx.iter().zip(&v.strides).map(|(i, s)| i * s).sum();
            f(unsafe { v.base.add(off * es) }, last);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    pub fn fill_zero(&mut self) {
        let es = self.dtype.size();
        self.for_rows(|p, n| unsafe { std::ptr::write_bytes(p, 0, n * es) });
    }

    /// Writes `src` (logical row-major order, converted to the array's type).
    pub fn write_f32(&mut self, src: &[f32]) {
        assert_eq!(src.len(), self.count(), "write_f32 size");
        let dt = self.dtype;
        let mut i = 0;
        self.for_rows(|p, n| {
            let s = &src[i..i + n];
            unsafe {
                match dt {
                    DataType::F16 => {
                        let d = std::slice::from_raw_parts_mut(p.cast::<u16>(), n);
                        for (d, &v) in d.iter_mut().zip(s) {
                            *d = half::f16::from_f32(v).to_bits();
                        }
                    }
                    DataType::F32 => std::slice::from_raw_parts_mut(p.cast::<f32>(), n).copy_from_slice(s),
                    DataType::F64 => {
                        for (d, &v) in std::slice::from_raw_parts_mut(p.cast::<f64>(), n).iter_mut().zip(s) {
                            *d = v as f64;
                        }
                    }
                    DataType::I32 => {
                        for (d, &v) in std::slice::from_raw_parts_mut(p.cast::<i32>(), n).iter_mut().zip(s) {
                            *d = v as i32;
                        }
                    }
                }
            }
            i += n;
        });
    }

    /// Writes raw fp16 bits (the array must be fp16).
    pub fn write_f16_bits(&mut self, src: &[u16]) {
        assert_eq!(self.dtype, DataType::F16);
        assert_eq!(src.len(), self.count(), "write_f16_bits size");
        let mut i = 0;
        self.for_rows(|p, n| {
            unsafe { std::slice::from_raw_parts_mut(p.cast::<u16>(), n) }.copy_from_slice(&src[i..i + n]);
            i += n;
        });
    }

    /// Writes integers (the array must be int32).
    pub fn write_i32(&mut self, src: &[i32]) {
        assert_eq!(self.dtype, DataType::I32);
        assert_eq!(src.len(), self.count(), "write_i32 size");
        let mut i = 0;
        self.for_rows(|p, n| {
            unsafe { std::slice::from_raw_parts_mut(p.cast::<i32>(), n) }.copy_from_slice(&src[i..i + n]);
            i += n;
        });
    }

    /// Appends the contents (logical row-major order) to `out` as f32.
    pub fn read_f32_into(&self, out: &mut Vec<f32>) {
        let dt = self.dtype;
        out.reserve(self.count());
        self.for_rows(|p, n| unsafe {
            match dt {
                DataType::F16 => out.extend(
                    std::slice::from_raw_parts(p.cast::<u16>(), n)
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32()),
                ),
                DataType::F32 => out.extend_from_slice(std::slice::from_raw_parts(p.cast::<f32>(), n)),
                DataType::F64 => out.extend(std::slice::from_raw_parts(p.cast::<f64>(), n).iter().map(|&v| v as f32)),
                DataType::I32 => out.extend(std::slice::from_raw_parts(p.cast::<i32>(), n).iter().map(|&v| v as f32)),
            }
        });
    }

    pub fn read_f32(&self) -> Vec<f32> {
        let mut v = Vec::new();
        self.read_f32_into(&mut v);
        v
    }

    /// The contents as integers (int32 arrays exactly; float arrays rounded toward zero).
    pub fn read_i32(&self) -> Vec<i32> {
        if self.dtype != DataType::I32 {
            return self.read_f32().into_iter().map(|v| v as i32).collect();
        }
        let mut out = Vec::with_capacity(self.count());
        self.for_rows(|p, n| out.extend_from_slice(unsafe { std::slice::from_raw_parts(p.cast::<i32>(), n) }));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiarray_roundtrip() {
        if available().is_err() {
            return;
        }
        let mut a = MultiArray::new(&[2, 3], DataType::F16).unwrap();
        a.write_f32(&[1.0, -2.0, 0.5, 3.0, 4.0, f32::NEG_INFINITY]);
        assert_eq!(a.read_f32(), vec![1.0, -2.0, 0.5, 3.0, 4.0, f32::NEG_INFINITY]);
        let mut s = MultiArray::surface(&[1, 3, 5]).unwrap();
        let v: Vec<f32> = (0..15).map(|i| i as f32).collect();
        s.write_f32(&v);
        assert_eq!(s.read_f32(), v);
        let mut i = MultiArray::new(&[4], DataType::I32).unwrap();
        i.write_i32(&[1, 2, 3, -4]);
        assert_eq!(i.read_i32(), vec![1, 2, 3, -4]);
    }
}
