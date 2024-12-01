#[cfg(target_family = "unix")] // Only compile this module on Unix-based systems
mod ffi {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int};

    use crate::api::{
        get_bpe_from_model, get_chat_completion_max_tokens, get_completion_max_tokens,
        num_tokens_from_messages,
    };
    use crate::patched_tiktoken::CoreBPE;
    use crate::tokenizer::{get_tokenizer, Tokenizer};

    /// Expose get_completion_max_tokens
    #[no_mangle]
    pub extern "C" fn ffi_get_completion_max_tokens(
        model: *const c_char,
        prompt: *const c_char,
    ) -> c_int {
        let model = unsafe { CStr::from_ptr(model).to_str().unwrap_or_default() };
        let prompt = unsafe { CStr::from_ptr(prompt).to_str().unwrap_or_default() };

        match get_completion_max_tokens(model, prompt) {
            Ok(tokens) => tokens as c_int,
            Err(_) => 0,
        }
    }

    /// Expose get_chat_completion_max_tokens
    #[no_mangle]
    pub extern "C" fn ffi_get_chat_completion_max_tokens(
        model: *const c_char,
        messages: *const c_char,
    ) -> c_int {
        let model = unsafe { CStr::from_ptr(model).to_str().unwrap_or_default() };
        let messages_json = unsafe { CStr::from_ptr(messages).to_str().unwrap_or_default() };

        let messages: Vec<_> = serde_json::from_str(messages_json).unwrap_or_default();

        match get_chat_completion_max_tokens(model, &messages) {
            Ok(tokens) => tokens as c_int,
            Err(_) => 0,
        }
    }

    /// Expose get_bpe_from_model
    #[no_mangle]
    pub extern "C" fn ffi_get_bpe_from_model(model: *const c_char) -> *mut CoreBPE {
        let model = unsafe { CStr::from_ptr(model).to_str().unwrap_or_default() };

        match get_bpe_from_model(model) {
            Ok(bpe) => Box::into_raw(Box::new(bpe)),
            Err(_) => std::ptr::null_mut(),
        }
    }

    /// Expose get_tokenizer
    #[no_mangle]
    pub extern "C" fn ffi_get_tokenizer(model_name: *const c_char) -> c_int {
        let model_name = unsafe { CStr::from_ptr(model_name).to_str().unwrap_or_default() };

        match get_tokenizer(model_name) {
            Some(tokenizer) => tokenizer as c_int,
            None => -1,
        }
    }

    /// Expose tokenize
    #[no_mangle]
    pub extern "C" fn tokenize(input: *const c_char, token_count: *mut c_int) -> *mut c_int {
        let input = unsafe { CStr::from_ptr(input) }
            .to_str()
            .unwrap_or_else(|_| {
                *token_count = 0;
                return "";
            });

        let bpe = match get_bpe_from_tokenizer(Tokenizer::Cl100kBase) {
            Ok(bpe) => bpe,
            Err(_) => {
                *token_count = 0;
                return std::ptr::null_mut();
            }
        };

        let tokens = bpe.encode_with_special_tokens(input);

        unsafe {
            *token_count = tokens.len() as c_int;
        }

        let mut token_array = tokens.into_iter().map(|t| t as c_int).collect::<Vec<_>>();
        let ptr = token_array.as_mut_ptr();
        std::mem::forget(token_array); // Prevent Rust from deallocating

        ptr
    }

    /// Expose free_tokens
    #[no_mangle]
    pub extern "C" fn free_tokens(ptr: *mut c_int) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 0); // Reclaim memory
        }
    }

    /// Expose encode_with_special_tokens
    #[no_mangle]
    pub extern "C" fn ffi_encode_with_special_tokens(bpe_ptr: *mut CoreBPE, input: *const c_char) -> *mut c_int {
        if bpe_ptr.is_null() {
            return std::ptr::null_mut();
        }
        let bpe = unsafe { &*bpe_ptr };
        let input = unsafe { CStr::from_ptr(input).to_str().unwrap_or_default() };

        let tokens = bpe.encode_with_special_tokens(input);
        let mut token_array = tokens.into_iter().map(|t| t as c_int).collect::<Vec<_>>();
        let ptr = token_array.as_mut_ptr();
        std::mem::forget(token_array);

        ptr
    }

    /// Free CoreBPE
    #[no_mangle]
    pub extern "C" fn free_core_bpe(ptr: *mut CoreBPE) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            let _ = Box::from_raw(ptr); // Reclaim memory
        }
    }
}
