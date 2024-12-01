#[cfg(target_family = "unix")]
mod ffi_tests {
    extern crate tiktoken_rs;
    use std::ffi::{CString, CStr};
    use std::os::raw::{c_char, c_int};

    use tiktoken_rs::{
        ffi_get_completion_max_tokens, ffi_get_chat_completion_max_tokens,
        ffi_get_bpe_from_model, ffi_get_tokenizer, tokenize, free_tokens,
        ffi_encode_with_special_tokens, free_core_bpe,
    };


    #[test]
    fn test_ffi_get_completion_max_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let prompt = CString::new("Hello, world!").unwrap();

        let tokens = unsafe { ffi_get_completion_max_tokens(model.as_ptr(), prompt.as_ptr()) };
        assert!(tokens > 0, "Should return a positive token count");
    }

    #[test]
    fn test_ffi_get_chat_completion_max_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let messages = CString::new(r#"[{"role":"user","content":"Hi"}]"#).unwrap();

        let tokens = unsafe { ffi_get_chat_completion_max_tokens(model.as_ptr(), messages.as_ptr()) };
        assert!(tokens > 0, "Expected a positive token count for valid messages");
    }

    #[test]
    fn test_ffi_get_bpe_from_model() {
        let model = CString::new("gpt-4").unwrap();

        let bpe_ptr = unsafe { ffi_get_bpe_from_model(model.as_ptr()) };
        assert!(!bpe_ptr.is_null(), "Expected a valid CoreBPE pointer");

        unsafe { free_core_bpe(bpe_ptr) }; // Ensure memory is cleaned up
    }

    #[test]
    fn test_ffi_tokenize() {
        let input = CString::new("This is a test").unwrap();
        let mut token_count: c_int = 0;

        let tokens_ptr = unsafe { tokenize(input.as_ptr(), &mut token_count) };
        assert!(!tokens_ptr.is_null(), "Tokenize should return a valid pointer");
        assert!(token_count > 0, "Token count should be positive");

        unsafe {
            free_tokens(tokens_ptr);
        }
    }

    #[test]
    fn test_ffi_get_tokenizer() {
        let model = CString::new("gpt-4").unwrap();

        let tokenizer = unsafe { ffi_get_tokenizer(model.as_ptr()) };
        assert!(tokenizer >= 0, "Tokenizer should be valid for the model");
    }

    #[test]
    fn test_ffi_encode_with_special_tokens() {
        let model = CString::new("gpt-4").unwrap();
        let input = CString::new("Special token test").unwrap();

        let bpe_ptr = unsafe { ffi_get_bpe_from_model(model.as_ptr()) };
        assert!(!bpe_ptr.is_null(), "Expected a valid CoreBPE pointer");

        let tokens_ptr = unsafe { ffi_encode_with_special_tokens(bpe_ptr, input.as_ptr()) };
        assert!(!tokens_ptr.is_null(), "Encoding should return a valid pointer");

        unsafe { free_tokens(tokens_ptr) }; // Clean up tokens
        unsafe { free_core_bpe(bpe_ptr) };  // Clean up CoreBPE
    }
}
