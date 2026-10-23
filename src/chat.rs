//! Gemma 4 chat formatting (mirrors the canonical chat template for text-only turns).

pub const EOT: &str = "<turn|>";

/// Text for the start of a conversation: `<bos>` plus the optional system turn.
pub fn conversation_start(system: Option<&str>, think: bool) -> String {
    let mut s = String::from("<bos>");
    if think || system.is_some() {
        s.push_str("<|turn>system\n");
        if think {
            s.push_str("<|think|>\n");
        }
        if let Some(sys) = system {
            s.push_str(sys.trim());
        }
        s.push_str("<turn|>\n");
    }
    s
}

/// A user turn followed by the generation prompt for the model.
pub fn user_turn(content: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", content.trim())
}

/// Text to append after the model has finished its reply (before the next user turn).
pub fn end_of_model_turn() -> String {
    format!("{EOT}\n")
}

/// Chat formatting for a model family.
pub trait ChatFormat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String;
    /// A user turn followed by the generation prompt.
    fn user_turn(&self, content: &str, think: bool) -> String;
    fn end_of_model_turn(&self) -> String;
    /// Token ids that open / close a thinking block in generated text, and whether
    /// generation starts inside one when thinking is enabled.
    fn think_ids(&self) -> (u32, u32);
    fn starts_in_thinking(&self, think: bool) -> bool;
}

pub struct GemmaChat;
impl ChatFormat for GemmaChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        conversation_start(system, think)
    }
    fn user_turn(&self, content: &str, _think: bool) -> String {
        user_turn(content)
    }
    fn end_of_model_turn(&self) -> String {
        end_of_model_turn()
    }
    fn think_ids(&self) -> (u32, u32) {
        (100, 101)
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
}

/// Qwen3.5 / 3.8 ChatML template: thinking is on by default in the model, so a non-thinking
/// request closes the think block in the generation prompt.
pub struct QwenChat;
const QWEN_REASONING_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
impl ChatFormat for QwenChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        let mut s = String::new();
        match (think, system) {
            (true, Some(sys)) => s.push_str(&format!("<|im_start|>system\n{QWEN_REASONING_XHIGH}\n\n{}<|im_end|>\n", sys.trim())),
            (true, None) => s.push_str(&format!("<|im_start|>system\n{QWEN_REASONING_XHIGH}<|im_end|>\n")),
            (false, Some(sys)) => s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.trim())),
            (false, None) => {}
        }
        s
    }
    fn user_turn(&self, content: &str, think: bool) -> String {
        let tail = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{tail}", content.trim())
    }
    fn end_of_model_turn(&self) -> String {
        "<|im_end|>\n".to_string()
    }
    fn think_ids(&self) -> (u32, u32) {
        (248068, 248069)
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
}
