//! Chat formatting for each model family (mirrors their chat templates), including tool calling.

use crate::tools::{ToolCall, ToolResult, ToolSpec};

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
    /// The special token that closes a model turn (generation stops on it).
    fn end_of_turn_token(&self) -> &'static str;
    /// Token ids that open / close a thinking block in generated text, and whether
    /// generation starts inside one when thinking is enabled.
    fn think_ids(&self) -> (u32, u32);
    fn starts_in_thinking(&self, think: bool) -> bool;

    /// Conversation start with tool declarations in the system turn (as the model's template).
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String;
    /// The text (a special token for Gemma) that opens a tool call in generated output.
    fn tool_call_marker(&self) -> &'static str;
    /// Tool calls in a reply decoded with its special tokens.
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall>;
    /// What to feed after a reply that ended on its tool calls (its stop token was sampled but not
    /// fed): the results, up to where the model continues.
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String;
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
    fn end_of_turn_token(&self) -> &'static str {
        EOT
    }
    fn think_ids(&self) -> (u32, u32) {
        (100, 101)
    }
    fn starts_in_thinking(&self, _think: bool) -> bool {
        false
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        if tools.is_empty() {
            return conversation_start(system, think);
        }
        let mut s = String::from("<bos><|turn>system\n");
        if think {
            s.push_str("<|think|>\n");
        }
        if let Some(sys) = system {
            s.push_str(sys.trim());
        }
        for t in tools {
            s.push_str(&crate::tools::gemma::declaration(t));
        }
        s.push_str("<turn|>\n");
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        "<|tool_call>"
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::gemma::parse_calls(raw)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The model stopped on `<|tool_response>`; the responses continue the same model turn.
        let mut s: String = results.iter().map(crate::tools::gemma::response).collect();
        if think {
            s.push_str("<|channel>thought\n");
        }
        s
    }
}

/// Qwen3.5 / 3.8 ChatML template: thinking is on by default in the model, so a non-thinking
/// request closes the think block in the generation prompt. Templates differ in one respect:
/// Qwen3.8's adds a reasoning-effort instruction to the system turn in thinking mode, Qwen3.5's
/// does not (and a small model given that instruction anyway can deliberate about it endlessly).
pub struct QwenChat {
    pub reasoning_effort: bool,
}

/// Qwen formats without and with the reasoning-effort instruction.
pub static QWEN: QwenChat = QwenChat { reasoning_effort: false };
pub static QWEN_EFFORT: QwenChat = QwenChat { reasoning_effort: true };

impl QwenChat {
    /// The format matching the model's own chat template (`chat_template.jinja`, or the template
    /// embedded in `tokenizer_config.json`).
    pub fn for_model(dir: &std::path::Path) -> &'static QwenChat {
        let jinja = crate::fsx::read(&dir.join("chat_template.jinja"))
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        let template = jinja.or_else(|| {
            let cfg: serde_json::Value = serde_json::from_slice(&crate::fsx::read(&dir.join("tokenizer_config.json")).ok()?).ok()?;
            cfg["chat_template"].as_str().map(str::to_string)
        });
        if template.is_some_and(|t| t.contains("reasoning_effort")) {
            &QWEN_EFFORT
        } else {
            &QWEN
        }
    }

    /// The instruction this template puts first in the system turn when thinking.
    fn effort(&self, think: bool) -> Option<&'static str> {
        (think && self.reasoning_effort).then_some(QWEN_REASONING_XHIGH)
    }
}
const QWEN_REASONING_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
impl ChatFormat for QwenChat {
    fn conversation_start(&self, system: Option<&str>, think: bool) -> String {
        let mut s = String::new();
        match (self.effort(think), system) {
            (Some(e), Some(sys)) => s.push_str(&format!("<|im_start|>system\n{e}\n\n{}<|im_end|>\n", sys.trim())),
            (Some(e), None) => s.push_str(&format!("<|im_start|>system\n{e}<|im_end|>\n")),
            (None, Some(sys)) => s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys.trim())),
            (None, None) => {}
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
    fn end_of_turn_token(&self) -> &'static str {
        "<|im_end|>"
    }
    fn think_ids(&self) -> (u32, u32) {
        (248068, 248069)
    }
    fn starts_in_thinking(&self, think: bool) -> bool {
        think
    }
    fn conversation_start_tools(&self, system: Option<&str>, think: bool, tools: &[ToolSpec]) -> String {
        if tools.is_empty() {
            return self.conversation_start(system, think);
        }
        let mut s = String::from("<|im_start|>system\n");
        if let Some(e) = self.effort(think) {
            s.push_str(e);
            s.push_str("\n\n");
        }
        s.push_str(&crate::tools::qwen::tools_block(tools));
        if let Some(sys) = system.map(str::trim).filter(|s| !s.is_empty()) {
            s.push_str("\n\n");
            s.push_str(sys);
        }
        s.push_str("<|im_end|>\n");
        s
    }
    fn tool_call_marker(&self) -> &'static str {
        "<tool_call>"
    }
    fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCall> {
        crate::tools::qwen::parse_calls(raw)
    }
    fn tool_results(&self, results: &[ToolResult], think: bool) -> String {
        // The assistant turn ended (<|im_end|> was sampled, not fed); results come as a user turn.
        let mut s = String::from("<|im_end|>\n<|im_start|>user");
        for r in results {
            s.push_str("\n<tool_response>\n");
            s.push_str(&r.content);
            s.push_str("\n</tool_response>");
        }
        s.push_str("<|im_end|>\n<|im_start|>assistant\n");
        s.push_str(if think { "<think>\n" } else { "<think>\n\n</think>\n\n" });
        s
    }
}

/// `eos` plus the chat format's end-of-turn token, looked up in the tokenizer.
pub fn with_turn_end(fmt: &dyn ChatFormat, tok: &tokenizers::Tokenizer, mut eos: Vec<u32>) -> Vec<u32> {
    if let Some(id) = tok.token_to_id(fmt.end_of_turn_token())
        && !eos.contains(&id)
    {
        eos.push(id);
    }
    eos
}
