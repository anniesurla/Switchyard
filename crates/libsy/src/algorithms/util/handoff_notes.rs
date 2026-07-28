// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router handoff notes.
//!
//! When a turn escalates to the strong tier (or hands back to the weak tier),
//! a short **deterministic** note can be handed to the model taking over so it
//! knows *why* — without re-diagnosing (weak→strong) or re-architecting settled
//! work (strong→weak). This is not a model call; the note text comes from config.
//!
//! [`HandoffNoteConfig`] owns the note text and the gate deciding which note (if
//! any) a turn earns; [`inject_note`] splices it into the outbound
//! request. Both are driven by
//! [`StageClassifier`](super::stage_router::StageClassifier) — the one component
//! that knows both the chosen tier and why the signals chose it.
//!
//! Stateless, with no per-session tier tracking: the note is a statement about
//! the turn's own signals, so every turn those signals drive carries one.
//!
//! The note is **ephemeral**: it rides in the single forwarded request and is
//! never written back into the caller's history, so notes cannot accumulate
//! across turns.

use switchyard_protocol::{ContentBlock, Message, Request, Role};

/// The notes a stage router hands to the model taking over, and the gate that
/// decides when the escalation note applies.
pub struct HandoffNoteConfig {
    escalation_note: String,
    deescalation_note: Option<String>,
    only_on_wrong_signal_escalation: bool,
}

impl HandoffNoteConfig {
    /// Configure the notes: the `escalation_note` handed to the strong tier, an
    /// optional `deescalation_note` handed back to the weak tier, and whether
    /// the escalation note fires only on a signal-driven escalation
    /// (`override` / `dimensions`) rather than a `fall_open` default.
    pub fn new(
        escalation_note: impl Into<String>,
        deescalation_note: Option<String>,
        only_on_wrong_signal_escalation: bool,
    ) -> Self {
        Self {
            escalation_note: escalation_note.into(),
            deescalation_note,
            only_on_wrong_signal_escalation,
        }
    }

    /// The note for a turn routed to `tier` with picker `source`, or `None` when
    /// no note applies.
    pub(crate) fn note_for(&self, tier: &str, source: Option<&str>) -> Option<String> {
        match tier {
            // Escalation to the strong tier. When gated, only a signal-driven
            // escalation qualifies — never a `fall_open` default, which would
            // tell the strong model the weak one was stalling when it wasn't.
            "strong" => {
                let signal_driven = matches!(source, Some("override") | Some("dimensions"));
                (!self.only_on_wrong_signal_escalation || signal_driven)
                    .then(|| self.escalation_note.clone())
            }
            // Hand-back to the weak tier, when a de-escalation note is configured.
            "weak" => self.deescalation_note.clone(),
            _ => None,
        }
    }
}

/// Splices `note` into the request the routed model is about to receive.
///
/// The note is appended to the trailing user message when there is one, rather
/// than sent as its own turn: Anthropic rejects two consecutive user messages,
/// and a `tool_result` must stay first within its message. Appending a text
/// block after it satisfies both and keeps the addition a cache-safe suffix. Any
/// other trailing role — an empty conversation, or one ending on an assistant
/// turn — takes a fresh user message.
pub(crate) fn inject_note(request: &mut Request, note: &str) {
    match request.llm_request.messages.last_mut() {
        Some(last) if last.role == Role::User => last.content.push(ContentBlock::Text {
            text: note.to_string(),
        }),
        _ => request
            .llm_request
            .messages
            .push(Message::text(Role::User, note)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{text_request, LlmRequest, ToolResult};

    const ESCALATION: &str = "recovering from an error";
    const DEESCALATION: &str = "settled — carry on";

    fn config(only_on_wrong_signal_escalation: bool) -> HandoffNoteConfig {
        HandoffNoteConfig::new(
            ESCALATION,
            Some(DEESCALATION.to_string()),
            only_on_wrong_signal_escalation,
        )
    }

    #[test]
    fn escalation_note_applies_to_signal_driven_strong() {
        for source in ["override", "dimensions"] {
            assert_eq!(
                config(true).note_for("strong", Some(source)),
                Some(ESCALATION.to_string())
            );
        }
    }

    #[test]
    fn no_escalation_note_on_fall_open_default_when_gated() {
        assert_eq!(config(true).note_for("strong", Some("fall_open")), None);
    }

    #[test]
    fn escalation_note_on_fall_open_when_not_gated() {
        assert_eq!(
            config(false).note_for("strong", Some("fall_open")),
            Some(ESCALATION.to_string())
        );
    }

    #[test]
    fn deescalation_note_applies_to_weak_when_configured() {
        assert_eq!(
            config(true).note_for("weak", Some("tests_passed")),
            Some(DEESCALATION.to_string())
        );
    }

    #[test]
    fn no_deescalation_note_when_unconfigured() {
        let config = HandoffNoteConfig::new(ESCALATION, None, true);
        assert_eq!(config.note_for("weak", Some("tests_passed")), None);
    }

    fn request_with(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn note_appends_to_a_trailing_user_turn_after_its_tool_result() {
        // The shape a coding-agent turn actually arrives in: the tool result
        // leads the trailing user message, so the note has to follow it.
        let tool_result = ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call_1".to_string(),
            content: vec![ContentBlock::Text {
                text: "exit 1".to_string(),
            }],
            is_error: Some(true),
        });
        let mut request = request_with(vec![Message {
            role: Role::User,
            content: vec![tool_result.clone()],
        }]);

        inject_note(&mut request, ESCALATION);

        let messages = &request.llm_request.messages;
        assert_eq!(messages.len(), 1, "no second consecutive user turn");
        assert_eq!(
            messages[0].content,
            vec![
                tool_result,
                ContentBlock::Text {
                    text: ESCALATION.to_string()
                }
            ]
        );
    }

    #[test]
    fn note_becomes_a_new_user_turn_after_an_assistant_turn() {
        let mut request = request_with(vec![Message::text(Role::Assistant, "done")]);

        inject_note(&mut request, ESCALATION);

        let messages = &request.llm_request.messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].text_content(""), Some(ESCALATION.to_string()));
    }

    #[test]
    fn note_leaves_the_rest_of_the_conversation_untouched() {
        let mut request = Request {
            llm_request: text_request(Some("auto".to_string()), "fix the build"),
            raw_request: None,
            metadata: None,
        };

        inject_note(&mut request, ESCALATION);

        let trail: Vec<String> = request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("|"))
            .collect();
        assert_eq!(trail, vec![format!("fix the build|{ESCALATION}")]);
    }
}
