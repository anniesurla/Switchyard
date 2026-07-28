// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for handoff notes on a running [`StageRouter`].
//!
//! The unit tests cover note selection and the splice in isolation. These drive
//! whole turns through the public API to pin the part only the assembled router
//! can show: which tier served the turn, and what the *model* actually received
//! — the note has to survive the classifier that wrote it, the rest of the
//! cascade, and the routed call.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;

use switchyard_libsy::algorithms::StageRouter;
use switchyard_libsy::stage_router::{
    HandoffNoteConfig, LlmFallback, PickerMode, StageRouterConfig, TierPrompts,
};
use switchyard_libsy::{
    Algorithm, Context, Decision, LlmResponse, LlmTarget, LlmTargetSet, Metadata, Request,
    Response, Result, RoutedLlmClient, SharedState,
};
use switchyard_protocol::{
    text_response, ContentBlock, LlmRequest, Message, Role, ToolCall, ToolResult, WireFormat,
};

const ESCALATION: &str = "the previous model was stalling; pick up the diagnosis";
const STRONG_PROMPT: &str = "diagnose before you edit";
const WEAK_PROMPT: &str = "follow the settled plan";
/// Semantic name of the judge target. It is called through its own target and is
/// never a routing destination.
const JUDGE: &str = "judge";

/// One model call as the client saw it.
#[derive(Clone, Debug)]
struct Call {
    /// Target the call was routed to.
    target: String,
    /// Text of each message, so a test can assert on the note.
    messages: Vec<String>,
    /// Text of each system instruction, so a test can assert on the tier prompt.
    instructions: Vec<String>,
}

/// A client that records what each target was handed, so a test can assert on
/// what reached the model rather than on router internals.
///
/// It also plays the judge: a call routed to the judge target answers with a
/// verdict, which is how the fallback classifier gets an answer without a real
/// model.
#[derive(Default)]
struct RecordingClient {
    calls: Mutex<Vec<Call>>,
    /// `p_solve` the judge reports. High keeps the turn on the weak tier.
    judge_p_solve: Mutex<f64>,
}

impl RecordingClient {
    /// The calls that routed to a tier, dropping the judge's own.
    fn routed(&self) -> Vec<Call> {
        self.calls
            .lock()
            .iter()
            .filter(|call| call.target != JUDGE)
            .cloned()
            .collect()
    }
}

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(
        &self,
        _ctx: Context,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
        let target = decision.selected_model().to_string();
        self.calls.lock().push(Call {
            target: target.clone(),
            messages: request
                .llm_request
                .messages
                .iter()
                .filter_map(|message| message.text_content("|"))
                .collect(),
            instructions: request
                .llm_request
                .instructions
                .iter()
                .filter_map(|block| {
                    block.content.iter().find_map(|content| match content {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                })
                .collect(),
        });
        let completion = if target == JUDGE {
            let p_solve = *self.judge_p_solve.lock();
            format!(
                r#"{{"recommended_route":"efficient","p_solve":{p_solve},"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}}"#
            )
        } else {
            target
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        })
    }
}

fn target(client: &Arc<RecordingClient>, name: &str) -> LlmTarget {
    LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone() as Arc<dyn RoutedLlmClient>),
    }
}

fn router(client: Arc<RecordingClient>, config: StageRouterConfig) -> Result<Arc<StageRouter>> {
    // Only the two tiers are routing destinations; the judge has its own target.
    let targets = LlmTargetSet::new(vec![target(&client, "strong"), target(&client, "weak")]);
    Ok(Arc::new(StageRouter::new(targets, config)?))
}

/// The signal-only configuration these tests start from.
fn config() -> StageRouterConfig {
    StageRouterConfig::new(PickerMode::EfficientFirst, 0.5)
}

/// That configuration plus handoff notes.
fn config_with_notes() -> StageRouterConfig {
    let mut config = config();
    config.handoff_notes = Some(HandoffNoteConfig::new(ESCALATION, None, true));
    config
}

/// One turn of a coding-agent conversation, as the wire delivers it: an
/// assistant tool call answered by a tool result the signal extractor reads.
///
/// `failed` makes the tool result a critical error, the hard override that
/// escalates a turn on the signal alone.
fn turn_request(failed: bool) -> Request {
    let tool_call = json!({
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"},
        }],
    });
    let content = if failed {
        "fatal runtime error: out of memory"
    } else {
        "ok"
    };
    let raw_request = json!({
        "model": "auto",
        "messages": [
            {"role": "user", "content": "fix the build"},
            tool_call,
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ],
    });
    // The neutral IR the router forwards, alongside the raw body the signal
    // extractor parses. An OpenAI tool result is its own `tool` turn, so a note
    // is appended as a fresh user message rather than folded into it.
    let messages = vec![
        Message::text(Role::User, "fix the build"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "Bash".to_string(),
                arguments: json!({"command": "cargo test"}),
            })],
        },
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: content.to_string(),
                }],
                is_error: Some(failed),
            })],
        },
    ];
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages,
            ..LlmRequest::default()
        },
        raw_request: Some(raw_request),
        metadata: Some(Metadata {
            wire_format: Some(WireFormat::OpenAiChat),
            ..Default::default()
        }),
    }
}

#[tokio::test]
async fn a_tier_change_hands_the_note_to_the_model_taking_over() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config_with_notes())?;
    // One session: both turns share the context, so the second sees the tier the
    // first routed to.
    let ctx = Context::<SharedState>::default();

    // Turn 1 — a clean tool result keeps the session on the weak tier.
    router.clone().run(ctx.clone(), turn_request(false)).await?;
    // Turn 2 — a failing tool result escalates, which is a real handover.
    router.run(ctx, turn_request(true)).await?;

    let calls = client.routed();
    assert_eq!(calls[0].target, "weak");
    assert_eq!(calls[1].target, "strong");
    assert!(
        !calls[0]
            .messages
            .iter()
            .any(|text| text.contains(ESCALATION)),
        "the steady-state turn should carry no note: {:?}",
        calls[0].messages
    );
    // The note rides on the trailing turn, after the tool result.
    assert!(
        calls[1]
            .messages
            .last()
            .is_some_and(|text| text.ends_with(ESCALATION)),
        "the escalating turn should carry the note last: {:?}",
        calls[1].messages
    );
    Ok(())
}

#[tokio::test]
async fn the_note_does_not_persist_into_the_next_turn() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config_with_notes())?;
    let ctx = Context::<SharedState>::default();

    router.clone().run(ctx.clone(), turn_request(false)).await?;
    router.clone().run(ctx.clone(), turn_request(true)).await?;
    // A third turn on the same escalated tier: no change, and the note from the
    // previous turn was ephemeral, so nothing carries over.
    router.run(ctx, turn_request(true)).await?;

    let calls = client.routed();
    assert_eq!(calls[2].target, "strong");
    assert!(
        !calls[2]
            .messages
            .iter()
            .any(|text| text.contains(ESCALATION)),
        "the note should not accumulate: {:?}",
        calls[2].messages
    );
    Ok(())
}

#[tokio::test]
async fn an_unconfigured_router_injects_nothing() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config())?;
    let ctx = Context::<SharedState>::default();

    router.clone().run(ctx.clone(), turn_request(false)).await?;
    router.run(ctx, turn_request(true)).await?;

    let calls = client.routed();
    assert_eq!(calls[1].target, "strong", "the escalation still happens");
    assert!(calls
        .iter()
        .all(|call| !call.messages.iter().any(|text| text.contains(ESCALATION))));
    Ok(())
}

/// That configuration plus the capability judge behind the signals.
fn config_with_judge(client: &Arc<RecordingClient>, p_solve: f64) -> StageRouterConfig {
    *client.judge_p_solve.lock() = p_solve;
    let mut config = config();
    config.llm_fallback = Some(LlmFallback {
        judge_target: target(client, JUDGE),
        threshold: 0.5,
    });
    config
}

#[tokio::test]
async fn the_judge_decides_a_turn_the_signals_leave_undecided() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // A low p_solve means the judge does not trust the weak tier with the task.
    let router = router(client.clone(), config_with_judge(&client, 0.1))?;

    // A clean turn is under threshold, so without a judge it would fall open to
    // the configured weak default. The judge overrides that.
    router
        .run(Context::<SharedState>::default(), turn_request(false))
        .await?;

    let judged = client.calls.lock().iter().any(|call| call.target == JUDGE);
    assert!(judged, "the judge should be consulted on an undecided turn");
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn a_decisive_signal_never_reaches_the_judge() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    // The judge would say weak; the signals must win before it is asked at all.
    let router = router(client.clone(), config_with_judge(&client, 0.9))?;

    router
        .run(Context::<SharedState>::default(), turn_request(true))
        .await?;

    assert!(
        !client.calls.lock().iter().any(|call| call.target == JUDGE),
        "a resolved turn should not pay for a judge call"
    );
    assert_eq!(client.routed()[0].target, "strong");
    Ok(())
}

#[tokio::test]
async fn the_judges_verdict_is_not_pinned_to_the_session() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let router = router(client.clone(), config_with_judge(&client, 0.1))?;
    let ctx = Context::<SharedState>::default();

    // First undecided turn: the judge sends it to the strong tier.
    router.clone().run(ctx.clone(), turn_request(false)).await?;
    // The judge changes its mind; a second undecided turn asks again rather than
    // replaying the first verdict.
    *client.judge_p_solve.lock() = 0.9;
    router.run(ctx, turn_request(false)).await?;

    let routed = client.routed();
    assert_eq!(routed[0].target, "strong");
    assert_eq!(routed[1].target, "weak");
    assert_eq!(
        client
            .calls
            .lock()
            .iter()
            .filter(|call| call.target == JUDGE)
            .count(),
        2,
        "each undecided turn is its own question"
    );
    Ok(())
}

#[tokio::test]
async fn each_tier_is_handed_its_own_system_prompt() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let mut config = config();
    config.tier_prompts = Some(TierPrompts::new(
        Some(STRONG_PROMPT.to_string()),
        Some(WEAK_PROMPT.to_string()),
    ));
    let router = router(client.clone(), config)?;
    let ctx = Context::<SharedState>::default();

    router.clone().run(ctx.clone(), turn_request(false)).await?;
    router.run(ctx, turn_request(true)).await?;

    let routed = client.routed();
    assert_eq!(routed[0].target, "weak");
    assert_eq!(routed[0].instructions, vec![WEAK_PROMPT.to_string()]);
    assert_eq!(routed[1].target, "strong");
    assert_eq!(routed[1].instructions, vec![STRONG_PROMPT.to_string()]);
    Ok(())
}

#[tokio::test]
async fn a_judge_decided_turn_still_gets_its_tier_prompt() -> Result<()> {
    let client = Arc::new(RecordingClient::default());
    let mut config = config_with_judge(&client, 0.1);
    config.tier_prompts = Some(TierPrompts::new(Some(STRONG_PROMPT.to_string()), None));
    let router = router(client.clone(), config)?;

    router
        .run(Context::<SharedState>::default(), turn_request(false))
        .await?;

    // The prompt follows the tier the cascade settled on, not the classifier that
    // picked it.
    let routed = client.routed();
    assert_eq!(routed[0].target, "strong");
    assert_eq!(routed[0].instructions, vec![STRONG_PROMPT.to_string()]);
    Ok(())
}
