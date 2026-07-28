// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-tier system prompts for stage routing.
//!
//! A strong and a weak model rarely want the same instructions: the capable tier
//! is worth telling to slow down and diagnose, the efficient one to stay on the
//! settled plan. [`TierPromptProcessor`] appends the prompt configured for the
//! tier a turn routed to, so the instruction follows the model rather than the
//! session.
//!
//! Unlike a handoff note, the prompt applies to *every* turn on that tier, not
//! just the turn that switched to it, and it applies however the tier was
//! chosen — signals, the LLM fallback, or falling open.

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, InstructionBlock, Role};

use super::stage_router::LAST_TIER_KEY;
use crate::{Event, Processor, Result, State, StateValue};

/// The system prompt to hand each tier. A tier left unset is routed untouched.
#[derive(Clone, Debug, Default)]
pub struct TierPrompts {
    strong: Option<String>,
    weak: Option<String>,
}

impl TierPrompts {
    /// Configure the prompt for either tier, or both.
    pub fn new(strong: Option<String>, weak: Option<String>) -> Self {
        Self { strong, weak }
    }

    fn prompt_for(&self, tier: &str) -> Option<&str> {
        match tier {
            "strong" => self.strong.as_deref(),
            "weak" => self.weak.as_deref(),
            _ => None,
        }
    }
}

/// Appends the routed tier's system prompt to the outbound request.
pub struct TierPromptProcessor {
    prompts: TierPrompts,
}

impl TierPromptProcessor {
    /// Hand each tier the prompt configured for it.
    pub fn new(prompts: TierPrompts) -> Self {
        Self { prompts }
    }
}

#[async_trait]
impl Processor for TierPromptProcessor {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        let Event::ModelRequest(request) = event else {
            return Ok(());
        };
        // The decision replay stamps the routed tier before the outbound request
        // is offered, so this reads the tier of the turn being sent — whichever
        // classifier in the cascade picked it.
        let Some(StateValue::String(tier)) = state.extra.get(LAST_TIER_KEY) else {
            return Ok(());
        };
        let Some(prompt) = self.prompts.prompt_for(tier) else {
            return Ok(());
        };
        // Appended after the client's own instructions: the caller's prompt still
        // leads, and the addition stays a suffix of the cached prefix.
        request.llm_request.instructions.push(InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: prompt.to_string(),
            }],
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::Request;

    const STRONG_PROMPT: &str = "diagnose before you edit";
    const WEAK_PROMPT: &str = "follow the settled plan";

    fn prompts() -> TierPrompts {
        TierPrompts::new(
            Some(STRONG_PROMPT.to_string()),
            Some(WEAK_PROMPT.to_string()),
        )
    }

    /// A state whose routed tier is `tier`, as the decision replay stamps it.
    fn state_on(tier: &str) -> State {
        let mut state = State::default();
        state.extra.insert(
            LAST_TIER_KEY.to_string(),
            StateValue::String(tier.to_string()),
        );
        state
    }

    /// The instruction text the request carries.
    fn instructions(request: &Request) -> Vec<String> {
        request
            .llm_request
            .instructions
            .iter()
            .filter_map(|block| {
                block.content.iter().find_map(|content| match content {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect()
    }

    async fn run(processor: &TierPromptProcessor, state: &mut State) -> Result<Request> {
        let mut request = Request::default();
        processor
            .process(state, Event::ModelRequest(&mut request))
            .await?;
        Ok(request)
    }

    #[tokio::test]
    async fn each_tier_gets_its_own_prompt() -> Result<()> {
        let processor = TierPromptProcessor::new(prompts());
        for (tier, expected) in [("strong", STRONG_PROMPT), ("weak", WEAK_PROMPT)] {
            let request = run(&processor, &mut state_on(tier)).await?;
            assert_eq!(instructions(&request), vec![expected.to_string()]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_unconfigured_tier_is_left_untouched() -> Result<()> {
        let processor =
            TierPromptProcessor::new(TierPrompts::new(Some(STRONG_PROMPT.to_string()), None));
        let request = run(&processor, &mut state_on("weak")).await?;
        assert!(instructions(&request).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn nothing_is_injected_before_a_tier_is_known() -> Result<()> {
        let processor = TierPromptProcessor::new(prompts());
        let request = run(&processor, &mut State::default()).await?;
        assert!(instructions(&request).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn the_prompt_follows_the_client_instructions() -> Result<()> {
        let processor = TierPromptProcessor::new(prompts());
        let mut state = state_on("strong");
        let mut request = Request::default();
        request.llm_request.instructions.push(InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "you are a coding agent".to_string(),
            }],
        });

        processor
            .process(&mut state, Event::ModelRequest(&mut request))
            .await?;

        assert_eq!(
            instructions(&request),
            vec![
                "you are a coding agent".to_string(),
                STRONG_PROMPT.to_string()
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn other_events_inject_nothing() -> Result<()> {
        let processor = TierPromptProcessor::new(prompts());
        let mut state = state_on("strong");
        let mut request = Request::default();

        // Only the post-decision hook knows the tier; the inbound one runs before
        // the cascade has picked anything.
        processor
            .process(&mut state, Event::Request(&mut request))
            .await?;

        assert!(instructions(&request).is_empty());
        Ok(())
    }
}
