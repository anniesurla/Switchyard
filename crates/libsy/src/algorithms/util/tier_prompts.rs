// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-tier system prompts for stage routing.
//!
//! A strong and a weak model rarely want the same instructions: the capable tier
//! is worth telling to slow down and diagnose, the efficient one to stay on the
//! settled plan. [`TierPromptProcessor`] prepends the routed tier's prompt, so
//! the instruction follows the model rather than the session.
//!
//! Unlike a handoff note, it applies on every turn that tier serves, however the
//! tier was chosen — signals, the judge, or falling open.

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, InstructionBlock, Role};

use super::stage_router::Tier;
use crate::{Event, Processor, Result, State, StateValue};

/// `State.extra` key bridging this processor's two hooks: the decision replay
/// records the routed tier, the outbound request hook reads it back.
const ROUTED_TIER_KEY: &str = "routed_tier";

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
        match Tier::from_target_name(tier)? {
            Tier::Capable => self.strong.as_deref(),
            Tier::Efficient => self.weak.as_deref(),
        }
    }
}

/// Prepends the routed tier's system prompt to the outbound request.
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
        match event {
            // Replayed once the whole cascade has run, so this is the tier the
            // turn really routed to — whichever classifier picked it. Recorded
            // only when there is a prompt for the request hook to apply.
            Event::Decision(decision) => {
                let tier = decision.selected_model();
                if self.prompts.prompt_for(tier).is_some() {
                    state.extra.insert(
                        ROUTED_TIER_KEY.to_string(),
                        StateValue::String(tier.to_string()),
                    );
                }
            }
            Event::ModelRequest(request) => {
                let Some(StateValue::String(tier)) = state.extra.get(ROUTED_TIER_KEY) else {
                    return Ok(());
                };
                let Some(prompt) = self.prompts.prompt_for(tier) else {
                    return Ok(());
                };
                // Ahead of the client's own instructions, so the tier framing is
                // what the model reads first.
                request.llm_request.instructions.insert(
                    0,
                    InstructionBlock {
                        role: Role::System,
                        content: vec![ContentBlock::Text {
                            text: prompt.to_string(),
                        }],
                    },
                );
            }
            _ => {}
        }
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
            ROUTED_TIER_KEY.to_string(),
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
    async fn the_prompt_leads_the_client_instructions() -> Result<()> {
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
                STRONG_PROMPT.to_string(),
                "you are a coding agent".to_string()
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_inbound_request_is_left_alone() -> Result<()> {
        let processor = TierPromptProcessor::new(prompts());
        let mut state = state_on("strong");
        let mut request = Request::default();

        // The inbound hook runs before the cascade has picked anything.
        processor
            .process(&mut state, Event::Request(&mut request))
            .await?;

        assert!(instructions(&request).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn the_decision_replay_records_the_routed_tier() -> Result<()> {
        struct FakeDecision;
        impl crate::Decision for FakeDecision {
            fn selected_model(&self) -> &str {
                "strong"
            }
            fn reasoning(&self) -> Option<&str> {
                None
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let processor = TierPromptProcessor::new(prompts());
        let mut state = State::default();
        processor
            .process(&mut state, Event::Decision(&FakeDecision))
            .await?;

        let request = run(&processor, &mut state).await?;
        assert_eq!(instructions(&request), vec![STRONG_PROMPT.to_string()]);
        Ok(())
    }
}
