// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signal-driven stage routing for coding agents.
//!
//! [`StageRouter`] is the assembled algorithm: a [`FallThrough`] pre-wired with
//! the tool-signal processor that reads each turn's tool results and the
//! [`StageClassifier`] that scores them onto the strong/weak tiers.
//!
//! Signals do not decide every turn. When the scorer is not confident the
//! classifier abstains, and the cascade falls through to whatever is configured
//! behind it — the same [`LlmTaskClassifier`] the capability route runs, joined
//! in unchanged, and finally the tier the turn fell open to. The judge is
//! consulted per turn and its verdict is not pinned to the session: an
//! under-threshold turn is a fresh question, so nothing carries a stale answer
//! forward.
//!
//! Callers that need a different composition can still build the `FallThrough`
//! from these parts themselves.

use std::sync::Arc;

use async_trait::async_trait;

use super::util::handoff_notes::HandoffNoteConfig;
use super::util::stage_router::{PickerMode, StageClassifier, DEFAULT_TARGET_KEY};
use super::util::tier_prompts::{TierPromptProcessor, TierPrompts};
use super::util::tool_signals::ToolSignalProcessor;
use super::{FallThrough, LlmTaskClassifier};
use crate::{
    Algorithm, Classification, Classifier, Context, Driver, LibsyError, LlmTarget, LlmTargetSet,
    Request, Response, Result, Score, SharedState, State, StateValue, DEFAULT_RECENT_WINDOW,
};

/// The judge consulted when the signals are not decisive.
pub struct LlmFallback {
    /// Target the judge model is called through. It is not a routing
    /// destination, so it does not belong in the router's target set.
    pub judge_target: LlmTarget,
    /// Lowest `p_solve` the judge must report to route to the weak tier.
    pub threshold: f64,
}

/// How a [`StageRouter`] scores turns, and what it hands the model it picks.
pub struct StageRouterConfig {
    /// Tier a turn falls open to when the scorer is not confident.
    pub mode: PickerMode,
    /// How much corroboration a decisive pick needs, in `[0.0, 1.0]`.
    pub confidence_threshold: f64,
    /// Trailing tool results the signals are computed over. `None` uses
    /// [`DEFAULT_RECENT_WINDOW`].
    pub recent_window: Option<usize>,
    /// Note handed to the model on a signal-driven escalation, and on a
    /// hand-back to the weak tier when a de-escalation note is configured.
    pub handoff_notes: Option<HandoffNoteConfig>,
    /// System prompt handed to each tier, on every turn it serves.
    pub tier_prompts: Option<TierPrompts>,
    /// Judge consulted on turns the signals leave undecided.
    pub llm_fallback: Option<LlmFallback>,
}

impl StageRouterConfig {
    /// The signal-only configuration: no notes, no per-tier prompts, no judge.
    /// Set the optional fields to add them.
    pub fn new(mode: PickerMode, confidence_threshold: f64) -> Self {
        Self {
            mode,
            confidence_threshold,
            recent_window: None,
            handoff_notes: None,
            tier_prompts: None,
            llm_fallback: None,
        }
    }
}

/// Routes coding-agent turns between a strong and a weak tier from tool-result
/// signals. A thin wrapper over the [`FallThrough`] it composes.
pub struct StageRouter {
    inner: Arc<FallThrough>,
}

/// Terminal classifier resolving the tier an under-threshold turn falls open to.
///
/// [`StageClassifier`] leaves such a turn *ambiguous* — its own score is not
/// decisive enough to route on, so anything configured behind it gets to decide
/// first. This closes the cascade with the default tier that turn recorded, so
/// the router never abstains, with or without a judge.
struct FallOpen;

#[async_trait]
impl Classifier for FallOpen {
    async fn score(
        &self,
        state: &mut State,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<Classification> {
        let Some(StateValue::String(target)) = state.extra.get(DEFAULT_TARGET_KEY) else {
            return Err(LibsyError::AlgorithmError {
                message: "stage classifier left no default target to fall open to".to_string(),
            });
        };
        Ok(Classification::Scores(vec![Score {
            target: target.clone(),
            confidence: 0.0,
        }]))
    }
}

impl StageRouter {
    /// Routes over `targets`, which must hold the `strong` and `weak` tiers the
    /// classifier scores onto.
    ///
    /// Errors if either threshold in `config` is outside `[0.0, 1.0]` or a tier
    /// is missing from `targets`.
    pub fn new(targets: LlmTargetSet, config: StageRouterConfig) -> Result<Self> {
        if !(0.0..=1.0).contains(&config.confidence_threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "confidence_threshold must be between 0 and 1, got {}",
                    config.confidence_threshold
                ),
            });
        }
        // The classifier scores onto these two names, so a missing tier would
        // only surface as a failed target lookup mid-turn.
        let strong = targets.get_target("strong")?;
        let weak = targets.get_target("weak")?;

        let mut classifier = StageClassifier::new(config.mode, config.confidence_threshold);
        if let Some(notes) = config.handoff_notes {
            classifier = classifier.with_handoff_notes(notes);
        }
        let signals = ToolSignalProcessor {
            recent_window: config.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
        };

        let mut router = FallThrough::new(targets)
            .with_processor(Arc::new(signals))
            .with_classifier(Arc::new(classifier));
        if let Some(fallback) = config.llm_fallback {
            // The capability judge, as the capability route builds it: the weak
            // tier is the efficient target, the strong tier the capable one.
            router = router.with_classifier(Arc::new(LlmTaskClassifier::new(
                fallback.judge_target,
                weak,
                strong,
                fallback.threshold,
            )?));
        }
        router = router.with_classifier(Arc::new(FallOpen));
        if let Some(prompts) = config.tier_prompts {
            // Runs on the post-decision hook, so it applies to the tier the
            // cascade settled on, whichever classifier picked it.
            router = router.with_processor(Arc::new(TierPromptProcessor::new(prompts)));
        }

        Ok(Self {
            inner: Arc::new(router),
        })
    }
}

#[async_trait]
impl Algorithm<SharedState> for StageRouter {
    fn name(&self) -> &str {
        "stage_router"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context<SharedState>,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.inner
            .clone()
            .create_run_task(ctx, driver, request)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(names: &[&str]) -> LlmTargetSet {
        LlmTargetSet::new(
            names
                .iter()
                .map(|name| LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: None,
                })
                .collect(),
        )
    }

    fn config() -> StageRouterConfig {
        StageRouterConfig::new(PickerMode::EfficientFirst, 0.5)
    }

    #[test]
    fn rejects_an_out_of_range_confidence_threshold() {
        let mut config = config();
        config.confidence_threshold = 1.5;
        assert!(matches!(
            StageRouter::new(targets(&["strong", "weak"]), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_an_out_of_range_judge_threshold() {
        let mut config = config();
        config.llm_fallback = Some(LlmFallback {
            judge_target: LlmTarget {
                semantic_name: "judge".to_string(),
                llm_client: None,
            },
            threshold: -0.1,
        });
        assert!(matches!(
            StageRouter::new(targets(&["strong", "weak"]), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_a_target_set_missing_a_tier() {
        assert!(matches!(
            StageRouter::new(targets(&["strong"]), config()),
            Err(LibsyError::TargetNotFound { .. })
        ));
    }

    #[test]
    fn builds_over_both_tiers() -> Result<()> {
        let router = StageRouter::new(targets(&["strong", "weak"]), config())?;
        assert_eq!(router.name(), "stage_router");
        Ok(())
    }
}
