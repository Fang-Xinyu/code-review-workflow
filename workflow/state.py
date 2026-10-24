from typing import TypedDict

from schemas import (Inputs, TestResult, FixTask, Attempt, Candidate, ReviewReport,
                     FixAssessment, Goal, Target, Event, Feedback, GeneratedTests, ReviewedFinding)


class WorkflowState(TypedDict, total=False):
    inputs: Inputs
    run_id: str
    original_code: str
    original_hash: str
    working_code: str
    working_hash: str
    tests_hash: str
    tests_origin: str
    generated_tests: GeneratedTests
    requirement_hash: str
    baseline_test: TestResult
    working_test_result: TestResult
    initial_review: ReviewReport
    final_review: ReviewReport
    current_review: ReviewReport
    review_count: int
    review_limit: int
    pending_history: list[ReviewedFinding]
    fix_assessment: FixAssessment
    active_task: FixTask
    active_goals: list[Goal]
    goals: list[Goal]
    targets: list[Target]
    goal_history: list[FixTask]
    included_tasks: list[FixTask]
    handled: list[str]
    attempt_count: int
    max_attempts: int
    attempt_history: list[Attempt]
    candidate: Candidate
    accepted_candidate: Candidate
    status: str
    stop_reason: str
    route: str
    seen_candidate_hashes: list[str]
    feedback: list[Feedback]
    coverage_flags: list[str]
    events: list[Event]
    environment: dict
    tool_calls: int
    tool_stats_complete: bool
