from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from lean_dojo import Dojo, LeanError, LeanGitRepo, ProofFinished, ProofGivenUp, TacticState, Theorem


@dataclass
class GoalSample:
    theorem: str
    premises: list[str]
    prompt: str


@dataclass
class ProofResult:
    """
    Wrapper for LeanDojo results to maintain backward compatibility.
    Mimics the old ProofResult interface.
    """
    tactic_state: Optional[TacticState]
    proved: bool
    crashed: bool
    error: Optional[str] = None


class LeanProofEnv:
    """
    Thin wrapper over LeanDojo providing reset/step for RL.
    """

    def __init__(
        self,
        repo: str,
        commit: str,
        theorems: Iterable[str],
        max_steps: int = 64,
        workdir: str | Path = ".lean_env",
    ) -> None:
        self.repo = repo
        self.commit = commit
        self.theorems = list(theorems)
        self.max_steps = max_steps
        self.workdir = Path(workdir)
        self._lean_repo = LeanGitRepo(repo, commit)
        self._dojo: Optional[Dojo] = None
        self._state: Optional[TacticState] = None
        self._steps = 0
        self._current_theorem: Optional[str] = None

    def reset(self, theorem_name: str) -> TacticState:
        """
        Reset to a new theorem. Creates a new Dojo instance for the theorem.
        Note: theorem_name should be in format "file_path:theorem_name" or just "theorem_name"
        """
        # Clean up existing Dojo if switching theorems
        if self._dojo is not None and self._current_theorem != theorem_name:
            self._cleanup_dojo()

        # Parse theorem_name - it might be "file:theorem" or just "theorem"
        if ":" in theorem_name:
            file_path, thm_name = theorem_name.rsplit(":", 1)
        else:
            # Assume the first file in the repo if not specified
            # This is a fallback; ideally theorem_name should include file path
            file_path = None
            thm_name = theorem_name

        # Create new Dojo if needed
        if self._dojo is None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            # Create theorem entry - using tuple format if we have file_path
            if file_path:
                theorem = Theorem(self._lean_repo, file_path, thm_name)
            else:
                # Fallback: try to use theorem_name directly
                # This may fail if file path is needed
                theorem = thm_name

            # Initialize Dojo using __enter__() for manual context management
            self._dojo, self._state = Dojo(theorem).__enter__()
            self._current_theorem = theorem_name
        else:
            # Reuse existing Dojo - get initial state for the theorem
            # Note: This assumes we can query states from the same Dojo instance
            # If this doesn't work, we may need to recreate Dojo for each theorem
            if file_path:
                theorem = Theorem(self._lean_repo, file_path, thm_name)
            else:
                theorem = thm_name
            self._dojo, self._state = Dojo(theorem).__enter__()
            self._current_theorem = theorem_name

        self._steps = 0
        return self._state

    def _cleanup_dojo(self) -> None:
        """Clean up the current Dojo instance."""
        if self._dojo is not None:
            try:
                self._dojo.__exit__(None, None, None)
            except:
                pass  # Ignore cleanup errors
            self._dojo = None
            self._state = None

    def step(self, tactic: str) -> tuple[Optional[TacticState], float, bool, ProofResult]:
        """
        Apply a tactic, return next state, reward, done, and lean result.
        """
        assert self._dojo is not None and self._state is not None

        # Run the tactic - returns TacticState, ProofFinished, ProofGivenUp, or LeanError
        raw_result = self._dojo.run_tac(self._state, tactic)
        self._steps += 1

        # Convert to ProofResult for backward compatibility
        if isinstance(raw_result, ProofFinished):
            result = ProofResult(tactic_state=None, proved=True, crashed=False)
            self._state = None
            done = True
            reward = 1.0
        elif isinstance(raw_result, ProofGivenUp):
            result = ProofResult(tactic_state=None, proved=False, crashed=True, error="proof given up")
            self._state = None
            done = True
            reward = 0.0
        elif isinstance(raw_result, LeanError):
            result = ProofResult(tactic_state=self._state, proved=False, crashed=True, error=raw_result.error)
            done = True
            reward = 0.0
        elif isinstance(raw_result, TacticState):
            result = ProofResult(tactic_state=raw_result, proved=False, crashed=False)
            self._state = raw_result
            done = self._steps >= self.max_steps
            reward = 0.0
        else:
            # Unknown result type - treat as error
            result = ProofResult(tactic_state=self._state, proved=False, crashed=True, error=f"unknown result type: {type(raw_result)}")
            done = True
            reward = 0.0

        return result.tactic_state, reward, done, result

    def close(self) -> None:
        """Clean up resources."""
        self._cleanup_dojo()

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.close()
