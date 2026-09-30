"""
Decision Hysteresis State Machine.

Architectural Design:
    Prevents UI badge flickering and false-alarm twitching across transient frame drops or
    brief biometric fluctuations by enforcing asymmetric hysteresis across temporal evaluation windows.
"""

from collections import deque
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class HysteresisState:
    badge: str              # "CONSISTENT", "UNCERTAIN", "LIKELY SYNTHETIC"
    recent_probabilities: List[float]
    high_prob_count: int    # Number of windows with P >= 0.75 in last 5
    low_prob_count: int     # Number of windows with P <= 0.50 in last 5
    transition_occurred: bool


class HysteresisFilter:
    """
    Asymmetric temporal hysteresis filter:
      - Enter LIKELY SYNTHETIC when P >= 0.75 in at least 3 of 5 windows.
      - Exit LIKELY SYNTHETIC to CONSISTENT when P <= 0.50 in at least 4 of 5 windows.
      - Otherwise UNCERTAIN.
    """

    def __init__(
        self,
        window_size: int = 5,
        enter_threshold: float = 0.75,
        enter_min_windows: int = 3,
        exit_threshold: float = 0.50,
        exit_min_windows: int = 4,
    ):
        self.window_size = window_size
        self.enter_threshold = enter_threshold
        self.enter_min_windows = enter_min_windows
        self.exit_threshold = exit_threshold
        self.exit_min_windows = exit_min_windows

        self._history: deque[float] = deque(maxlen=window_size)
        self._current_state = "CONSISTENT"

    def update(self, probability: float) -> HysteresisState:
        """
        Record the newest window probability and calculate the updated trust state.
        """
        prob = max(0.0, min(1.0, float(probability)))
        self._history.append(prob)

        high_count = sum(1 for p in self._history if p >= self.enter_threshold)
        low_count = sum(1 for p in self._history if p <= self.exit_threshold)

        prev_state = self._current_state
        new_state = prev_state

        if prev_state == "LIKELY SYNTHETIC":
            # Condition to EXIT alert state: must have P <= exit_threshold on at least exit_min_windows
            if low_count >= self.exit_min_windows:
                new_state = "CONSISTENT"
            else:
                new_state = "LIKELY SYNTHETIC"
        elif prev_state == "CONSISTENT":
            # Condition to ENTER alert state: must have P >= enter_threshold on at least enter_min_windows
            if high_count >= self.enter_min_windows:
                new_state = "LIKELY SYNTHETIC"
            elif prob > self.exit_threshold:
                new_state = "UNCERTAIN"
            else:
                new_state = "CONSISTENT"
        else: # UNCERTAIN
            if high_count >= self.enter_min_windows:
                new_state = "LIKELY SYNTHETIC"
            elif low_count >= self.exit_min_windows:
                new_state = "CONSISTENT"
            else:
                new_state = "UNCERTAIN"

        self._current_state = new_state
        transition = (prev_state != new_state)

        return HysteresisState(
            badge=new_state,
            recent_probabilities=list(self._history),
            high_prob_count=high_count,
            low_prob_count=low_count,
            transition_occurred=transition,
        )

    def reset(self):
        """Reset historical window queue."""
        self._history.clear()
        self._current_state = "CONSISTENT"
