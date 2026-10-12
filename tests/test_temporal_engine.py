import pytest
from src.models.memory_schema import MemoryRecord
from src.analytics.temporal_engine import TemporalTrendEngine


def generate_7day_synthetic_records():
    """Generates realistic 7-day synthetic patient records."""
    days = [
        # (date, sleep, ex_type, ex_mins, ex_done, energy, quote)
        ("2026-10-04", 7.0, "walk", 30, True, 4, "Slept 7 hours, walked for 30 minutes, felt great."),
        ("2026-10-05", 6.5, "walk", 20, True, 3, "Slept 6.5 hours, 20 minute walk, normal day."),
        ("2026-10-06", 5.0, "walk", 0, False, 2, "Slept 5 hours, skipped morning walk, feeling very sluggish."),
        ("2026-10-07", 8.0, "jog", 25, True, 5, "Slept 8 hours, morning jog for 25 mins, high energy!"),
        ("2026-10-08", None, None, None, None, None, "Doctor appointment today, forgot to note sleep."),
        ("2026-10-09", 5.5, "walk", 0, False, 2, "Slept 5.5 hours, skipped walk again, tired after lunch."),
        ("2026-10-10", 7.0, "walk", 35, True, 4, "Slept 7 hours, good 35 min walk in the park."),
    ]

    records = []
    for d, sleep, ex_type, ex_mins, ex_done, energy, quote in days:
        extracted = {}
        if sleep is not None:
            extracted["sleep_hours"] = sleep
        if ex_type is not None:
            extracted["exercise_type"] = ex_type
        if ex_mins is not None:
            extracted["exercise_minutes"] = ex_mins
        if ex_done is not None:
            extracted["exercise_completed"] = ex_done
        if energy is not None:
            extracted["energy_score"] = energy

        records.append(MemoryRecord(
            user_id="synth_user",
            date_str=d,
            original_transcript=quote,
            extracted_data=extracted
        ))
    return records


def test_sleep_trends_calculation():
    records = generate_7day_synthetic_records()
    analytics = TemporalTrendEngine.analyze_sleep_trends(records, target_days=7)

    # 6 recorded days out of 7 (2026-10-08 had no sleep data)
    assert analytics["recorded_days"] == 6
    assert analytics["target_days"] == 7
    assert analytics["is_statistically_reliable"] is True

    # Expected average: (7.0 + 6.5 + 5.0 + 8.0 + 5.5 + 7.0) / 6 = 39.0 / 6 = 6.50
    assert analytics["average_hours"] == 6.50
    assert analytics["min_hours"] == 5.0
    assert analytics["max_hours"] == 8.0

    # Ensure evidence is attached with quotes
    assert len(analytics["evidence"]) == 6
    assert analytics["evidence"][0]["quote"] == "Slept 7 hours, walked for 30 minutes, felt great."


def test_exercise_frequency_calculation():
    records = generate_7day_synthetic_records()
    ex = TemporalTrendEngine.analyze_exercise_frequency(records)

    # Completed on 4 days (Oct 4: 30, Oct 5: 20, Oct 7: 25, Oct 10: 35)
    # Total minutes = 30 + 20 + 25 + 35 = 110 minutes
    assert ex["days_completed"] == 4
    assert ex["days_skipped"] == 2  # Oct 6 and Oct 9
    assert ex["total_minutes"] == 110
    # Average minutes when active = 110 / 4 = 27.5
    assert ex["average_minutes_when_active"] == 27.5


def test_missing_data_guardrail():
    # Only 1 record (N < 3)
    sparse_records = [
        MemoryRecord(
            user_id="sparse_user",
            date_str="2026-10-10",
            original_transcript="Slept 6 hours.",
            extracted_data={"sleep_hours": 6.0}
        )
    ]
    analytics = TemporalTrendEngine.analyze_sleep_trends(sparse_records, target_days=7)
    assert analytics["recorded_days"] == 1
    # Must flag as NOT statistically reliable
    assert analytics["is_statistically_reliable"] is False
    assert "Minimum 3 days needed" in analytics["message"]


def test_period_comparison_deltas():
    records = generate_7day_synthetic_records()
    # Split into period A (first 3 days) and period B (last 4 days)
    period_a = records[:3]  # Oct 4 (7.0), Oct 5 (6.5), Oct 6 (5.0) -> avg = 18.5/3 = 6.17
    period_b = records[3:]  # Oct 7 (8.0), Oct 8 (None), Oct 9 (5.5), Oct 10 (7.0) -> avg = 20.5/3 = 6.83

    comparison = TemporalTrendEngine.compare_periods(
        period_a_records=period_a,
        period_b_records=period_b,
        label_a="Week_1",
        label_b="Week_2"
    )

    sleep_comp = comparison["sleep_comparison"]
    assert sleep_comp["Week_1_avg"] == 6.17
    assert sleep_comp["Week_2_avg"] == 6.83
    # Delta should be positive +0.66
    assert sleep_comp["delta_hours"] == 0.66
    assert sleep_comp["percent_change"] is not None
