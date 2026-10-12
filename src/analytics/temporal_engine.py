from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.memory_schema import MemoryRecord


class TemporalTrendEngine:
    """
    Deterministic mathematical analytics layer.
    Performs time-series calculations in pure Python without LLM hallucinations.
    """

    @staticmethod
    def analyze_sleep_trends(records: List[MemoryRecord], target_days: int = 7) -> Dict[str, Any]:
        """Calculates sleep averages, variance, and logs missing days."""
        sleep_data = []
        for r in records:
            hours = r.extracted_data.get("sleep_hours")
            if hours is not None and isinstance(hours, (int, float)):
                sleep_data.append({
                    "date": r.date_str,
                    "hours": float(hours),
                    "record_id": r.record_id,
                    "quote": r.original_transcript
                })

        if not sleep_data:
            return {
                "recorded_days": 0,
                "target_days": target_days,
                "average_hours": None,
                "min_hours": None,
                "max_hours": None,
                "is_statistically_reliable": False,
                "message": "No sleep observations recorded.",
                "evidence": []
            }

        df = pd.DataFrame(sleep_data)
        # In case of multiple logs per day, take the morning/earliest sleep log
        daily_df = df.groupby("date").first().reset_index()

        avg_hours = round(float(daily_df["hours"].mean()), 2)
        min_hours = round(float(daily_df["hours"].min()), 2)
        max_hours = round(float(daily_df["hours"].max()), 2)
        recorded_days = len(daily_df)

        is_reliable = recorded_days >= 3

        evidence = [
            {"date": row["date"], "hours": row["hours"], "quote": row["quote"], "record_id": row["record_id"]}
            for _, row in daily_df.iterrows()
        ]

        return {
            "recorded_days": recorded_days,
            "target_days": target_days,
            "average_hours": avg_hours,
            "min_hours": min_hours,
            "max_hours": max_hours,
            "is_statistically_reliable": is_reliable,
            "message": f"Average sleep of {avg_hours}h across {recorded_days} recorded days."
                       if is_reliable else f"Only {recorded_days} day(s) recorded. Minimum 3 days needed for reliable trend.",
            "evidence": evidence
        }

    @staticmethod
    def analyze_exercise_frequency(records: List[MemoryRecord]) -> Dict[str, Any]:
        """Calculates active days, skipped days, and total minutes."""
        exercise_events = []
        for r in records:
            data = r.extracted_data
            minutes = data.get("exercise_minutes")
            completed = data.get("exercise_completed")
            ex_type = data.get("exercise_type")

            # Check if exercise was mentioned
            if minutes is not None or completed is not None or ex_type is not None:
                is_completed = completed is True or (minutes is not None and minutes > 0)
                is_skipped = completed is False or (minutes == 0)

                exercise_events.append({
                    "date": r.date_str,
                    "type": ex_type or "general exercise",
                    "minutes": int(minutes) if minutes else 0,
                    "completed": is_completed,
                    "skipped": is_skipped,
                    "quote": r.original_transcript,
                    "record_id": r.record_id
                })

        if not exercise_events:
            return {
                "total_exercise_days": 0,
                "days_completed": 0,
                "days_skipped": 0,
                "total_minutes": 0,
                "average_minutes_when_active": 0,
                "evidence": []
            }

        df = pd.DataFrame(exercise_events)
        completed_df = df[df["completed"] == True]

        total_mins = int(completed_df["minutes"].sum())
        days_completed = int(completed_df["date"].nunique())
        days_skipped = int(df[df["skipped"] == True]["date"].nunique())
        avg_mins = round(total_mins / days_completed, 1) if days_completed > 0 else 0

        return {
            "total_exercise_events": len(df),
            "days_completed": days_completed,
            "days_skipped": days_skipped,
            "total_minutes": total_mins,
            "average_minutes_when_active": avg_mins,
            "evidence": exercise_events
        }

    @staticmethod
    def analyze_energy_and_stress(records: List[MemoryRecord]) -> Dict[str, Any]:
        """Aggregates self-reported energy (1-5) and stress (1-5) ratings."""
        ratings = []
        for r in records:
            energy = r.extracted_data.get("energy_score")
            stress = r.extracted_data.get("stress_score")
            symptoms = r.extracted_data.get("symptoms", [])

            if energy is not None or stress is not None or symptoms:
                ratings.append({
                    "date": r.date_str,
                    "energy": int(energy) if energy is not None else None,
                    "stress": int(stress) if stress is not None else None,
                    "symptoms": symptoms,
                    "quote": r.original_transcript,
                    "record_id": r.record_id
                })

        if not ratings:
            return {
                "recorded_days": 0,
                "average_energy": None,
                "average_stress": None,
                "common_symptoms": [],
                "evidence": []
            }

        df = pd.DataFrame(ratings)
        avg_energy = round(float(df["energy"].dropna().mean()), 2) if not df["energy"].dropna().empty else None
        avg_stress = round(float(df["stress"].dropna().mean()), 2) if not df["stress"].dropna().empty else None

        all_symptoms = [s for row in ratings for s in row.get("symptoms", [])]
        symptom_counts = pd.Series(all_symptoms).value_counts().to_dict() if all_symptoms else {}

        return {
            "recorded_days": len(df),
            "average_energy": avg_energy,
            "average_stress": avg_stress,
            "common_symptoms": symptom_counts,
            "evidence": ratings
        }

    @classmethod
    def compare_periods(
        cls,
        period_a_records: List[MemoryRecord],
        period_b_records: List[MemoryRecord],
        label_a: str = "Previous Period",
        label_b: str = "Current Period"
    ) -> Dict[str, Any]:
        """
        Deterministic comparison between two chronological periods.
        Calculates exact delta differences and percentage shifts.
        """
        sleep_a = cls.analyze_sleep_trends(period_a_records)
        sleep_b = cls.analyze_sleep_trends(period_b_records)

        exercise_a = cls.analyze_exercise_frequency(period_a_records)
        exercise_b = cls.analyze_exercise_frequency(period_b_records)

        energy_a = cls.analyze_energy_and_stress(period_a_records)
        energy_b = cls.analyze_energy_and_stress(period_b_records)

        # Compute Sleep Delta
        sleep_delta = None
        sleep_pct_change = None
        if sleep_a["average_hours"] is not None and sleep_b["average_hours"] is not None:
            sleep_delta = round(sleep_b["average_hours"] - sleep_a["average_hours"], 2)
            if sleep_a["average_hours"] > 0:
                sleep_pct_change = round((sleep_delta / sleep_a["average_hours"]) * 100, 1)

        # Compute Exercise Delta
        ex_delta_mins = exercise_b["total_minutes"] - exercise_a["total_minutes"]
        ex_delta_days = exercise_b["days_completed"] - exercise_a["days_completed"]

        # Compute Energy Delta
        energy_delta = None
        if energy_a["average_energy"] is not None and energy_b["average_energy"] is not None:
            energy_delta = round(energy_b["average_energy"] - energy_a["average_energy"], 2)

        return {
            "period_labels": {"a": label_a, "b": label_b},
            "sleep_comparison": {
                f"{label_a}_avg": sleep_a["average_hours"],
                f"{label_b}_avg": sleep_b["average_hours"],
                "delta_hours": sleep_delta,
                "percent_change": sleep_pct_change,
            },
            "exercise_comparison": {
                f"{label_a}_completed_days": exercise_a["days_completed"],
                f"{label_b}_completed_days": exercise_b["days_completed"],
                "delta_completed_days": ex_delta_days,
                "delta_total_minutes": ex_delta_mins,
            },
            "energy_comparison": {
                f"{label_a}_avg_energy": energy_a["average_energy"],
                f"{label_b}_avg_energy": energy_b["average_energy"],
                "delta_energy": energy_delta,
            }
        }
