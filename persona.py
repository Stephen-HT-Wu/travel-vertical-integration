"""Traveler persona: the profile that drives every agent's decisions."""
import argparse
from typing import Literal

from pydantic import BaseModel

AGE_GROUPS = ["18-25", "26-35", "36-50", "51+"]
GENDERS = ["male", "female", "unspecified"]
TRIP_LENGTHS = ["half_day", "one_day", "multi_day"]


class Persona(BaseModel):
    age_group: Literal["18-25", "26-35", "36-50", "51+"] = "26-35"
    gender: Literal["male", "female", "unspecified"] = "unspecified"
    home_location: str
    destination_location: str
    trip_length_type: Literal["half_day", "one_day", "multi_day"] = "one_day"
    days: int = 1
    party_size: int = 1
    companion_notes: str = ""

    def summary_zh(self) -> str:
        length_label = {
            "half_day": "半日遊",
            "one_day": "一日遊",
            "multi_day": f"{self.days} 天多日遊",
        }[self.trip_length_type]
        gender_label = {"male": "男性", "female": "女性", "unspecified": "未指定性別"}[self.gender]
        summary = (
            f"{self.age_group} 歲、{gender_label}、從 {self.home_location} 出發前往 {self.destination_location}、"
            f"{length_label}、{self.party_size} 人同行"
        )
        if self.companion_notes.strip():
            summary += f"，同行需求：{self.companion_notes.strip()}"
        return summary


def build_persona_from_args(args: argparse.Namespace) -> Persona:
    return Persona(
        age_group=args.age_group,
        gender=args.gender,
        home_location=args.location,
        destination_location=args.destination,
        trip_length_type=_trip_length_flag_to_type(args.trip_length),
        days=args.days,
        party_size=args.party_size,
        companion_notes=args.companion_notes,
    )


def _trip_length_flag_to_type(flag: str) -> str:
    return {"half-day": "half_day", "one-day": "one_day", "multi-day": "multi_day"}[flag]


def add_persona_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--age-group", choices=AGE_GROUPS, default="26-35")
    parser.add_argument("--gender", choices=GENDERS, default="unspecified")
    parser.add_argument("--location", required=True, help="Departure / home city, e.g. 台北")
    parser.add_argument("--destination", required=True, help="Long-distance travel destination, e.g. 礁溪")
    parser.add_argument(
        "--trip-length", choices=["half-day", "one-day", "multi-day"], default="one-day"
    )
    parser.add_argument(
        "--days", type=int, default=1, help="Number of days, only meaningful for --trip-length multi-day"
    )
    parser.add_argument("--party-size", type=int, default=1)
    parser.add_argument(
        "--companion-notes", default="",
        help="特殊同行需求，例如長輩/幼兒/寵物同行（僅供 schema 對齊，CLI 不主動詢問）",
    )
