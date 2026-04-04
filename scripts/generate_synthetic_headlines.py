import csv
import os
import random
from itertools import cycle
from typing import Dict, List, Tuple

random.seed(42)

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_headlines.csv")

TOPICS = [
    "Environment and renewable energy",
    "Health",
    "Science",
    "Community and people",
    "Animals",
    "Sports",
    "Other",
]

HEDGES = [
    "may",
    "could",
    "appears to",
    "reportedly",
    "on track to",
    "set to",
    "poised to",
    "expected to",
    "quietly",
    "surprisingly",
    "after years",
    "amid debate",
    "despite hurdles",
    "at last",
    "at first glance",
]

LEADS = [
    "Local",
    "Regional",
    "National",
    "Global",
    "Small-town",
    "University",
    "Community",
    "Grassroots",
    "Coalition",
    "Volunteer",
    "Family",
    "Youth",
    "Student",
    "Startup",
    "Public",
    "Private",
]

AMBIGUITIES = [
    "according to locals",
    "sources say",
    "early signs",
    "initial reports",
    "it seems",
    "observers note",
    "onlookers say",
    "by some accounts",
]

UPLIFTING_TEMPLATES: Dict[str, List[str]] = {
    "Environment and renewable energy": [
        "{lead} co-op {hedge} flips old parking lot into solar microgrid",
        "{lead} project {hedge} plants native trees along bus corridors",
        "Wind array {hedge} keeps lights on during storm season",
        "Neighborhood roofs {hedge} supply surplus power to food bank",
        "Farmers market {hedge} pilots solar-powered cold storage",
    ],
    "Health": [
        "Mobile clinic {hedge} adds weekend hours for new parents",
        "Nurses {hedge} launch texting line for quick triage",
        "Lab team {hedge} shares early data on one-shot flu booster",
        "Town pharmacists {hedge} run pop-up blood pressure checks",
        "Teen volunteers {hedge} guide elders through telehealth signups",
    ],
    "Science": [
        "Citizen scientists {hedge} map light pollution block by block",
        "Battery prototype {hedge} charges buses in minutes during trials",
        "Open hardware telescope {hedge} finds faint exoplanet hint",
        "Reef DNA library {hedge} helps match corals to cooler waters",
        "Bioplastic circuit board {hedge} survives drop tests",
    ],
    "Community and people": [
        "Neighbors {hedge} repair storm damage on shared block in a day",
        "Barbershop library {hedge} swaps haircuts for kids' book reports",
        "Cafe {hedge} keeps pay-what-you-can nights after full houses",
        "High school build crew {hedge} designs inclusive playground",
        "Librarians {hedge} bike story time to porches",
    ],
    "Animals": [
        "Wildlife overpass {hedge} cuts collisions on canyon road",
        "Rescue boats {hedge} release rehabbed turtles at sunrise",
        "Barn owls {hedge} replace pesticides in trial vineyards",
        "Shelter {hedge} matches senior pets with fosters",
        "Citizen logs {hedge} show orcas revisiting cleaned fjord",
    ],
    "Sports": [
        "Underdog team {hedge} edges into finals after comeback",
        "Marathon {hedge} adds inclusive heats and sells out",
        "Youth league {hedge} doubles signups after scholarship push",
        "Park rink {hedge} opens free night lessons",
        "Veterans climb club {hedge} clears new adaptive route",
    ],
    "Other": [
        "Food bank kitchen {hedge} turns surplus into freezer meals",
        "Rural mesh network {hedge} brings stable Wi-Fi to farms",
        "Makerspace {hedge} trains newcomers for 3D-print jobs",
        "Prison ed cohort {hedge} celebrates first AA degrees",
        "Alley art walk {hedge} brightens downtown weekends",
    ],
}

NOT_TEMPLATES: Dict[str, List[str]] = {
    "Environment and renewable energy": [
        "Spill response {hedge} struggles as oil reaches marsh",
        "Reservoir levels {hedge} dip near record lows",
        "Coal retrofit {hedge} delayed over budget fight",
        "Illegal dumping {hedge} clouds river cleanup plans",
        "Heatwave {hedge} strains aging grid",
    ],
    "Health": [
        "Clinic staffing {hedge} stretched, patients wait",
        "Asthma cases {hedge} rise near busy corridor",
        "Drug recall {hedge} widens after new batch tests",
        "Rural hospital {hedge} pauses surgeries",
        "Flu surge {hedge} closes several schools",
    ],
    "Science": [
        "Deep-space probe {hedge} loses main signal",
        "Research data {hedge} exposed in breach inquiry",
        "Budget cuts {hedge} threaten climate monitors",
        "Lab mishap {hedge} triggers evacuation",
        "Launch window {hedge} scrubs again",
    ],
    "Community and people": [
        "Bus routes {hedge} suspended, commuters stranded",
        "Rent hikes {hedge} push families out",
        "Food pantry {hedge} shelves thin after donation drop",
        "Water boil notice {hedge} extends into weekend",
        "Youth center {hedge} closes early over safety review",
    ],
    "Animals": [
        "Avian flu {hedge} leads to flock culls",
        "Bat counts {hedge} fall after cave disturbance",
        "Wildfire smoke {hedge} drives elk toward suburbs",
        "Driftnets {hedge} injure protected dolphins",
        "Zoo habitat {hedge} opening pushed back",
    ],
    "Sports": [
        "Star forward {hedge} out for season",
        "Stadium upgrade {hedge} over budget again",
        "Youth season {hedge} canceled over referee shortage",
        "Heat warning {hedge} postpones tournament",
        "Club {hedge} faces sanctions for salary cap mixup",
    ],
    "Other": [
        "Cyberattack {hedge} knocks out town Wi-Fi",
        "Postal delays {hedge} leave medicine late",
        "Downtown blackout {hedge} stalls businesses",
        "Rail strike {hedge} strands commuters",
        "Sprinkler failure {hedge} closes museum wing",
    ],
}


def allocate_per_topic(total: int) -> Dict[str, int]:
    base = total // len(TOPICS)
    remainder = total - base * len(TOPICS)
    return {topic: base + (i < remainder) for i, topic in enumerate(TOPICS)}


def generate_rows() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    per_topic = allocate_per_topic(1000)
    for label, templates in [("uplifting", UPLIFTING_TEMPLATES), ("not_uplifting", NOT_TEMPLATES)]:
        for topic in TOPICS:
            count = per_topic[topic]
            templ_cycle = cycle(templates[topic])
            for _ in range(count):
                template = next(templ_cycle)
                hedge = random.choice(HEDGES)
                lead = random.choice(LEADS)
                ambiguity = random.choice(AMBIGUITIES)
                headline = template.format(hedge=hedge, lead=lead)
                # light ambiguity appended without symbols
                headline = f"{headline} {ambiguity}"
                # Normalize spacing
                headline = " ".join(headline.split())
                rows.append((topic, headline, label))
    return rows


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = generate_rows()
    # Verify size
    assert len(rows) == 2000, f"Expected 2000 rows, got {len(rows)}"
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "news_headline", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
