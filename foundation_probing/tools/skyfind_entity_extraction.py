"""Deterministic SkyFind entity extraction for Experiment 2.

The extractor is inspired by LazyMCoT's entity-decomposition idea, but it does
not call an LLM. It keeps the behavior auditable before any optional VLM/LLM
entity extraction is introduced.
"""

import re


COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "dark",
    "dark-colored",
    "gray",
    "green",
    "grey",
    "light",
    "light-colored",
    "orange",
    "purple",
    "red",
    "red-and-yellow",
    "red-and-white",
    "silver",
    "white",
    "white-and-green",
    "yellow",
    "orange-yellow",
    "yellow-and-white",
}

SIZE_WORDS = {
    "big",
    "large",
    "little",
    "long",
    "narrow",
    "short",
    "small",
    "smaller",
    "tiny",
}

APPEARANCE_WORDS = {
    "bright",
    "colored",
    "dark",
    "distant",
    "facing",
    "floating",
    "moving",
    "parked",
    "standing",
    "striped",
    "visible",
}

MODIFIER_WORDS = COLOR_WORDS | SIZE_WORDS | APPEARANCE_WORDS

LEFT_BOUNDARY_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "below",
    "between",
    "by",
    "closest",
    "counted",
    "counting",
    "directly",
    "from",
    "has",
    "in",
    "is",
    "located",
    "near",
    "next",
    "of",
    "on",
    "opposite",
    "positioned",
    "second",
    "slightly",
    "the",
    "their",
    "this",
    "to",
    "towards",
    "upper",
    "wearing",
    "with",
}

ENTITY_HEADS = [
    "traffic sign",
    "traffic light",
    "traffic cone",
    "baseball field",
    "soccer field",
    "parking space",
    "parking spot",
    "parking lot",
    "life jacket",
    "life vest",
    "wetsuit",
    "swimwear",
    "swimsuit",
    "motor boat",
    "motorboat",
    "speed boat",
    "small boat",
    "jet ski",
    "electric scooter",
    "red buoy",
    "orange buoy",
    "yellow buoy",
    "white buoy",
    "ocean wave",
    "ocean waves",
    "planter box",
    "building entrance",
    "double-decker bus",
    "sports car",
    "hatchback",
    "pickup truck",
    "small truck",
    "white truck",
    "black truck",
    "white suv",
    "black suv",
    "white sedan",
    "black sedan",
    "blue sedan",
    "gray sedan",
    "grey sedan",
    "silver sedan",
    "red sedan",
    "dark gray sedan",
    "dark-colored car",
    "light-colored sedan",
    "white car",
    "black car",
    "red car",
    "blue car",
    "silver car",
    "green tricycle",
    "white van",
    "black van",
    "yellow-and-black taxi",
    "red-and-white taxi",
    "streetlamp",
    "street light",
    "crosswalk",
    "sidewalk",
    "walkway",
    "pathway",
    "roundabout",
    "skyscraper",
    "building",
    "bridge",
    "water",
    "road",
    "lane",
    "tree",
    "steps",
    "bus",
    "truck",
    "suv",
    "sedan",
    "vehicle",
    "car",
    "van",
    "taxi",
    "tricycle",
    "bicycle",
    "motorcycle",
    "scooter",
    "boat",
    "ship",
    "vessel",
    "yacht",
    "kayak",
    "buoy",
    "canopy",
    "swimmer",
    "pedestrian",
    "pedestrians",
    "person",
    "people",
    "woman",
    "man",
    "child",
    "adult",
    "passenger",
    "individual",
    "drone",
    "uav",
]


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
HEAD_PATTERNS = [
    (head, re.compile(r"\b" + re.escape(head).replace(r"\ ", r"\s+") + r"s?\b"))
    for head in sorted(ENTITY_HEADS, key=lambda value: (-len(value.split()), -len(value)))
]


def tokenize_with_spans(text):
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in TOKEN_RE.finditer(text.lower())
    ]


def token_indices_for_span(tokens, start, end):
    return [
        index
        for index, token in enumerate(tokens)
        if token["start"] >= start and token["end"] <= end
    ]


def find_matching_tokens(tokens, head):
    head_parts = head.split()
    matches = []
    for index in range(0, len(tokens) - len(head_parts) + 1):
        parts = [tokens[index + offset]["text"] for offset in range(len(head_parts))]
        plural_parts = list(head_parts)
        if not plural_parts[-1].endswith("s"):
            plural_parts[-1] = plural_parts[-1] + "s"
        if parts == head_parts or parts == plural_parts:
            matches.append((index, index + len(head_parts) - 1))
    return matches


def expand_left(tokens, start_index):
    current = start_index
    while current > 0:
        previous = tokens[current - 1]["text"]
        if previous in MODIFIER_WORDS:
            current -= 1
            continue
        break
    return current


def normalize_surface_phrase(text):
    phrase = re.sub(r"\s+", " ", text.lower()).strip(" ,.;:")
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase)
    return phrase


def canonicalize_phrase(surface_phrase, head):
    words = surface_phrase.split()
    head_words = head.split()
    if len(words) <= len(head_words):
        return surface_phrase

    modifiers = words[: -len(head_words)]
    canonical_parts = [head]
    for modifier in modifiers:
        if modifier in COLOR_WORDS:
            canonical_parts.extend(["with", modifier, "color"])
        elif modifier in SIZE_WORDS:
            canonical_parts.extend(["with", modifier, "size"])
        else:
            canonical_parts.extend(["with", modifier, "attribute"])
    return " ".join(canonical_parts)


def overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def extract_entities(expression):
    tokens = tokenize_with_spans(expression)
    candidates = []

    for head, _ in HEAD_PATTERNS:
        for start_index, end_index in find_matching_tokens(tokens, head):
            phrase_start_index = expand_left(tokens, start_index)
            start = tokens[phrase_start_index]["start"]
            end = tokens[end_index]["end"]
            surface = normalize_surface_phrase(expression[start:end])
            if not surface:
                continue
            candidates.append(
                {
                    "surface_phrase": surface,
                    "canonical_phrase": canonicalize_phrase(surface, head),
                    "head": head,
                    "char_start": start,
                    "char_end": end,
                    "token_start": phrase_start_index,
                    "token_end": end_index + 1,
                }
            )

    candidates.sort(
        key=lambda item: (
            item["char_start"],
            -(item["char_end"] - item["char_start"]),
            item["surface_phrase"],
        )
    )

    entities = []
    occupied = []
    seen = set()
    for candidate in candidates:
        if candidate["surface_phrase"] in seen:
            continue
        if any(
            overlap(candidate["char_start"], candidate["char_end"], start, end)
            for start, end in occupied
        ):
            continue
        entities.append(candidate)
        occupied.append((candidate["char_start"], candidate["char_end"]))
        seen.add(candidate["surface_phrase"])

    return entities


def entity_set_text(entities, key="surface_phrase"):
    return ", ".join(entity[key] for entity in entities)


def select_wrong_phrase(target_head):
    vehicle_heads = {
        "bus",
        "car",
        "hatchback",
        "sedan",
        "suv",
        "taxi",
        "truck",
        "van",
        "vehicle",
    }
    person_heads = {
        "adult",
        "child",
        "individual",
        "man",
        "passenger",
        "pedestrian",
        "pedestrians",
        "people",
        "person",
        "swimmer",
        "woman",
    }
    boat_heads = {
        "boat",
        "jet ski",
        "kayak",
        "motor boat",
        "motorboat",
        "ship",
        "vessel",
        "yacht",
    }
    if target_head in vehicle_heads:
        return "person"
    if target_head in person_heads:
        return "white sedan"
    if target_head in boat_heads:
        return "black car"
    return "person"
