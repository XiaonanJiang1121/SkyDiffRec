"""Deterministic SkyFind entity extraction for Experiment 2.

The extractor follows LazyMCoT's entity-decomposition idea, but it does not
depend on a fixed positive object lexicon. It extracts noun-like phrases from
the expression with auditable syntactic rules, then places the likely target
entity first and the referring entities after it.
"""

import re


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

DETERMINERS = {
    "a",
    "an",
    "another",
    "each",
    "every",
    "its",
    "several",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "two",
}

ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "topmost",
    "bottommost",
}

QUANTITY_WORDS = {
    "many",
    "multiple",
    "several",
    "single",
    "two",
    "three",
    "four",
    "five",
}

SPATIAL_MODIFIERS = {
    "bottom",
    "bottom-left",
    "bottom-right",
    "central",
    "center",
    "center-left",
    "center-right",
    "left",
    "lower",
    "lower-left",
    "lower-right",
    "middle",
    "right",
    "top",
    "top-left",
    "top-right",
    "upper",
    "upper-left",
    "upper-right",
}

GENERIC_LEADING_WORDS = (
    DETERMINERS
    | ORDINAL_WORDS
    | QUANTITY_WORDS
    | SPATIAL_MODIFIERS
    | {
        "counted",
        "counting",
        "directly",
        "farther",
        "immediate",
        "nearest",
        "next",
        "partially",
        "precisely",
        "slightly",
    }
)

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
    "maroon-colored",
    "orange",
    "purple",
    "red",
    "red-and-white",
    "silver",
    "white",
    "white-and-green",
    "yellow",
    "yellow-and-black",
}

ATTRIBUTE_WORDS = {
    "big",
    "bright",
    "colored",
    "compact",
    "covered",
    "curved",
    "dark",
    "dashed",
    "distant",
    "elderly",
    "elevated",
    "exterior",
    "front",
    "gray-roofed",
    "greenery",
    "horizontal",
    "intact",
    "large",
    "light",
    "long",
    "main",
    "motor",
    "narrow",
    "obscured",
    "orange-yellow",
    "parked",
    "rectangular",
    "rightmost",
    "sandy",
    "semi-trailer",
    "slender",
    "small",
    "striped",
    "tall",
    "uncovered",
    "upright",
    "vertical",
    "visible",
    "white-roofed",
}

DESCRIPTOR_WORDS = COLOR_WORDS | ATTRIBUTE_WORDS

BOUNDARY_WORDS = {
    "above",
    "across",
    "adjacent",
    "ahead",
    "aligned",
    "along",
    "among",
    "and",
    "appearing",
    "appears",
    "are",
    "around",
    "as",
    "at",
    "away",
    "be",
    "because",
    "been",
    "before",
    "behind",
    "being",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "casting",
    "close",
    "closer",
    "closest",
    "counted",
    "counting",
    "directly",
    "down",
    "downward",
    "downwards",
    "dressed",
    "driving",
    "facing",
    "farther",
    "features",
    "followed",
    "from",
    "has",
    "have",
    "having",
    "holding",
    "in",
    "indicates",
    "inside",
    "is",
    "left",
    "leading",
    "located",
    "mounted",
    "moving",
    "near",
    "nearest",
    "next",
    "of",
    "on",
    "opposite",
    "or",
    "parallel",
    "parked",
    "placed",
    "positioned",
    "pointing",
    "pushing",
    "pushed",
    "reads",
    "riding",
    "right",
    "shows",
    "situated",
    "slightly",
    "spanning",
    "standing",
    "starting",
    "surrounded",
    "to",
    "toward",
    "towards",
    "traveling",
    "under",
    "up",
    "upward",
    "upwards",
    "walking",
    "wearing",
    "while",
    "with",
    "without",
}

STOP_HEADS = {
    "alone",
    "area",
    "backdrop",
    "blurry",
    "bottom",
    "center",
    "column",
    "corner",
    "direction",
    "east",
    "edge",
    "far",
    "frame",
    "half",
    "he",
    "his",
    "image",
    "immediately",
    "it",
    "left",
    "line",
    "middle",
    "north",
    "part",
    "position",
    "proximity",
    "right",
    "row",
    "section",
    "side",
    "south",
    "someone",
    "them",
    "there",
    "they",
    "top",
    "upward",
    "upwards",
    "view",
    "when",
    "which",
    "west",
}

GENERIC_HEADS = {
    "object",
    "thing",
}

REFERENCE_PREPOSITIONS = {
    "above",
    "adjacent",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "from",
    "near",
    "of",
    "on",
    "opposite",
    "to",
    "under",
    "with",
}

ATTRIBUTE_CONTEXT_WORDS = {
    "dressed",
    "holding",
    "pushed",
    "pushing",
    "riding",
    "wearing",
}

TARGET_FOLLOW_WORDS = {
    "appears",
    "are",
    "driving",
    "features",
    "has",
    "is",
    "located",
    "moving",
    "parked",
    "positioned",
    "reads",
    "situated",
    "standing",
    "traveling",
    "walking",
}


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


def has_punctuation_boundary(text, previous_token, next_token):
    gap = text[previous_token["end"] : next_token["start"]]
    return any(char in gap for char in ",.;:")


def previous_token(tokens, index):
    return tokens[index - 1]["text"] if index > 0 else None


def next_token(tokens, index):
    return tokens[index + 1]["text"] if index + 1 < len(tokens) else None


def is_boundary(text, tokens, index):
    token = tokens[index]["text"]
    if token in BOUNDARY_WORDS:
        return True
    if index > 0 and has_punctuation_boundary(text, tokens[index - 1], tokens[index]):
        return True
    return False


def can_start_phrase(text, tokens, index):
    token = tokens[index]["text"]
    if token in BOUNDARY_WORDS:
        return False
    if token in GENERIC_LEADING_WORDS and token not in DETERMINERS and token not in DESCRIPTOR_WORDS:
        return False
    if index == 0:
        return True

    prev = previous_token(tokens, index)
    if prev in DETERMINERS:
        return True
    if prev in BOUNDARY_WORDS:
        return True
    if has_punctuation_boundary(text, tokens[index - 1], tokens[index]):
        return True
    return False


def phrase_end(text, tokens, start_index, max_tokens=8):
    end_index = start_index
    while end_index + 1 < len(tokens) and end_index - start_index + 1 < max_tokens:
        probe = end_index + 1
        if is_boundary(text, tokens, probe):
            break
        end_index = probe
    return end_index


def strip_phrase_words(words):
    words = list(words)
    while words and words[0] in GENERIC_LEADING_WORDS:
        words.pop(0)
    while words and words[-1] in GENERIC_LEADING_WORDS:
        words.pop()
    return words


def normalize_surface_phrase(text):
    phrase = re.sub(r"\s+", " ", text.lower()).strip(" ,.;:")
    phrase = re.sub(r"^(a|an|the|this|that|these|those|another)\s+", "", phrase)
    phrase = re.sub(r"^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+", "", phrase)
    return phrase


def singularize(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def phrase_head(words):
    return singularize(words[-1]) if words else ""


def is_valid_phrase(words):
    if not words:
        return False
    head = phrase_head(words)
    if head in STOP_HEADS:
        return False
    if head in GENERIC_HEADS and len(words) == 1:
        return False
    if all(word in DESCRIPTOR_WORDS or word in SPATIAL_MODIFIERS for word in words):
        return False
    if len(words) == 1 and words[0] in (DESCRIPTOR_WORDS | SPATIAL_MODIFIERS):
        return False
    return True


def canonicalize_phrase(surface_phrase, head):
    return surface_phrase


def overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def previous_content_word(tokens, index):
    probe = index - 1
    while probe >= 0:
        text = tokens[probe]["text"]
        if text not in DETERMINERS and text not in ORDINAL_WORDS:
            return text
        probe -= 1
    return None


def next_content_word(tokens, index):
    probe = index + 1
    while probe < len(tokens):
        text = tokens[probe]["text"]
        if text not in DETERMINERS and text not in ORDINAL_WORDS:
            return text
        probe += 1
    return None


def target_score(entity, tokens):
    score = entity["char_start"] / 1000.0
    prev_word = previous_content_word(tokens, entity["token_start"])
    next_word = next_content_word(tokens, entity["token_end"] - 1)
    if prev_word in REFERENCE_PREPOSITIONS:
        score += 20.0
    if prev_word in ATTRIBUTE_CONTEXT_WORDS:
        score += 15.0
    if next_word in TARGET_FOLLOW_WORDS:
        score -= 5.0
    if entity["head"] in GENERIC_HEADS:
        score += 10.0
    return score


def candidate_from_span(expression, tokens, start_index, end_index):
    raw_words = [tokens[index]["text"] for index in range(start_index, end_index + 1)]
    raw_head = phrase_head(raw_words)
    if raw_head in STOP_HEADS or raw_head in GENERIC_HEADS:
        return None
    words = strip_phrase_words(raw_words)
    if not is_valid_phrase(words):
        return None

    leading_trim = len(raw_words) - len(strip_phrase_words(raw_words))
    phrase_start_index = start_index + leading_trim
    phrase_end_index = phrase_start_index + len(words) - 1
    start = tokens[phrase_start_index]["start"]
    end = tokens[phrase_end_index]["end"]
    surface = normalize_surface_phrase(expression[start:end])
    if not surface:
        return None
    head = phrase_head(surface.split())
    return {
        "surface_phrase": surface,
        "canonical_phrase": canonicalize_phrase(surface, head),
        "head": head,
        "char_start": start,
        "char_end": end,
        "token_start": phrase_start_index,
        "token_end": phrase_end_index + 1,
    }


def phrase_contained_by_existing(candidate, entities):
    c_words = candidate["surface_phrase"].split()
    for entity in entities:
        e_words = entity["surface_phrase"].split()
        if candidate["surface_phrase"] == entity["surface_phrase"]:
            return True
        if len(c_words) == 1 and c_words[0] == entity["head"]:
            return True
        if len(e_words) == 1 and e_words[0] == candidate["head"]:
            continue
        if candidate["surface_phrase"] in entity["surface_phrase"]:
            return True
    return False


def extract_entities(expression):
    tokens = tokenize_with_spans(expression)
    candidates = []

    for index in range(len(tokens)):
        if not can_start_phrase(expression, tokens, index):
            continue
        end_index = phrase_end(expression, tokens, index)
        candidate = candidate_from_span(expression, tokens, index, end_index)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item["char_start"],
            -(item["char_end"] - item["char_start"]),
            item["surface_phrase"],
        )
    )

    entities = []
    occupied = []
    for candidate in candidates:
        if phrase_contained_by_existing(candidate, entities):
            continue
        if any(
            overlap(candidate["char_start"], candidate["char_end"], start, end)
            for start, end in occupied
        ):
            continue
        entities.append(candidate)
        occupied.append((candidate["char_start"], candidate["char_end"]))

    if not entities:
        return []

    target = min(entities, key=lambda item: target_score(item, tokens))
    referring = [entity for entity in entities if entity is not target]
    referring.sort(key=lambda item: item["char_start"])
    return [target] + referring


def entity_set_text(entities, key="surface_phrase"):
    return ", ".join(entity[key] for entity in entities)


def select_wrong_phrase(target_head):
    head = singularize(target_head)
    if head in {"adult", "child", "man", "pedestrian", "people", "person", "woman"}:
        return "white sedan"
    if head in {"boat", "kayak", "ship", "vessel", "yacht"}:
        return "black car"
    if head in {"bus", "car", "sedan", "suv", "taxi", "truck", "van", "vehicle"}:
        return "person"
    return "person"
