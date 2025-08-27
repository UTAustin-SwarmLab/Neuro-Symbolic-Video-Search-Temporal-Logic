def process_proposition_set(proposition_set: list[str]) -> list:
    """Process proposition set."""
    new_set = []
    for proposition in proposition_set:
        new_set.append(proposition.replace(" ", "_"))
    return new_set
