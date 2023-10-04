def combine_consecutive_lists(data):
    if len(data) > 0:
        # Normalize data to ensure all elements are lists
        data = [[x] if not isinstance(x, list) else x for x in data]

        # Sort the data based on the first element of each sublist
        data.sort(key=lambda x: x[0])

        combined_lists = [data[0]]

        for sublist in data[1:]:
            # Check if the last number of the previous sublist is consecutive to the first number of the current sublist
            if sublist[0] - combined_lists[-1][-1] == 1:
                # If the current sublist is single-item and the previous sublist is also single-item, combine them
                if len(sublist) == len(combined_lists[-1]) == 1:
                    combined_lists[-1].extend(sublist)
                # If the current sublist is single-item but the previous sublist is multi-item, append it
                elif len(sublist) == 1 and len(combined_lists[-1]) > 1:
                    combined_lists[-1].append(sublist[0])
                # Otherwise, start a new group
                else:
                    combined_lists.append(sublist)
            else:
                combined_lists.append(sublist)

        return combined_lists
    else:
        return []


def prop1_u_prop2_grouping(my_list):
    # To hold the indices of non-None elements
    groups = []
    current_group = []

    for i, elem in enumerate(my_list):
        if elem is not None:
            current_group.append(i)
        elif current_group:
            groups.append(current_group)
            current_group = []

    # Append the last group if it exists
    if current_group:
        groups.append(current_group)
    return groups


if __name__ == "__main__":
    # data = [1, 2, [3], [4, 5], 9, 21]
    # # data = [[2, 4, 6], [9, 21]]
    # # data = [[1], [2], [3], [5], [7], [9], [10], [21]]
    # print(combine_consecutive_lists(data))
    my_list = [
        ["book", "toilet"],
        ["book", "toilet"],
        ["book", "toilet"],
        None,
        ["book", "toilet"],
        "hello",
        None,
        None,
        None,
        None,
        ["book", "toilet"],
        ["book", "toilet"],
        None,
        "hello",
        None,
    ]
    print(prop1_u_prop2_grouping(my_list))
